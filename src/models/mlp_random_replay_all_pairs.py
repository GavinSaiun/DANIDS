import argparse
import copy
import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PAIR_DIR


# ============================================================
# Experiment constants
# ============================================================
DATASETS = [
    "NF-UNSW-NB15-v3",
    "NF-ToN-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3",
]

DEFAULT_SEEDS = [42, 123, 456]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# Reproducibility / loading
# ============================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_directed_pairs() -> list[str]:
    return [
        f"{src}__TO__{tgt}"
        for src, tgt in permutations(DATASETS, 2)
    ]


def load_pair(pair_dir: Path) -> dict[str, np.ndarray]:
    required = [
        "Xs_train", "ys_train",
        "Xs_val", "ys_val",
        "Xs_test", "ys_test",
        "Xt_adapt", "yt_adapt",
        "Xt_test", "yt_test",
    ]

    data = {}
    for name in required:
        path = pair_dir / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        data[name] = np.load(path)

    return data


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ============================================================
# Model / metrics
# ============================================================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    chunks = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())

    return np.concatenate(chunks)


def compute_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> dict:
    probs = predict_proba(model, X, device)
    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y, probs)),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    label: str,
    val_name: str,
) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        train_loss = total_loss / max(total_n, 1)
        val_metrics = compute_metrics(model, X_val, y_val, device)
        val_auroc = val_metrics["auroc"]

        print(
            f"[{label}] Epoch {epoch + 1:02d} | "
            f"loss={train_loss:.4f} | "
            f"{val_name}_auroc={val_auroc:.4f} | "
            f"{val_name}_f1={val_metrics['f1']:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{label}] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
    }


# ============================================================
# Sampling helpers
# ============================================================
def stratified_sample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    selected_parts = []

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]

        if len(idx) < samples_per_class:
            raise ValueError(
                f"Class {cls} has only {len(idx)} rows, "
                f"but {samples_per_class} were requested."
            )

        chosen = rng.choice(
            idx,
            size=samples_per_class,
            replace=False,
        )
        selected_parts.append(chosen)

    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)

    return X[selected], y[selected], selected


def split_target_budget(
    X_budget: np.ndarray,
    y_budget: np.ndarray,
    random_state: int,
    val_fraction: float = 0.20,
):
    return train_test_split(
        X_budget,
        y_budget,
        test_size=val_fraction,
        stratify=y_budget,
        random_state=random_state,
    )


def build_adaptation_set(
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_replay: np.ndarray | None,
    y_replay: np.ndarray | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X_replay is None or len(X_replay) == 0:
        X = X_target_train.copy()
        y = y_target_train.copy()
    else:
        X = np.vstack([X_target_train, X_replay])
        y = np.concatenate([y_target_train, y_replay])

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(y))

    return X[order], y[order]


# ============================================================
# Analysis metrics
# ============================================================
def forgetting_stats(
    source_before_auroc: float,
    source_after_auroc: float,
) -> dict:
    drop = source_before_auroc - source_after_auroc

    return {
        "absolute_source_auroc_drop": float(drop),
        "relative_source_auroc_drop_pct": float(
            100.0 * drop / source_before_auroc
            if source_before_auroc != 0 else np.nan
        ),
    }


def joint_scores(
    target_auroc: float,
    source_auroc: float,
) -> dict:
    mean_auroc = (target_auroc + source_auroc) / 2.0

    harmonic = (
        2.0 * target_auroc * source_auroc
        / (target_auroc + source_auroc)
        if (target_auroc + source_auroc) > 0
        else np.nan
    )

    return {
        "mean_source_target_auroc": float(mean_auroc),
        "harmonic_source_target_auroc": float(harmonic),
        "worst_domain_auroc": float(
            min(target_auroc, source_auroc)
        ),
    }


# ============================================================
# One pair / one seed
# ============================================================
def run_pair_seed(
    pair: str,
    seed: int,
    data: dict,
    stage1_variant: str,
    target_budget_per_class: int,
    replay_per_class: int,
    batch_size: int,
    source_epochs: int,
    adapt_epochs: int,
    source_lr: float,
    adapt_lr: float,
    weight_decay: float,
    patience: int,
    dropout: float,
    hidden_dims: tuple[int, ...],
    device: torch.device,
) -> list[dict]:

    print("\n" + "=" * 100)
    print(f"PAIR: {pair} | SEED: {seed}")
    print("=" * 100)

    set_seed(seed)

    input_dim = data["Xs_train"].shape[1]

    # --------------------------------------------------------
    # 1) Train source model ONCE for this pair+seed
    # --------------------------------------------------------
    source_model = MLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    source_loader = make_loader(
        data["Xs_train"],
        data["ys_train"],
        batch_size=batch_size,
        shuffle=True,
    )

    source_training = train_model(
        model=source_model,
        train_loader=source_loader,
        X_val=data["Xs_val"],
        y_val=data["ys_val"],
        device=device,
        lr=source_lr,
        weight_decay=weight_decay,
        max_epochs=source_epochs,
        patience=patience,
        label=f"{pair}_source_seed_{seed}",
        val_name="source_val",
    )

    source_before = compute_metrics(
        source_model,
        data["Xs_test"],
        data["ys_test"],
        device,
    )

    target_zero_shot = compute_metrics(
        source_model,
        data["Xt_test"],
        data["yt_test"],
        device,
    )

    source_state = copy.deepcopy(source_model.state_dict())

    # --------------------------------------------------------
    # 2) Fixed target adaptation subset / split
    # --------------------------------------------------------
    (
        X_target_budget,
        y_target_budget,
        _,
    ) = stratified_sample_per_class(
        data["Xt_adapt"],
        data["yt_adapt"],
        samples_per_class=target_budget_per_class,
        random_state=seed + 1_000,
    )

    (
        X_target_train,
        X_target_val,
        y_target_train,
        y_target_val,
    ) = split_target_budget(
        X_target_budget,
        y_target_budget,
        random_state=seed + 2_000,
        val_fraction=0.20,
    )

    # --------------------------------------------------------
    # 3) Fixed random source replay memory
    # --------------------------------------------------------
    (
        X_replay,
        y_replay,
        _,
    ) = stratified_sample_per_class(
        data["Xs_train"],
        data["ys_train"],
        samples_per_class=replay_per_class,
        random_state=seed + 3_000,
    )

    results = []

    # --------------------------------------------------------
    # 4) Compare naive fine-tuning vs random replay
    #
    # FAIRNESS CONTROL:
    # For the same pair+seed, both conditions:
    # - start from exact same source weights
    # - use exact same target adaptation train/val split
    # - use exact same adaptation RNG seed
    # The deliberate difference is replay memory inclusion.
    # --------------------------------------------------------
    for method in ["naive_finetune", "random_replay"]:
        print(
            f"\n--- {pair} | seed={seed} | method={method} ---"
        )

        if method == "naive_finetune":
            X_adapt, y_adapt = build_adaptation_set(
                X_target_train,
                y_target_train,
                None,
                None,
                random_state=seed + 4_000,
            )
            replay_total = 0

        else:
            X_adapt, y_adapt = build_adaptation_set(
                X_target_train,
                y_target_train,
                X_replay,
                y_replay,
                random_state=seed + 4_000,
            )
            replay_total = int(len(y_replay))

        # Paired adaptation RNG for naive vs replay.
        set_seed(seed + 10_000)

        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(copy.deepcopy(source_state))

        adapt_loader = make_loader(
            X_adapt,
            y_adapt,
            batch_size=min(batch_size, len(y_adapt)),
            shuffle=True,
        )

        adaptation_training = train_model(
            model=model,
            train_loader=adapt_loader,
            X_val=X_target_val,
            y_val=y_target_val,
            device=device,
            lr=adapt_lr,
            weight_decay=weight_decay,
            max_epochs=adapt_epochs,
            patience=patience,
            label=f"{pair}_{method}_seed_{seed}",
            val_name="target_val",
        )

        source_after = compute_metrics(
            model,
            data["Xs_test"],
            data["ys_test"],
            device,
        )

        target_after = compute_metrics(
            model,
            data["Xt_test"],
            data["yt_test"],
            device,
        )

        forgetting = forgetting_stats(
            source_before["auroc"],
            source_after["auroc"],
        )

        joint = joint_scores(
            target_after["auroc"],
            source_after["auroc"],
        )

        result = {
            "pair": pair,
            "source_dataset": pair.split("__TO__")[0],
            "target_dataset": pair.split("__TO__")[1],
            "stage1_variant": stage1_variant,
            "seed": int(seed),
            "method": method,
            "target_budget_per_class": int(
                target_budget_per_class
            ),
            "target_adapt_train_total": int(
                len(y_target_train)
            ),
            "source_replay_per_class": (
                int(replay_per_class)
                if method == "random_replay"
                else 0
            ),
            "source_replay_total": replay_total,
            "source_replay_fraction_of_training": float(
                replay_total / len(y_adapt)
                if len(y_adapt) else 0.0
            ),
            "source_auroc_before": float(
                source_before["auroc"]
            ),
            "source_f1_before": float(
                source_before["f1"]
            ),
            "target_auroc_zero_shot": float(
                target_zero_shot["auroc"]
            ),
            "target_f1_zero_shot": float(
                target_zero_shot["f1"]
            ),
            "target_auroc_after": float(
                target_after["auroc"]
            ),
            "target_f1_after": float(
                target_after["f1"]
            ),
            "target_auroc_gain": float(
                target_after["auroc"]
                - target_zero_shot["auroc"]
            ),
            "source_auroc_after": float(
                source_after["auroc"]
            ),
            "source_f1_after": float(
                source_after["f1"]
            ),
            **forgetting,
            **joint,
            "source_best_val_auroc": float(
                source_training["best_val_auroc"]
            ),
            "source_best_epoch": int(
                source_training["best_epoch"]
            ),
            "target_best_val_auroc": float(
                adaptation_training["best_val_auroc"]
            ),
            "target_best_epoch": int(
                adaptation_training["best_epoch"]
            ),
        }

        results.append(result)

        print(
            f"Target AUROC={target_after['auroc']:.4f} | "
            f"Source AUROC={source_after['auroc']:.4f} | "
            f"Source drop="
            f"{forgetting['absolute_source_auroc_drop']:.4f}"
        )

    return results


# ============================================================
# Aggregation
# ============================================================
def aggregate_by_pair_method(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "source_auroc_before",
        "source_f1_before",
        "target_auroc_zero_shot",
        "target_f1_zero_shot",
        "target_auroc_after",
        "target_f1_after",
        "target_auroc_gain",
        "source_auroc_after",
        "source_f1_after",
        "absolute_source_auroc_drop",
        "relative_source_auroc_drop_pct",
        "mean_source_target_auroc",
        "harmonic_source_target_auroc",
        "worst_domain_auroc",
        "target_best_val_auroc",
        "target_best_epoch",
    ]

    rows = []

    grouped = raw_df.groupby(
        ["pair", "source_dataset", "target_dataset", "method"],
        sort=False,
    )

    for (
        pair,
        source_dataset,
        target_dataset,
        method,
    ), group in grouped:

        row = {
            "pair": pair,
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "method": method,
            "num_seeds": int(len(group)),
            "target_budget_per_class": int(
                group["target_budget_per_class"].iloc[0]
            ),
            "source_replay_per_class": int(
                group["source_replay_per_class"].iloc[0]
            ),
            "source_replay_total": int(
                group["source_replay_total"].iloc[0]
            ),
            "source_replay_fraction_of_training": float(
                group[
                    "source_replay_fraction_of_training"
                ].iloc[0]
            ),
        }

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
            else:
                row[f"{metric}_mean"] = float(
                    values.mean()
                )
                row[f"{metric}_std"] = (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0
                )

        rows.append(row)

    return pd.DataFrame(rows)


def build_pair_comparison(
    aggregate_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for pair in aggregate_df["pair"].unique():
        pair_df = aggregate_df[
            aggregate_df["pair"] == pair
        ]

        naive = pair_df[
            pair_df["method"] == "naive_finetune"
        ].iloc[0]

        replay = pair_df[
            pair_df["method"] == "random_replay"
        ].iloc[0]

        naive_forgetting = float(
            naive["absolute_source_auroc_drop_mean"]
        )
        replay_forgetting = float(
            replay["absolute_source_auroc_drop_mean"]
        )

        forgetting_reduction = (
            naive_forgetting - replay_forgetting
        )

        forgetting_reduction_pct = (
            100.0 * forgetting_reduction / naive_forgetting
            if naive_forgetting > 0
            else np.nan
        )

        rows.append({
            "pair": pair,
            "source_dataset": naive["source_dataset"],
            "target_dataset": naive["target_dataset"],
            "target_auroc_zero_shot_mean": float(
                naive["target_auroc_zero_shot_mean"]
            ),
            "source_auroc_before_mean": float(
                naive["source_auroc_before_mean"]
            ),

            "naive_target_auroc_mean": float(
                naive["target_auroc_after_mean"]
            ),
            "naive_target_auroc_std": float(
                naive["target_auroc_after_std"]
            ),
            "naive_source_auroc_mean": float(
                naive["source_auroc_after_mean"]
            ),
            "naive_source_auroc_std": float(
                naive["source_auroc_after_std"]
            ),
            "naive_source_drop_mean": naive_forgetting,
            "naive_source_drop_std": float(
                naive["absolute_source_auroc_drop_std"]
            ),
            "naive_worst_domain_auroc_mean": float(
                naive["worst_domain_auroc_mean"]
            ),

            "replay_target_auroc_mean": float(
                replay["target_auroc_after_mean"]
            ),
            "replay_target_auroc_std": float(
                replay["target_auroc_after_std"]
            ),
            "replay_source_auroc_mean": float(
                replay["source_auroc_after_mean"]
            ),
            "replay_source_auroc_std": float(
                replay["source_auroc_after_std"]
            ),
            "replay_source_drop_mean": replay_forgetting,
            "replay_source_drop_std": float(
                replay["absolute_source_auroc_drop_std"]
            ),
            "replay_worst_domain_auroc_mean": float(
                replay["worst_domain_auroc_mean"]
            ),

            "source_auroc_recovered_by_replay": float(
                replay["source_auroc_after_mean"]
                - naive["source_auroc_after_mean"]
            ),
            "target_auroc_change_with_replay": float(
                replay["target_auroc_after_mean"]
                - naive["target_auroc_after_mean"]
            ),
            "forgetting_reduction_absolute": float(
                forgetting_reduction
            ),
            "forgetting_reduction_pct": float(
                forgetting_reduction_pct
            ),
            "worst_domain_auroc_improvement": float(
                replay["worst_domain_auroc_mean"]
                - naive["worst_domain_auroc_mean"]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================
def short_pair(pair: str) -> str:
    aliases = {
        "NF-UNSW-NB15-v3": "UNSW",
        "NF-ToN-IoT-v3": "ToN",
        "NF-CSE-CIC-IDS2018-v3": "CIC",
    }

    src, tgt = pair.split("__TO__")
    return f"{aliases[src]}→{aliases[tgt]}"


def plot_pair_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    df = comparison_df.copy()
    labels = [short_pair(p) for p in df["pair"]]
    x = np.arange(len(df))

    # --------------------------------------------------------
    # Source AUROC after adaptation
    # --------------------------------------------------------
    width = 0.36

    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        df["naive_source_auroc_mean"],
        width=width,
        yerr=df["naive_source_auroc_std"],
        capsize=3,
        label="Naive fine-tuning",
    )
    plt.bar(
        x + width / 2,
        df["replay_source_auroc_mean"],
        width=width,
        yerr=df["replay_source_auroc_std"],
        capsize=3,
        label="Random replay (250/class)",
    )
    plt.xticks(x, labels)
    plt.ylabel("Source AUROC after adaptation")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=False)
    savefig(
        output_dir
        / "all_pairs_source_retention.png"
    )

    # --------------------------------------------------------
    # Target AUROC after adaptation
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        df["naive_target_auroc_mean"],
        width=width,
        yerr=df["naive_target_auroc_std"],
        capsize=3,
        label="Naive fine-tuning",
    )
    plt.bar(
        x + width / 2,
        df["replay_target_auroc_mean"],
        width=width,
        yerr=df["replay_target_auroc_std"],
        capsize=3,
        label="Random replay (250/class)",
    )
    plt.xticks(x, labels)
    plt.ylabel("Target AUROC after adaptation")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=False)
    savefig(
        output_dir
        / "all_pairs_target_adaptation.png"
    )

    # --------------------------------------------------------
    # Forgetting reduction
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plt.bar(
        x,
        df["forgetting_reduction_absolute"],
    )
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels)
    plt.ylabel(
        "Reduction in source AUROC drop\n"
        "(naive forgetting - replay forgetting)"
    )
    savefig(
        output_dir
        / "all_pairs_forgetting_reduction.png"
    )

    # --------------------------------------------------------
    # Worst-domain AUROC
    # --------------------------------------------------------
    plt.figure(figsize=(11, 6))
    plt.bar(
        x - width / 2,
        df["naive_worst_domain_auroc_mean"],
        width=width,
        label="Naive fine-tuning",
    )
    plt.bar(
        x + width / 2,
        df["replay_worst_domain_auroc_mean"],
        width=width,
        label="Random replay (250/class)",
    )
    plt.xticks(x, labels)
    plt.ylabel("Worst-domain AUROC")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=False)
    savefig(
        output_dir
        / "all_pairs_worst_domain_auroc.png"
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage1_variant",
        default="balanced_100k",
        type=str,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )
    parser.add_argument(
        "--target_budget_per_class",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--replay_per_class",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--source_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--adapt_epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--source_lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--adapt_lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[256, 128],
    )

    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help=(
            "Optional explicit pair names. "
            "Default: all 6 directed pairs."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    pairs = (
        args.pairs
        if args.pairs
        else all_directed_pairs()
    )

    variant_dir = (
        PAIR_DIR
        / args.stage1_variant
    )

    missing = [
        pair
        for pair in pairs
        if not (variant_dir / pair).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing pair directories:\n"
            + "\n".join(missing)
        )

    output_dir = (
        variant_dir
        / "mlp_random_replay_all_pairs_results"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    protocol = {
        "stage1_variant": args.stage1_variant,
        "pairs": pairs,
        "seeds": [int(v) for v in args.seeds],
        "target_budget_per_class": int(
            args.target_budget_per_class
        ),
        "random_replay_per_class": int(
            args.replay_per_class
        ),
        "comparison": (
            "Naive target fine-tuning vs balanced random "
            "source replay."
        ),
        "fairness_control": (
            "Within each pair and seed, naive fine-tuning "
            "and random replay start from the same source "
            "model, use the same target adaptation subset "
            "and validation split, and use the same "
            "adaptation RNG seed."
        ),
        "evaluation": (
            "Source and target test sets are evaluation only "
            "and are never used for adaptation, replay "
            "selection, or model selection."
        ),
    }

    all_rows = []

    for pair_index, pair in enumerate(pairs, start=1):
        print("\n" + "#" * 100)
        print(
            f"PAIR {pair_index}/{len(pairs)}: {pair}"
        )
        print("#" * 100)

        pair_dir = variant_dir / pair
        data = load_pair(pair_dir)

        for seed in args.seeds:
            rows = run_pair_seed(
                pair=pair,
                seed=int(seed),
                data=data,
                stage1_variant=args.stage1_variant,
                target_budget_per_class=(
                    args.target_budget_per_class
                ),
                replay_per_class=(
                    args.replay_per_class
                ),
                batch_size=args.batch_size,
                source_epochs=args.source_epochs,
                adapt_epochs=args.adapt_epochs,
                source_lr=args.source_lr,
                adapt_lr=args.adapt_lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                dropout=args.dropout,
                hidden_dims=tuple(args.hidden_dims),
                device=device,
            )

            all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)

    aggregate_df = aggregate_by_pair_method(
        raw_df
    )

    comparison_df = build_pair_comparison(
        aggregate_df
    )

    raw_path = (
        output_dir
        / "all_pairs_raw_results.csv"
    )
    aggregate_path = (
        output_dir
        / "all_pairs_aggregate_by_method.csv"
    )
    comparison_path = (
        output_dir
        / "all_pairs_replay_comparison.csv"
    )
    json_path = (
        output_dir
        / "all_pairs_experiment_summary.json"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )
    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )
    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "protocol": protocol,
                "pair_comparison": comparison_df.to_dict(
                    orient="records"
                ),
            },
            f,
            indent=2,
        )

    plot_pair_comparison(
        comparison_df,
        output_dir,
    )

    print("\n" + "=" * 120)
    print("ALL-PAIRS RANDOM REPLAY EXPERIMENT COMPLETE")
    print("=" * 120)

    display = comparison_df[[
        "pair",
        "target_auroc_zero_shot_mean",
        "naive_target_auroc_mean",
        "naive_source_auroc_mean",
        "replay_target_auroc_mean",
        "replay_source_auroc_mean",
        "naive_source_drop_mean",
        "replay_source_drop_mean",
        "forgetting_reduction_absolute",
        "target_auroc_change_with_replay",
        "worst_domain_auroc_improvement",
    ]].copy()

    display["pair"] = display["pair"].map(
        short_pair
    )

    print(display.to_string(index=False))

    print(f"\nRaw results:       {raw_path}")
    print(f"Method aggregates: {aggregate_path}")
    print(f"Pair comparison:   {comparison_path}")
    print(f"Summary JSON:      {json_path}")
    print("Generated plots:")
    print("- all_pairs_source_retention.png")
    print("- all_pairs_target_adaptation.png")
    print("- all_pairs_forgetting_reduction.png")
    print("- all_pairs_worst_domain_auroc.png")


if __name__ == "__main__":
    main()