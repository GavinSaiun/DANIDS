import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import PAIR_DIR
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pair(pair_dir: Path) -> dict[str, np.ndarray]:
    return {
        "Xs_train": np.load(pair_dir / "Xs_train.npy"),
        "ys_train": np.load(pair_dir / "ys_train.npy"),
        "Xs_val": np.load(pair_dir / "Xs_val.npy"),
        "ys_val": np.load(pair_dir / "ys_val.npy"),
        "Xs_test": np.load(pair_dir / "Xs_test.npy"),
        "ys_test": np.load(pair_dir / "ys_test.npy"),
        "Xt_adapt": np.load(pair_dir / "Xt_adapt.npy"),
        "yt_adapt": np.load(pair_dir / "yt_adapt.npy"),
        "Xt_test": np.load(pair_dir / "Xt_test.npy"),
        "yt_test": np.load(pair_dir / "yt_test.npy"),
    }


def make_loader(X, y, batch_size=512, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.2):
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

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_proba(model, X, device, batch_size=8192):
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


def compute_metrics(model, X, y, device):
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
    model,
    train_loader,
    X_val,
    y_val,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=30,
    patience=5,
    label="train",
):
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
    history = []

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

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "target_val_auroc": float(val_auroc),
            "target_val_f1": float(val_metrics["f1"]),
        })

        print(
            f"[{label}] Epoch {epoch + 1:02d} | "
            f"loss={train_loss:.4f} | "
            f"target_val_auroc={val_auroc:.4f} | "
            f"target_val_f1={val_metrics['f1']:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{label}] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_target_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def stratified_sample_per_class(X, y, samples_per_class, random_state):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0 = min(samples_per_class, len(idx0))
    n1 = min(samples_per_class, len(idx1))

    s0 = rng.choice(idx0, size=n0, replace=False)
    s1 = rng.choice(idx1, size=n1, replace=False)

    selected = np.concatenate([s0, s1])
    rng.shuffle(selected)

    return X[selected], y[selected], selected


def split_target_budget(X_budget, y_budget, random_state, val_fraction=0.20):
    return train_test_split(
        X_budget,
        y_budget,
        test_size=val_fraction,
        stratify=y_budget,
        random_state=random_state,
    )


def make_nested_replay_pool(X, y, max_samples_per_class, random_state):
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n = min(max_samples_per_class, len(idx0), len(idx1))

    return {
        "class0_indices": rng.choice(idx0, size=n, replace=False),
        "class1_indices": rng.choice(idx1, size=n, replace=False),
        "max_samples_per_class": int(n),
    }


def get_nested_replay_sample(
    X,
    y,
    replay_pool,
    samples_per_class,
    random_state,
):
    if samples_per_class == 0:
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=y.dtype),
            np.empty((0,), dtype=int),
        )

    idx0 = replay_pool["class0_indices"][:samples_per_class]
    idx1 = replay_pool["class1_indices"][:samples_per_class]

    selected = np.concatenate([idx0, idx1])
    rng = np.random.default_rng(random_state)
    rng.shuffle(selected)

    return X[selected], y[selected], selected


def build_replay_training_set(
    X_target_train,
    y_target_train,
    X_replay,
    y_replay,
    random_state,
):
    if len(X_replay) == 0:
        X = X_target_train.copy()
        y = y_target_train.copy()
    else:
        X = np.vstack([X_target_train, X_replay])
        y = np.concatenate([y_target_train, y_replay])

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(y))
    return X[order], y[order]


def forgetting_stats(source_before_auroc, source_after_auroc):
    absolute_drop = source_before_auroc - source_after_auroc

    relative_drop = (
        absolute_drop / source_before_auroc
        if source_before_auroc != 0
        else np.nan
    )

    useful_denominator = source_before_auroc - 0.5
    useful_relative_drop = (
        absolute_drop / useful_denominator
        if useful_denominator > 0
        else np.nan
    )

    return {
        "absolute_source_auroc_drop": float(absolute_drop),
        "relative_source_auroc_drop": float(relative_drop),
        "relative_source_auroc_drop_pct": float(relative_drop * 100),
        "useful_relative_source_auroc_drop": float(useful_relative_drop),
        "useful_relative_source_auroc_drop_pct": float(
            useful_relative_drop * 100
        ),
    }


def adaptation_retention_scores(target_auroc, source_auroc):
    mean_auroc = (target_auroc + source_auroc) / 2.0
    harmonic = (
        2.0 * target_auroc * source_auroc / (target_auroc + source_auroc)
        if (target_auroc + source_auroc) > 0
        else np.nan
    )

    return {
        "mean_source_target_auroc": float(mean_auroc),
        "harmonic_source_target_auroc": float(harmonic),
        "worst_domain_auroc": float(min(target_auroc, source_auroc)),
    }


# --------------------------------------------------
# Plot style
# --------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def run_one_seed(
    *,
    seed: int,
    data: dict,
    pair: str,
    stage1_variant: str,
    target_budget_per_class: int,
    replay_budgets_per_class: list[int],
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
) -> tuple[list[dict], dict]:
    """
    Run the complete replay sweep for ONE random seed.

    Within a seed:
    - source model is trained once
    - target budget is sampled once
    - target train/validation split is fixed
    - one nested source replay pool is fixed
    - every replay condition starts from identical source weights

    Across seeds:
    - source initialisation/training changes
    - target budget sample changes
    - replay pool changes
    """
    print("\n" + "=" * 90)
    print(f"SEED {seed}")
    print("=" * 90)

    set_seed(seed)

    input_dim = data["Xs_train"].shape[1]

    # --------------------------------------------------
    # 1) Source model
    # --------------------------------------------------
    print("\n=== Source-only pretraining ===")

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
        label=f"source_seed_{seed}",
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

    source_state = {
        k: v.detach().clone()
        for k, v in source_model.state_dict().items()
    }

    # --------------------------------------------------
    # 2) Fixed target budget for this seed
    # --------------------------------------------------
    X_target_budget, y_target_budget, target_indices = (
        stratified_sample_per_class(
            data["Xt_adapt"],
            data["yt_adapt"],
            samples_per_class=target_budget_per_class,
            random_state=seed + 1_000,
        )
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

    print(
        f"Target budget/class={target_budget_per_class} | "
        f"total={len(y_target_budget)} | "
        f"train={len(y_target_train)} | "
        f"val={len(y_target_val)}"
    )

    # --------------------------------------------------
    # 3) One nested replay pool for this seed
    # --------------------------------------------------
    max_replay = max(replay_budgets_per_class)

    replay_pool = make_nested_replay_pool(
        data["Xs_train"],
        data["ys_train"],
        max_samples_per_class=max_replay,
        random_state=seed + 3_000,
    )

    rows = []
    seed_detail = {
        "seed": int(seed),
        "source_training": source_training,
        "source_before_adaptation": source_before,
        "target_zero_shot": target_zero_shot,
        "target_budget": {
            "samples_per_class": int(target_budget_per_class),
            "selected_total": int(len(y_target_budget)),
            "train_total": int(len(y_target_train)),
            "validation_total": int(len(y_target_val)),
            "indices_within_Xt_adapt": target_indices.tolist(),
        },
        "nested_replay_pool": {
            "max_samples_per_class": int(
                replay_pool["max_samples_per_class"]
            ),
            "class0_ordered_indices": (
                replay_pool["class0_indices"].tolist()
            ),
            "class1_ordered_indices": (
                replay_pool["class1_indices"].tolist()
            ),
        },
        "replay_results": {},
    }

    # --------------------------------------------------
    # 4) Replay sweep
    # --------------------------------------------------
    for replay_budget in replay_budgets_per_class:
        print(
            f"\n=== Seed {seed} | replay "
            f"{replay_budget}/class ==="
        )

        X_replay, y_replay, replay_indices = (
            get_nested_replay_sample(
                data["Xs_train"],
                data["ys_train"],
                replay_pool=replay_pool,
                samples_per_class=replay_budget,
                random_state=seed + 4_000 + replay_budget,
            )
        )

        X_adapt_train, y_adapt_train = (
            build_replay_training_set(
                X_target_train,
                y_target_train,
                X_replay,
                y_replay,
                random_state=seed + 5_000 + replay_budget,
            )
        )

        # Reset stochastic training stream in a deterministic way.
        # All conditions still start from identical source weights.
        set_seed(seed + 10_000 + replay_budget)

        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)
        model.load_state_dict(source_state)

        adapt_loader = make_loader(
            X_adapt_train,
            y_adapt_train,
            batch_size=min(batch_size, len(y_adapt_train)),
            shuffle=True,
        )

        training = train_model(
            model=model,
            train_loader=adapt_loader,
            X_val=X_target_val,
            y_val=y_target_val,
            device=device,
            lr=adapt_lr,
            weight_decay=weight_decay,
            max_epochs=adapt_epochs,
            patience=patience,
            label=f"seed_{seed}_replay_{replay_budget}",
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

        f_stats = forgetting_stats(
            source_before_auroc=source_before["auroc"],
            source_after_auroc=source_after["auroc"],
        )

        joint = adaptation_retention_scores(
            target_auroc=target_after["auroc"],
            source_auroc=source_after["auroc"],
        )

        replay_total = len(y_replay)
        combined_total = len(y_adapt_train)

        row = {
            "seed": int(seed),
            "pair": pair,
            "stage1_variant": stage1_variant,
            "target_budget_per_class": int(
                target_budget_per_class
            ),
            "target_budget_total": int(len(y_target_budget)),
            "target_train_total": int(len(y_target_train)),
            "source_replay_per_class": int(replay_budget),
            "source_replay_total": int(replay_total),
            "combined_adaptation_train_total": int(
                combined_total
            ),
            "source_replay_fraction_of_training": float(
                replay_total / combined_total
                if combined_total > 0 else 0.0
            ),
            "source_auroc_before_replay": float(
                source_before["auroc"]
            ),
            "target_auroc_zero_shot": float(
                target_zero_shot["auroc"]
            ),
            "target_auroc_after_replay": float(
                target_after["auroc"]
            ),
            "target_auroc_gain": float(
                target_after["auroc"]
                - target_zero_shot["auroc"]
            ),
            "source_auroc_after_replay": float(
                source_after["auroc"]
            ),
            **f_stats,
            **joint,
            "target_f1_after_replay": float(
                target_after["f1"]
            ),
            "source_f1_after_replay": float(
                source_after["f1"]
            ),
            "best_target_val_auroc": float(
                training["best_target_val_auroc"]
            ),
            "best_epoch": int(training["best_epoch"]),
        }

        rows.append(row)

        seed_detail["replay_results"][str(replay_budget)] = {
            "source_replay_indices_within_Xs_train": (
                replay_indices.tolist()
            ),
            "training": training,
            "source_after_replay": source_after,
            "target_after_replay": target_after,
            "comparison": row,
        }

        print(
            f"Target AUROC={target_after['auroc']:.4f} | "
            f"Source AUROC={source_after['auroc']:.4f} | "
            f"Source drop={f_stats['absolute_source_auroc_drop']:.4f}"
        )

    return rows, seed_detail


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_auroc_zero_shot",
        "source_auroc_before_replay",
        "target_auroc_after_replay",
        "target_auroc_gain",
        "source_auroc_after_replay",
        "absolute_source_auroc_drop",
        "target_f1_after_replay",
        "source_f1_after_replay",
        "mean_source_target_auroc",
        "harmonic_source_target_auroc",
        "worst_domain_auroc",
        "best_target_val_auroc",
        "best_epoch",
    ]

    grouped = raw_df.groupby(
        "source_replay_per_class",
        sort=True,
    )

    rows = []
    for replay_budget, group in grouped:
        row = {
            "source_replay_per_class": int(replay_budget),
            "num_seeds": int(len(group)),
            "source_replay_total": int(
                group["source_replay_total"].iloc[0]
            ),
            "source_replay_fraction_of_training": float(
                group["source_replay_fraction_of_training"].iloc[0]
            ),
        }

        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)


def plot_mean_std(
    aggregate_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = np.arange(len(aggregate_df))
    labels = (
        aggregate_df["source_replay_per_class"]
        .astype(int)
        .astype(str)
        .tolist()
    )

    # Target + source AUROC.
    plt.figure(figsize=(9, 5.5))

    plt.errorbar(
        x,
        aggregate_df["target_auroc_after_replay_mean"],
        yerr=aggregate_df["target_auroc_after_replay_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Target AUROC",
    )

    plt.errorbar(
        x,
        aggregate_df["source_auroc_after_replay_mean"],
        yerr=aggregate_df["source_auroc_after_replay_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Source AUROC",
    )

    plt.xticks(x, labels)
    plt.xlabel("Source replay samples per class")
    plt.ylabel("AUROC (mean ± SD)")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=False)
    savefig(output_dir / "multiseed_replay_auroc_mean_std.png")

    # Forgetting.
    plt.figure(figsize=(9, 5.5))

    plt.errorbar(
        x,
        aggregate_df[
            "absolute_source_auroc_drop_mean"
        ],
        yerr=aggregate_df[
            "absolute_source_auroc_drop_std"
        ],
        marker="o",
        linewidth=2,
        capsize=4,
    )

    plt.xticks(x, labels)
    plt.xlabel("Source replay samples per class")
    plt.ylabel("Source AUROC drop (mean ± SD)")
    plt.axhline(0.0, linewidth=1)
    savefig(output_dir / "multiseed_replay_forgetting_mean_std.png")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair", required=True, type=str)
    parser.add_argument(
        "--stage1_variant",
        default="balanced_100k",
        type=str,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
    )

    parser.add_argument(
        "--target_budget_per_class",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--replay_budgets_per_class",
        type=int,
        nargs="+",
        default=[0, 50, 250, 500, 1000],
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
        type=int,
        nargs="+",
        default=[256, 128],
    )

    args = parser.parse_args()

    replay_budgets = sorted(
        set(args.replay_budgets_per_class)
    )

    if replay_budgets[0] < 0:
        raise ValueError(
            "Replay budgets must be >= 0."
        )

    if args.target_budget_per_class <= 0:
        raise ValueError(
            "Target budget/class must be > 0."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    pair_dir = (
        PAIR_DIR
        / args.stage1_variant
        / args.pair
    )

    if not pair_dir.exists():
        raise FileNotFoundError(
            f"Pair directory not found: {pair_dir}"
        )

    data = load_pair(pair_dir)

    output_dir = (
        pair_dir
        / "mlp_replay_multiseed_results"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []
    all_details = {
        "pair": args.pair,
        "stage1_variant": args.stage1_variant,
        "device": str(device),
        "seeds": [int(s) for s in args.seeds],
        "target_budget_per_class": int(
            args.target_budget_per_class
        ),
        "replay_budgets_per_class": [
            int(v) for v in replay_budgets
        ],
        "protocol": {
            "within_seed": (
                "One source model, one fixed target budget/split, "
                "and one nested replay pool are shared by every "
                "replay condition."
            ),
            "across_seeds": (
                "Source initialisation/training, target budget sampling, "
                "and replay-pool sampling are repeated independently."
            ),
            "source_test": (
                "Evaluation only; never used for adaptation or "
                "model selection."
            ),
            "target_test": (
                "Evaluation only; never used for adaptation or "
                "model selection."
            ),
            "aggregation": (
                "Mean and sample standard deviation (ddof=1) "
                "are reported across seeds."
            ),
        },
        "seed_results": {},
    }

    for seed in args.seeds:
        rows, details = run_one_seed(
            seed=int(seed),
            data=data,
            pair=args.pair,
            stage1_variant=args.stage1_variant,
            target_budget_per_class=(
                args.target_budget_per_class
            ),
            replay_budgets_per_class=(
                replay_budgets
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
        all_details["seed_results"][str(seed)] = details

    raw_df = pd.DataFrame(all_rows)
    aggregate_df = aggregate_results(raw_df)

    raw_csv = output_dir / "multiseed_raw_results.csv"
    aggregate_csv = (
        output_dir
        / "multiseed_aggregate_summary.csv"
    )
    json_path = (
        output_dir
        / "multiseed_detailed_results.json"
    )

    raw_df.to_csv(raw_csv, index=False)
    aggregate_df.to_csv(
        aggregate_csv,
        index=False,
    )

    all_details["aggregate_summary"] = (
        aggregate_df.to_dict(orient="records")
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_details,
            f,
            indent=2,
        )

    plot_mean_std(
        aggregate_df,
        output_dir,
    )

    print("\n" + "=" * 90)
    print("3-SEED REPLAY VALIDATION COMPLETE")
    print("=" * 90)

    display_cols = [
        "source_replay_per_class",
        "target_auroc_after_replay_mean",
        "target_auroc_after_replay_std",
        "source_auroc_after_replay_mean",
        "source_auroc_after_replay_std",
        "absolute_source_auroc_drop_mean",
        "absolute_source_auroc_drop_std",
        "worst_domain_auroc_mean",
        "worst_domain_auroc_std",
    ]

    print(
        aggregate_df[
            display_cols
        ].to_string(index=False)
    )

    print(f"\nRaw per-seed CSV: {raw_csv}")
    print(f"Aggregate CSV:    {aggregate_csv}")
    print(f"Detailed JSON:    {json_path}")
    print("Generated plots:")
    print("- multiseed_replay_auroc_mean_std.png")
    print("- multiseed_replay_forgetting_mean_std.png")


if __name__ == "__main__":
    main()