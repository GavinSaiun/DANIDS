import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PAIR_DIR, RANDOM_STATE


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


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
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
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
    """
    Early stopping uses the SAME fixed target validation set for every replay
    condition. This keeps replay size as the only changing variable.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        "best_target_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def stratified_sample_per_class(X, y, samples_per_class: int, random_state: int):
    if samples_per_class < 0:
        raise ValueError("samples_per_class must be >= 0")

    if samples_per_class == 0:
        return (
            np.empty((0, X.shape[1]), dtype=X.dtype),
            np.empty((0,), dtype=y.dtype),
            np.empty((0,), dtype=int),
        )

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


def split_target_budget(X_budget, y_budget, random_state):
    if len(y_budget) < 10 or len(np.unique(y_budget)) < 2:
        raise ValueError("Target budget is too small or lacks both classes")

    return train_test_split(
        X_budget,
        y_budget,
        test_size=0.20,
        stratify=y_budget,
        random_state=random_state,
    )


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
        "useful_relative_source_auroc_drop_pct": float(useful_relative_drop * 100),
    }


def joint_scores(target_auroc, source_auroc):
    mean_auroc = (target_auroc + source_auroc) / 2.0
    harmonic = (
        2.0 * target_auroc * source_auroc / (target_auroc + source_auroc)
        if target_auroc + source_auroc > 0
        else np.nan
    )
    return {
        "mean_source_target_auroc": float(mean_auroc),
        "harmonic_source_target_auroc": float(harmonic),
        "worst_domain_auroc": float(min(target_auroc, source_auroc)),
    }


def categorical_x(values):
    x = np.arange(len(values))
    labels = [str(v) for v in values]
    return x, labels


def plot_replay_tradeoff(df: pd.DataFrame, results_dir: Path):
    budgets = df["source_replay_per_class"].astype(int).tolist()
    x, labels = categorical_x(budgets)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        x,
        df["target_auroc_after_replay"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Target AUROC",
    )
    plt.plot(
        x,
        df["source_auroc_after_replay"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Source AUROC",
    )
    plt.axhline(
        df["target_auroc_zero_shot"].iloc[0],
        linestyle="--",
        linewidth=1.5,
        label="Target zero-shot",
    )
    plt.axhline(
        df["source_auroc_before_replay"].iloc[0],
        linestyle=":",
        linewidth=1.8,
        label="Source before adaptation",
    )
    plt.xticks(x, labels)
    plt.xlabel("Source replay samples per class")
    plt.ylabel("AUROC")
    plt.ylim(0.0, 1.05)
    plt.legend(frameon=False, loc="best")
    savefig(results_dir / "replay_tradeoff_auroc.png")


def plot_replay_forgetting(df: pd.DataFrame, results_dir: Path):
    budgets = df["source_replay_per_class"].astype(int).tolist()
    x, labels = categorical_x(budgets)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        x,
        df["absolute_source_auroc_drop"],
        marker="o",
        linewidth=2.5,
        markersize=7,
    )
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, labels)
    plt.xlabel("Source replay samples per class")
    plt.ylabel("Source AUROC drop")
    savefig(results_dir / "replay_forgetting.png")


def plot_adaptation_vs_forgetting(df: pd.DataFrame, results_dir: Path):
    plt.figure(figsize=(8, 5.5))
    plt.scatter(
        df["absolute_source_auroc_drop"],
        df["target_auroc_gain"],
        s=85,
        alpha=0.85,
    )

    for _, row in df.iterrows():
        plt.annotate(
            str(int(row["source_replay_per_class"])),
            (row["absolute_source_auroc_drop"], row["target_auroc_gain"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=10,
        )

    plt.xlabel("Source AUROC drop")
    plt.ylabel("Target AUROC gain")
    savefig(results_dir / "replay_adaptation_vs_forgetting.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, type=str)
    parser.add_argument("--stage1_variant", default="balanced_100k", type=str)
    parser.add_argument(
        "--target_budget_per_class",
        default=1000,
        type=int,
        help="Fixed target samples/class selected from Xt_adapt before 80/20 split",
    )
    parser.add_argument(
        "--replay_budgets_per_class",
        type=int,
        nargs="+",
        default=[0, 50, 100, 250, 500, 1000],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--adapt_epochs", type=int, default=40)
    parser.add_argument("--source_lr", type=float, default=1e-3)
    parser.add_argument("--adapt_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    args = parser.parse_args()

    if args.target_budget_per_class <= 0:
        raise ValueError("--target_budget_per_class must be > 0")

    replay_budgets = sorted(set(args.replay_budgets_per_class))
    if any(v < 0 for v in replay_budgets):
        raise ValueError("Replay budgets must be >= 0")

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pair_dir = PAIR_DIR / args.stage1_variant / args.pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    data = load_pair(pair_dir)
    input_dim = data["Xs_train"].shape[1]
    results_dir = pair_dir / "mlp_replay_budget_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pair: {args.pair}")
    print(f"Input dim: {input_dim}")
    print(f"Fixed target budget/class: {args.target_budget_per_class}")
    print(f"Replay budgets/class: {replay_budgets}")

    # 1) Train source model once.
    print("\n=== Source-only pretraining ===")
    source_model = MLP(
        input_dim=input_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    source_loader = make_loader(
        data["Xs_train"],
        data["ys_train"],
        batch_size=args.batch_size,
        shuffle=True,
    )

    source_training = train_model(
        source_model,
        source_loader,
        data["Xs_val"],
        data["ys_val"],
        device,
        lr=args.source_lr,
        weight_decay=args.weight_decay,
        max_epochs=args.source_epochs,
        patience=args.patience,
        label="source",
    )

    source_before = compute_metrics(source_model, data["Xs_test"], data["ys_test"], device)
    target_zero_shot = compute_metrics(source_model, data["Xt_test"], data["yt_test"], device)

    source_state = copy.deepcopy(source_model.state_dict())
    torch.save(source_state, results_dir / "source_model.pt")

    # 2) Select target budget once and reuse it for every replay condition.
    print("\n=== Fixed target adaptation budget ===")
    X_target_budget, y_target_budget, target_idx = stratified_sample_per_class(
        data["Xt_adapt"],
        data["yt_adapt"],
        args.target_budget_per_class,
        RANDOM_STATE + args.target_budget_per_class,
    )

    X_target_train, X_target_val, y_target_train, y_target_val = split_target_budget(
        X_target_budget,
        y_target_budget,
        RANDOM_STATE,
    )

    print(
        f"Target total={len(y_target_budget)} | "
        f"train={len(y_target_train)} | val={len(y_target_val)}"
    )

    detailed = {
        "pair": args.pair,
        "stage1_variant": args.stage1_variant,
        "device": str(device),
        "protocol": {
            "source_model": "Trained once; exact same weights initialise every replay condition.",
            "target_budget": "Selected once from Xt_adapt and reused across all replay conditions.",
            "target_validation": "Fixed target-only validation set reused across all replay conditions.",
            "replay_source": "Replay samples come only from Xs_train/ys_train.",
            "source_test": "Evaluation only.",
            "target_test": "Evaluation only.",
            "replay_zero": "Replay=0 is naive target-only fine-tuning under the same protocol.",
        },
        "target_budget": {
            "samples_per_class_requested": int(args.target_budget_per_class),
            "selected_total": int(len(y_target_budget)),
            "train_total": int(len(y_target_train)),
            "validation_total": int(len(y_target_val)),
            "indices_within_Xt_adapt": target_idx.tolist(),
        },
        "source_training": source_training,
        "source_before_adaptation": source_before,
        "target_zero_shot": target_zero_shot,
        "replay_results": {},
    }

    rows = []

    # 3) Build ONE random source ordering per class so replay memories are nested.
    #    Example: the 50/class memory is a subset of 100/class, which is a
    #    subset of 250/class. This isolates memory SIZE from memory COMPOSITION.
    rng_replay = np.random.default_rng(RANDOM_STATE + 10_000)
    source_idx0 = np.where(data["ys_train"] == 0)[0].copy()
    source_idx1 = np.where(data["ys_train"] == 1)[0].copy()
    rng_replay.shuffle(source_idx0)
    rng_replay.shuffle(source_idx1)

    # 4) Sweep replay memory size.
    for replay_budget in replay_budgets:
        print(f"\n=== Replay budget: {replay_budget} per class ===")

        if replay_budget == 0:
            replay_idx = np.empty((0,), dtype=int)
        else:
            n0 = min(replay_budget, len(source_idx0))
            n1 = min(replay_budget, len(source_idx1))
            replay_idx = np.concatenate([source_idx0[:n0], source_idx1[:n1]])
            rng_condition = np.random.default_rng(RANDOM_STATE + 20_000 + replay_budget)
            rng_condition.shuffle(replay_idx)

        X_replay = data["Xs_train"][replay_idx]
        y_replay = data["ys_train"][replay_idx]

        X_adapt_train, y_adapt_train = build_replay_training_set(
            X_target_train,
            y_target_train,
            X_replay,
            y_replay,
            RANDOM_STATE + 20_000 + replay_budget,
        )

        model = MLP(
            input_dim=input_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(copy.deepcopy(source_state))

        adapt_loader = make_loader(
            X_adapt_train,
            y_adapt_train,
            batch_size=min(args.batch_size, len(y_adapt_train)),
            shuffle=True,
        )

        training = train_model(
            model,
            adapt_loader,
            X_target_val,
            y_target_val,
            device,
            lr=args.adapt_lr,
            weight_decay=args.weight_decay,
            max_epochs=args.adapt_epochs,
            patience=args.patience,
            label=f"replay_{replay_budget}",
        )

        source_after = compute_metrics(model, data["Xs_test"], data["ys_test"], device)
        target_after = compute_metrics(model, data["Xt_test"], data["yt_test"], device)

        replay_total = int(len(y_replay))
        combined_total = int(len(y_adapt_train))

        row = {
            "pair": args.pair,
            "stage1_variant": args.stage1_variant,
            "target_budget_per_class": int(args.target_budget_per_class),
            "target_budget_total": int(len(y_target_budget)),
            "target_train_total": int(len(y_target_train)),
            "source_replay_per_class": int(replay_budget),
            "source_replay_total": replay_total,
            "combined_adaptation_train_total": combined_total,
            "source_replay_fraction_of_training": float(replay_total / combined_total),
            "target_auroc_zero_shot": target_zero_shot["auroc"],
            "target_auroc_after_replay": target_after["auroc"],
            "target_auroc_gain": target_after["auroc"] - target_zero_shot["auroc"],
            "source_auroc_before_replay": source_before["auroc"],
            "source_auroc_after_replay": source_after["auroc"],
            **forgetting_stats(source_before["auroc"], source_after["auroc"]),
            **joint_scores(target_after["auroc"], source_after["auroc"]),
            "target_f1_after_replay": target_after["f1"],
            "source_f1_after_replay": source_after["f1"],
            "best_target_val_auroc": training["best_target_val_auroc"],
            "best_epoch": training["best_epoch"],
        }
        rows.append(row)

        detailed["replay_results"][str(replay_budget)] = {
            "source_replay_indices_within_Xs_train": replay_idx.tolist(),
            "training": training,
            "source_after_replay": source_after,
            "target_after_replay": target_after,
            "comparison": row,
        }

        torch.save(model.state_dict(), results_dir / f"replay_{replay_budget}_model.pt")
        print(json.dumps(row, indent=2))

    # 5) Save results and plots.
    json_path = results_dir / "replay_budget_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    csv_path = results_dir / "replay_budget_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows).sort_values("source_replay_per_class")
    plot_replay_tradeoff(df, results_dir)
    plot_replay_forgetting(df, results_dir)
    plot_adaptation_vs_forgetting(df, results_dir)

    best_worst_idx = df["worst_domain_auroc"].idxmax()
    best_harmonic_idx = df["harmonic_source_target_auroc"].idxmax()

    print("\n" + "=" * 80)
    print("REPLAY BUDGET SWEEP COMPLETE")
    print("=" * 80)
    print(
        df[[
            "source_replay_per_class",
            "target_auroc_after_replay",
            "source_auroc_after_replay",
            "absolute_source_auroc_drop",
            "worst_domain_auroc",
        ]].to_string(index=False)
    )
    print(
        "\nBest worst-domain AUROC: replay/class="
        f"{int(df.loc[best_worst_idx, 'source_replay_per_class'])}, "
        f"score={df.loc[best_worst_idx, 'worst_domain_auroc']:.4f}"
    )
    print(
        "Best harmonic source/target AUROC: replay/class="
        f"{int(df.loc[best_harmonic_idx, 'source_replay_per_class'])}, "
        f"score={df.loc[best_harmonic_idx, 'harmonic_source_target_auroc']:.4f}"
    )
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print("Generated:")
    print("- replay_tradeoff_auroc.png")
    print("- replay_forgetting.png")
    print("- replay_adaptation_vs_forgetting.png")


if __name__ == "__main__":
    main()