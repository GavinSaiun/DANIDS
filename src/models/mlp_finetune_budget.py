import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PAIR_DIR, RANDOM_STATE


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pair(pair_dir: Path):
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


def predict_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        return torch.sigmoid(logits).cpu().numpy()


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
            "val_auroc": float(val_auroc),
            "val_f1": float(val_metrics["f1"]),
        })

        print(
            f"[{label}] Epoch {epoch + 1:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_auroc={val_auroc:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def stratified_budget_sample(X, y, samples_per_class: int, random_state: int):
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


def split_budget_train_val(X_budget, y_budget, random_state):
    if len(y_budget) < 10 or len(np.unique(y_budget)) < 2:
        raise ValueError("Budget sample too small or lacks both classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        X_budget,
        y_budget,
        test_size=0.20,
        stratify=y_budget,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, type=str)
    parser.add_argument("--stage1_variant", default="balanced_100k", type=str)
    parser.add_argument(
        "--budgets_per_class",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 1000, 5000, 10000],
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--source_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    args = parser.parse_args()

    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pair_dir = PAIR_DIR / args.stage1_variant / args.pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    data = load_pair(pair_dir)
    input_dim = data["Xs_train"].shape[1]

    results_dir = pair_dir / "mlp_finetune_budget_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Source-only pretraining once
    # -----------------------------
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
        model=source_model,
        train_loader=source_loader,
        X_val=data["Xs_val"],
        y_val=data["ys_val"],
        device=device,
        lr=args.source_lr,
        max_epochs=args.source_epochs,
        patience=args.patience,
        label="source",
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

    torch.save(source_model.state_dict(), results_dir / "source_model.pt")

    rows = []
    detailed = {
        "pair": args.pair,
        "stage1_variant": args.stage1_variant,
        "device": str(device),
        "budgets_per_class": args.budgets_per_class,
        "source_training": source_training,
        "source_before_adaptation": source_before,
        "target_zero_shot": target_zero_shot,
        "budget_results": {},
        "leakage_control": {
            "source_test_usage": "Only used for evaluation before and after adaptation.",
            "target_test_usage": "Only used for final evaluation, never for adaptation or early stopping.",
            "target_budget_sampling": "Budget samples are drawn only from Xt_adapt.",
        },
    }

    # -----------------------------
    # Fine-tuning budget sweep
    # -----------------------------
    for budget in args.budgets_per_class:
        print(f"\n=== Fine-tuning budget: {budget} per class ===")

        X_budget, y_budget, selected_idx = stratified_budget_sample(
            data["Xt_adapt"],
            data["yt_adapt"],
            samples_per_class=budget,
            random_state=RANDOM_STATE + budget,
        )

        X_ft_train, X_ft_val, y_ft_train, y_ft_val = split_budget_train_val(
            X_budget,
            y_budget,
            random_state=RANDOM_STATE,
        )

        model = MLP(
            input_dim=input_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(copy.deepcopy(source_model.state_dict()))

        ft_loader = make_loader(
            X_ft_train,
            y_ft_train,
            batch_size=min(args.batch_size, len(y_ft_train)),
            shuffle=True,
        )

        ft_training = train_model(
            model=model,
            train_loader=ft_loader,
            X_val=X_ft_val,
            y_val=y_ft_val,
            device=device,
            lr=args.finetune_lr,
            max_epochs=args.finetune_epochs,
            patience=args.patience,
            label=f"ft_{budget}",
        )

        source_after = compute_metrics(model, data["Xs_test"], data["ys_test"], device)
        target_after = compute_metrics(model, data["Xt_test"], data["yt_test"], device)

        f_stats = forgetting_stats(
            source_before_auroc=source_before["auroc"],
            source_after_auroc=source_after["auroc"],
        )

        row = {
            "pair": args.pair,
            "stage1_variant": args.stage1_variant,
            "target_budget_per_class": int(budget),
            "target_budget_total": int(len(y_budget)),
            "target_auroc_zero_shot": target_zero_shot["auroc"],
            "target_auroc_after_finetune": target_after["auroc"],
            "target_auroc_gain": target_after["auroc"] - target_zero_shot["auroc"],
            "source_auroc_before_finetune": source_before["auroc"],
            "source_auroc_after_finetune": source_after["auroc"],
            **f_stats,
            "target_f1_after_finetune": target_after["f1"],
            "source_f1_after_finetune": source_after["f1"],
        }
        rows.append(row)

        detailed["budget_results"][str(budget)] = {
            "budget_indices_within_Xt_adapt": selected_idx.tolist(),
            "training": ft_training,
            "source_after_finetune": source_after,
            "target_after_finetune": target_after,
            "comparison": row,
        }

        print(json.dumps(row, indent=2))

    # Save detailed JSON
    with open(results_dir / "budget_summary.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    # Save compact CSV
    csv_path = results_dir / "budget_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved detailed results to: {results_dir / 'budget_summary.json'}")
    print(f"Saved compact results to: {csv_path}")


if __name__ == "__main__":
    main()