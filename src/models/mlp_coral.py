import argparse
import copy
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
    data = {
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
    return data


def split_target_adapt(Xt_adapt, yt_adapt, random_state=RANDOM_STATE):
    X_train, X_val, y_train, y_val = train_test_split(
        Xt_adapt,
        yt_adapt,
        test_size=0.20,
        stratify=yt_adapt,
        random_state=random_state,
    )
    return X_train, X_val, y_train, y_val


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def make_loader(X, y, batch_size=512, shuffle=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class MLPWithFeatures(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()

        feature_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h

        self.feature_extractor = nn.Sequential(*feature_layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        logits = self.classifier(features).squeeze(1)

        if return_features:
            return logits, features
        return logits


def predict_proba(model, X, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def compute_metrics(model, X, y, device):
    probs = predict_proba(model, X, device)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y, probs)),
    }
    return metrics


def coral_loss(source_features, target_features):
    """
    Deep CORAL loss:
    Align second-order statistics between source and target features.
    """
    d = source_features.size(1)

    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    ns = source_features.size(0)
    nt = target_features.size(0)

    if ns <= 1 or nt <= 1:
        return torch.tensor(0.0, device=source_features.device)

    cov_source = (source_centered.T @ source_centered) / (ns - 1)
    cov_target = (target_centered.T @ target_centered) / (nt - 1)

    loss = torch.mean((cov_source - cov_target) ** 2)
    loss = loss / (4.0 * d * d)
    return loss


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def train_source_only(
    model,
    train_loader,
    X_val,
    y_val,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=30,
    patience=5,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

        train_loss = running_loss / max(num_samples, 1)
        val_metrics = compute_metrics(model, X_val, y_val, device)
        val_auroc = val_metrics["auroc"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_auroc": float(val_auroc),
            "val_f1": float(val_metrics["f1"]),
        })

        print(
            f"[Source] Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auroc={val_auroc:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[Source] Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def train_coral_adaptation(
    model,
    source_loader,
    target_loader,
    X_val,
    y_val,
    device,
    coral_lambda=1.0,
    lr=5e-4,
    weight_decay=1e-4,
    max_epochs=20,
    patience=5,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    target_iter = cycle_loader(target_loader)
    steps_per_epoch = len(source_loader)

    for epoch in range(max_epochs):
        model.train()

        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_coral_loss = 0.0
        num_source_samples = 0

        for step, (xs, ys) in enumerate(source_loader):
            xt, _ = next(target_iter)

            xs = xs.to(device)
            ys = ys.to(device)
            xt = xt.to(device)

            optimizer.zero_grad()

            source_logits, source_features = model(xs, return_features=True)
            _, target_features = model(xt, return_features=True)

            cls_loss = criterion(source_logits, ys)
            align_loss = coral_loss(source_features, target_features)
            total_loss = cls_loss + coral_lambda * align_loss

            total_loss.backward()
            optimizer.step()

            batch_size = xs.size(0)
            running_total_loss += total_loss.item() * batch_size
            running_cls_loss += cls_loss.item() * batch_size
            running_coral_loss += align_loss.item() * batch_size
            num_source_samples += batch_size

        avg_total_loss = running_total_loss / max(num_source_samples, 1)
        avg_cls_loss = running_cls_loss / max(num_source_samples, 1)
        avg_coral_loss = running_coral_loss / max(num_source_samples, 1)

        val_metrics = compute_metrics(model, X_val, y_val, device)
        val_auroc = val_metrics["auroc"]

        history.append({
            "epoch": epoch + 1,
            "total_loss": float(avg_total_loss),
            "classification_loss": float(avg_cls_loss),
            "coral_loss": float(avg_coral_loss),
            "val_auroc": float(val_auroc),
            "val_f1": float(val_metrics["f1"]),
        })

        print(
            f"[CORAL] Epoch {epoch + 1:02d} | "
            f"total_loss={avg_total_loss:.4f} | "
            f"cls_loss={avg_cls_loss:.4f} | "
            f"coral_loss={avg_coral_loss:.6f} | "
            f"val_auroc={val_auroc:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[CORAL] Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        type=str,
        default="NF-ToN-IoT-v3__TO__NF-UNSW-NB15-v3",
        help="Pair folder name inside artifacts/pairs/<stage1_variant>/",
    )
    parser.add_argument("--stage1_variant", type=str, default="balanced_100k")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--source_epochs", type=int, default=30)
    parser.add_argument("--coral_epochs", type=int, default=20)
    parser.add_argument("--source_lr", type=float, default=1e-3)
    parser.add_argument("--coral_lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--coral_lambda", type=float, default=1.0)
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
    print(f"Loaded pair: {args.pair}")
    print(f"Input dim: {input_dim}")

    results_dir = pair_dir / "mlp_coral_results"
    ensure_dir(results_dir)

    # Split target adapt into train/val for adaptation
    Xt_adapt_train, Xt_adapt_val, yt_adapt_train, yt_adapt_val = split_target_adapt(
        data["Xt_adapt"], data["yt_adapt"]
    )

    # ---------------------------------------------
    # 1) SOURCE-ONLY PRETRAINING
    # ---------------------------------------------
    print("\n=== Source-only pretraining ===")
    source_model = MLPWithFeatures(
        input_dim=input_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    source_train_loader = make_loader(
        data["Xs_train"], data["ys_train"],
        batch_size=args.batch_size,
        shuffle=True,
    )

    source_train_info = train_source_only(
        model=source_model,
        train_loader=source_train_loader,
        X_val=data["Xs_val"],
        y_val=data["ys_val"],
        device=device,
        lr=args.source_lr,
        max_epochs=args.source_epochs,
        patience=args.patience,
    )

    source_only_metrics = {
        "source_test": compute_metrics(source_model, data["Xs_test"], data["ys_test"], device),
        "target_test_zero_shot": compute_metrics(source_model, data["Xt_test"], data["yt_test"], device),
    }

    torch.save(source_model.state_dict(), results_dir / "source_only_model.pt")

    # ---------------------------------------------
    # 2) CORAL ADAPTATION
    # ---------------------------------------------
    print("\n=== CORAL adaptation ===")
    coral_model = MLPWithFeatures(
        input_dim=input_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    coral_model.load_state_dict(copy.deepcopy(source_model.state_dict()))

    # Source stays labeled, target is used unlabeled in the CORAL loss.
    source_adapt_loader = make_loader(
        data["Xs_train"], data["ys_train"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    target_adapt_loader = make_loader(
        Xt_adapt_train,
        yt_adapt_train,   # labels are ignored during CORAL adaptation
        batch_size=args.batch_size,
        shuffle=True,
    )

    coral_train_info = train_coral_adaptation(
        model=coral_model,
        source_loader=source_adapt_loader,
        target_loader=target_adapt_loader,
        X_val=Xt_adapt_val,
        y_val=yt_adapt_val,
        device=device,
        coral_lambda=args.coral_lambda,
        lr=args.coral_lr,
        max_epochs=args.coral_epochs,
        patience=args.patience,
    )

    coral_metrics = {
        "source_test_after_coral": compute_metrics(coral_model, data["Xs_test"], data["ys_test"], device),
        "target_test_after_coral": compute_metrics(coral_model, data["Xt_test"], data["yt_test"], device),
    }

    torch.save(coral_model.state_dict(), results_dir / "coral_model.pt")

    # ---------------------------------------------
    # 3) SUMMARY
    # ---------------------------------------------
    source_auroc_before = source_only_metrics["source_test"]["auroc"]
    source_auroc_after = coral_metrics["source_test_after_coral"]["auroc"]
    target_auroc_zero_shot = source_only_metrics["target_test_zero_shot"]["auroc"]
    target_auroc_coral = coral_metrics["target_test_after_coral"]["auroc"]

    summary = {
        "pair": args.pair,
        "stage1_variant": args.stage1_variant,
        "device": str(device),
        "model": {
            "type": "MLPWithFeatures",
            "input_dim": int(input_dim),
            "hidden_dims": list(args.hidden_dims),
            "dropout": float(args.dropout),
        },
        "training_config": {
            "batch_size": int(args.batch_size),
            "source_epochs": int(args.source_epochs),
            "coral_epochs": int(args.coral_epochs),
            "source_lr": float(args.source_lr),
            "coral_lr": float(args.coral_lr),
            "coral_lambda": float(args.coral_lambda),
            "patience": int(args.patience),
            "random_state": int(RANDOM_STATE),
        },
        "source_only": {
            "training": source_train_info,
            "metrics": source_only_metrics,
        },
        "coral_adaptation": {
            "training": coral_train_info,
            "metrics": coral_metrics,
        },
        "comparison": {
            "target_auroc_zero_shot": float(target_auroc_zero_shot),
            "target_auroc_after_coral": float(target_auroc_coral),
            "target_auroc_gain_from_coral": float(target_auroc_coral - target_auroc_zero_shot),
            "source_auroc_before_coral": float(source_auroc_before),
            "source_auroc_after_coral": float(source_auroc_after),
            "source_auroc_drop_after_coral": float(source_auroc_before - source_auroc_after),
        },
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Final summary ===")
    print(json.dumps(summary["comparison"], indent=2))
    print(f"\nSaved results to: {results_dir}")


if __name__ == "__main__":
    main()