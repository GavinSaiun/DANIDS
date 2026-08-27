import argparse
import copy
import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PAIR_DIR


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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def all_directed_pairs() -> list[str]:
    return [f"{s}__TO__{t}" for s, t in permutations(DATASETS, 2)]


def load_pair(pair_dir: Path) -> dict[str, np.ndarray]:
    names = [
        "Xs_train", "ys_train", "Xs_val", "ys_val", "Xs_test", "ys_test",
        "Xt_adapt", "yt_adapt", "Xt_test", "yt_test",
    ]
    out = {}
    for name in names:
        p = pair_dir / f"{name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
        out[name] = np.load(p)
    return out


def make_loader(X, y=None, batch_size=512, shuffle=True):
    xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(xt)
    else:
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class MLPWithFeatures(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        z = self.extract_features(x)
        return self.classifier(z).squeeze(1)


def predict_proba(model, X, device, batch_size=8192):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            parts.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(parts)


def metrics(model, X, y, device):
    p = predict_proba(model, X, device)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
    }


def train_source(model, loader, X_val, y_val, device, lr, weight_decay, max_epochs, patience, label):
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_auc, best_epoch, stale = None, -np.inf, -1, 0

    for epoch in range(max_epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        val = metrics(model, X_val, y_val, device)
        print(
            f"[{label}] Epoch {epoch+1:02d} | loss={total_loss/max(total_n,1):.4f} | "
            f"source_val_auroc={val['auroc']:.4f} | source_val_f1={val['f1']:.4f}"
        )
        if val["auroc"] > best_auc:
            best_auc = val["auroc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"[{label}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_auroc": float(best_auc), "best_epoch": int(best_epoch)}


def covariance_matrix(z):
    if z.size(0) <= 1:
        return torch.zeros((z.size(1), z.size(1)), dtype=z.dtype, device=z.device)
    zc = z - z.mean(dim=0, keepdim=True)
    return zc.T @ zc / (z.size(0) - 1)


def coral_loss(zs, zt):
    cs = covariance_matrix(zs)
    ct = covariance_matrix(zt)
    d = zs.size(1)
    return torch.sum((cs - ct) ** 2) / (4.0 * d * d)


def train_coral(
    model,
    source_loader,
    target_loader,
    X_target_val,
    y_target_val,
    device,
    coral_lambda,
    lr,
    weight_decay,
    max_epochs,
    patience,
    label,
):
    """
    Objective = source BCE + lambda * CORAL(source hidden features, target hidden features).
    Target TRAIN labels are never used in the loss.
    Target VAL labels are used for early stopping/model selection, so this is not strictly
    fully-unsupervised domain adaptation.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_auc, best_epoch, stale = None, -np.inf, -1, 0

    for epoch in range(max_epochs):
        model.train()
        s_iter = iter(source_loader)
        t_iter = iter(target_loader)
        steps = max(len(source_loader), len(target_loader))
        sum_loss = sum_bce = sum_coral = 0.0

        for _ in range(steps):
            try:
                xs, ys = next(s_iter)
            except StopIteration:
                s_iter = iter(source_loader)
                xs, ys = next(s_iter)
            try:
                (xt,) = next(t_iter)
            except StopIteration:
                t_iter = iter(target_loader)
                (xt,) = next(t_iter)

            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            opt.zero_grad()
            zs = model.extract_features(xs)
            zt = model.extract_features(xt)
            logits_s = model.classifier(zs).squeeze(1)
            bce = loss_fn(logits_s, ys)
            c = coral_loss(zs, zt)
            loss = bce + coral_lambda * c
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            sum_bce += bce.item()
            sum_coral += c.item()

        val = metrics(model, X_target_val, y_target_val, device)
        print(
            f"[{label}] Epoch {epoch+1:02d} | loss={sum_loss/steps:.4f} | "
            f"bce={sum_bce/steps:.4f} | coral={sum_coral/steps:.6f} | "
            f"target_val_auroc={val['auroc']:.4f} | target_val_f1={val['f1']:.4f}"
        )
        if val["auroc"] > best_auc:
            best_auc = val["auroc"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print(f"[{label}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_target_val_auroc": float(best_auc), "best_epoch": int(best_epoch)}


def sample_per_class(X, y, n_per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in [0, 1]:
        idx = np.where(np.asarray(y) == cls)[0]
        if len(idx) < n_per_class:
            raise ValueError(f"Class {cls} has {len(idx)} rows, requested {n_per_class}.")
        chosen.append(rng.choice(idx, size=n_per_class, replace=False))
    idx = np.concatenate(chosen)
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def split_target_budget(X, y, seed):
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


def derived_scores(source_before, source_after, target_after):
    drop = source_before - source_after
    harmonic = (
        2 * source_after * target_after / (source_after + target_after)
        if source_after + target_after > 0 else np.nan
    )
    return {
        "absolute_source_auroc_drop": float(drop),
        "relative_source_auroc_drop_pct": float(100 * drop / source_before) if source_before else np.nan,
        "mean_source_target_auroc": float((source_after + target_after) / 2),
        "harmonic_source_target_auroc": float(harmonic),
        "worst_domain_auroc": float(min(source_after, target_after)),
    }


def run_pair_seed(
    pair,
    seed,
    data,
    stage1_variant,
    target_budget_per_class,
    coral_lambdas,
    batch_size,
    source_epochs,
    coral_epochs,
    source_lr,
    coral_lr,
    weight_decay,
    patience,
    dropout,
    hidden_dims,
    device,
):
    print("\n" + "=" * 110)
    print(f"PAIR: {pair} | SEED: {seed}")
    print("=" * 110)

    set_seed(seed)
    input_dim = data["Xs_train"].shape[1]
    source_model = MLPWithFeatures(input_dim, hidden_dims, dropout).to(device)
    source_loader = make_loader(data["Xs_train"], data["ys_train"], batch_size, True)
    source_train = train_source(
        source_model, source_loader, data["Xs_val"], data["ys_val"], device,
        source_lr, weight_decay, source_epochs, patience, f"{pair}_source_seed_{seed}"
    )

    source_before = metrics(source_model, data["Xs_test"], data["ys_test"], device)
    target_zero = metrics(source_model, data["Xt_test"], data["yt_test"], device)
    source_state = copy.deepcopy(source_model.state_dict())

    Xtb, ytb, _ = sample_per_class(
        data["Xt_adapt"], data["yt_adapt"], target_budget_per_class, seed + 1000
    )
    Xt_train, Xt_val, yt_train, yt_val = split_target_budget(Xtb, ytb, seed + 2000)

    rows = []
    for lam in coral_lambdas:
        print(f"\n--- {pair} | seed={seed} | CORAL lambda={lam} ---")

        # Separate deterministic adaptation stream for each lambda.
        adapt_seed = seed + 10000 + int(round(float(lam) * 1000))
        set_seed(adapt_seed)

        model = MLPWithFeatures(input_dim, hidden_dims, dropout).to(device)
        model.load_state_dict(copy.deepcopy(source_state))

        s_loader = make_loader(data["Xs_train"], data["ys_train"], batch_size, True)
        # IMPORTANT: target TRAIN labels are intentionally omitted here.
        t_loader = make_loader(Xt_train, None, batch_size, True)

        coral_train = train_coral(
            model, s_loader, t_loader, Xt_val, yt_val, device,
            float(lam), coral_lr, weight_decay, coral_epochs, patience,
            f"{pair}_coral_{lam}_seed_{seed}"
        )

        source_after = metrics(model, data["Xs_test"], data["ys_test"], device)
        target_after = metrics(model, data["Xt_test"], data["yt_test"], device)
        extra = derived_scores(source_before["auroc"], source_after["auroc"], target_after["auroc"])

        rows.append({
            "pair": pair,
            "source_dataset": pair.split("__TO__")[0],
            "target_dataset": pair.split("__TO__")[1],
            "stage1_variant": stage1_variant,
            "seed": int(seed),
            "method": "coral",
            "coral_lambda": float(lam),
            "target_budget_per_class": int(target_budget_per_class),
            "target_adapt_train_total": int(len(yt_train)),
            "target_adapt_val_total": int(len(yt_val)),
            "source_auroc_before": source_before["auroc"],
            "source_f1_before": source_before["f1"],
            "target_auroc_zero_shot": target_zero["auroc"],
            "target_f1_zero_shot": target_zero["f1"],
            "target_auroc_after": target_after["auroc"],
            "target_f1_after": target_after["f1"],
            "target_auroc_gain": target_after["auroc"] - target_zero["auroc"],
            "source_auroc_after": source_after["auroc"],
            "source_f1_after": source_after["f1"],
            **extra,
            "source_best_val_auroc": source_train["best_val_auroc"],
            "source_best_epoch": source_train["best_epoch"],
            "target_best_val_auroc": coral_train["best_target_val_auroc"],
            "target_best_epoch": coral_train["best_epoch"],
        })

        print(
            f"RESULT | target={target_after['auroc']:.4f} | source={source_after['auroc']:.4f} | "
            f"source_drop={extra['absolute_source_auroc_drop']:.4f}"
        )

    return rows


def aggregate(raw_df):
    metrics_cols = [
        "source_auroc_before", "source_f1_before", "target_auroc_zero_shot", "target_f1_zero_shot",
        "target_auroc_after", "target_f1_after", "target_auroc_gain", "source_auroc_after",
        "source_f1_after", "absolute_source_auroc_drop", "relative_source_auroc_drop_pct",
        "mean_source_target_auroc", "harmonic_source_target_auroc", "worst_domain_auroc",
        "target_best_val_auroc", "target_best_epoch",
    ]
    rows = []
    keys = ["pair", "source_dataset", "target_dataset", "coral_lambda"]
    for key, g in raw_df.groupby(keys, sort=False):
        pair, src, tgt, lam = key
        row = {
            "pair": pair,
            "source_dataset": src,
            "target_dataset": tgt,
            "method": "coral",
            "coral_lambda": float(lam),
            "num_seeds": int(len(g)),
            "target_budget_per_class": int(g["target_budget_per_class"].iloc[0]),
        }
        for col in metrics_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def best_lambda_per_pair(agg):
    rows = []
    for pair in agg["pair"].unique():
        d = agg[agg["pair"] == pair].sort_values(
            ["target_best_val_auroc_mean", "worst_domain_auroc_mean"], ascending=[False, False]
        )
        rows.append(d.iloc[0].to_dict())
    return pd.DataFrame(rows)


def short_pair(pair):
    alias = {
        "NF-UNSW-NB15-v3": "UNSW",
        "NF-ToN-IoT-v3": "ToN",
        "NF-CSE-CIC-IDS2018-v3": "CIC",
    }
    s, t = pair.split("__TO__")
    return f"{alias[s]}→{alias[t]}"


def make_plots(agg, best, outdir):
    for metric, ylabel, fname in [
        ("target_auroc_after_mean", "Target AUROC after CORAL", "coral_lambda_sweep_target_auroc.png"),
        ("source_auroc_after_mean", "Source AUROC after CORAL", "coral_lambda_sweep_source_auroc.png"),
        ("worst_domain_auroc_mean", "Worst-domain AUROC", "coral_lambda_sweep_worst_domain.png"),
    ]:
        plt.figure(figsize=(11, 6))
        for pair in agg["pair"].unique():
            d = agg[agg["pair"] == pair].sort_values("coral_lambda")
            plt.plot(d["coral_lambda"], d[metric], marker="o", linewidth=1.8, label=short_pair(pair))
        plt.xscale("log")
        plt.xlabel("CORAL lambda")
        plt.ylabel(ylabel)
        plt.legend(frameon=False, ncol=2)
        savefig(outdir / fname)

    x = np.arange(len(best))
    width = 0.36
    plt.figure(figsize=(11, 6))
    plt.bar(x - width/2, best["source_auroc_after_mean"], width=width, label="Source AUROC")
    plt.bar(x + width/2, best["target_auroc_after_mean"], width=width, label="Target AUROC")
    plt.xticks(x, [short_pair(p) for p in best["pair"]])
    plt.ylabel("AUROC")
    plt.ylim(0, 1.05)
    plt.legend(frameon=False)
    savefig(outdir / "coral_best_source_target_auroc.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_variant", default="balanced_100k")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--target_budget_per_class", type=int, default=10000)
    p.add_argument("--coral_lambdas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--source_epochs", type=int, default=30)
    p.add_argument("--coral_epochs", type=int, default=40)
    p.add_argument("--source_lr", type=float, default=1e-3)
    p.add_argument("--coral_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 128])
    p.add_argument("--pairs", nargs="*", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pairs = args.pairs if args.pairs else all_directed_pairs()
    variant_dir = PAIR_DIR / args.stage1_variant

    missing = [pair for pair in pairs if not (variant_dir / pair).exists()]
    if missing:
        raise FileNotFoundError("Missing pair directories:\n" + "\n".join(missing))

    outdir = variant_dir / "mlp_coral_all_pairs_results"
    outdir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "stage1_variant": args.stage1_variant,
        "pairs": pairs,
        "seeds": args.seeds,
        "target_budget_per_class": args.target_budget_per_class,
        "coral_lambdas": args.coral_lambdas,
        "objective": "source BCE + lambda * Deep CORAL covariance alignment",
        "target_train_labels": "not used in CORAL loss",
        "target_validation_labels": "used for early stopping and lambda selection; therefore not strictly fully-unsupervised DA",
        "test_sets": "source/target test sets are evaluation only",
    }

    rows = []
    for i, pair in enumerate(pairs, 1):
        print("\n" + "#" * 110)
        print(f"PAIR {i}/{len(pairs)}: {pair}")
        print("#" * 110)
        data = load_pair(variant_dir / pair)
        for seed in args.seeds:
            rows += run_pair_seed(
                pair, int(seed), data, args.stage1_variant, args.target_budget_per_class,
                args.coral_lambdas, args.batch_size, args.source_epochs, args.coral_epochs,
                args.source_lr, args.coral_lr, args.weight_decay, args.patience,
                args.dropout, tuple(args.hidden_dims), device,
            )

    raw = pd.DataFrame(rows)
    agg = aggregate(raw)
    best = best_lambda_per_pair(agg)

    raw_path = outdir / "coral_all_pairs_raw_results.csv"
    agg_path = outdir / "coral_all_pairs_lambda_summary.csv"
    best_path = outdir / "coral_all_pairs_best_lambda.csv"
    json_path = outdir / "coral_all_pairs_summary.json"
    raw.to_csv(raw_path, index=False)
    agg.to_csv(agg_path, index=False)
    best.to_csv(best_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "best_lambda_per_pair": best.to_dict(orient="records")}, f, indent=2)

    make_plots(agg, best, outdir)

    print("\n" + "=" * 120)
    print("ALL-PAIRS CORAL EXPERIMENT COMPLETE")
    print("=" * 120)
    show = best[[
        "pair", "coral_lambda", "target_auroc_zero_shot_mean", "target_auroc_after_mean",
        "source_auroc_before_mean", "source_auroc_after_mean", "absolute_source_auroc_drop_mean",
        "worst_domain_auroc_mean", "target_best_val_auroc_mean",
    ]].copy()
    show["pair"] = show["pair"].map(short_pair)
    print(show.to_string(index=False))
    print(f"\nRaw results:      {raw_path}")
    print(f"Lambda summary:   {agg_path}")
    print(f"Best lambda/pair: {best_path}")
    print(f"Summary JSON:     {json_path}")


if __name__ == "__main__":
    main()