import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PAIR_DIR, RANDOM_STATE


# --------------------------------------------------
# Presentation-quality plot style
# --------------------------------------------------
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


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_pair(pair_dir: Path) -> dict[str, np.ndarray]:
    return {
        "Xs_train": np.load(pair_dir / "Xs_train.npy"),
        "ys_train": np.load(pair_dir / "ys_train.npy"),
        "Xs_val": np.load(pair_dir / "Xs_val.npy"),
        "ys_val": np.load(pair_dir / "ys_val.npy"),
        "Xs_test": np.load(pair_dir / "Xs_test.npy"),
        "ys_test": np.load(pair_dir / "ys_test.npy"),
        "Xt_test": np.load(pair_dir / "Xt_test.npy"),
        "yt_test": np.load(pair_dir / "yt_test.npy"),
    }


def make_x_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# --------------------------------------------------
# Leakage-safe robust preprocessing
# --------------------------------------------------
def fit_quantile_bounds(
    X_source_train: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature clipping bounds using SOURCE TRAINING data only."""
    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError("Require 0 <= lower_q < upper_q <= 1.")

    lower = np.quantile(X_source_train, lower_q, axis=0)
    upper = np.quantile(X_source_train, upper_q, axis=0)
    return lower.astype(np.float32), upper.astype(np.float32)


def apply_quantile_clip(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply fixed clipping bounds and return (clipped_X, element_clip_mask)."""
    mask = (X < lower) | (X > upper)
    clipped = np.clip(X, lower, upper).astype(np.float32)
    return clipped, mask


def clipping_split_summary(mask: np.ndarray) -> dict:
    if mask.size == 0:
        return {
            "values_clipped": 0,
            "pct_values_clipped": 0.0,
            "rows_affected": 0,
            "pct_rows_affected": 0.0,
        }

    row_mask = mask.any(axis=1)
    return {
        "values_clipped": int(mask.sum()),
        "pct_values_clipped": float(100.0 * mask.mean()),
        "rows_affected": int(row_mask.sum()),
        "pct_rows_affected": float(100.0 * row_mask.mean()),
    }


def build_feature_clipping_diagnostics(
    original_X: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> pd.DataFrame:
    rows = []
    num_features = original_X["Xs_train"].shape[1]

    for feature_idx in range(num_features):
        row = {
            "feature_index": feature_idx,
            "source_lower_bound": float(lower[feature_idx]),
            "source_upper_bound": float(upper[feature_idx]),
        }
        for split_name in ["Xs_train", "Xs_val", "Xs_test", "Xt_test"]:
            m = masks[split_name][:, feature_idx]
            row[f"{split_name}_pct_clipped"] = float(100.0 * m.mean())
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------
# Autoencoder
# --------------------------------------------------
class Autoencoder(nn.Module):
    """
    Simple deterministic autoencoder.

    Default architecture for the 49 common NetFlow features:
        49 -> 32 -> 16 -> 8 -> 16 -> 32 -> 49

    The bottleneck output is the learned latent representation Z.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (32, 16),
        latent_dim: int = 8,
    ):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h

        # No output activation: pair features were already StandardScaled
        # using the source-training scaler in prepare_pair.py.
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def evaluate_reconstruction_loss(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    loader = make_x_loader(X, batch_size=batch_size, shuffle=False)
    criterion = nn.MSELoss(reduction="sum")

    total_squared_error = 0.0
    total_elements = 0

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, _ = model(xb)
            total_squared_error += criterion(x_hat, xb).item()
            total_elements += xb.numel()

    return float(total_squared_error / max(total_elements, 1))


def train_autoencoder(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 50,
    patience: int = 7,
) -> dict:
    """
    Train ONLY on source-domain data.

    Model selection also uses source validation reconstruction loss only.
    No target examples or target labels are used for AE training/selection.
    """
    train_loader = make_x_loader(X_train, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val_loss = np.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        total_n = 0

        for (xb,) in train_loader:
            xb = xb.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            total_n += bs

        train_loss = running_loss / max(total_n, 1)
        val_loss = evaluate_reconstruction_loss(
            model,
            X_val,
            device=device,
            batch_size=batch_size,
        )

        history.append({
            "epoch": epoch + 1,
            "train_reconstruction_mse": float(train_loss),
            "source_val_reconstruction_mse": float(val_loss),
        })

        print(
            f"[AE] Epoch {epoch + 1:02d} | "
            f"train_mse={train_loss:.6f} | "
            f"source_val_mse={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[AE] Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_source_val_reconstruction_mse": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "history": history,
    }


# --------------------------------------------------
# Batched inference
# --------------------------------------------------
def encode_array(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    loader = make_x_loader(X, batch_size=batch_size, shuffle=False)
    chunks = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            z = model.encode(xb)
            chunks.append(z.cpu().numpy())

    return np.vstack(chunks).astype(np.float32)


def reconstruction_errors(
    model: Autoencoder,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Per-sample MSE in the source-standardised 49-D feature space."""
    model.eval()
    loader = make_x_loader(X, batch_size=batch_size, shuffle=False)
    chunks = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, _ = model(xb)
            err = torch.mean((x_hat - xb) ** 2, dim=1)
            chunks.append(err.cpu().numpy())

    return np.concatenate(chunks).astype(np.float64)


# --------------------------------------------------
# Latent-space shift metrics
# --------------------------------------------------
def source_standardise_latent(
    Zs_train: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Standardise latent dimensions using SOURCE-TRAIN latent statistics only.

    This makes latent Wasserstein values interpretable in source-SD units and
    avoids fitting any normalisation parameters to target data.
    """
    mean = Zs_train.mean(axis=0, keepdims=True)
    std = Zs_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    return tuple(((Z - mean) / std).astype(np.float32) for Z in arrays)


def clipped_wasserstein_1d(
    xs: np.ndarray,
    xt: np.ndarray,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> float:
    combined = np.concatenate([xs, xt])
    lo = np.quantile(combined, lower_q)
    hi = np.quantile(combined, upper_q)

    xs_clip = np.clip(xs, lo, hi)
    xt_clip = np.clip(xt, lo, hi)
    return float(wasserstein_distance(xs_clip, xt_clip))


def latent_wasserstein_stats(Zs: np.ndarray, Zt: np.ndarray) -> dict:
    distances = np.array([
        clipped_wasserstein_1d(Zs[:, i], Zt[:, i])
        for i in range(Zs.shape[1])
    ])

    return {
        "per_dimension": [float(v) for v in distances],
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def balanced_domain_classifier_auroc(
    Zs: np.ndarray,
    Zt: np.ndarray,
    random_state: int,
    max_per_domain: int = 20_000,
) -> dict:
    """
    Measures how easily a linear classifier can distinguish source vs target.

    AUROC ~0.5 -> domains overlap strongly in this representation.
    AUROC ->1.0 -> domains remain strongly separable.
    """
    rng = np.random.default_rng(random_state)
    n = min(len(Zs), len(Zt), max_per_domain)

    idx_s = rng.choice(len(Zs), size=n, replace=False)
    idx_t = rng.choice(len(Zt), size=n, replace=False)

    X = np.vstack([Zs[idx_s], Zt[idx_t]])
    y = np.concatenate([
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    return {
        "auroc": float(roc_auc_score(y_test, probs)),
        "samples_per_domain": int(n),
    }


def class_conditional_latent_shift(
    Zs: np.ndarray,
    ys: np.ndarray,
    Zt: np.ndarray,
    yt: np.ndarray,
    random_state: int,
) -> dict:
    result = {}

    for class_value, class_name in [(0, "benign"), (1, "attack")]:
        Zs_c = Zs[ys == class_value]
        Zt_c = Zt[yt == class_value]

        if len(Zs_c) < 2 or len(Zt_c) < 2:
            result[class_name] = None
            continue

        result[class_name] = {
            "source_samples": int(len(Zs_c)),
            "target_samples": int(len(Zt_c)),
            "wasserstein": latent_wasserstein_stats(Zs_c, Zt_c),
            "domain_classifier": balanced_domain_classifier_auroc(
                Zs_c,
                Zt_c,
                random_state=random_state + class_value + 1,
            ),
        }

    benign_w = (
        result["benign"]["wasserstein"]["mean"]
        if result.get("benign") is not None
        else np.nan
    )
    attack_w = (
        result["attack"]["wasserstein"]["mean"]
        if result.get("attack") is not None
        else np.nan
    )

    result["summary"] = {
        "benign_latent_wasserstein_mean": float(benign_w),
        "attack_latent_wasserstein_mean": float(attack_w),
        "attack_minus_benign_wasserstein": float(attack_w - benign_w),
        "overall_class_conditional_latent_shift_mean": float(
            np.nanmean([benign_w, attack_w])
        ),
    }

    return result


# --------------------------------------------------
# Does latent space preserve attack information?
# --------------------------------------------------
def latent_attack_classifier(
    Zs_train: np.ndarray,
    ys_train: np.ndarray,
    Zs_test: np.ndarray,
    ys_test: np.ndarray,
    Zt_test: np.ndarray,
    yt_test: np.ndarray,
    random_state: int,
) -> dict:
    """
    Train a simple linear attack classifier on SOURCE latent codes only.

    This is diagnostic, not the main NIDS model. It asks whether the latent
    representation keeps security-relevant class information and whether that
    information transfers to the target domain.
    """
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(Zs_train, ys_train)

    source_probs = clf.predict_proba(Zs_test)[:, 1]
    target_probs = clf.predict_proba(Zt_test)[:, 1]

    return {
        "source_test_auroc": float(roc_auc_score(ys_test, source_probs)),
        "target_zero_shot_auroc": float(roc_auc_score(yt_test, target_probs)),
    }


# --------------------------------------------------
# Reconstruction-error analysis
# --------------------------------------------------
def error_summary(errors: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.quantile(errors, 0.90)),
        "p95": float(np.quantile(errors, 0.95)),
        "p99": float(np.quantile(errors, 0.99)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def reconstruction_shift_summary(
    source_errors: np.ndarray,
    ys: np.ndarray,
    target_errors: np.ndarray,
    yt: np.ndarray,
) -> dict:
    source = error_summary(source_errors)
    target = error_summary(target_errors)

    result = {
        "source": source,
        "target": target,
        "target_minus_source_mean": float(target["mean"] - source["mean"]),
        "target_to_source_mean_ratio": safe_ratio(target["mean"], source["mean"]),
        "target_to_source_median_ratio": safe_ratio(target["median"], source["median"]),
        "target_to_source_p90_ratio": safe_ratio(target["p90"], source["p90"]),
        "target_to_source_p95_ratio": safe_ratio(target["p95"], source["p95"]),
        "target_to_source_p99_ratio": safe_ratio(target["p99"], source["p99"]),
        "by_class": {},
    }

    for class_value, class_name in [(0, "benign"), (1, "attack")]:
        s = source_errors[ys == class_value]
        tt = target_errors[yt == class_value]

        s_summary = error_summary(s)
        t_summary = error_summary(tt)

        result["by_class"][class_name] = {
            "source": s_summary,
            "target": t_summary,
            "target_minus_source_mean": float(t_summary["mean"] - s_summary["mean"]),
            "target_to_source_mean_ratio": safe_ratio(t_summary["mean"], s_summary["mean"]),
            "target_to_source_median_ratio": safe_ratio(t_summary["median"], s_summary["median"]),
            "target_to_source_p90_ratio": safe_ratio(t_summary["p90"], s_summary["p90"]),
            "target_to_source_p95_ratio": safe_ratio(t_summary["p95"], s_summary["p95"]),
            "target_to_source_p99_ratio": safe_ratio(t_summary["p99"], s_summary["p99"]),
        }

    return result


# --------------------------------------------------
# Visualisations
# --------------------------------------------------
def sample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def plot_latent_by_domain(
    Zs: np.ndarray,
    Zt: np.ndarray,
    save_path: Path,
    random_state: int,
    max_points_per_domain: int = 5000,
) -> None:
    rng = np.random.default_rng(random_state)
    idx_s = sample_indices(len(Zs), max_points_per_domain, rng)
    idx_t = sample_indices(len(Zt), max_points_per_domain, rng)

    Z = np.vstack([Zs[idx_s], Zt[idx_t]])
    labels = np.array(["Source"] * len(idx_s) + ["Target"] * len(idx_t))

    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(Z)

    plt.figure(figsize=(8.5, 5.5))
    for name in ["Source", "Target"]:
        mask = labels == name
        plt.scatter(
            Zp[mask, 0],
            Zp[mask, 1],
            s=14,
            alpha=0.40,
            label=name,
        )

    plt.xlabel(f"Latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"Latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(frameon=False)
    savefig(save_path)


def plot_latent_by_domain_and_class(
    Zs: np.ndarray,
    ys: np.ndarray,
    Zt: np.ndarray,
    yt: np.ndarray,
    save_path: Path,
    random_state: int,
    max_points_per_group: int = 2500,
) -> None:
    rng = np.random.default_rng(random_state)

    groups = []
    for Z, y, domain in [(Zs, ys, "Source"), (Zt, yt, "Target")]:
        for class_value, class_name in [(0, "Benign"), (1, "Attack")]:
            group_idx = np.where(y == class_value)[0]
            chosen_local = sample_indices(len(group_idx), max_points_per_group, rng)
            chosen_idx = group_idx[chosen_local]
            groups.append((f"{domain} {class_name}", Z[chosen_idx]))

    Z = np.vstack([g[1] for g in groups])
    group_labels = np.concatenate([
        np.array([name] * len(values))
        for name, values in groups
    ])

    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(Z)

    plt.figure(figsize=(9, 5.8))
    for name, _ in groups:
        mask = group_labels == name
        plt.scatter(
            Zp[mask, 0],
            Zp[mask, 1],
            s=13,
            alpha=0.38,
            label=name,
        )

    plt.xlabel(f"Latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"Latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(frameon=False, ncol=2)
    savefig(save_path)


def plot_reconstruction_errors(
    source_errors: np.ndarray,
    target_errors: np.ndarray,
    save_path: Path,
) -> None:
    # Clip only for visualisation so a few extreme errors do not flatten the plot.
    combined = np.concatenate([source_errors, target_errors])
    upper = np.quantile(combined, 0.99)

    plt.figure(figsize=(8.5, 5.5))
    plt.hist(
        np.clip(source_errors, 0, upper),
        bins=60,
        alpha=0.55,
        density=True,
        label="Source",
    )
    plt.hist(
        np.clip(target_errors, 0, upper),
        bins=60,
        alpha=0.55,
        density=True,
        label="Target",
    )
    plt.xlabel("Per-sample reconstruction MSE (clipped at 99th percentile for display)")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    savefig(save_path)


def plot_latent_by_domain_robust(
    Zs: np.ndarray,
    Zt: np.ndarray,
    clip_lower: np.ndarray,
    clip_upper: np.ndarray,
    save_path: Path,
    random_state: int,
    max_points_per_domain: int = 5000,
) -> None:
    """
    Robust visualisation only. Latent dimensions are clipped using SOURCE-TRAIN
    latent bounds before PCA so a few target extremes do not determine the plot.
    This does not alter any reported latent-shift metrics.
    """
    rng = np.random.default_rng(random_state)
    idx_s = sample_indices(len(Zs), max_points_per_domain, rng)
    idx_t = sample_indices(len(Zt), max_points_per_domain, rng)

    Zs_plot = np.clip(Zs[idx_s], clip_lower, clip_upper)
    Zt_plot = np.clip(Zt[idx_t], clip_lower, clip_upper)
    Z = np.vstack([Zs_plot, Zt_plot])
    labels = np.array(["Source"] * len(idx_s) + ["Target"] * len(idx_t))

    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(Z)

    plt.figure(figsize=(8.5, 5.5))
    for name in ["Source", "Target"]:
        mask = labels == name
        plt.scatter(Zp[mask, 0], Zp[mask, 1], s=14, alpha=0.40, label=name)

    plt.xlabel(f"Robust latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"Robust latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(frameon=False)
    savefig(save_path)


def plot_latent_by_domain_and_class_robust(
    Zs: np.ndarray,
    ys: np.ndarray,
    Zt: np.ndarray,
    yt: np.ndarray,
    clip_lower: np.ndarray,
    clip_upper: np.ndarray,
    save_path: Path,
    random_state: int,
    max_points_per_group: int = 2500,
) -> None:
    rng = np.random.default_rng(random_state)

    groups = []
    for Z, y, domain in [(Zs, ys, "Source"), (Zt, yt, "Target")]:
        for class_value, class_name in [(0, "Benign"), (1, "Attack")]:
            group_idx = np.where(y == class_value)[0]
            chosen_local = sample_indices(len(group_idx), max_points_per_group, rng)
            chosen_idx = group_idx[chosen_local]
            values = np.clip(Z[chosen_idx], clip_lower, clip_upper)
            groups.append((f"{domain} {class_name}", values))

    Z = np.vstack([g[1] for g in groups])
    group_labels = np.concatenate([
        np.array([name] * len(values))
        for name, values in groups
    ])

    pca = PCA(n_components=2, random_state=random_state)
    Zp = pca.fit_transform(Z)

    plt.figure(figsize=(9, 5.8))
    for name, _ in groups:
        mask = group_labels == name
        plt.scatter(Zp[mask, 0], Zp[mask, 1], s=13, alpha=0.38, label=name)

    plt.xlabel(f"Robust latent PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"Robust latent PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(frameon=False, ncol=2)
    savefig(save_path)


def plot_reconstruction_errors_log10(
    source_errors: np.ndarray,
    target_errors: np.ndarray,
    save_path: Path,
) -> None:
    eps = 1e-12
    source_log = np.log10(source_errors + eps)
    target_log = np.log10(target_errors + eps)

    combined = np.concatenate([source_log, target_log])
    lo = np.quantile(combined, 0.005)
    hi = np.quantile(combined, 0.995)

    plt.figure(figsize=(8.5, 5.5))
    plt.hist(np.clip(source_log, lo, hi), bins=60, alpha=0.55, density=True, label="Source")
    plt.hist(np.clip(target_log, lo, hi), bins=60, alpha=0.55, density=True, label="Target")
    plt.xlabel("log10 per-sample reconstruction MSE")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    savefig(save_path)


def load_previous_unclipped_summary(pair_dir: Path) -> dict | None:
    path = pair_dir / "autoencoder_latent_shift_results" / "summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_with_previous_unclipped(previous: dict | None, current: dict) -> dict | None:
    if previous is None:
        return None

    try:
        old_recon = previous["reconstruction_shift"]
        old_median_ratio = safe_ratio(
            old_recon["target"]["median"],
            old_recon["source"]["median"],
        )
        return {
            "previous_latent_domain_auroc": float(previous["latent_shift"]["domain_classifier"]["auroc"]),
            "robust_latent_domain_auroc": float(current["latent_shift"]["domain_classifier"]["auroc"]),
            "latent_domain_auroc_change": float(
                current["latent_shift"]["domain_classifier"]["auroc"]
                - previous["latent_shift"]["domain_classifier"]["auroc"]
            ),
            "previous_reconstruction_median_ratio": float(old_median_ratio),
            "robust_reconstruction_median_ratio": float(
                current["reconstruction_shift"]["target_to_source_median_ratio"]
            ),
        }
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------
# Existing DANIDS results for comparison
# --------------------------------------------------
def load_existing_comparison(pair_dir: Path, pair: str, stage1_variant: str) -> dict:
    result = {
        "raw_shift": None,
        "mlp_baseline": None,
    }

    shift_path = PAIR_DIR / stage1_variant / "shift_summary.json"
    if shift_path.exists():
        with open(shift_path, "r", encoding="utf-8") as f:
            shift_data = json.load(f)

        if pair in shift_data:
            info = shift_data[pair]
            result["raw_shift"] = {
                "clipped_wasserstein_mean": info["wasserstein"]["clipped_mean"],
                "domain_classifier_auroc": info["domain_classifier"]["auroc"],
                "class_conditional_shift_mean": info["class_conditional_shift"]["summary"][
                    "overall_class_conditional_shift_mean"
                ],
                "shift_type": info["shift_classification"]["shift_type"],
            }

    baseline_path = pair_dir / "mlp_baseline_results" / "summary.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        comp = baseline["comparison"]
        result["mlp_baseline"] = {
            "target_zero_shot_auroc": comp["target_auroc_zero_shot"],
            "target_finetuned_auroc": comp["target_auroc_after_finetune"],
            "target_gain_from_finetune": comp["target_auroc_gain_from_finetune"],
            "source_auroc_before_finetune": comp["source_auroc_before_finetune"],
            "source_auroc_after_finetune": comp["source_auroc_after_finetune"],
            "source_auroc_drop_after_finetune": comp["source_auroc_drop_after_finetune"],
        }

    return result


# --------------------------------------------------
# One-pair experiment
# --------------------------------------------------
def analyse_pair(args, pair: str, device: torch.device) -> dict:
    print("\n" + "=" * 80)
    print(f"ROBUST AUTOENCODER LATENT-SHIFT ANALYSIS: {pair}")
    print("=" * 80)

    pair_dir = PAIR_DIR / args.stage1_variant / pair
    if not pair_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {pair_dir}")

    data = load_pair(pair_dir)
    input_dim = data["Xs_train"].shape[1]

    results_dir = pair_dir / "autoencoder_latent_shift_robust_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 0) Leakage-safe robust input clipping.
    #    Bounds are fitted on Xs_train only and then frozen.
    # --------------------------------------------------
    input_lower, input_upper = fit_quantile_bounds(
        data["Xs_train"],
        lower_q=args.input_clip_lower,
        upper_q=args.input_clip_upper,
    )

    original_X = {
        key: data[key]
        for key in ["Xs_train", "Xs_val", "Xs_test", "Xt_test"]
    }
    clipped_X = {}
    clip_masks = {}
    clipping_summary = {}

    for key, X in original_X.items():
        clipped_X[key], clip_masks[key] = apply_quantile_clip(X, input_lower, input_upper)
        clipping_summary[key] = clipping_split_summary(clip_masks[key])

    np.save(results_dir / "source_feature_clip_lower.npy", input_lower)
    np.save(results_dir / "source_feature_clip_upper.npy", input_upper)

    feature_diag = build_feature_clipping_diagnostics(
        original_X=original_X,
        masks=clip_masks,
        lower=input_lower,
        upper=input_upper,
    )
    feature_diag.to_csv(results_dir / "feature_clipping_diagnostics.csv", index=False)

    # 1) Train AE on robustly clipped SOURCE data only.
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
    ).to(device)

    training_info = train_autoencoder(
        model=model,
        X_train=clipped_X["Xs_train"],
        X_val=clipped_X["Xs_val"],
        device=device,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
    )

    torch.save(model.state_dict(), results_dir / "source_autoencoder.pt")

    # 2) Encode clipped source train/test and target test.
    Zs_train_raw = encode_array(
        model, clipped_X["Xs_train"], device, args.inference_batch_size
    )
    Zs_test_raw = encode_array(
        model, clipped_X["Xs_test"], device, args.inference_batch_size
    )
    Zt_test_raw = encode_array(
        model, clipped_X["Xt_test"], device, args.inference_batch_size
    )

    # Source-train-only normalisation of latent dimensions.
    Zs_train, Zs_test, Zt_test = source_standardise_latent(
        Zs_train_raw,
        Zs_train_raw,
        Zs_test_raw,
        Zt_test_raw,
    )

    # Source-only bounds used only for robust PCA visualisation.
    latent_plot_lower, latent_plot_upper = fit_quantile_bounds(
        Zs_train,
        lower_q=args.latent_plot_clip_lower,
        upper_q=args.latent_plot_clip_upper,
    )

    # 3) Global and class-conditional latent shift.
    #    Metrics use the actual latent codes; robust PCA clipping is display-only.
    global_wasserstein = latent_wasserstein_stats(Zs_test, Zt_test)
    domain_separability = balanced_domain_classifier_auroc(
        Zs_test,
        Zt_test,
        random_state=RANDOM_STATE,
        max_per_domain=args.max_domain_samples,
    )
    class_conditional = class_conditional_latent_shift(
        Zs_test,
        data["ys_test"],
        Zt_test,
        data["yt_test"],
        random_state=RANDOM_STATE,
    )

    # 4) Reconstruction error in the same clipped feature space used to train AE.
    source_errors = reconstruction_errors(
        model, clipped_X["Xs_test"], device, args.inference_batch_size
    )
    target_errors = reconstruction_errors(
        model, clipped_X["Xt_test"], device, args.inference_batch_size
    )
    reconstruction = reconstruction_shift_summary(
        source_errors,
        data["ys_test"],
        target_errors,
        data["yt_test"],
    )

    # 5) Diagnostic: benign-vs-attack information in the latent representation.
    attack_classifier = latent_attack_classifier(
        Zs_train,
        data["ys_train"],
        Zs_test,
        data["ys_test"],
        Zt_test,
        data["yt_test"],
        random_state=RANDOM_STATE,
    )

    existing = load_existing_comparison(
        pair_dir=pair_dir,
        pair=pair,
        stage1_variant=args.stage1_variant,
    )

    summary = {
        "pair": pair,
        "stage1_variant": args.stage1_variant,
        "device": str(device),
        "protocol": {
            "autoencoder_training_data": "Xs_train only after source-derived clipping",
            "autoencoder_early_stopping": "Xs_val reconstruction MSE only",
            "target_data_used_during_autoencoder_training": False,
            "target_labels_used_during_autoencoder_training": False,
            "input_clipping": (
                f"per-feature quantiles [{args.input_clip_lower}, {args.input_clip_upper}] "
                "fitted on Xs_train only and frozen for every split"
            ),
            "latent_standardisation": "mean/std fitted on source-train latent codes only",
            "robust_pca_display": (
                f"latent [{args.latent_plot_clip_lower}, {args.latent_plot_clip_upper}] "
                "bounds fitted on source-train latent codes only; display only"
            ),
            "interpretation": (
                "Latent domain AUROC near 0.5 indicates source/target overlap; "
                "values near 1.0 indicate persistent domain separability."
            ),
        },
        "model": {
            "input_dim": int(input_dim),
            "hidden_dims": list(args.hidden_dims),
            "latent_dim": int(args.latent_dim),
        },
        "training_config": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "random_state": int(RANDOM_STATE),
            "input_clip_lower": float(args.input_clip_lower),
            "input_clip_upper": float(args.input_clip_upper),
        },
        "input_clipping": {
            "split_summary": clipping_summary,
        },
        "training": training_info,
        "latent_shift": {
            "global_wasserstein": global_wasserstein,
            "domain_classifier": domain_separability,
            "class_conditional": class_conditional,
        },
        "reconstruction_shift": reconstruction,
        "latent_attack_classifier": attack_classifier,
        "existing_danids_results": existing,
    }

    previous = load_previous_unclipped_summary(pair_dir)
    summary["robustness_comparison_to_previous_unclipped_run"] = (
        compare_with_previous_unclipped(previous, summary)
    )

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame({
        "latent_dimension": np.arange(args.latent_dim),
        "clipped_wasserstein": global_wasserstein["per_dimension"],
    }).to_csv(results_dir / "latent_wasserstein_by_dimension.csv", index=False)

    # Full plots preserve the complete latent spread.
    plot_latent_by_domain(
        Zs_test,
        Zt_test,
        results_dir / "latent_pca_by_domain_full.png",
        random_state=RANDOM_STATE,
    )
    plot_latent_by_domain_and_class(
        Zs_test,
        data["ys_test"],
        Zt_test,
        data["yt_test"],
        results_dir / "latent_pca_by_domain_and_class_full.png",
        random_state=RANDOM_STATE,
    )

    # Robust PCA views suppress extreme target influence for interpretation only.
    plot_latent_by_domain_robust(
        Zs_test,
        Zt_test,
        latent_plot_lower,
        latent_plot_upper,
        results_dir / "latent_pca_by_domain_robust.png",
        random_state=RANDOM_STATE,
    )
    plot_latent_by_domain_and_class_robust(
        Zs_test,
        data["ys_test"],
        Zt_test,
        data["yt_test"],
        latent_plot_lower,
        latent_plot_upper,
        results_dir / "latent_pca_by_domain_and_class_robust.png",
        random_state=RANDOM_STATE,
    )

    plot_reconstruction_errors(
        source_errors,
        target_errors,
        results_dir / "reconstruction_error_source_vs_target.png",
    )
    plot_reconstruction_errors_log10(
        source_errors,
        target_errors,
        results_dir / "reconstruction_error_log10_source_vs_target.png",
    )

    print("\nKey robust results")
    print(f"Latent Wasserstein mean:       {global_wasserstein['mean']:.4f}")
    print(f"Latent domain AUROC:           {domain_separability['auroc']:.4f}")
    print(
        "Benign latent shift:           "
        f"{class_conditional['summary']['benign_latent_wasserstein_mean']:.4f}"
    )
    print(
        "Attack latent shift:           "
        f"{class_conditional['summary']['attack_latent_wasserstein_mean']:.4f}"
    )
    print(
        "Target/source recon mean:      "
        f"{reconstruction['target_to_source_mean_ratio']:.3f}x"
    )
    print(
        "Target/source recon median:    "
        f"{reconstruction['target_to_source_median_ratio']:.3f}x"
    )
    print(
        "Latent attack AUROC:           "
        f"source={attack_classifier['source_test_auroc']:.4f}, "
        f"target={attack_classifier['target_zero_shot_auroc']:.4f}"
    )
    print(
        "Target feature values clipped: "
        f"{clipping_summary['Xt_test']['pct_values_clipped']:.3f}%"
    )
    print(
        "Target rows affected by clip:  "
        f"{clipping_summary['Xt_test']['pct_rows_affected']:.3f}%"
    )

    comparison = summary["robustness_comparison_to_previous_unclipped_run"]
    if comparison is not None:
        print("\nComparison with previous unclipped run")
        print(
            "Latent domain AUROC: "
            f"{comparison['previous_latent_domain_auroc']:.4f} -> "
            f"{comparison['robust_latent_domain_auroc']:.4f}"
        )
        print(
            "Median recon ratio:  "
            f"{comparison['previous_reconstruction_median_ratio']:.3f}x -> "
            f"{comparison['robust_reconstruction_median_ratio']:.3f}x"
        )

    print(f"Saved to: {results_dir}")
    return summary


# --------------------------------------------------
# Aggregate six-pair analysis
# --------------------------------------------------
def summary_to_row(summary: dict) -> dict:
    latent = summary["latent_shift"]
    cc = latent["class_conditional"]["summary"]
    recon = summary["reconstruction_shift"]
    attack = summary["latent_attack_classifier"]
    existing = summary["existing_danids_results"]

    raw = existing.get("raw_shift") or {}
    baseline = existing.get("mlp_baseline") or {}

    return {
        "pair": summary["pair"],
        "raw_shift_type": raw.get("shift_type"),
        "raw_wasserstein_mean": raw.get("clipped_wasserstein_mean"),
        "raw_domain_auroc": raw.get("domain_classifier_auroc"),
        "raw_class_conditional_shift": raw.get("class_conditional_shift_mean"),
        "latent_wasserstein_mean": latent["global_wasserstein"]["mean"],
        "latent_domain_auroc": latent["domain_classifier"]["auroc"],
        "latent_benign_wasserstein": cc["benign_latent_wasserstein_mean"],
        "latent_attack_wasserstein": cc["attack_latent_wasserstein_mean"],
        "latent_attack_minus_benign": cc["attack_minus_benign_wasserstein"],
        "target_source_reconstruction_mean_ratio": recon["target_to_source_mean_ratio"],
        "target_source_reconstruction_median_ratio": recon["target_to_source_median_ratio"],
        "latent_attack_source_auroc": attack["source_test_auroc"],
        "latent_attack_target_zero_shot_auroc": attack["target_zero_shot_auroc"],
        "mlp_target_zero_shot_auroc": baseline.get("target_zero_shot_auroc"),
        "mlp_target_finetuned_auroc": baseline.get("target_finetuned_auroc"),
        "mlp_source_forgetting": baseline.get("source_auroc_drop_after_finetune"),
    }


def short_pair(pair: str) -> str:
    return (
        pair.replace("NF-", "")
        .replace("-v3", "")
        .replace("CSE-CIC-IDS2018", "CIC")
        .replace("UNSW-NB15", "UNSW")
        .replace("ToN-IoT", "ToN")
        .replace("__TO__", " → ")
    )


def plot_aggregate_relationships(df: pd.DataFrame, output_dir: Path) -> None:
    valid = df.dropna(subset=["latent_domain_auroc", "mlp_target_zero_shot_auroc"])
    if len(valid) >= 2:
        plt.figure(figsize=(8.5, 5.5))
        plt.scatter(
            valid["latent_domain_auroc"],
            valid["mlp_target_zero_shot_auroc"],
            s=80,
            alpha=0.85,
        )
        for _, row in valid.iterrows():
            plt.annotate(
                short_pair(row["pair"]),
                (row["latent_domain_auroc"], row["mlp_target_zero_shot_auroc"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=10,
            )
        plt.xlabel("Latent-space domain classifier AUROC")
        plt.ylabel("MLP zero-shot target AUROC")
        savefig(output_dir / "latent_domain_shift_vs_zero_shot.png")

    valid = df.dropna(subset=["latent_attack_wasserstein", "mlp_source_forgetting"])
    if len(valid) >= 2:
        plt.figure(figsize=(8.5, 5.5))
        plt.scatter(
            valid["latent_attack_wasserstein"],
            valid["mlp_source_forgetting"],
            s=80,
            alpha=0.85,
        )
        for _, row in valid.iterrows():
            plt.annotate(
                short_pair(row["pair"]),
                (row["latent_attack_wasserstein"], row["mlp_source_forgetting"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=10,
            )
        plt.xlabel("Attack-class latent Wasserstein shift")
        plt.ylabel("Source AUROC drop after fine-tuning")
        savefig(output_dir / "latent_attack_shift_vs_forgetting.png")


def discover_pairs(stage1_variant: str) -> list[str]:
    base = PAIR_DIR / stage1_variant
    if not base.exists():
        raise FileNotFoundError(f"Pair base directory not found: {base}")

    pairs = []
    for path in sorted(base.iterdir()):
        if (
            path.is_dir()
            and "__TO__" in path.name
            and (path / "Xs_train.npy").exists()
            and (path / "Xt_test.npy").exists()
        ):
            pairs.append(path.name)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pair", type=str)
    mode.add_argument("--all_pairs", action="store_true")

    parser.add_argument("--stage1_variant", default="balanced_100k", type=str)
    parser.add_argument("--latent_dim", default=8, type=int)
    parser.add_argument("--hidden_dims", default=[32, 16], type=int, nargs="+")
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--inference_batch_size", default=4096, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--patience", default=7, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--max_domain_samples", default=20_000, type=int)
    parser.add_argument("--input_clip_lower", default=0.01, type=float)
    parser.add_argument("--input_clip_upper", default=0.99, type=float)
    parser.add_argument("--latent_plot_clip_lower", default=0.01, type=float)
    parser.add_argument("--latent_plot_clip_upper", default=0.99, type=float)
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.pair is not None:
        analyse_pair(args, args.pair, device)
        return

    pairs = discover_pairs(args.stage1_variant)
    if not pairs:
        raise RuntimeError("No prepared source-target pair directories were found.")

    rows = []
    for pair in pairs:
        # Reset seeds so each pair run is independently reproducible.
        set_seed(RANDOM_STATE)
        summary = analyse_pair(args, pair, device)
        rows.append(summary_to_row(summary))

    df = pd.DataFrame(rows)
    output_dir = PAIR_DIR / args.stage1_variant
    csv_path = output_dir / "autoencoder_latent_shift_robust_summary.csv"
    df.to_csv(csv_path, index=False)
    plot_aggregate_relationships(df, output_dir)

    print("\n" + "=" * 80)
    print("ALL-PAIR LATENT-SHIFT ANALYSIS COMPLETE")
    print("=" * 80)
    print(df.to_string(index=False))
    print(f"\nSaved aggregate summary to: {csv_path}")
    print("Generated aggregate plots:")
    print("- latent_domain_shift_vs_zero_shot.png")
    print("- latent_attack_shift_vs_forgetting.png")


if __name__ == "__main__":
    main()