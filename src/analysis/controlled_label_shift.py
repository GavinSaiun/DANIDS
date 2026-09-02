"""
controlled_label_shift.py

Controlled pure label / class-prior shift experiment for DANIDS.

This revision is designed to answer not only WHETHER label shift matters,
but WHEN it matters and WHICH correction is appropriate.

Research questions
------------------
RQ2a. Under pure class-prior shift, which metrics change even though
      P(X|Y) is fixed?

RQ2b. Can prior correction recover probability calibration / decision quality
      without representation retraining?

RQ2c. Does the practical impact of label shift depend on classifier
      uncertainty / class overlap?

Construction
------------
A single underlying NIDS dataset is used (default: NF-UNSW-NB15-v3).

By construction:
    P_T(Y) != P_S(Y)
    P_T(X|Y) ~= P_S(X|Y)

The script:
- uses Stage-1 balanced_100k only as a large class-balanced RESERVOIR;
- reconstructs the source at the dataset's natural full-data attack prior;
- creates mutually disjoint source train / validation / test sets;
- creates target-only adaptation and target-test banks disjoint from source;
- changes only the target attack prevalence;
- validates class-conditional invariance with source-vs-target domain AUROC.

Hypotheses
----------
H1. AUROC remains approximately stable as target prevalence changes because
    class-conditional score distributions are unchanged.

H2. Calibration / threshold-dependent behaviour can change under prior shift
    even when AUROC remains stable.

H3. Prior correction can improve calibration / decision quality without
    feature or classifier retraining.

H4. The operational impact of label shift is larger for classifiers with
    more source-class overlap / predictive uncertainty.

Classifiers
-----------
1. MLP:
       input -> 256 -> 128 -> 1
   ReLU, dropout=0.2, BCEWithLogitsLoss, Adam.

2. Logistic regression:
   Lower-capacity probabilistic baseline trained on the exact same source
   split and source-only StandardScaler.

Probability methods
-------------------
For EACH classifier:
1. raw
   Original source posterior probabilities.

2. temperature_scaled
   Temperature scaling fitted ONLY on source validation logits:
       sigmoid(logit / T), T > 0
   No intercept is learned, so calibration cannot arbitrarily shift the
   decision boundary.

3. raw_oracle_prior_corrected
   Binary label-shift prior correction applied directly to RAW probabilities,
   using the known designed target prior. Oracle/reference only.

4. raw_em_prior_corrected
   Target prior estimated by EM from UNLABELLED target-adaptation RAW
   probabilities, then prior correction applied to untouched target test.

Important
---------
The previous Platt-scaling path is deliberately removed from the correction
pipeline because one seed showed severe intercept-induced calibration failure.
This script tests prior correction directly from the source model probabilities
and keeps temperature scaling as a separate conservative calibration baseline.

Outputs
-------
- controlled_label_shift_per_seed.csv
- controlled_label_shift_summary.csv
- controlled_label_shift_source_metrics.csv
- controlled_label_shift_diagnostics_per_seed.csv
- controlled_label_shift_diagnostics_summary.csv
- controlled_label_shift_paired_differences.csv
- controlled_label_shift_paired_difference_summary.csv
- controlled_label_shift_protocol.json
- separate metric plots for MLP and logistic regression

Run
---
python -B -m src.analysis.controlled_label_shift

Suggested location
------------------
src/analysis/controlled_label_shift.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import RANDOM_STATE, STAGE1_DIR


# ============================================================
# Defaults
# ============================================================
DEFAULT_DATASET = "NF-UNSW-NB15-v3"
DEFAULT_POOL_VARIANT = "balanced_100k"

# Five seeds for headline controlled experiments.
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]

DEFAULT_TARGET_PRIORS = [
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
]

DEFAULT_SOURCE_TRAIN_SIZE = 50_000
DEFAULT_SOURCE_VAL_SIZE = 10_000
DEFAULT_SOURCE_TEST_SIZE = 10_000

DEFAULT_TARGET_ADAPT_SIZE = 5_000
DEFAULT_TARGET_TEST_SIZE = 20_000

METHODS = [
    "raw",
    "temperature_scaled",
    "raw_oracle_prior_corrected",
    "raw_em_prior_corrected",
]

CLASSIFIERS = [
    "mlp",
    "logistic_regression",
]

EPS = 1e-7

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


# ============================================================
# Loading
# ============================================================
def load_stage1_summary() -> dict[str, Any]:
    path = STAGE1_DIR / "stage1_summary.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run Stage-1 dataset construction first."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_pool(
    dataset: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    balanced_100k is used only as an underlying reservoir.

    The experiment reconstructs the requested source/target priors itself.
    """
    base = STAGE1_DIR / variant
    x_path = base / f"{dataset}_X.parquet"
    y_path = base / f"{dataset}_y.npy"

    if not x_path.exists():
        raise FileNotFoundError(
            f"Missing feature pool: {x_path}"
        )
    if not y_path.exists():
        raise FileNotFoundError(
            f"Missing labels: {y_path}"
        )

    X = pd.read_parquet(x_path).to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y = np.load(y_path).astype(np.int8)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label mismatch: X={len(X)}, y={len(y)}"
        )

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0,1}}, got {np.unique(y)}"
        )

    return X, y


def natural_attack_prior(
    stage1_summary: dict[str, Any],
    dataset: str,
) -> float:
    try:
        full = (
            stage1_summary["datasets"][dataset]
            ["variants"]["full"]
        )
        return float(
            full["attack"] / full["rows"]
        )
    except KeyError as exc:
        raise KeyError(
            f"Could not find full Stage-1 metadata for {dataset}"
        ) from exc


# ============================================================
# Exact-prior sampling
# ============================================================
def class_counts(
    total: int,
    attack_prior: float,
) -> tuple[int, int]:
    if not 0.0 < attack_prior < 1.0:
        raise ValueError(
            "attack_prior must be strictly between 0 and 1."
        )

    n_attack = int(
        round(total * attack_prior)
    )
    n_attack = min(
        max(n_attack, 1),
        total - 1,
    )
    n_benign = total - n_attack

    return n_benign, n_attack


def take_slice(
    indices: np.ndarray,
    start: int,
    count: int,
    label: str,
) -> tuple[np.ndarray, int]:
    end = start + count

    if end > len(indices):
        raise ValueError(
            f"Not enough {label}: requested through {end:,}, "
            f"but only {len(indices):,} available."
        )

    return indices[start:end], end


def build_source_splits_and_target_banks(
    X: np.ndarray,
    y: np.ndarray,
    source_prior: float,
    source_train_size: int,
    source_val_size: int,
    source_test_size: int,
    target_priors: list[float],
    target_adapt_size: int,
    target_test_size: int,
    seed: int,
) -> dict[str, Any]:
    """
    Build mutually disjoint source and target pools.

    Each class is shuffled ONCE per seed. Source splits consume prefixes.
    Remaining rows are target-only.

    Target conditions use prefixes of fixed class-specific target banks.
    This gives a paired/nested design: the principal manipulated variable is
    class prevalence rather than a completely new random target sample.
    """
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0].copy()
    idx1 = np.where(y == 1)[0].copy()

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    p0 = 0
    p1 = 0

    source_parts: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}

    for split_name, total in [
        ("train", source_train_size),
        ("val", source_val_size),
        ("test", source_test_size),
    ]:
        n0, n1 = class_counts(
            total,
            source_prior,
        )

        selected0, p0 = take_slice(
            idx0,
            p0,
            n0,
            f"source benign ({split_name})",
        )
        selected1, p1 = take_slice(
            idx1,
            p1,
            n1,
            f"source attack ({split_name})",
        )

        selected = np.concatenate(
            [selected0, selected1]
        )

        split_rng = np.random.default_rng(
            seed
            + {
                "train": 11,
                "val": 22,
                "test": 33,
            }[split_name]
        )
        split_rng.shuffle(selected)

        source_parts[split_name] = (
            X[selected],
            y[selected],
            selected,
        )

    remaining0 = idx0[p0:]
    remaining1 = idx1[p1:]

    max_adapt0 = max(
        class_counts(
            target_adapt_size,
            p,
        )[0]
        for p in target_priors
    )
    max_adapt1 = max(
        class_counts(
            target_adapt_size,
            p,
        )[1]
        for p in target_priors
    )
    max_test0 = max(
        class_counts(
            target_test_size,
            p,
        )[0]
        for p in target_priors
    )
    max_test1 = max(
        class_counts(
            target_test_size,
            p,
        )[1]
        for p in target_priors
    )

    required0 = max_adapt0 + max_test0
    required1 = max_adapt1 + max_test1

    if required0 > len(remaining0):
        raise ValueError(
            f"Not enough target-only benign examples. "
            f"Need {required0:,}, have {len(remaining0):,}."
        )

    if required1 > len(remaining1):
        raise ValueError(
            f"Not enough target-only attack examples. "
            f"Need {required1:,}, have {len(remaining1):,}."
        )

    adapt_bank0 = remaining0[:max_adapt0]
    test_bank0 = remaining0[
        max_adapt0 : max_adapt0 + max_test0
    ]

    adapt_bank1 = remaining1[:max_adapt1]
    test_bank1 = remaining1[
        max_adapt1 : max_adapt1 + max_test1
    ]

    all_source_indices = np.concatenate(
        [
            source_parts["train"][2],
            source_parts["val"][2],
            source_parts["test"][2],
        ]
    )

    all_target_bank_indices = np.concatenate(
        [
            adapt_bank0,
            adapt_bank1,
            test_bank0,
            test_bank1,
        ]
    )

    if np.intersect1d(
        all_source_indices,
        all_target_bank_indices,
    ).size > 0:
        raise RuntimeError(
            "Source and target banks overlap."
        )

    if np.intersect1d(
        np.concatenate([adapt_bank0, adapt_bank1]),
        np.concatenate([test_bank0, test_bank1]),
    ).size > 0:
        raise RuntimeError(
            "Target adaptation and target test banks overlap."
        )

    return {
        "source_train": source_parts["train"][:2],
        "source_val": source_parts["val"][:2],
        "source_test": source_parts["test"][:2],
        "target_adapt_bank0": adapt_bank0,
        "target_adapt_bank1": adapt_bank1,
        "target_test_bank0": test_bank0,
        "target_test_bank1": test_bank1,
    }


def construct_target_condition(
    X: np.ndarray,
    y: np.ndarray,
    banks: dict[str, Any],
    target_prior: float,
    target_adapt_size: int,
    target_test_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    adapt0, adapt1 = class_counts(
        target_adapt_size,
        target_prior,
    )
    test0, test1 = class_counts(
        target_test_size,
        target_prior,
    )

    adapt_idx = np.concatenate(
        [
            banks["target_adapt_bank0"][:adapt0],
            banks["target_adapt_bank1"][:adapt1],
        ]
    )

    test_idx = np.concatenate(
        [
            banks["target_test_bank0"][:test0],
            banks["target_test_bank1"][:test1],
        ]
    )

    adapt_rng = np.random.default_rng(
        seed + 101
    )
    test_rng = np.random.default_rng(
        seed + 202
    )

    adapt_rng.shuffle(adapt_idx)
    test_rng.shuffle(test_idx)

    if np.intersect1d(
        adapt_idx,
        test_idx,
    ).size > 0:
        raise RuntimeError(
            "Condition target adaptation/test overlap."
        )

    return {
        "X_adapt": X[adapt_idx],
        "y_adapt": y[adapt_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }


# ============================================================
# MLP
# ============================================================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h

        layers.append(
            nn.Linear(prev, 1)
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(
                X,
                dtype=torch.float32,
            ),
            torch.tensor(
                y,
                dtype=torch.float32,
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def predict_mlp_logits(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    chunks = []

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            batch_size,
        ):
            xb = torch.tensor(
                X[
                    start : start + batch_size
                ],
                dtype=torch.float32,
                device=device,
            )

            chunks.append(
                model(xb)
                .detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(chunks)


def sigmoid(
    logits: np.ndarray,
) -> np.ndarray:
    logits = np.clip(
        logits,
        -50.0,
        50.0,
    )
    return (
        1.0
        / (1.0 + np.exp(-logits))
    )


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[MLP, dict[str, Any]]:
    set_seed(seed)

    model = MLP(
        input_dim=X_train.shape[1],
        hidden_dims=tuple(
            args.hidden_dims
        ),
        dropout=args.dropout,
    ).to(device)

    loader = make_loader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(
        args.max_epochs
    ):
        model.train()

        total_loss = 0.0
        total_n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(
                logits,
                yb,
            )
            loss.backward()
            optimizer.step()

            batch_n = xb.size(0)
            total_loss += (
                loss.item() * batch_n
            )
            total_n += batch_n

        train_loss = (
            total_loss / max(total_n, 1)
        )

        val_logits = predict_mlp_logits(
            model,
            X_val,
            device,
        )
        val_probs = sigmoid(
            val_logits
        )
        val_auroc = float(
            roc_auc_score(
                y_val,
                val_probs,
            )
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    train_loss
                ),
                "val_auroc": val_auroc,
            }
        )

        print(
            f"[MLP seed={seed}] "
            f"epoch={epoch + 1:02d} "
            f"loss={train_loss:.5f} "
            f"val_auroc={val_auroc:.5f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(
                f"[MLP seed={seed}] "
                f"early stopping at "
                f"epoch {epoch + 1}"
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model, {
        "best_val_auroc": float(
            best_val_auroc
        ),
        "best_epoch": int(
            best_epoch
        ),
        "history": history,
    }


# ============================================================
# Logistic baseline
# ============================================================
def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[
    LogisticRegression,
    dict[str, Any],
]:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        clf.fit(
            X_train,
            y_train,
        )

    convergence_warnings = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    n_iter = int(
        np.max(clf.n_iter_)
    )

    return clf, {
        "converged": (
            len(convergence_warnings)
            == 0
        ),
        "iterations": n_iter,
        "warnings": sorted(
            set(convergence_warnings)
        ),
    }


# ============================================================
# Temperature scaling
# ============================================================
def binary_nll_from_logits(
    logits: np.ndarray,
    y: np.ndarray,
    temperature: float,
) -> float:
    probs = sigmoid(
        logits / temperature
    )

    return float(
        log_loss(
            y,
            np.clip(
                probs,
                EPS,
                1.0 - EPS,
            ),
            labels=[0, 1],
        )
    )


def fit_temperature(
    val_logits: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, Any]:
    """
    Fit one positive scalar T on source validation only.

    Optimisation occurs in log(T) so positivity is guaranteed.
    """
    def objective(
        log_temperature: float,
    ) -> float:
        temperature = float(
            np.exp(log_temperature)
        )

        return binary_nll_from_logits(
            val_logits,
            val_y,
            temperature,
        )

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={
            "xatol": 1e-6,
        },
    )

    temperature = float(
        np.exp(result.x)
    )

    return {
        "temperature": temperature,
        "success": bool(
            result.success
        ),
        "source_val_nll_before": binary_nll_from_logits(
            val_logits,
            val_y,
            1.0,
        ),
        "source_val_nll_after": binary_nll_from_logits(
            val_logits,
            val_y,
            temperature,
        ),
    }


def temperature_probs(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    return sigmoid(
        logits / temperature
    )


# ============================================================
# Label-shift correction
# ============================================================
def prior_correct_probs(
    source_probs: np.ndarray,
    source_prior: float,
    target_prior: float,
) -> np.ndarray:
    """
    Binary label-shift posterior correction.

    target odds = source odds
                  * target-prior-odds / source-prior-odds
    """
    ps = float(
        np.clip(
            source_prior,
            EPS,
            1.0 - EPS,
        )
    )
    pt = float(
        np.clip(
            target_prior,
            EPS,
            1.0 - EPS,
        )
    )

    p = np.clip(
        source_probs,
        EPS,
        1.0 - EPS,
    )

    source_odds = (
        p / (1.0 - p)
    )

    prior_odds_ratio = (
        (pt / (1.0 - pt))
        / (ps / (1.0 - ps))
    )

    target_odds = (
        source_odds
        * prior_odds_ratio
    )

    corrected = (
        target_odds
        / (1.0 + target_odds)
    )

    return np.clip(
        corrected,
        EPS,
        1.0 - EPS,
    )


def estimate_target_prior_em(
    source_probs_on_target_adapt: np.ndarray,
    source_prior: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """
    Binary Saerens-style EM prior estimation.

    Uses:
      - source posterior probabilities on target adaptation examples
      - known source prior
      - NO target labels
    """
    ps = float(
        np.clip(
            source_prior,
            EPS,
            1.0 - EPS,
        )
    )

    p = np.clip(
        source_probs_on_target_adapt,
        EPS,
        1.0 - EPS,
    )

    pi = ps
    converged = False

    for iteration in range(
        1,
        max_iter + 1,
    ):
        w1 = pi / ps
        w0 = (
            (1.0 - pi)
            / (1.0 - ps)
        )

        numerator = w1 * p
        denominator = (
            numerator
            + w0 * (1.0 - p)
        )

        posterior = (
            numerator
            / np.clip(
                denominator,
                EPS,
                None,
            )
        )

        pi_new = float(
            np.mean(posterior)
        )
        pi_new = float(
            np.clip(
                pi_new,
                EPS,
                1.0 - EPS,
            )
        )

        if abs(
            pi_new - pi
        ) < tol:
            pi = pi_new
            converged = True
            break

        pi = pi_new

    return {
        "estimated_target_prior": float(
            pi
        ),
        "converged": bool(
            converged
        ),
        "iterations": int(
            iteration
        ),
    }


# ============================================================
# Metrics / uncertainty
# ============================================================
def expected_calibration_error(
    y: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = np.asarray(y)
    probs = np.asarray(probs)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (probs >= edges[i])
                & (probs <= edges[i + 1])
            )
        else:
            mask = (
                (probs >= edges[i])
                & (probs < edges[i + 1])
            )

        count = int(
            np.sum(mask)
        )

        if count == 0:
            continue

        confidence = float(
            np.mean(probs[mask])
        )
        observed_rate = float(
            np.mean(y[mask])
        )

        ece += (
            count / len(y)
        ) * abs(
            confidence
            - observed_rate
        )

    return float(ece)


def binary_entropy(
    probs: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        probs,
        EPS,
        1.0 - EPS,
    )

    return -(
        p * np.log(p)
        + (1.0 - p)
        * np.log(1.0 - p)
    )


def uncertainty_metrics(
    probs: np.ndarray,
) -> dict[str, float]:
    p = np.asarray(
        probs,
        dtype=float,
    )

    return {
        "uncertain_fraction_0_1_to_0_9": float(
            np.mean(
                (p >= 0.10)
                & (p <= 0.90)
            )
        ),
        "uncertain_fraction_0_25_to_0_75": float(
            np.mean(
                (p >= 0.25)
                & (p <= 0.75)
            )
        ),
        "mean_binary_entropy": float(
            np.mean(
                binary_entropy(p)
            )
        ),
        "mean_distance_from_0_5": float(
            np.mean(
                np.abs(
                    p - 0.5
                )
            )
        ),
    }


def probability_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(
        y,
        dtype=int,
    )

    probs = np.clip(
        np.asarray(
            probs,
            dtype=float,
        ),
        EPS,
        1.0 - EPS,
    )

    pred = (
        probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    return {
        "accuracy": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "fpr": float(
            fpr
        ),
        "auroc": float(
            roc_auc_score(
                y,
                probs,
            )
        ),
        "auprc": float(
            average_precision_score(
                y,
                probs,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                probs,
            )
        ),
        "log_loss": float(
            log_loss(
                y,
                probs,
                labels=[0, 1],
            )
        ),
        "ece": float(
            expected_calibration_error(
                y,
                probs,
            )
        ),
        "true_attack_rate": float(
            np.mean(y)
        ),
        "predicted_attack_probability_mean": float(
            np.mean(probs)
        ),
        "predicted_attack_rate_at_0_5": float(
            np.mean(pred)
        ),
        "absolute_prevalence_probability_error": float(
            abs(
                np.mean(probs)
                - np.mean(y)
            )
        ),
    }


# ============================================================
# Construction diagnostics
# ============================================================
def balanced_domain_auroc(
    Xs: np.ndarray,
    Xt: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """
    Linear source-vs-target classifier within one class.

    Pure prior shift should yield AUROC around 0.5.
    """
    n = min(
        len(Xs),
        len(Xt),
    )

    if n < 100:
        return {
            "auroc": np.nan,
            "converged": False,
        }

    rng = np.random.default_rng(
        seed
    )

    source_idx = rng.choice(
        len(Xs),
        size=n,
        replace=False,
    )
    target_idx = rng.choice(
        len(Xt),
        size=n,
        replace=False,
    )

    Xd = np.vstack(
        [
            Xs[source_idx],
            Xt[target_idx],
        ]
    )

    d = np.concatenate(
        [
            np.zeros(
                n,
                dtype=int,
            ),
            np.ones(
                n,
                dtype=int,
            ),
        ]
    )

    X_train, X_test, d_train, d_test = train_test_split(
        Xd,
        d,
        test_size=0.30,
        stratify=d,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(
        X_train
    )
    X_test = scaler.transform(
        X_test
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        random_state=seed,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        clf.fit(
            X_train,
            d_train,
        )

    conv_warnings = [
        w
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    auroc = float(
        roc_auc_score(
            d_test,
            clf.predict_proba(
                X_test
            )[:, 1],
        )
    )

    return {
        "auroc": auroc,
        "converged": (
            len(conv_warnings) == 0
        ),
    }


def construction_diagnostics(
    X_source_reference: np.ndarray,
    y_source_reference: np.ndarray,
    X_target_test: np.ndarray,
    y_target_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    benign = balanced_domain_auroc(
        X_source_reference[
            y_source_reference == 0
        ],
        X_target_test[
            y_target_test == 0
        ],
        seed + 1,
    )

    attack = balanced_domain_auroc(
        X_source_reference[
            y_source_reference == 1
        ],
        X_target_test[
            y_target_test == 1
        ],
        seed + 2,
    )

    return {
        "benign_conditional_domain_auroc": benign[
            "auroc"
        ],
        "benign_domain_classifier_converged": benign[
            "converged"
        ],
        "attack_conditional_domain_auroc": attack[
            "auroc"
        ],
        "attack_domain_classifier_converged": attack[
            "converged"
        ],
        "mean_conditional_domain_auroc": float(
            np.nanmean(
                [
                    benign["auroc"],
                    attack["auroc"],
                ]
            )
        ),
    }


# ============================================================
# Classifier abstraction
# ============================================================
def get_logits(
    classifier_name: str,
    classifier: Any,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if classifier_name == "mlp":
        return predict_mlp_logits(
            classifier,
            X,
            device,
        )

    if classifier_name == "logistic_regression":
        return classifier.decision_function(
            X
        ).astype(float)

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


# ============================================================
# One seed
# ============================================================
def run_seed(
    seed: int,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    source_prior: float,
    target_priors: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print("\n" + "=" * 115)
    print(
        f"CONTROLLED LABEL SHIFT | "
        f"dataset={args.dataset} | "
        f"seed={seed}"
    )
    print("=" * 115)

    set_seed(seed)

    banks = build_source_splits_and_target_banks(
        X=X_pool,
        y=y_pool,
        source_prior=source_prior,
        source_train_size=args.source_train_size,
        source_val_size=args.source_val_size,
        source_test_size=args.source_test_size,
        target_priors=target_priors,
        target_adapt_size=args.target_adapt_size,
        target_test_size=args.target_test_size,
        seed=seed,
    )

    Xs_train_raw, ys_train = banks[
        "source_train"
    ]
    Xs_val_raw, ys_val = banks[
        "source_val"
    ]
    Xs_test_raw, ys_test = banks[
        "source_test"
    ]

    # Source-only preprocessing.
    scaler = StandardScaler()
    scaler.fit(
        Xs_train_raw
    )

    Xs_train = scaler.transform(
        Xs_train_raw
    ).astype(np.float32)

    Xs_val = scaler.transform(
        Xs_val_raw
    ).astype(np.float32)

    Xs_test = scaler.transform(
        Xs_test_raw
    ).astype(np.float32)

    source_train_prior = float(
        np.mean(ys_train)
    )

    # Train both classifiers on identical source splits.
    mlp, mlp_training = train_mlp(
        Xs_train,
        ys_train,
        Xs_val,
        ys_val,
        args,
        device,
        seed,
    )

    logistic, logistic_training = train_logistic(
        Xs_train,
        ys_train,
        seed,
    )

    classifiers: dict[str, Any] = {
        "mlp": mlp,
        "logistic_regression": logistic,
    }

    training_info = {
        "mlp": mlp_training,
        "logistic_regression": logistic_training,
    }

    result_rows: list[
        dict[str, Any]
    ] = []

    diagnostic_rows: list[
        dict[str, Any]
    ] = []

    source_rows: list[
        dict[str, Any]
    ] = []

    temperature_by_classifier: dict[
        str,
        dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # Fit source-only temperature + source diagnostics.
    # --------------------------------------------------------
    for classifier_name in CLASSIFIERS:
        classifier = classifiers[
            classifier_name
        ]

        val_logits = get_logits(
            classifier_name,
            classifier,
            Xs_val,
            device,
        )

        temp_info = fit_temperature(
            val_logits,
            ys_val,
        )

        temperature_by_classifier[
            classifier_name
        ] = temp_info

        test_logits = get_logits(
            classifier_name,
            classifier,
            Xs_test,
            device,
        )

        raw_probs = sigmoid(
            test_logits
        )

        temp_probs = temperature_probs(
            test_logits,
            temp_info[
                "temperature"
            ],
        )

        raw_metrics = probability_metrics(
            ys_test,
            raw_probs,
        )

        temp_metrics = probability_metrics(
            ys_test,
            temp_probs,
        )

        uncertainty = uncertainty_metrics(
            raw_probs
        )

        source_rows.append(
            {
                "seed": int(seed),
                "dataset": args.dataset,
                "classifier": classifier_name,
                "requested_source_prior": float(
                    source_prior
                ),
                "source_train_prior": source_train_prior,
                "source_val_prior": float(
                    np.mean(ys_val)
                ),
                "source_test_prior": float(
                    np.mean(ys_test)
                ),
                "source_train_size": int(
                    len(ys_train)
                ),
                "source_val_size": int(
                    len(ys_val)
                ),
                "source_test_size": int(
                    len(ys_test)
                ),
                "temperature": float(
                    temp_info[
                        "temperature"
                    ]
                ),
                "temperature_fit_success": bool(
                    temp_info[
                        "success"
                    ]
                ),
                "source_val_nll_before_temperature": float(
                    temp_info[
                        "source_val_nll_before"
                    ]
                ),
                "source_val_nll_after_temperature": float(
                    temp_info[
                        "source_val_nll_after"
                    ]
                ),
                "training_best_epoch": (
                    int(
                        training_info[
                            classifier_name
                        ]["best_epoch"]
                    )
                    if classifier_name
                    == "mlp"
                    else np.nan
                ),
                "training_best_val_auroc": (
                    float(
                        training_info[
                            classifier_name
                        ]["best_val_auroc"]
                    )
                    if classifier_name
                    == "mlp"
                    else np.nan
                ),
                "training_converged": (
                    bool(
                        training_info[
                            classifier_name
                        ]["converged"]
                    )
                    if classifier_name
                    == "logistic_regression"
                    else True
                ),
                "training_iterations": (
                    int(
                        training_info[
                            classifier_name
                        ]["iterations"]
                    )
                    if classifier_name
                    == "logistic_regression"
                    else np.nan
                ),
                **{
                    f"raw_{k}": v
                    for k, v in raw_metrics.items()
                },
                **{
                    f"temperature_{k}": v
                    for k, v in temp_metrics.items()
                },
                **uncertainty,
            }
        )

    # --------------------------------------------------------
    # Target prior conditions.
    # --------------------------------------------------------
    for prior_index, target_prior in enumerate(
        target_priors
    ):
        condition = construct_target_condition(
            X=X_pool,
            y=y_pool,
            banks=banks,
            target_prior=target_prior,
            target_adapt_size=args.target_adapt_size,
            target_test_size=args.target_test_size,
            seed=(
                seed
                + prior_index * 10_000
            ),
        )

        Xt_adapt = scaler.transform(
            condition["X_adapt"]
        ).astype(np.float32)

        Xt_test = scaler.transform(
            condition["X_test"]
        ).astype(np.float32)

        yt_adapt = condition[
            "y_adapt"
        ]
        yt_test = condition[
            "y_test"
        ]

        actual_adapt_prior = float(
            np.mean(yt_adapt)
        )
        actual_test_prior = float(
            np.mean(yt_test)
        )

        print(
            f"\nTarget prior="
            f"{target_prior:.3f} | "
            f"adapt={actual_adapt_prior:.4f} | "
            f"test={actual_test_prior:.4f}"
        )

        # Validate pure-prior construction once per condition.
        construction = construction_diagnostics(
            X_source_reference=Xs_train,
            y_source_reference=ys_train,
            X_target_test=Xt_test,
            y_target_test=yt_test,
            seed=(
                seed
                + prior_index * 20_000
            ),
        )

        diagnostic_rows.append(
            {
                "seed": int(seed),
                "dataset": args.dataset,
                "requested_target_prior": float(
                    target_prior
                ),
                "actual_target_test_prior": actual_test_prior,
                **construction,
            }
        )

        print(
            "  construction AUROC | "
            f"benign="
            f"{construction['benign_conditional_domain_auroc']:.3f} | "
            f"attack="
            f"{construction['attack_conditional_domain_auroc']:.3f}"
        )

        for classifier_name in CLASSIFIERS:
            classifier = classifiers[
                classifier_name
            ]

            adapt_logits = get_logits(
                classifier_name,
                classifier,
                Xt_adapt,
                device,
            )

            test_logits = get_logits(
                classifier_name,
                classifier,
                Xt_test,
                device,
            )

            raw_adapt_probs = sigmoid(
                adapt_logits
            )
            raw_test_probs = sigmoid(
                test_logits
            )

            temperature = (
                temperature_by_classifier[
                    classifier_name
                ]["temperature"]
            )

            temp_test_probs = temperature_probs(
                test_logits,
                temperature,
            )

            em_info = estimate_target_prior_em(
                raw_adapt_probs,
                source_prior=source_train_prior,
            )

            oracle_test_probs = prior_correct_probs(
                raw_test_probs,
                source_prior=source_train_prior,
                target_prior=target_prior,
            )

            em_test_probs = prior_correct_probs(
                raw_test_probs,
                source_prior=source_train_prior,
                target_prior=em_info[
                    "estimated_target_prior"
                ],
            )

            method_probs = {
                "raw": raw_test_probs,
                "temperature_scaled": temp_test_probs,
                "raw_oracle_prior_corrected": oracle_test_probs,
                "raw_em_prior_corrected": em_test_probs,
            }

            print(
                f"  {classifier_name}: "
                f"EM prior="
                f"{em_info['estimated_target_prior']:.4f} "
                f"(error="
                f"{abs(em_info['estimated_target_prior'] - actual_test_prior):.4f})"
            )

            for method, probs in method_probs.items():
                metrics = probability_metrics(
                    yt_test,
                    probs,
                )

                if (
                    method
                    == "raw_oracle_prior_corrected"
                ):
                    estimated_target_prior = float(
                        target_prior
                    )
                elif (
                    method
                    == "raw_em_prior_corrected"
                ):
                    estimated_target_prior = float(
                        em_info[
                            "estimated_target_prior"
                        ]
                    )
                else:
                    estimated_target_prior = np.nan

                result_rows.append(
                    {
                        "seed": int(seed),
                        "dataset": args.dataset,
                        "classifier": classifier_name,
                        "method": method,
                        "requested_source_prior": float(
                            source_prior
                        ),
                        "actual_source_train_prior": source_train_prior,
                        "requested_target_prior": float(
                            target_prior
                        ),
                        "actual_target_adapt_prior": actual_adapt_prior,
                        "actual_target_test_prior": actual_test_prior,
                        "absolute_prior_shift": float(
                            abs(
                                actual_test_prior
                                - source_train_prior
                            )
                        ),
                        "temperature": float(
                            temperature
                        ),
                        "estimated_target_prior": estimated_target_prior,
                        "estimated_prior_absolute_error": (
                            float(
                                abs(
                                    estimated_target_prior
                                    - actual_test_prior
                                )
                            )
                            if np.isfinite(
                                estimated_target_prior
                            )
                            else np.nan
                        ),
                        "em_converged": (
                            bool(
                                em_info[
                                    "converged"
                                ]
                            )
                            if method
                            == "raw_em_prior_corrected"
                            else np.nan
                        ),
                        "em_iterations": (
                            int(
                                em_info[
                                    "iterations"
                                ]
                            )
                            if method
                            == "raw_em_prior_corrected"
                            else np.nan
                        ),
                        **metrics,
                    }
                )

    return (
        result_rows,
        diagnostic_rows,
        source_rows,
    )


# ============================================================
# Aggregation
# ============================================================
def aggregate_results(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "requested_target_prior",
    ]

    metric_cols = [
        "actual_target_test_prior",
        "absolute_prior_shift",
        "estimated_target_prior",
        "estimated_prior_absolute_error",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "auroc",
        "auprc",
        "brier",
        "log_loss",
        "ece",
        "predicted_attack_probability_mean",
        "predicted_attack_rate_at_0_5",
        "absolute_prevalence_probability_error",
    ]

    rows = []

    for keys, group in per_seed.groupby(
        group_cols,
        dropna=False,
    ):
        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row["n_seeds"] = int(
            group["seed"].nunique()
        )

        for metric in metric_cols:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(values)
            ]

            if len(finite) == 0:
                row[
                    f"{metric}_mean"
                ] = np.nan
                row[
                    f"{metric}_std"
                ] = np.nan
            else:
                row[
                    f"{metric}_mean"
                ] = float(
                    np.mean(finite)
                )
                row[
                    f"{metric}_std"
                ] = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                    if len(finite) > 1
                    else 0.0
                )

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "requested_target_prior",
            "method",
        ]
    )


def aggregate_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for prior, group in diagnostics.groupby(
        "requested_target_prior"
    ):
        row = {
            "requested_target_prior": float(
                prior
            ),
            "n_seeds": int(
                group["seed"].nunique()
            ),
            "all_benign_domain_classifiers_converged": bool(
                group[
                    "benign_domain_classifier_converged"
                ].all()
            ),
            "all_attack_domain_classifiers_converged": bool(
                group[
                    "attack_domain_classifier_converged"
                ].all()
            ),
        }

        for metric in [
            "benign_conditional_domain_auroc",
            "attack_conditional_domain_auroc",
            "mean_conditional_domain_auroc",
        ]:
            vals = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(vals)
            )

            row[
                f"{metric}_std"
            ] = float(
                np.nanstd(
                    vals,
                    ddof=1,
                )
                if np.sum(
                    np.isfinite(vals)
                ) > 1
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        "requested_target_prior"
    )


# ============================================================
# Paired method differences
# ============================================================
def build_paired_differences(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Paired differences against RAW within the exact same:
      seed + classifier + target prior.

    Positive brier_improvement / ece_improvement means LOWER error than raw.
    Positive f1_improvement means higher F1 than raw.
    AUROC difference should be ~0 for monotonic prior corrections.
    """
    key_cols = [
        "seed",
        "dataset",
        "classifier",
        "requested_target_prior",
    ]

    raw = (
        per_seed[
            per_seed["method"]
            == "raw"
        ][
            key_cols
            + [
                "brier",
                "ece",
                "f1",
                "auroc",
                "log_loss",
                "absolute_prevalence_probability_error",
            ]
        ]
        .copy()
        .rename(
            columns={
                "brier": "raw_brier",
                "ece": "raw_ece",
                "f1": "raw_f1",
                "auroc": "raw_auroc",
                "log_loss": "raw_log_loss",
                "absolute_prevalence_probability_error":
                    "raw_prevalence_probability_error",
            }
        )
    )

    compared = per_seed[
        per_seed["method"].isin(
            [
                "temperature_scaled",
                "raw_oracle_prior_corrected",
                "raw_em_prior_corrected",
            ]
        )
    ].copy()

    merged = compared.merge(
        raw,
        on=key_cols,
        how="left",
        validate="many_to_one",
    )

    merged[
        "brier_improvement_vs_raw"
    ] = (
        merged["raw_brier"]
        - merged["brier"]
    )

    merged[
        "ece_improvement_vs_raw"
    ] = (
        merged["raw_ece"]
        - merged["ece"]
    )

    merged[
        "log_loss_improvement_vs_raw"
    ] = (
        merged["raw_log_loss"]
        - merged["log_loss"]
    )

    merged[
        "prevalence_error_improvement_vs_raw"
    ] = (
        merged[
            "raw_prevalence_probability_error"
        ]
        - merged[
            "absolute_prevalence_probability_error"
        ]
    )

    merged[
        "f1_improvement_vs_raw"
    ] = (
        merged["f1"]
        - merged["raw_f1"]
    )

    merged[
        "auroc_difference_vs_raw"
    ] = (
        merged["auroc"]
        - merged["raw_auroc"]
    )

    return merged


def paired_difference_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "requested_target_prior",
    ]

    metrics = [
        "brier_improvement_vs_raw",
        "ece_improvement_vs_raw",
        "log_loss_improvement_vs_raw",
        "prevalence_error_improvement_vs_raw",
        "f1_improvement_vs_raw",
        "auroc_difference_vs_raw",
    ]

    rows = []

    for keys, group in paired.groupby(
        group_cols
    ):
        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        n = int(
            group["seed"].nunique()
        )
        row["n_seeds"] = n

        for metric in metrics:
            vals = group[
                metric
            ].to_numpy(
                dtype=float
            )
            vals = vals[
                np.isfinite(vals)
            ]

            if len(vals) == 0:
                mean = np.nan
                std = np.nan
                low = np.nan
                high = np.nan
            else:
                mean = float(
                    np.mean(vals)
                )
                std = float(
                    np.std(
                        vals,
                        ddof=1,
                    )
                    if len(vals) > 1
                    else 0.0
                )

                if len(vals) > 1:
                    critical = float(
                        student_t.ppf(
                            0.975,
                            df=len(vals) - 1,
                        )
                    )
                    half = (
                        critical
                        * std
                        / math.sqrt(
                            len(vals)
                        )
                    )
                    low = mean - half
                    high = mean + half
                else:
                    low = mean
                    high = mean

            row[
                f"{metric}_mean"
            ] = mean
            row[
                f"{metric}_std"
            ] = std
            row[
                f"{metric}_ci95_low"
            ] = float(low)
            row[
                f"{metric}_ci95_high"
            ] = float(high)

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "requested_target_prior",
            "method",
        ]
    )


# ============================================================
# Plots
# ============================================================
def plot_metric_for_classifier(
    summary: pd.DataFrame,
    classifier_name: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    classifier_df = summary[
        summary["classifier"]
        == classifier_name
    ]

    for method in METHODS:
        subset = classifier_df[
            classifier_df["method"]
            == method
        ].sort_values(
            "requested_target_prior"
        )

        plt.errorbar(
            subset[
                "requested_target_prior"
            ],
            subset[
                f"{metric}_mean"
            ],
            yerr=subset[
                f"{metric}_std"
            ],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel(
        "Target attack prevalence"
    )
    plt.ylabel(
        ylabel
    )
    plt.legend()
    savefig(
        output_path
    )


def plot_predicted_prevalence_for_classifier(
    summary: pd.DataFrame,
    classifier_name: str,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    classifier_df = summary[
        summary["classifier"]
        == classifier_name
    ]

    priors = np.sort(
        classifier_df[
            "requested_target_prior"
        ].unique()
    )

    plt.plot(
        priors,
        priors,
        linestyle="--",
        label="ideal",
    )

    for method in METHODS:
        subset = classifier_df[
            classifier_df["method"]
            == method
        ].sort_values(
            "requested_target_prior"
        )

        plt.errorbar(
            subset[
                "requested_target_prior"
            ],
            subset[
                "predicted_attack_probability_mean_mean"
            ],
            yerr=subset[
                "predicted_attack_probability_mean_std"
            ],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel(
        "True target attack prevalence"
    )
    plt.ylabel(
        "Mean predicted attack probability"
    )
    plt.legend()
    savefig(
        output_path
    )


def plot_construction_diagnostic(
    diagnostic_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="indistinguishable (0.5)",
    )

    plt.errorbar(
        diagnostic_summary[
            "requested_target_prior"
        ],
        diagnostic_summary[
            "benign_conditional_domain_auroc_mean"
        ],
        yerr=diagnostic_summary[
            "benign_conditional_domain_auroc_std"
        ],
        marker="o",
        capsize=3,
        label="Benign P(X|Y=0)",
    )

    plt.errorbar(
        diagnostic_summary[
            "requested_target_prior"
        ],
        diagnostic_summary[
            "attack_conditional_domain_auroc_mean"
        ],
        yerr=diagnostic_summary[
            "attack_conditional_domain_auroc_std"
        ],
        marker="o",
        capsize=3,
        label="Attack P(X|Y=1)",
    )

    plt.xlabel(
        "Target attack prevalence"
    )
    plt.ylabel(
        "Source-vs-target domain AUROC"
    )
    plt.ylim(
        0.40,
        0.70,
    )
    plt.legend()
    savefig(
        output_path
    )


# ============================================================
# Console summaries
# ============================================================
def print_source_uncertainty(
    source_df: pd.DataFrame,
) -> None:
    print(
        "\nSOURCE CLASSIFIER UNCERTAINTY"
    )
    print(
        "=" * 100
    )

    grouped = (
        source_df.groupby(
            "classifier"
        )[
            [
                "raw_auroc",
                "raw_f1",
                "raw_brier",
                "raw_ece",
                "uncertain_fraction_0_1_to_0_9",
                "uncertain_fraction_0_25_to_0_75",
                "mean_binary_entropy",
                "temperature",
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    print(
        grouped.to_string()
    )


def print_target_summary(
    summary: pd.DataFrame,
) -> None:
    print(
        "\nTARGET PRIOR-SHIFT SUMMARY"
    )
    print(
        "=" * 130
    )

    cols = [
        "classifier",
        "method",
        "requested_target_prior",
        "auroc_mean",
        "f1_mean",
        "brier_mean",
        "ece_mean",
        "predicted_attack_probability_mean_mean",
        "estimated_target_prior_mean",
        "estimated_prior_absolute_error_mean",
    ]

    print(
        summary[
            cols
        ].to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--pool_variant",
        type=str,
        default=DEFAULT_POOL_VARIANT,
    )

    parser.add_argument(
        "--source_prior",
        type=float,
        default=None,
        help=(
            "Default: dataset natural attack prevalence "
            "from Stage-1 full metadata."
        ),
    )

    parser.add_argument(
        "--target_priors",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_PRIORS,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--source_train_size",
        type=int,
        default=DEFAULT_SOURCE_TRAIN_SIZE,
    )
    parser.add_argument(
        "--source_val_size",
        type=int,
        default=DEFAULT_SOURCE_VAL_SIZE,
    )
    parser.add_argument(
        "--source_test_size",
        type=int,
        default=DEFAULT_SOURCE_TEST_SIZE,
    )

    parser.add_argument(
        "--target_adapt_size",
        type=int,
        default=DEFAULT_TARGET_ADAPT_SIZE,
    )
    parser.add_argument(
        "--target_test_size",
        type=int,
        default=DEFAULT_TARGET_TEST_SIZE,
    )

    # MLP settings align with previous DANIDS MLP experiments.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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

    target_priors = sorted(
        set(
            float(p)
            for p in args.target_priors
        )
    )

    if any(
        p <= 0.0 or p >= 1.0
        for p in target_priors
    ):
        raise ValueError(
            "All target priors must lie strictly between 0 and 1."
        )

    stage1_summary = (
        load_stage1_summary()
    )

    natural_prior = (
        natural_attack_prior(
            stage1_summary,
            args.dataset,
        )
    )

    source_prior = (
        natural_prior
        if args.source_prior is None
        else float(
            args.source_prior
        )
    )

    if not 0.0 < source_prior < 1.0:
        raise ValueError(
            "Source prior must lie strictly between 0 and 1."
        )

    print(
        "=" * 115
    )
    print(
        "CONTROLLED PURE LABEL / CLASS-PRIOR SHIFT EXPERIMENT"
    )
    print(
        "=" * 115
    )
    print(
        f"Dataset: {args.dataset}"
    )
    print(
        f"Pool: {args.pool_variant}"
    )
    print(
        f"Natural source attack prior: "
        f"{natural_prior:.6f}"
    )
    print(
        f"Experiment source attack prior: "
        f"{source_prior:.6f}"
    )
    print(
        f"Target priors: "
        f"{target_priors}"
    )
    print(
        f"Classifiers: "
        f"{CLASSIFIERS}"
    )
    print(
        f"Methods: "
        f"{METHODS}"
    )
    print(
        f"Seeds: {args.seeds}"
    )

    X_pool, y_pool = (
        load_dataset_pool(
            args.dataset,
            args.pool_variant,
        )
    )

    print(
        f"Loaded pool: "
        f"{len(y_pool):,} rows | "
        f"benign="
        f"{int((y_pool == 0).sum()):,} | "
        f"attack="
        f"{int((y_pool == 1).sum()):,} | "
        f"features="
        f"{X_pool.shape[1]}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    out_dir = (
        STAGE1_DIR
        / "controlled_label_shift"
        / args.dataset
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    all_diagnostics = []
    all_source = []

    for seed in args.seeds:
        (
            result_rows,
            diagnostic_rows,
            source_rows,
        ) = run_seed(
            seed=int(seed),
            X_pool=X_pool,
            y_pool=y_pool,
            source_prior=source_prior,
            target_priors=target_priors,
            args=args,
            device=device,
        )

        all_results.extend(
            result_rows
        )
        all_diagnostics.extend(
            diagnostic_rows
        )
        all_source.extend(
            source_rows
        )

    per_seed_df = pd.DataFrame(
        all_results
    )
    diagnostics_df = pd.DataFrame(
        all_diagnostics
    )
    source_df = pd.DataFrame(
        all_source
    )

    summary_df = aggregate_results(
        per_seed_df
    )
    diagnostic_summary_df = (
        aggregate_diagnostics(
            diagnostics_df
        )
    )

    paired_df = build_paired_differences(
        per_seed_df
    )
    paired_summary_df = (
        paired_difference_summary(
            paired_df
        )
    )

    # --------------------------------------------------------
    # Save tables.
    # --------------------------------------------------------
    per_seed_path = (
        out_dir
        / "controlled_label_shift_per_seed.csv"
    )
    summary_path = (
        out_dir
        / "controlled_label_shift_summary.csv"
    )
    source_path = (
        out_dir
        / "controlled_label_shift_source_metrics.csv"
    )
    diagnostics_path = (
        out_dir
        / "controlled_label_shift_diagnostics_per_seed.csv"
    )
    diagnostics_summary_path = (
        out_dir
        / "controlled_label_shift_diagnostics_summary.csv"
    )
    paired_path = (
        out_dir
        / "controlled_label_shift_paired_differences.csv"
    )
    paired_summary_path = (
        out_dir
        / "controlled_label_shift_paired_difference_summary.csv"
    )

    per_seed_df.to_csv(
        per_seed_path,
        index=False,
    )
    summary_df.to_csv(
        summary_path,
        index=False,
    )
    source_df.to_csv(
        source_path,
        index=False,
    )
    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )
    diagnostic_summary_df.to_csv(
        diagnostics_summary_path,
        index=False,
    )
    paired_df.to_csv(
        paired_path,
        index=False,
    )
    paired_summary_df.to_csv(
        paired_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Protocol.
    # --------------------------------------------------------
    protocol = {
        "research_questions": {
            "RQ2a": (
                "Under pure class-prior shift, which evaluation metrics "
                "change while P(X|Y) remains fixed?"
            ),
            "RQ2b": (
                "Can prior correction recover calibration / decision quality "
                "without retraining representation or classifier parameters?"
            ),
            "RQ2c": (
                "Does label-shift impact depend on classifier uncertainty "
                "and source-class overlap?"
            ),
        },
        "hypotheses": {
            "H1": (
                "AUROC remains approximately stable as target prior changes."
            ),
            "H2": (
                "Calibration and threshold-dependent behaviour can degrade "
                "with increasing prior mismatch."
            ),
            "H3": (
                "Prior correction can improve calibration / decision quality "
                "without classifier retraining."
            ),
            "H4": (
                "Label-shift impact is larger for classifiers with greater "
                "predictive uncertainty / class overlap."
            ),
        },
        "dataset": args.dataset,
        "pool_variant": args.pool_variant,
        "natural_source_prior": natural_prior,
        "experiment_source_prior": source_prior,
        "target_priors": target_priors,
        "seeds": [
            int(s)
            for s in args.seeds
        ],
        "classifiers": {
            "mlp": {
                "architecture": [
                    int(
                        X_pool.shape[1]
                    ),
                    *[
                        int(h)
                        for h in args.hidden_dims
                    ],
                    1,
                ],
                "dropout": args.dropout,
                "loss": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
            },
            "logistic_regression": {
                "solver": "lbfgs",
                "max_iter": 5000,
                "class_weight": None,
            },
        },
        "probability_methods": {
            "raw": (
                "Original source classifier probability."
            ),
            "temperature_scaled": (
                "Source-validation-only temperature scaling sigmoid(logit/T). "
                "Separate calibration baseline; not used as mandatory input "
                "to prior correction."
            ),
            "raw_oracle_prior_corrected": (
                "Prior correction directly from raw source probabilities "
                "using known designed target prior. Oracle/reference only."
            ),
            "raw_em_prior_corrected": (
                "EM estimates target prior from unlabelled target-adaptation "
                "raw probabilities; correction is then applied to held-out target test."
            ),
        },
        "source_sizes": {
            "train": args.source_train_size,
            "validation": args.source_val_size,
            "test": args.source_test_size,
        },
        "target_sizes_per_condition": {
            "unlabelled_adaptation": args.target_adapt_size,
            "test": args.target_test_size,
        },
        "leakage_controls": [
            "Source train, validation, and test are mutually disjoint.",
            "Every target example is disjoint from every source example.",
            "Target adaptation and target test banks are disjoint.",
            "The StandardScaler fits source training features only.",
            "Temperature scaling fits source validation logits/labels only.",
            "EM uses target-adaptation probabilities without target labels.",
            "Target test labels are evaluation only.",
            "All classifiers use the exact same source split per seed.",
            "All methods use the exact same target condition per seed.",
        ],
        "pure_shift_validation": (
            "Within-class source-vs-target domain classifier AUROC should "
            "remain approximately 0.5 for benign and attack separately. "
            "If this fails, the condition must not be interpreted as pure label shift."
        ),
        "paired_statistics": (
            "Correction-vs-raw differences are paired within seed, classifier, "
            "and target prior. The summary reports mean, SD, and t-based 95% CI. "
            "With five seeds these intervals are descriptive rather than definitive."
        ),
        "previous_platt_issue": (
            "Platt calibration was removed from the correction pipeline after "
            "one seed showed severe intercept-induced threshold/calibration failure."
        ),
    }

    protocol_path = (
        out_dir
        / "controlled_label_shift_protocol.json"
    )

    with open(
        protocol_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Plots.
    # --------------------------------------------------------
    for classifier_name in CLASSIFIERS:
        for metric, ylabel in [
            (
                "auroc",
                "Target AUROC",
            ),
            (
                "f1",
                "Target F1 @ 0.5",
            ),
            (
                "brier",
                "Brier score (lower is better)",
            ),
            (
                "ece",
                "ECE (lower is better)",
            ),
            (
                "log_loss",
                "Log loss (lower is better)",
            ),
        ]:
            plot_metric_for_classifier(
                summary_df,
                classifier_name,
                metric,
                ylabel,
                out_dir
                / (
                    f"label_shift_"
                    f"{classifier_name}_"
                    f"{metric}.png"
                ),
            )

        plot_predicted_prevalence_for_classifier(
            summary_df,
            classifier_name,
            out_dir
            / (
                f"label_shift_"
                f"{classifier_name}_"
                f"predicted_prevalence.png"
            ),
        )

    plot_construction_diagnostic(
        diagnostic_summary_df,
        out_dir
        / "label_shift_conditional_invariance.png",
    )

    # --------------------------------------------------------
    # Console.
    # --------------------------------------------------------
    print(
        "\n" + "=" * 130
    )
    print(
        "CONTROLLED LABEL-SHIFT EXPERIMENT COMPLETE"
    )
    print(
        "=" * 130
    )

    print_source_uncertainty(
        source_df
    )
    print_target_summary(
        summary_df
    )

    print(
        "\nCONSTRUCTION DIAGNOSTIC"
    )
    print(
        "=" * 100
    )
    print(
        diagnostic_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nINTERPRETATION GUARDRAILS"
    )
    print(
        "- First verify benign/attack conditional domain AUROC remains near 0.5."
    )
    print(
        "- AUROC should be essentially unchanged by temperature or prior correction "
        "because these are monotonic score transformations."
    )
    print(
        "- Oracle correction is a mechanism/reference condition, not deployable."
    )
    print(
        "- EM is the practical unlabelled-target prior-estimation condition."
    )
    print(
        "- Compare MLP vs logistic uncertainty before interpreting H4."
    )
    print(
        "- Use paired_difference_summary for correction-vs-raw claims."
    )

    print(
        "\nOUTPUTS"
    )
    for path in [
        per_seed_path,
        summary_path,
        source_path,
        diagnostics_path,
        diagnostics_summary_path,
        paired_path,
        paired_summary_path,
        protocol_path,
    ]:
        print(
            f"- {path}"
        )


if __name__ == "__main__":
    main()
