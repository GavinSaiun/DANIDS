"""
controlled_covariate_shift.py

Controlled pure covariate-shift experiment for DANIDS.

Purpose
-------
Natural UNSW / CIC / ToN cross-domain shifts are mixed. This benchmark
isolates covariate shift using sample-selection bias:

    P_T(X) != P_S(X)
    P_T(Y | X) = P_S(Y | X)        by construction

Target selection depends ONLY on X, never Y.

Synthetic shift mechanism
-------------------------
1. Reserve a fixed, label-free design subset from Stage-1 balanced_100k.
2. Fit StandardScaler + PCA(1) on DESIGN FEATURES ONLY.
3. Fix the PCA sign deterministically.
4. Define a scalar shift score g(X) = PC1.
5. Define target domains as upper-tail subpopulations:

       A_tau = {x : g(x) >= q_(1-tau)}

   where tau is the retained tail fraction.

Default tau values:
    1.00  -> no shift
    0.70  -> mild
    0.40  -> moderate
    0.20  -> strong

Target examples are sampled uniformly from A_tau. Therefore:

    p_T(x) = p_S(x | x in A_tau)

and the oracle density ratio is

    w*(x) = I[x in A_tau] / P_S(A_tau).

This is an exact covariate-shift mechanism at the population level.
Class prior P(Y) is allowed to change as a CONSEQUENCE of changing P(X);
that does not make this pure label shift.

Classifiers
-----------
- MLP: input -> 256 -> 128 -> 1, ReLU, dropout 0.2
- Logistic regression

Adaptation methods
------------------
1. source_only
   Train once on source.

2. uniform_extra_training
   MLP-only matched optimisation control: warm-start from the source MLP and
   continue training with all source weights equal to 1 using the same
   adaptation seed / LR / epoch budget as the IW runs. This isolates gains
   due to weighting from gains due merely to additional optimisation.

3. oracle_iw
   Importance-weight the labelled SOURCE loss with the known synthetic
   density ratio. This is a mechanism/reference condition.

4. estimated_iw
   Estimate p_T(x)/p_S(x) from an UNLABELLED source-vs-target domain
   classifier trained with balanced domain labels. Reweight source loss.

5. coral_target_to_source
   Unsupervised target-to-source CORAL using Ledoit-Wolf shrinkage covariance
   estimates, then apply the unchanged source classifier. The experiment
   records transform-to-identity and distribution-distortion diagnostics.

Hypotheses
----------
H1. Marginal source-vs-target domain separability should increase as the
    retained target tail fraction decreases.

H2. Source-only target performance can degrade under covariate shift even
    though P(Y|X) is unchanged, due to model misspecification / changed
    emphasis over feature space.

H3. Importance weighting should improve target risk by emphasizing
    source regions that resemble the target distribution.

H4. Estimated importance weighting should approach oracle importance
    weighting when the density-ratio estimator is accurate and weight
    effective sample size remains adequate.

H5. Very severe covariate shift can reduce the effective sample size of
    importance weights, limiting or reversing adaptation benefit.

Rigor / leakage controls
------------------------
- Shift-design PCA uses a fixed DESIGN subset excluded from all model data.
- Shift design never uses Y.
- Source train/val/test are mutually disjoint.
- Target adaptation and target test candidate banks are disjoint.
- Every target row is disjoint from every source row.
- NIDS StandardScaler fits source train only.
- Estimated density ratio uses source X + target-adaptation X only.
- CORAL uses source train X + target-adaptation X only.
- Target adaptation labels are never used by adaptation.
- Target test labels are evaluation only.
- All methods share the exact same target test condition within seed/tau.
- Pairwise method comparisons are paired within seed/classifier/tau.
- MLP IW is additionally compared against matched uniform extra training.
- CORAL reports transform identity / mean / covariance displacement diagnostics.

Recommended first run
---------------------
ToN is the default because the previous controlled label-shift experiment
showed substantially more natural classifier uncertainty than UNSW.

    python -B -m src.analysis.controlled_covariate_shift

Other datasets:
    python -B -m src.analysis.controlled_covariate_shift --dataset NF-CSE-CIC-IDS2018-v3
    python -B -m src.analysis.controlled_covariate_shift --dataset NF-UNSW-NB15-v3
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
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
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
DEFAULT_DATASET = "NF-ToN-IoT-v3"
DEFAULT_POOL_VARIANT = "balanced_100k"

DEFAULT_SEEDS = [42, 123, 456, 789, 2026]

# Retained upper-tail fractions of the fixed covariate score.
# shift_severity = 1 - tail_fraction
DEFAULT_TAIL_FRACTIONS = [1.0, 0.70, 0.40, 0.20]

DESIGN_SEED = 314159
DEFAULT_DESIGN_SIZE = 20_000

DEFAULT_SOURCE_TRAIN_SIZE = 40_000
DEFAULT_SOURCE_VAL_SIZE = 10_000
DEFAULT_SOURCE_TEST_SIZE = 10_000

DEFAULT_TARGET_ADAPT_CANDIDATE_SIZE = 40_000
DEFAULT_TARGET_TEST_CANDIDATE_SIZE = 80_000

DEFAULT_TARGET_ADAPT_SIZE = 5_000
DEFAULT_TARGET_TEST_SIZE = 10_000

CLASSIFIERS = [
    "mlp",
    "logistic_regression",
]

METHODS = [
    "source_only",
    "uniform_extra_training",
    "oracle_iw",
    "estimated_iw",
    "coral_target_to_source",
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
def load_dataset_pool(
    dataset: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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

    frame = pd.read_parquet(x_path)

    feature_names = [
        str(c)
        for c in frame.columns
    ]

    X = frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    y = np.load(
        y_path
    ).astype(np.int8)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label mismatch: X={len(X)}, y={len(y)}"
        )

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0,1}}, got {np.unique(y)}"
        )

    return X, y, feature_names


# ============================================================
# Fixed covariate-shift design
# ============================================================
class ShiftDesign:
    """
    Fixed label-free covariate score.

    A reserved design subset is excluded from all downstream source/target
    model data. The design uses X only.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        pca: PCA,
        thresholds: dict[float, float],
        design_indices: np.ndarray,
        available_indices: np.ndarray,
        design_scores: np.ndarray,
    ) -> None:
        self.scaler = scaler
        self.pca = pca
        self.thresholds = thresholds
        self.design_indices = design_indices
        self.available_indices = available_indices
        self.design_scores = design_scores

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        Xs = self.scaler.transform(
            X
        )

        return self.pca.transform(
            Xs
        )[:, 0]


def build_shift_design(
    X: np.ndarray,
    tail_fractions: list[float],
    design_size: int,
) -> ShiftDesign:
    """
    Reserve a fixed X-only design subset and fit a 1-D PCA shift score.

    PCA sign is fixed by requiring the largest-absolute loading to be
    positive. This avoids arbitrary sign reversal.
    """
    if design_size >= len(X):
        raise ValueError(
            "design_size must be smaller than dataset pool."
        )

    rng = np.random.default_rng(
        DESIGN_SEED
    )

    perm = rng.permutation(
        len(X)
    )

    design_indices = perm[
        :design_size
    ]
    available_indices = perm[
        design_size:
    ]

    X_design = X[
        design_indices
    ]

    scaler = StandardScaler()
    X_design_scaled = scaler.fit_transform(
        X_design
    )

    pca = PCA(
        n_components=1,
        svd_solver="full",
    )
    pca.fit(
        X_design_scaled
    )

    component = pca.components_[
        0
    ].copy()

    anchor = int(
        np.argmax(
            np.abs(component)
        )
    )

    if component[
        anchor
    ] < 0:
        pca.components_[
            0
        ] *= -1.0

    design_scores = pca.transform(
        X_design_scaled
    )[:, 0]

    thresholds: dict[
        float,
        float,
    ] = {}

    for tail_fraction in tail_fractions:
        if tail_fraction >= 1.0:
            thresholds[
                float(tail_fraction)
            ] = -np.inf
        else:
            thresholds[
                float(tail_fraction)
            ] = float(
                np.quantile(
                    design_scores,
                    1.0 - tail_fraction,
                )
            )

    return ShiftDesign(
        scaler=scaler,
        pca=pca,
        thresholds=thresholds,
        design_indices=design_indices,
        available_indices=available_indices,
        design_scores=design_scores,
    )


def shift_loading_table(
    design: ShiftDesign,
    feature_names: list[str],
) -> pd.DataFrame:
    loadings = design.pca.components_[
        0
    ]

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "pc1_loading": loadings,
            "absolute_pc1_loading": np.abs(
                loadings
            ),
        }
    )

    return df.sort_values(
        "absolute_pc1_loading",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# Per-seed source / target candidate split
# ============================================================
def build_seed_split(
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    seed: int,
    source_train_size: int,
    source_val_size: int,
    source_test_size: int,
    target_adapt_candidate_size: int,
    target_test_candidate_size: int,
) -> dict[str, Any]:
    required = (
        source_train_size
        + source_val_size
        + source_test_size
        + target_adapt_candidate_size
        + target_test_candidate_size
    )

    if required > len(
        design.available_indices
    ):
        raise ValueError(
            f"Need {required:,} non-design rows, "
            f"but only {len(design.available_indices):,} available."
        )

    rng = np.random.default_rng(
        seed
    )

    indices = design.available_indices.copy()
    rng.shuffle(
        indices
    )

    cursor = 0

    def take(
        n: int,
    ) -> np.ndarray:
        nonlocal cursor

        out = indices[
            cursor : cursor + n
        ]
        cursor += n
        return out

    idx_source_train = take(
        source_train_size
    )
    idx_source_val = take(
        source_val_size
    )
    idx_source_test = take(
        source_test_size
    )
    idx_target_adapt_candidate = take(
        target_adapt_candidate_size
    )
    idx_target_test_candidate = take(
        target_test_candidate_size
    )

    # Fixed random priority orders for paired/nested target conditions.
    adapt_order_rng = np.random.default_rng(
        seed + 10_001
    )
    test_order_rng = np.random.default_rng(
        seed + 20_001
    )

    adapt_priority = (
        idx_target_adapt_candidate.copy()
    )
    test_priority = (
        idx_target_test_candidate.copy()
    )

    adapt_order_rng.shuffle(
        adapt_priority
    )
    test_order_rng.shuffle(
        test_priority
    )

    return {
        "X_source_train": X[
            idx_source_train
        ],
        "y_source_train": y[
            idx_source_train
        ],
        "X_source_val": X[
            idx_source_val
        ],
        "y_source_val": y[
            idx_source_val
        ],
        "X_source_test": X[
            idx_source_test
        ],
        "y_source_test": y[
            idx_source_test
        ],
        "idx_source_train": idx_source_train,
        "idx_source_val": idx_source_val,
        "idx_source_test": idx_source_test,
        "target_adapt_priority": adapt_priority,
        "target_test_priority": test_priority,
    }


def select_target_condition(
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    split: dict[str, Any],
    tail_fraction: float,
    target_adapt_size: int,
    target_test_size: int,
) -> dict[str, Any]:
    threshold = design.thresholds[
        float(tail_fraction)
    ]

    adapt_priority = split[
        "target_adapt_priority"
    ]
    test_priority = split[
        "target_test_priority"
    ]

    adapt_scores = design.score(
        X[
            adapt_priority
        ]
    )
    test_scores = design.score(
        X[
            test_priority
        ]
    )

    if np.isneginf(
        threshold
    ):
        eligible_adapt = np.ones(
            len(adapt_priority),
            dtype=bool,
        )
        eligible_test = np.ones(
            len(test_priority),
            dtype=bool,
        )
    else:
        eligible_adapt = (
            adapt_scores >= threshold
        )
        eligible_test = (
            test_scores >= threshold
        )

    adapt_idx = adapt_priority[
        eligible_adapt
    ][
        :target_adapt_size
    ]

    test_idx = test_priority[
        eligible_test
    ][
        :target_test_size
    ]

    if len(
        adapt_idx
    ) < target_adapt_size:
        raise ValueError(
            f"tail_fraction={tail_fraction}: "
            f"only {len(adapt_idx):,} eligible target-adapt rows; "
            f"need {target_adapt_size:,}."
        )

    if len(
        test_idx
    ) < target_test_size:
        raise ValueError(
            f"tail_fraction={tail_fraction}: "
            f"only {len(test_idx):,} eligible target-test rows; "
            f"need {target_test_size:,}."
        )

    if np.intersect1d(
        adapt_idx,
        test_idx,
    ).size > 0:
        raise RuntimeError(
            "Target adaptation/test overlap."
        )

    return {
        "X_adapt_raw": X[
            adapt_idx
        ],
        "y_adapt": y[
            adapt_idx
        ],
        "X_test_raw": X[
            test_idx
        ],
        "y_test": y[
            test_idx
        ],
        "adapt_indices": adapt_idx,
        "test_indices": test_idx,
        "threshold": float(
            threshold
        ),
        "eligible_adapt_fraction": float(
            np.mean(
                eligible_adapt
            )
        ),
        "eligible_test_fraction": float(
            np.mean(
                eligible_test
            )
        ),
    }


# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: list[
            nn.Module
        ] = []

        previous = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        previous,
                        hidden,
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        dropout
                    ),
                ]
            )
            previous = hidden

        layers.append(
            nn.Linear(
                previous,
                1,
            )
        )

        self.net = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            x
        ).squeeze(
            1
        )


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
        / (
            1.0
            + np.exp(
                -logits
            )
        )
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
                    start : start
                    + batch_size
                ],
                dtype=torch.float32,
                device=device,
            )

            chunks.append(
                model(
                    xb
                )
                .detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        chunks
    )


def make_weighted_loader(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
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
            torch.tensor(
                weights,
                dtype=torch.float32,
            ),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def weighted_auroc(
    y: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        roc_auc_score(
            y,
            probs,
            sample_weight=weights,
        )
    )


def train_source_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[
    MLP,
    dict[str, Any],
]:
    set_seed(
        seed
    )

    model = MLP(
        input_dim=X_train.shape[
            1
        ],
        hidden_dims=tuple(
            args.hidden_dims
        ),
        dropout=args.dropout,
    ).to(
        device
    )

    weights_train = np.ones(
        len(y_train),
        dtype=np.float32,
    )

    loader = make_weighted_loader(
        X_train,
        y_train,
        weights_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(
        args.max_epochs
    ):
        model.train()

        total_loss = 0.0
        total_weight = 0.0

        for xb, yb, wb in loader:
            xb = xb.to(
                device
            )
            yb = yb.to(
                device
            )
            wb = wb.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                xb
            )

            losses = criterion(
                logits,
                yb,
            )

            loss = (
                (
                    losses
                    * wb
                ).sum()
                / wb.sum().clamp_min(
                    1e-8
                )
            )

            loss.backward()
            optimizer.step()

            total_loss += float(
                (
                    losses
                    * wb
                ).sum()
                .detach()
                .cpu()
            )
            total_weight += float(
                wb.sum()
                .detach()
                .cpu()
            )

        val_logits = predict_mlp_logits(
            model,
            X_val,
            device,
        )

        val_auroc = float(
            roc_auc_score(
                y_val,
                sigmoid(
                    val_logits
                ),
            )
        )

        print(
            f"[source MLP seed={seed}] "
            f"epoch={epoch + 1:02d} "
            f"loss="
            f"{total_loss / max(total_weight, 1e-8):.5f} "
            f"val_auroc={val_auroc:.5f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = (
                val_auroc
            )
            best_epoch = (
                epoch + 1
            )
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
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
    }


def adapt_mlp_with_weights(
    source_model: MLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_weights: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[
    MLP,
    dict[str, Any],
]:
    """
    Warm-start from the source model and optimise the importance-weighted
    source risk. No target labels are used.
    """
    set_seed(
        seed
    )

    model = copy.deepcopy(
        source_model
    ).to(
        device
    )

    loader = make_weighted_loader(
        X_train,
        y_train,
        train_weights.astype(
            np.float32
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.adapt_lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    initial_val_logits = (
        predict_mlp_logits(
            model,
            X_val,
            device,
        )
    )

    best_val_auroc = weighted_auroc(
        y_val,
        sigmoid(
            initial_val_logits
        ),
        val_weights,
    )

    best_epoch = 0
    no_improve = 0

    for epoch in range(
        args.adapt_epochs
    ):
        model.train()

        for xb, yb, wb in loader:
            xb = xb.to(
                device
            )
            yb = yb.to(
                device
            )
            wb = wb.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                xb
            )

            losses = criterion(
                logits,
                yb,
            )

            loss = (
                (
                    losses
                    * wb
                ).sum()
                / wb.sum().clamp_min(
                    1e-8
                )
            )

            loss.backward()
            optimizer.step()

        val_logits = predict_mlp_logits(
            model,
            X_val,
            device,
        )

        val_probs = sigmoid(
            val_logits
        )

        val_auroc = weighted_auroc(
            y_val,
            val_probs,
            val_weights,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = (
                val_auroc
            )
            best_epoch = (
                epoch + 1
            )
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.adapt_patience:
            break

    model.load_state_dict(
        best_state
    )

    return model, {
        "best_weighted_val_auroc": float(
            best_val_auroc
        ),
        "best_epoch": int(
            best_epoch
        ),
        "label": label,
    }


# ============================================================
# Logistic regression
# ============================================================
def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
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
            sample_weight=sample_weight,
        )

    convergence_warnings = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    return clf, {
        "converged": (
            len(
                convergence_warnings
            ) == 0
        ),
        "iterations": int(
            np.max(
                clf.n_iter_
            )
        ),
        "warnings": sorted(
            set(
                convergence_warnings
            )
        ),
    }


def classifier_logits(
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
        ).astype(
            float
        )

    raise ValueError(
        classifier_name
    )


# ============================================================
# Importance weights
# ============================================================
def normalize_weights(
    weights: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(
        weights,
        dtype=float,
    )

    mean_weight = float(
        np.mean(
            weights
        )
    )

    if (
        not np.isfinite(
            mean_weight
        )
        or mean_weight <= 0
    ):
        raise ValueError(
            "Importance weights have invalid mean."
        )

    return (
        weights
        / mean_weight
    )


def effective_sample_size(
    weights: np.ndarray,
) -> float:
    w = np.asarray(
        weights,
        dtype=float,
    )

    numerator = (
        np.sum(
            w
        ) ** 2
    )

    denominator = np.sum(
        w ** 2
    )

    if denominator <= 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def oracle_importance_weights(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Exact source-to-target ratio for target = source conditioned on
    score >= threshold, up to empirical finite-sample normalization.
    """
    if np.isneginf(
        threshold
    ):
        return np.ones(
            len(scores),
            dtype=float,
        )

    indicator = (
        scores >= threshold
    ).astype(
        float
    )

    if indicator.sum() == 0:
        raise ValueError(
            "No source examples fall in target region."
        )

    return normalize_weights(
        indicator
    )


def estimate_importance_weights(
    X_source_train: np.ndarray,
    X_target_adapt: np.ndarray,
    X_source_eval_sets: dict[
        str,
        np.ndarray,
    ],
    seed: int,
    iw_clip: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Density-ratio estimation with balanced domain labels.

    Because the domain-classifier training set is balanced:
        p_T(x)/p_S(x) ~= p(D=1|x) / p(D=0|x)

    Ratios are clipped and then normalized to source-train mean 1.
    """
    n = min(
        len(
            X_source_train
        ),
        len(
            X_target_adapt
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    source_idx = rng.choice(
        len(
            X_source_train
        ),
        size=n,
        replace=False,
    )

    target_idx = rng.choice(
        len(
            X_target_adapt
        ),
        size=n,
        replace=False,
    )

    X_domain = np.vstack(
        [
            X_source_train[
                source_idx
            ],
            X_target_adapt[
                target_idx
            ],
        ]
    )

    y_domain = np.concatenate(
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

    (
        X_domain_train,
        X_domain_test,
        y_domain_train,
        y_domain_test,
    ) = train_test_split(
        X_domain,
        y_domain,
        test_size=0.30,
        stratify=y_domain,
        random_state=seed,
    )

    domain_scaler = StandardScaler()

    X_domain_train_scaled = (
        domain_scaler.fit_transform(
            X_domain_train
        )
    )

    X_domain_test_scaled = (
        domain_scaler.transform(
            X_domain_test
        )
    )

    domain_clf = LogisticRegression(
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

        domain_clf.fit(
            X_domain_train_scaled,
            y_domain_train,
        )

    convergence_warnings = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    test_probs = domain_clf.predict_proba(
        X_domain_test_scaled
    )[:, 1]

    domain_auroc = float(
        roc_auc_score(
            y_domain_test,
            test_probs,
        )
    )

    raw_weight_sets: dict[
        str,
        np.ndarray,
    ] = {}

    for name, X_eval in X_source_eval_sets.items():
        p_target = domain_clf.predict_proba(
            domain_scaler.transform(
                X_eval
            )
        )[:, 1]

        p_target = np.clip(
            p_target,
            EPS,
            1.0 - EPS,
        )

        odds = (
            p_target
            / (
                1.0
                - p_target
            )
        )

        odds = np.clip(
            odds,
            0.0,
            iw_clip,
        )

        raw_weight_sets[
            name
        ] = odds

    train_mean = float(
        np.mean(
            raw_weight_sets[
                "train"
            ]
        )
    )

    if train_mean <= 0:
        raise ValueError(
            "Estimated importance weights have non-positive train mean."
        )

    normalized_sets = {
        name: (
            values
            / train_mean
        )
        for name, values
        in raw_weight_sets.items()
    }

    return normalized_sets, {
        "domain_auroc": domain_auroc,
        "domain_classifier_converged": (
            len(
                convergence_warnings
            ) == 0
        ),
        "domain_classifier_iterations": int(
            np.max(
                domain_clf.n_iter_
            )
        ),
        "domain_classifier_warnings": sorted(
            set(
                convergence_warnings
            )
        ),
        "train_weight_mean_before_normalization": train_mean,
        "train_weight_max_after_normalization": float(
            np.max(
                normalized_sets[
                    "train"
                ]
            )
        ),
        "train_weight_ess": effective_sample_size(
            normalized_sets[
                "train"
            ]
        ),
    }


# ============================================================
# CORAL target -> source
# ============================================================
def psd_power(
    matrix: np.ndarray,
    power: float,
    eps: float,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(
        matrix
    )

    values = np.maximum(
        values,
        eps,
    )

    powered = (
        values ** power
    )

    return (
        vectors
        @ np.diag(
            powered
        )
        @ vectors.T
    )


def coral_target_to_source_fit(
    X_source: np.ndarray,
    X_target_adapt: np.ndarray,
    eps: float,
) -> dict[str, Any]:
    """
    Learn a numerically stable target->source CORAL transform.

    Covariances are estimated with Ledoit-Wolf shrinkage rather than the raw
    sample covariance. This is deliberately conservative in 49 dimensions
    with a ~5k target-adaptation sample and reduces unstable whitening along
    small-eigenvalue directions.

        X_t' = (X_t - mu_t) Ct^{-1/2} Cs^{1/2} + mu_s

    The returned dictionary also contains sanity diagnostics. Under the
    no-shift control, transform_identity_relative_frobenius and RMS
    displacement should be small, and covariance mismatch should not increase
    materially after transformation.
    """
    source_mean = np.mean(
        X_source,
        axis=0,
    )

    target_mean = np.mean(
        X_target_adapt,
        axis=0,
    )

    source_lw = LedoitWolf().fit(
        X_source
    )
    target_lw = LedoitWolf().fit(
        X_target_adapt
    )

    source_cov = source_lw.covariance_.astype(
        float,
        copy=True,
    )
    target_cov = target_lw.covariance_.astype(
        float,
        copy=True,
    )

    dim = X_source.shape[1]
    identity = np.eye(
        dim,
        dtype=float,
    )

    # A small floor remains useful even with shrinkage.
    source_cov = (
        source_cov
        + eps * identity
    )
    target_cov = (
        target_cov
        + eps * identity
    )

    target_whiten = psd_power(
        target_cov,
        -0.5,
        eps,
    )

    source_recolour = psd_power(
        source_cov,
        0.5,
        eps,
    )

    transform = (
        target_whiten
        @ source_recolour
    )

    transformed_adapt = (
        (
            X_target_adapt
            - target_mean
        )
        @ transform
        + source_mean
    )

    transformed_cov = LedoitWolf().fit(
        transformed_adapt
    ).covariance_

    source_cov_norm = max(
        float(
            np.linalg.norm(
                source_cov,
                ord="fro",
            )
        ),
        EPS,
    )

    identity_norm = max(
        float(
            np.linalg.norm(
                identity,
                ord="fro",
            )
        ),
        EPS,
    )

    diagnostics = {
        "source_covariance_shrinkage": float(
            source_lw.shrinkage_
        ),
        "target_covariance_shrinkage": float(
            target_lw.shrinkage_
        ),
        "transform_identity_relative_frobenius": float(
            np.linalg.norm(
                transform - identity,
                ord="fro",
            )
            / identity_norm
        ),
        "mean_displacement_rms": float(
            np.sqrt(
                np.mean(
                    (
                        source_mean
                        - target_mean
                    ) ** 2
                )
            )
        ),
        "sample_transform_rms_displacement": float(
            np.sqrt(
                np.mean(
                    (
                        transformed_adapt
                        - X_target_adapt
                    ) ** 2
                )
            )
        ),
        "covariance_relative_frobenius_before": float(
            np.linalg.norm(
                target_cov
                - source_cov,
                ord="fro",
            )
            / source_cov_norm
        ),
        "covariance_relative_frobenius_after": float(
            np.linalg.norm(
                transformed_cov
                - source_cov,
                ord="fro",
            )
            / source_cov_norm
        ),
    }

    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "transform": transform,
        "diagnostics": diagnostics,
    }


def coral_target_to_source_apply(
    X_target: np.ndarray,
    coral: dict[str, np.ndarray],
) -> np.ndarray:
    transformed = (
        (
            X_target
            - coral[
                "target_mean"
            ]
        )
        @ coral[
            "transform"
        ]
        + coral[
            "source_mean"
        ]
    )

    return transformed.astype(
        np.float32
    )


# ============================================================
# Metrics
# ============================================================
def expected_calibration_error(
    y: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = np.asarray(
        y
    )

    probs = np.asarray(
        probs
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(
        n_bins
    ):
        if i == n_bins - 1:
            mask = (
                (probs >= edges[i])
                & (
                    probs
                    <= edges[
                        i + 1
                    ]
                )
            )
        else:
            mask = (
                (probs >= edges[i])
                & (
                    probs
                    < edges[
                        i + 1
                    ]
                )
            )

        count = int(
            np.sum(
                mask
            )
        )

        if count == 0:
            continue

        mean_confidence = float(
            np.mean(
                probs[
                    mask
                ]
            )
        )

        observed_rate = float(
            np.mean(
                y[
                    mask
                ]
            )
        )

        ece += (
            count / len(y)
        ) * abs(
            mean_confidence
            - observed_rate
        )

    return float(
        ece
    )


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
    ).astype(
        int
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (
            fp + tn
        )
        if (
            fp + tn
        ) > 0
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
            np.mean(
                y
            )
        ),
        "predicted_attack_probability_mean": float(
            np.mean(
                probs
            )
        ),
        "predicted_attack_rate_at_0_5": float(
            np.mean(
                pred
            )
        ),
    }


# ============================================================
# Shift diagnostics
# ============================================================
def marginal_domain_diagnostic(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    n = min(
        len(
            X_source
        ),
        len(
            X_target
        ),
    )

    rng = np.random.default_rng(
        seed
    )

    source_idx = rng.choice(
        len(
            X_source
        ),
        size=n,
        replace=False,
    )

    target_idx = rng.choice(
        len(
            X_target
        ),
        size=n,
        replace=False,
    )

    X_domain = np.vstack(
        [
            X_source[
                source_idx
            ],
            X_target[
                target_idx
            ],
        ]
    )

    y_domain = np.concatenate(
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

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_domain,
        y_domain,
        test_size=0.30,
        stratify=y_domain,
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
            y_test,
            clf.predict_proba(
                X_test
            )[:, 1],
        )
    )

    return {
        "marginal_domain_auroc": auroc,
        "marginal_domain_classifier_converged": (
            len(
                conv_warnings
            ) == 0
        ),
    }


# ============================================================
# Paired summaries
# ============================================================
def aggregate_results(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "tail_fraction",
        "shift_severity",
    ]

    metric_cols = [
        "target_attack_prior",
        "attack_prior_change",
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

        row[
            "n_seeds"
        ] = int(
            group[
                "seed"
            ].nunique()
        )

        for metric in metric_cols:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
                if np.sum(
                    np.isfinite(
                        values
                    )
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "shift_severity",
            "method",
        ]
    )


def aggregate_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "tail_fraction",
        "shift_severity",
    ]

    metric_cols = [
        "source_attack_prior",
        "target_attack_prior",
        "attack_prior_change",
        "marginal_domain_auroc",
        "source_region_fraction_train",
        "target_shift_score_mean",
        "source_shift_score_mean",
        "shift_score_standardized_mean_difference",
        "oracle_weight_ess_ratio",
        "estimated_weight_ess_ratio",
        "estimated_vs_oracle_weight_correlation",
        "estimated_domain_auroc",
        "coral_transform_identity_relative_frobenius",
        "coral_mean_displacement_rms",
        "coral_sample_transform_rms_displacement",
        "coral_covariance_relative_frobenius_before",
        "coral_covariance_relative_frobenius_after",
        "coral_source_covariance_shrinkage",
        "coral_target_covariance_shrinkage",
    ]

    rows = []

    for keys, group in diagnostics.groupby(
        group_cols,
        dropna=False,
    ):
        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row[
            "n_seeds"
        ] = int(
            group[
                "seed"
            ].nunique()
        )

        row[
            "all_marginal_domain_classifiers_converged"
        ] = bool(
            group[
                "marginal_domain_classifier_converged"
            ].all()
        )

        row[
            "all_density_ratio_domain_classifiers_converged"
        ] = bool(
            group[
                "estimated_domain_classifier_converged"
            ].all()
        )

        for metric in metric_cols:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.nanmean(
                    values
                )
            )

            row[
                f"{metric}_std"
            ] = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
                if np.sum(
                    np.isfinite(
                        values
                    )
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "shift_severity"
    )


def build_paired_differences(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "tail_fraction",
        "shift_severity",
    ]

    source = (
        per_seed[
            per_seed[
                "method"
            ] == "source_only"
        ][
            keys
            + [
                "auroc",
                "auprc",
                "f1",
                "brier",
                "log_loss",
                "ece",
            ]
        ]
        .copy()
        .rename(
            columns={
                "auroc": "source_auroc",
                "auprc": "source_auprc",
                "f1": "source_f1",
                "brier": "source_brier",
                "log_loss": "source_log_loss",
                "ece": "source_ece",
            }
        )
    )

    adapted = per_seed[
        per_seed[
            "method"
        ] != "source_only"
    ].copy()

    merged = adapted.merge(
        source,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    merged[
        "auroc_improvement_vs_source"
    ] = (
        merged[
            "auroc"
        ]
        - merged[
            "source_auroc"
        ]
    )

    merged[
        "auprc_improvement_vs_source"
    ] = (
        merged[
            "auprc"
        ]
        - merged[
            "source_auprc"
        ]
    )

    merged[
        "f1_improvement_vs_source"
    ] = (
        merged[
            "f1"
        ]
        - merged[
            "source_f1"
        ]
    )

    merged[
        "brier_improvement_vs_source"
    ] = (
        merged[
            "source_brier"
        ]
        - merged[
            "brier"
        ]
    )

    merged[
        "log_loss_improvement_vs_source"
    ] = (
        merged[
            "source_log_loss"
        ]
        - merged[
            "log_loss"
        ]
    )

    merged[
        "ece_improvement_vs_source"
    ] = (
        merged[
            "source_ece"
        ]
        - merged[
            "ece"
        ]
    )

    return merged


def summarize_paired_differences(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "tail_fraction",
        "shift_severity",
    ]

    metrics = [
        "auroc_improvement_vs_source",
        "auprc_improvement_vs_source",
        "f1_improvement_vs_source",
        "brier_improvement_vs_source",
        "log_loss_improvement_vs_source",
        "ece_improvement_vs_source",
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

        for metric in metrics:
            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            n = len(
                values
            )

            if n == 0:
                mean = std = low = high = np.nan
            else:
                mean = float(
                    np.mean(
                        values
                    )
                )

                std = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    if n > 1
                    else 0.0
                )

                if n > 1:
                    critical = float(
                        student_t.ppf(
                            0.975,
                            df=n - 1,
                        )
                    )

                    half = (
                        critical
                        * std
                        / math.sqrt(
                            n
                        )
                    )

                    low = (
                        mean - half
                    )
                    high = (
                        mean + half
                    )
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
            ] = low
            row[
                f"{metric}_ci95_high"
            ] = high

        row[
            "n_seeds"
        ] = int(
            group[
                "seed"
            ].nunique()
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "shift_severity",
            "method",
        ]
    )


def build_mlp_iw_vs_uniform_differences(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    """
    MLP-only paired comparison that isolates the weighting effect from the
    effect of simply continuing optimisation.
    """
    keys = [
        "seed",
        "dataset",
        "classifier",
        "tail_fraction",
        "shift_severity",
    ]

    uniform = (
        per_seed[
            (per_seed["classifier"] == "mlp")
            & (per_seed["method"] == "uniform_extra_training")
        ][
            keys
            + [
                "auroc",
                "auprc",
                "f1",
                "brier",
                "log_loss",
                "ece",
            ]
        ]
        .copy()
        .rename(
            columns={
                "auroc": "uniform_auroc",
                "auprc": "uniform_auprc",
                "f1": "uniform_f1",
                "brier": "uniform_brier",
                "log_loss": "uniform_log_loss",
                "ece": "uniform_ece",
            }
        )
    )

    iw = per_seed[
        (per_seed["classifier"] == "mlp")
        & per_seed["method"].isin(
            ["oracle_iw", "estimated_iw"]
        )
    ].copy()

    merged = iw.merge(
        uniform,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    merged["auroc_improvement_vs_uniform"] = (
        merged["auroc"] - merged["uniform_auroc"]
    )
    merged["auprc_improvement_vs_uniform"] = (
        merged["auprc"] - merged["uniform_auprc"]
    )
    merged["f1_improvement_vs_uniform"] = (
        merged["f1"] - merged["uniform_f1"]
    )
    merged["brier_improvement_vs_uniform"] = (
        merged["uniform_brier"] - merged["brier"]
    )
    merged["log_loss_improvement_vs_uniform"] = (
        merged["uniform_log_loss"] - merged["log_loss"]
    )
    merged["ece_improvement_vs_uniform"] = (
        merged["uniform_ece"] - merged["ece"]
    )

    return merged


def summarize_mlp_iw_vs_uniform(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "tail_fraction",
        "shift_severity",
    ]

    metrics = [
        "auroc_improvement_vs_uniform",
        "auprc_improvement_vs_uniform",
        "f1_improvement_vs_uniform",
        "brier_improvement_vs_uniform",
        "log_loss_improvement_vs_uniform",
        "ece_improvement_vs_uniform",
    ]

    rows = []

    for keys, group in paired.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            n = len(values)

            if n == 0:
                mean = std = low = high = np.nan
            else:
                mean = float(np.mean(values))
                std = float(
                    np.std(values, ddof=1)
                    if n > 1
                    else 0.0
                )

                if n > 1:
                    critical = float(
                        student_t.ppf(
                            0.975,
                            df=n - 1,
                        )
                    )
                    half = critical * std / math.sqrt(n)
                    low = mean - half
                    high = mean + half
                else:
                    low = mean
                    high = mean

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["shift_severity", "method"]
    )


# ============================================================
# Plotting
# ============================================================
def plot_metric(
    summary: pd.DataFrame,
    classifier_name: str,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    subset_classifier = summary[
        summary[
            "classifier"
        ] == classifier_name
    ]

    for method in METHODS:
        subset = subset_classifier[
            subset_classifier[
                "method"
            ] == method
        ].sort_values(
            "shift_severity"
        )

        if subset.empty:
            continue

        plt.errorbar(
            subset[
                "shift_severity"
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
        "Covariate-shift severity (1 - retained tail fraction)"
    )

    plt.ylabel(
        ylabel
    )

    plt.legend()

    savefig(
        output_path
    )


def plot_shift_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    plt.errorbar(
        diagnostics[
            "shift_severity"
        ],
        diagnostics[
            "marginal_domain_auroc_mean"
        ],
        yerr=diagnostics[
            "marginal_domain_auroc_std"
        ],
        marker="o",
        capsize=3,
        label="Marginal domain AUROC",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="No separability",
    )

    plt.xlabel(
        "Covariate-shift severity"
    )

    plt.ylabel(
        "Source-vs-target domain AUROC"
    )

    plt.legend()

    savefig(
        output_path
    )


def plot_weight_ess(
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.0, 5.0)
    )

    plt.errorbar(
        diagnostics[
            "shift_severity"
        ],
        diagnostics[
            "oracle_weight_ess_ratio_mean"
        ],
        yerr=diagnostics[
            "oracle_weight_ess_ratio_std"
        ],
        marker="o",
        capsize=3,
        label="Oracle IW ESS / N",
    )

    plt.errorbar(
        diagnostics[
            "shift_severity"
        ],
        diagnostics[
            "estimated_weight_ess_ratio_mean"
        ],
        yerr=diagnostics[
            "estimated_weight_ess_ratio_std"
        ],
        marker="o",
        capsize=3,
        label="Estimated IW ESS / N",
    )

    plt.xlabel(
        "Covariate-shift severity"
    )

    plt.ylabel(
        "Effective sample size ratio"
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.legend()

    savefig(
        output_path
    )


# ============================================================
# One seed
# ============================================================
def run_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    tail_fractions: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print(
        "\n" + "=" * 120
    )
    print(
        f"CONTROLLED COVARIATE SHIFT | dataset={args.dataset} | seed={seed}"
    )
    print(
        "=" * 120
    )

    split = build_seed_split(
        X=X,
        y=y,
        design=design,
        seed=seed,
        source_train_size=args.source_train_size,
        source_val_size=args.source_val_size,
        source_test_size=args.source_test_size,
        target_adapt_candidate_size=args.target_adapt_candidate_size,
        target_test_candidate_size=args.target_test_candidate_size,
    )

    # --------------------------------------------------------
    # NIDS preprocessing: source train only.
    # --------------------------------------------------------
    model_scaler = StandardScaler()

    X_source_train = model_scaler.fit_transform(
        split[
            "X_source_train"
        ]
    ).astype(
        np.float32
    )

    X_source_val = model_scaler.transform(
        split[
            "X_source_val"
        ]
    ).astype(
        np.float32
    )

    X_source_test = model_scaler.transform(
        split[
            "X_source_test"
        ]
    ).astype(
        np.float32
    )

    y_source_train = split[
        "y_source_train"
    ]
    y_source_val = split[
        "y_source_val"
    ]
    y_source_test = split[
        "y_source_test"
    ]

    source_attack_prior = float(
        np.mean(
            y_source_train
        )
    )

    # Scores from the separate fixed shift-design mapping.
    source_train_shift_scores = design.score(
        split[
            "X_source_train"
        ]
    )

    source_val_shift_scores = design.score(
        split[
            "X_source_val"
        ]
    )

    # --------------------------------------------------------
    # Train source classifiers once.
    # --------------------------------------------------------
    source_mlp, source_mlp_info = train_source_mlp(
        X_source_train,
        y_source_train,
        X_source_val,
        y_source_val,
        args,
        device,
        seed,
    )

    source_logistic, source_logistic_info = fit_logistic(
        X_source_train,
        y_source_train,
        seed,
    )

    source_classifiers: dict[
        str,
        Any,
    ] = {
        "mlp": source_mlp,
        "logistic_regression": source_logistic,
    }

    source_rows: list[
        dict[str, Any]
    ] = []

    for classifier_name, classifier in source_classifiers.items():
        source_probs = sigmoid(
            classifier_logits(
                classifier_name,
                classifier,
                X_source_test,
                device,
            )
        )

        metrics = probability_metrics(
            y_source_test,
            source_probs,
        )

        source_rows.append(
            {
                "seed": seed,
                "dataset": args.dataset,
                "classifier": classifier_name,
                "source_attack_prior": source_attack_prior,
                "source_train_size": len(
                    y_source_train
                ),
                "source_val_size": len(
                    y_source_val
                ),
                "source_test_size": len(
                    y_source_test
                ),
                "training_best_epoch": (
                    source_mlp_info[
                        "best_epoch"
                    ]
                    if classifier_name
                    == "mlp"
                    else np.nan
                ),
                "training_best_val_auroc": (
                    source_mlp_info[
                        "best_val_auroc"
                    ]
                    if classifier_name
                    == "mlp"
                    else np.nan
                ),
                "training_converged": (
                    True
                    if classifier_name
                    == "mlp"
                    else source_logistic_info[
                        "converged"
                    ]
                ),
                "training_iterations": (
                    np.nan
                    if classifier_name
                    == "mlp"
                    else source_logistic_info[
                        "iterations"
                    ]
                ),
                **{
                    f"source_test_{k}": v
                    for k, v in metrics.items()
                },
            }
        )

    result_rows: list[
        dict[str, Any]
    ] = []

    diagnostic_rows: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Each synthetic target.
    # --------------------------------------------------------
    for condition_index, tail_fraction in enumerate(
        tail_fractions
    ):
        target = select_target_condition(
            X=X,
            y=y,
            design=design,
            split=split,
            tail_fraction=tail_fraction,
            target_adapt_size=args.target_adapt_size,
            target_test_size=args.target_test_size,
        )

        X_target_adapt = model_scaler.transform(
            target[
                "X_adapt_raw"
            ]
        ).astype(
            np.float32
        )

        X_target_test = model_scaler.transform(
            target[
                "X_test_raw"
            ]
        ).astype(
            np.float32
        )

        y_target_test = target[
            "y_test"
        ]

        target_attack_prior = float(
            np.mean(
                y_target_test
            )
        )

        shift_severity = float(
            1.0
            - tail_fraction
        )

        threshold = target[
            "threshold"
        ]

        # ----------------------------------------------------
        # Shift diagnostics.
        # ----------------------------------------------------
        marginal_diag = marginal_domain_diagnostic(
            X_source_test,
            X_target_test,
            seed=(
                seed
                + condition_index
                * 1000
                + 101
            ),
        )

        target_test_shift_scores = design.score(
            target[
                "X_test_raw"
            ]
        )

        pooled_std = float(
            np.std(
                np.concatenate(
                    [
                        source_train_shift_scores,
                        target_test_shift_scores,
                    ]
                )
            )
        )

        shift_score_smd = (
            float(
                np.mean(
                    target_test_shift_scores
                )
                - np.mean(
                    source_train_shift_scores
                )
            )
            / max(
                pooled_std,
                EPS,
            )
        )

        # ----------------------------------------------------
        # Oracle IW.
        # ----------------------------------------------------
        oracle_train_weights = (
            oracle_importance_weights(
                source_train_shift_scores,
                threshold,
            )
        )

        oracle_val_weights = (
            oracle_importance_weights(
                source_val_shift_scores,
                threshold,
            )
        )

        source_region_fraction_train = float(
            np.mean(
                source_train_shift_scores
                >= threshold
            )
            if not np.isneginf(
                threshold
            )
            else 1.0
        )

        # ----------------------------------------------------
        # Estimated IW from unlabelled target adapt.
        # ----------------------------------------------------
        (
            estimated_weight_sets,
            estimated_weight_info,
        ) = estimate_importance_weights(
            X_source_train=X_source_train,
            X_target_adapt=X_target_adapt,
            X_source_eval_sets={
                "train": X_source_train,
                "val": X_source_val,
            },
            seed=(
                seed
                + condition_index
                * 1000
                + 202
            ),
            iw_clip=args.iw_clip,
        )

        estimated_train_weights = (
            estimated_weight_sets[
                "train"
            ]
        )

        estimated_val_weights = (
            estimated_weight_sets[
                "val"
            ]
        )

        if (
            np.std(
                oracle_train_weights
            ) > 0
            and np.std(
                estimated_train_weights
            ) > 0
        ):
            weight_correlation = float(
                np.corrcoef(
                    oracle_train_weights,
                    estimated_train_weights,
                )[0, 1]
            )
        else:
            weight_correlation = (
                1.0
                if tail_fraction >= 1.0
                else np.nan
            )

        # ----------------------------------------------------
        # CORAL target -> source transform.
        # ----------------------------------------------------
        coral = coral_target_to_source_fit(
            X_source=X_source_train,
            X_target_adapt=X_target_adapt,
            eps=args.coral_eps,
        )

        X_target_test_coral = (
            coral_target_to_source_apply(
                X_target_test,
                coral,
            )
        )

        diagnostic_rows.append(
            {
                "seed": seed,
                "dataset": args.dataset,
                "tail_fraction": float(
                    tail_fraction
                ),
                "shift_severity": shift_severity,
                "threshold": float(
                    threshold
                ),
                "selection_uses_label": False,
                "source_attack_prior": source_attack_prior,
                "target_attack_prior": target_attack_prior,
                "attack_prior_change": float(
                    target_attack_prior
                    - source_attack_prior
                ),
                "source_region_fraction_train": source_region_fraction_train,
                "target_candidate_region_fraction_adapt": target[
                    "eligible_adapt_fraction"
                ],
                "target_candidate_region_fraction_test": target[
                    "eligible_test_fraction"
                ],
                "source_shift_score_mean": float(
                    np.mean(
                        source_train_shift_scores
                    )
                ),
                "target_shift_score_mean": float(
                    np.mean(
                        target_test_shift_scores
                    )
                ),
                "shift_score_standardized_mean_difference": shift_score_smd,
                "oracle_weight_ess": effective_sample_size(
                    oracle_train_weights
                ),
                "oracle_weight_ess_ratio": float(
                    effective_sample_size(
                        oracle_train_weights
                    )
                    / len(
                        oracle_train_weights
                    )
                ),
                "estimated_weight_ess": effective_sample_size(
                    estimated_train_weights
                ),
                "estimated_weight_ess_ratio": float(
                    effective_sample_size(
                        estimated_train_weights
                    )
                    / len(
                        estimated_train_weights
                    )
                ),
                "estimated_vs_oracle_weight_correlation": weight_correlation,
                "estimated_domain_auroc": estimated_weight_info[
                    "domain_auroc"
                ],
                "estimated_domain_classifier_converged": estimated_weight_info[
                    "domain_classifier_converged"
                ],
                "estimated_domain_classifier_iterations": estimated_weight_info[
                    "domain_classifier_iterations"
                ],
                "coral_transform_identity_relative_frobenius": coral[
                    "diagnostics"
                ][
                    "transform_identity_relative_frobenius"
                ],
                "coral_mean_displacement_rms": coral[
                    "diagnostics"
                ][
                    "mean_displacement_rms"
                ],
                "coral_sample_transform_rms_displacement": coral[
                    "diagnostics"
                ][
                    "sample_transform_rms_displacement"
                ],
                "coral_covariance_relative_frobenius_before": coral[
                    "diagnostics"
                ][
                    "covariance_relative_frobenius_before"
                ],
                "coral_covariance_relative_frobenius_after": coral[
                    "diagnostics"
                ][
                    "covariance_relative_frobenius_after"
                ],
                "coral_source_covariance_shrinkage": coral[
                    "diagnostics"
                ][
                    "source_covariance_shrinkage"
                ],
                "coral_target_covariance_shrinkage": coral[
                    "diagnostics"
                ][
                    "target_covariance_shrinkage"
                ],
                **marginal_diag,
            }
        )

        print(
            f"\nTail fraction={tail_fraction:.2f} "
            f"| severity={shift_severity:.2f} "
            f"| target prior={target_attack_prior:.3f} "
            f"| domain AUROC="
            f"{marginal_diag['marginal_domain_auroc']:.3f}"
        )

        print(
            f"  IW ESS ratio | oracle="
            f"{effective_sample_size(oracle_train_weights) / len(oracle_train_weights):.3f} "
            f"| estimated="
            f"{effective_sample_size(estimated_train_weights) / len(estimated_train_weights):.3f} "
            f"| weight corr={weight_correlation:.3f}"
        )

        # ----------------------------------------------------
        # Fit IW-adapted classifiers.
        # ----------------------------------------------------
        # Matched adaptation seed means batch order / dropout randomness are
        # identical across uniform, oracle-IW, and estimated-IW MLP runs.
        # At zero shift, uniform and oracle weights are both exactly one, so
        # those two runs should be numerically identical.
        matched_adapt_seed = (
            seed
            + condition_index
            * 1000
            + 300
        )

        uniform_mlp, uniform_mlp_info = (
            adapt_mlp_with_weights(
                source_model=source_mlp,
                X_train=X_source_train,
                y_train=y_source_train,
                train_weights=np.ones(
                    len(y_source_train),
                    dtype=float,
                ),
                X_val=X_source_val,
                y_val=y_source_val,
                val_weights=np.ones(
                    len(y_source_val),
                    dtype=float,
                ),
                args=args,
                device=device,
                seed=matched_adapt_seed,
                label="uniform_extra_training",
            )
        )

        oracle_mlp, oracle_mlp_info = (
            adapt_mlp_with_weights(
                source_model=source_mlp,
                X_train=X_source_train,
                y_train=y_source_train,
                train_weights=oracle_train_weights,
                X_val=X_source_val,
                y_val=y_source_val,
                val_weights=oracle_val_weights,
                args=args,
                device=device,
                seed=matched_adapt_seed,
                label="oracle_iw",
            )
        )

        estimated_mlp, estimated_mlp_info = (
            adapt_mlp_with_weights(
                source_model=source_mlp,
                X_train=X_source_train,
                y_train=y_source_train,
                train_weights=estimated_train_weights,
                X_val=X_source_val,
                y_val=y_source_val,
                val_weights=estimated_val_weights,
                args=args,
                device=device,
                seed=matched_adapt_seed,
                label="estimated_iw",
            )
        )

        oracle_logistic, oracle_logistic_info = (
            fit_logistic(
                X_source_train,
                y_source_train,
                seed=(
                    seed
                    + condition_index
                    * 1000
                    + 401
                ),
                sample_weight=oracle_train_weights,
            )
        )

        (
            estimated_logistic,
            estimated_logistic_info,
        ) = fit_logistic(
            X_source_train,
            y_source_train,
            seed=(
                seed
                + condition_index
                * 1000
                + 402
            ),
            sample_weight=estimated_train_weights,
        )

        method_models: dict[
            str,
            dict[str, Any],
        ] = {
            "mlp": {
                "source_only": source_mlp,
                "uniform_extra_training": uniform_mlp,
                "oracle_iw": oracle_mlp,
                "estimated_iw": estimated_mlp,
            },
            "logistic_regression": {
                "source_only": source_logistic,
                "oracle_iw": oracle_logistic,
                "estimated_iw": estimated_logistic,
            },
        }

        # ----------------------------------------------------
        # Evaluate exact same target test for every method.
        # ----------------------------------------------------
        for classifier_name in CLASSIFIERS:
            classifier_methods = (
                [
                    "source_only",
                    "uniform_extra_training",
                    "oracle_iw",
                    "estimated_iw",
                ]
                if classifier_name == "mlp"
                else [
                    "source_only",
                    "oracle_iw",
                    "estimated_iw",
                ]
            )

            for method in classifier_methods:
                classifier = method_models[
                    classifier_name
                ][
                    method
                ]

                probs = sigmoid(
                    classifier_logits(
                        classifier_name,
                        classifier,
                        X_target_test,
                        device,
                    )
                )

                metrics = probability_metrics(
                    y_target_test,
                    probs,
                )

                result_rows.append(
                    {
                        "seed": seed,
                        "dataset": args.dataset,
                        "classifier": classifier_name,
                        "method": method,
                        "tail_fraction": float(
                            tail_fraction
                        ),
                        "shift_severity": shift_severity,
                        "source_attack_prior": source_attack_prior,
                        "target_attack_prior": target_attack_prior,
                        "attack_prior_change": float(
                            target_attack_prior
                            - source_attack_prior
                        ),
                        "marginal_domain_auroc": marginal_diag[
                            "marginal_domain_auroc"
                        ],
                        "oracle_weight_ess_ratio": float(
                            effective_sample_size(
                                oracle_train_weights
                            )
                            / len(
                                oracle_train_weights
                            )
                        ),
                        "estimated_weight_ess_ratio": float(
                            effective_sample_size(
                                estimated_train_weights
                            )
                            / len(
                                estimated_train_weights
                            )
                        ),
                        **metrics,
                    }
                )

            # CORAL uses unchanged source classifier on transformed target.
            source_classifier = source_classifiers[
                classifier_name
            ]

            coral_probs = sigmoid(
                classifier_logits(
                    classifier_name,
                    source_classifier,
                    X_target_test_coral,
                    device,
                )
            )

            coral_metrics = probability_metrics(
                y_target_test,
                coral_probs,
            )

            result_rows.append(
                {
                    "seed": seed,
                    "dataset": args.dataset,
                    "classifier": classifier_name,
                    "method": "coral_target_to_source",
                    "tail_fraction": float(
                        tail_fraction
                    ),
                    "shift_severity": shift_severity,
                    "source_attack_prior": source_attack_prior,
                    "target_attack_prior": target_attack_prior,
                    "attack_prior_change": float(
                        target_attack_prior
                        - source_attack_prior
                    ),
                    "marginal_domain_auroc": marginal_diag[
                        "marginal_domain_auroc"
                    ],
                    "oracle_weight_ess_ratio": float(
                        effective_sample_size(
                            oracle_train_weights
                        )
                        / len(
                            oracle_train_weights
                        )
                    ),
                    "estimated_weight_ess_ratio": float(
                        effective_sample_size(
                            estimated_train_weights
                        )
                        / len(
                            estimated_train_weights
                        )
                    ),
                    **coral_metrics,
                }
            )

    return (
        result_rows,
        diagnostic_rows,
        source_rows,
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
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--tail_fractions",
        type=float,
        nargs="+",
        default=DEFAULT_TAIL_FRACTIONS,
    )

    parser.add_argument(
        "--design_size",
        type=int,
        default=DEFAULT_DESIGN_SIZE,
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
        "--target_adapt_candidate_size",
        type=int,
        default=DEFAULT_TARGET_ADAPT_CANDIDATE_SIZE,
    )

    parser.add_argument(
        "--target_test_candidate_size",
        type=int,
        default=DEFAULT_TARGET_TEST_CANDIDATE_SIZE,
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

    # Source MLP.
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

    # IW adaptation.
    parser.add_argument(
        "--adapt_epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--adapt_lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--adapt_patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--iw_clip",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--coral_eps",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    tail_fractions = sorted(
        set(
            float(x)
            for x in args.tail_fractions
        ),
        reverse=True,
    )

    if any(
        x <= 0.0
        or x > 1.0
        for x in tail_fractions
    ):
        raise ValueError(
            "tail_fractions must lie in (0, 1]."
        )

    print(
        "=" * 120
    )
    print(
        "CONTROLLED PURE COVARIATE-SHIFT EXPERIMENT"
    )
    print(
        "=" * 120
    )
    print(
        f"Dataset: {args.dataset}"
    )
    print(
        f"Pool: {args.pool_variant}"
    )
    print(
        f"Tail fractions: {tail_fractions}"
    )
    print(
        f"Seeds: {args.seeds}"
    )

    X, y, feature_names = load_dataset_pool(
        args.dataset,
        args.pool_variant,
    )

    print(
        f"Loaded pool: {len(y):,} rows "
        f"| benign={(y == 0).sum():,} "
        f"| attack={(y == 1).sum():,} "
        f"| features={X.shape[1]}"
    )

    design = build_shift_design(
        X,
        tail_fractions=tail_fractions,
        design_size=args.design_size,
    )

    print(
        f"Shift-design PCA explained variance ratio: "
        f"{design.pca.explained_variance_ratio_[0]:.6f}"
    )

    output_dir = (
        STAGE1_DIR
        / "controlled_covariate_shift"
        / args.dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    loadings_df = shift_loading_table(
        design,
        feature_names,
    )

    loading_path = (
        output_dir
        / "controlled_covariate_shift_direction.csv"
    )

    loadings_df.to_csv(
        loading_path,
        index=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
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
            seed=int(
                seed
            ),
            X=X,
            y=y,
            design=design,
            tail_fractions=tail_fractions,
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

    diagnostics_summary_df = (
        aggregate_diagnostics(
            diagnostics_df
        )
    )

    paired_df = build_paired_differences(
        per_seed_df
    )

    paired_summary_df = (
        summarize_paired_differences(
            paired_df
        )
    )

    mlp_iw_vs_uniform_df = (
        build_mlp_iw_vs_uniform_differences(
            per_seed_df
        )
    )

    mlp_iw_vs_uniform_summary_df = (
        summarize_mlp_iw_vs_uniform(
            mlp_iw_vs_uniform_df
        )
    )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------
    per_seed_path = (
        output_dir
        / "controlled_covariate_shift_per_seed.csv"
    )

    summary_path = (
        output_dir
        / "controlled_covariate_shift_summary.csv"
    )

    source_path = (
        output_dir
        / "controlled_covariate_shift_source_metrics.csv"
    )

    diagnostics_path = (
        output_dir
        / "controlled_covariate_shift_diagnostics_per_seed.csv"
    )

    diagnostics_summary_path = (
        output_dir
        / "controlled_covariate_shift_diagnostics_summary.csv"
    )

    paired_path = (
        output_dir
        / "controlled_covariate_shift_paired_differences.csv"
    )

    paired_summary_path = (
        output_dir
        / "controlled_covariate_shift_paired_difference_summary.csv"
    )

    mlp_iw_vs_uniform_path = (
        output_dir
        / "controlled_covariate_shift_mlp_iw_vs_uniform_per_seed.csv"
    )

    mlp_iw_vs_uniform_summary_path = (
        output_dir
        / "controlled_covariate_shift_mlp_iw_vs_uniform_summary.csv"
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

    diagnostics_summary_df.to_csv(
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

    mlp_iw_vs_uniform_df.to_csv(
        mlp_iw_vs_uniform_path,
        index=False,
    )

    mlp_iw_vs_uniform_summary_df.to_csv(
        mlp_iw_vs_uniform_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Protocol.
    # --------------------------------------------------------
    protocol = {
        "research_question": (
            "How do source-only, importance weighting, and CORAL behave "
            "under controlled covariate shift where target selection depends "
            "only on X and P(Y|X) is preserved by construction?"
        ),
        "hypotheses": {
            "H1": (
                "Marginal domain separability increases with shift severity."
            ),
            "H2": (
                "Source-only target performance can degrade as the target "
                "places more mass on a restricted feature-space region."
            ),
            "H3": (
                "Importance weighting improves target risk by emphasizing "
                "target-relevant source regions."
            ),
            "H4": (
                "Estimated IW approaches oracle IW when density-ratio "
                "estimation is accurate and ESS is adequate."
            ),
            "H5": (
                "Very severe shift reduces IW effective sample size and can "
                "limit adaptation benefit."
            ),
        },
        "dataset": args.dataset,
        "pool_variant": args.pool_variant,
        "seeds": [
            int(
                s
            )
            for s in args.seeds
        ],
        "tail_fractions": tail_fractions,
        "shift_severities": [
            float(
                1.0 - x
            )
            for x in tail_fractions
        ],
        "design": {
            "design_seed": DESIGN_SEED,
            "design_size": args.design_size,
            "uses_labels": False,
            "score": (
                "PC1 of StandardScaler-transformed fixed design subset"
            ),
            "pca_explained_variance_ratio": float(
                design.pca.explained_variance_ratio_[
                    0
                ]
            ),
            "sign_rule": (
                "largest-absolute loading is forced positive"
            ),
            "target_definition": (
                "upper-tail subpopulation score >= fixed design quantile"
            ),
        },
        "source_sizes": {
            "train": args.source_train_size,
            "validation": args.source_val_size,
            "test": args.source_test_size,
        },
        "target_candidate_sizes": {
            "adaptation": args.target_adapt_candidate_size,
            "test": args.target_test_candidate_size,
        },
        "target_evaluation_sizes": {
            "unlabelled_adaptation": args.target_adapt_size,
            "test": args.target_test_size,
        },
        "classifiers": {
            "mlp": {
                "architecture": [
                    int(
                        X.shape[
                            1
                        ]
                    ),
                    *[
                        int(
                            h
                        )
                        for h in args.hidden_dims
                    ],
                    1,
                ],
                "dropout": args.dropout,
                "loss": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "source_lr": args.lr,
                "adapt_lr": args.adapt_lr,
                "source_max_epochs": args.max_epochs,
                "adapt_max_epochs": args.adapt_epochs,
            },
            "logistic_regression": {
                "solver": "lbfgs",
                "max_iter": 5000,
            },
        },
        "methods": {
            "source_only": (
                "No target adaptation."
            ),
            "uniform_extra_training": (
                "MLP-only matched continuation-training control with all "
                "weights equal to one. Uses the same adaptation random seed, "
                "LR, epoch budget, and warm start as MLP IW."
            ),
            "oracle_iw": (
                "Known synthetic density ratio I[A]/P_S(A); "
                "oracle/reference only."
            ),
            "estimated_iw": (
                "Density ratio estimated from a balanced source-vs-target "
                "logistic domain classifier using target-adaptation X only; "
                f"weights clipped at {args.iw_clip} before normalization."
            ),
            "coral_target_to_source": (
                "Unlabelled target-to-source CORAL with Ledoit-Wolf shrinkage "
                "covariance estimates; source classifier remains unchanged. "
                "Identity-distance and covariance-displacement diagnostics are saved."
            ),
        },
        "important_covariate_shift_guardrail": (
            "P(Y) may change under covariate shift because integrating a fixed "
            "P(Y|X) against a different P(X) can change the marginal class prior. "
            "This is not evidence that the construction became pure label shift."
        ),
        "leakage_controls": [
            "Shift design subset is excluded from all model source/target data.",
            "Shift construction uses X only, never Y.",
            "Source train/validation/test are mutually disjoint.",
            "Target adaptation/test candidate banks are disjoint.",
            "All source and target rows are disjoint.",
            "NIDS StandardScaler fits source train only.",
            "Estimated IW uses target-adaptation X without target labels.",
            "CORAL uses target-adaptation X without target labels.",
            "Target test labels are evaluation only.",
        ],
        "paired_statistics": (
            "Method-vs-source differences are paired within exact "
            "seed/classifier/tail condition and summarized with mean, SD, "
            "and descriptive t-based 95% CI. MLP oracle/estimated IW are also "
            "paired directly against uniform_extra_training to isolate the "
            "effect of weighting from additional optimisation."
        ),
    }

    protocol_path = (
        output_dir
        / "controlled_covariate_shift_protocol.json"
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
                "auprc",
                "Target AUPRC",
            ),
            (
                "f1",
                "Target F1 @ 0.5",
            ),
            (
                "brier",
                "Brier score (lower is better)",
            ),
        ]:
            plot_metric(
                summary_df,
                classifier_name,
                metric,
                ylabel,
                output_dir
                / (
                    f"covariate_shift_"
                    f"{classifier_name}_"
                    f"{metric}.png"
                ),
            )

    plot_shift_diagnostics(
        diagnostics_summary_df,
        output_dir
        / "covariate_shift_domain_separability.png",
    )

    plot_weight_ess(
        diagnostics_summary_df,
        output_dir
        / "covariate_shift_weight_ess.png",
    )

    # --------------------------------------------------------
    # Console summary.
    # --------------------------------------------------------
    print(
        "\n" + "=" * 135
    )
    print(
        "CONTROLLED COVARIATE-SHIFT EXPERIMENT COMPLETE"
    )
    print(
        "=" * 135
    )

    print(
        "\nSHIFT DIAGNOSTICS"
    )

    diagnostic_display = diagnostics_summary_df[
        [
            "tail_fraction",
            "shift_severity",
            "marginal_domain_auroc_mean",
            "source_region_fraction_train_mean",
            "target_attack_prior_mean",
            "attack_prior_change_mean",
            "oracle_weight_ess_ratio_mean",
            "estimated_weight_ess_ratio_mean",
            "estimated_vs_oracle_weight_correlation_mean",
            "estimated_domain_auroc_mean",
            "coral_transform_identity_relative_frobenius_mean",
            "coral_sample_transform_rms_displacement_mean",
            "coral_covariance_relative_frobenius_before_mean",
            "coral_covariance_relative_frobenius_after_mean",
        ]
    ]

    print(
        diagnostic_display.to_string(
            index=False
        )
    )

    print(
        "\nTARGET PERFORMANCE"
    )

    performance_display = summary_df[
        [
            "classifier",
            "method",
            "tail_fraction",
            "shift_severity",
            "target_attack_prior_mean",
            "auroc_mean",
            "auprc_mean",
            "f1_mean",
            "brier_mean",
        ]
    ]

    print(
        performance_display.to_string(
            index=False
        )
    )

    print(
        "\nGUARDRAILS"
    )
    print(
        "- The synthetic target is selected using X only; Y is never used in the shift mechanism."
    )
    print(
        "- A target class-prior change is allowed under covariate shift and is not itself proof of label shift."
    )
    print(
        "- Check marginal domain AUROC rises with severity before interpreting adaptation results."
    )
    print(
        "- Check oracle/estimated IW ESS ratios; severe weight collapse can explain adaptation failure."
    )
    print(
        "- Use paired_difference_summary for method-vs-source claims."
    )
    print(
        "- For MLP IW claims, use mlp_iw_vs_uniform_summary to remove the extra-training confound."
    )
    print(
        "- At zero shift, uniform_extra_training and oracle_iw should be numerically identical for the MLP."
    )
    print(
        "- Inspect CORAL identity/displacement diagnostics at zero shift before interpreting CORAL failures mechanistically."
    )

    print(
        "\nOUTPUTS"
    )

    for path in [
        summary_path,
        source_path,
        diagnostics_summary_path,
        paired_summary_path,
        mlp_iw_vs_uniform_summary_path,
        mlp_iw_vs_uniform_path,
        per_seed_path,
        diagnostics_path,
        paired_path,
        loading_path,
        protocol_path,
    ]:
        print(
            f"- {path}"
        )


if __name__ == "__main__":
    main()
