"""
controlled_concept_shift_CHAT_NEW.py

Controlled boundary-localized concept-shift experiment for DANIDS.

Research question
-----------------
How do adaptation strategies behave when the predictive relationship changes
while the feature distribution and class prior are deliberately held fixed?

Controlled mechanism
--------------------
This experiment uses the Stage-1 balanced_100k reservoir.

A fixed held-out DESIGN subset is excluded from every model train/validation/
test split. A reference logistic classifier is trained ONLY on the design
subset. Its absolute decision margin |f_design(x)| provides a fixed definition
of "near the original decision boundary".

Source and target rows are then sampled as disjoint, class-balanced samples
from the SAME balanced reservoir. The target feature rows are identical across
all concept-severity conditions within a seed.

For target adaptation and target test only, labels are changed as follows:

    severity 0.00 -> flip  0% of each original class
    severity 0.10 -> flip 10% of each original class nearest design boundary
    severity 0.20 -> flip 20% of each original class nearest design boundary
    severity 0.30 -> flip 30% of each original class nearest design boundary

Exactly the same fraction is flipped in benign and attack, therefore:

    P_S(Y=1) = P_T(Y=1) = 0.5

and because NO target X row is added, removed, transformed or reweighted
between severity conditions:

    P_T(X) is exactly unchanged across concept-severity conditions.

Source and target are disjoint finite samples from the same balanced reservoir,
so source-vs-target X-domain AUROC should remain near 0.5.

The intervention directly changes labels on fixed X rows, therefore it changes
P(Y|X). This is a controlled boundary-localized concept-shift benchmark.

Important terminology
---------------------
This is not random label noise. The flipped rows are deterministically
concentrated near a fixed held-out reference decision boundary, making the
change structured and local in feature space.

It is also not claimed to reproduce every possible form of real-world concept
drift. It is a mechanism benchmark designed to isolate classifier adaptation.

Methods
-------
1. source_only
   Source model without adaptation.

2. oracle_prior_correction
   Mismatched sanity baseline. Because source and target class priors are both
   exactly 0.5, prior correction is the identity transform.

3. unconditional_estimated_iw
   Mismatched covariate-shift baseline. A source-vs-target domain classifier
   estimates p_T(X)/p_S(X) using unlabeled target adaptation X. Because X has
   not deliberately shifted, its domain AUROC should be near 0.5 and weights
   should contain little useful signal.

4. uniform_extra_training
   MLP only. Warm-start source model and continue training on source data with
   all weights equal to one. Uses the exact same optimizer/seed/epoch budget as
   unconditional_estimated_iw. This removes the extra-training confound.

5. naive_target_finetuning
   MLP only. Warm-start source MLP and fine-tune on labelled target adaptation
   data under the shifted concept.

6. finetuning_plus_replay
   MLP only. Same labelled-target fine-tuning, plus a fixed random replay
   buffer of 250 benign + 250 attack SOURCE-TRAIN examples. With the default
   target adaptation budget (10k/class, 80% train), replay is approximately
   500 / (16,000 + 500) = 3.03% of the adaptation training set.

7. target_only_reference
   Supervision-heavy target reference:
   - MLP trained from scratch on target adaptation train/validation labels.
   - Logistic regression trained on target adaptation train labels.
   This is a reference, not an "upper bound".

Hypotheses
----------
H1. Source-vs-target feature-domain AUROC remains near 0.5 and attack prior
    remains exactly 0.5 across severity.

H2. Source-only target performance degrades as the fraction of boundary-local
    concept changes increases.

H3. Prior correction does not repair concept shift when priors do not change.

H4. Unconditional importance weighting provides little benefit when P(X) has
    not deliberately changed.

H5. Labelled target fine-tuning repairs target performance because it updates
    the classifier decision function.

H6. Fine-tuning may reduce performance on the original source concept, while
    replay reduces this forgetting with limited target-performance cost.

H7. Adaptation effect is model-dependent; a nonlinear MLP may tolerate or
    adapt to the local concept change differently from logistic regression.

Leakage / rigor controls
------------------------
- Design subset is excluded from all modelling data.
- Design boundary is fitted once and never selected using target test results.
- Source train/val/test and target adapt/test are mutually disjoint.
- Target X membership is fixed across all severities within a seed.
- Only target labels change across severity.
- Target prior remains exactly 0.5.
- NIDS StandardScaler fits source train only.
- Target test labels are evaluation only.
- Unconditional IW receives target X only, never target labels.
- Prior-correction baseline uses known equal priors and must be identity.
- Target fine-tuning uses target-adaptation labels only.
- Replay examples come from SOURCE TRAIN only.
- Replay indices are fixed across severities within a seed.
- Naive FT and FT+replay use the same target train/validation split.
- Five default seeds; paired descriptive t-based 95% confidence intervals.
- MLP unconditional IW is interpreted relative to uniform_extra_training.
- Target-only is called a reference, not an upper bound.

Run
---
    python -B -m src.analysis.controlled_concept_shift_CHAT_NEW

Other datasets can be run with --dataset.
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

from src.config import STAGE1_DIR


# ============================================================
# Defaults
# ============================================================
DEFAULT_DATASET = "NF-ToN-IoT-v3"
DEFAULT_POOL_VARIANT = "balanced_100k"

DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
DEFAULT_CONCEPT_SEVERITIES = [0.0, 0.10, 0.20, 0.30]

DESIGN_SEED = 314159

# Per-class allocation. The balanced_100k reservoir has 100k/class.
DEFAULT_DESIGN_PER_CLASS = 10_000
DEFAULT_SOURCE_TRAIN_PER_CLASS = 20_000
DEFAULT_SOURCE_VAL_PER_CLASS = 5_000
DEFAULT_SOURCE_TEST_PER_CLASS = 5_000
DEFAULT_TARGET_ADAPT_PER_CLASS = 10_000
DEFAULT_TARGET_TEST_PER_CLASS = 20_000

DEFAULT_REPLAY_PER_CLASS = 250

CLASSIFIERS = ["mlp", "logistic_regression"]

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
        raise FileNotFoundError(f"Missing feature pool: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"Missing label pool: {y_path}")

    frame = pd.read_parquet(x_path)
    X = frame.to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y = np.load(y_path).astype(np.int8)

    if len(X) != len(y):
        raise ValueError(
            f"X/y length mismatch: {len(X)} vs {len(y)}"
        )

    unique = set(np.unique(y).tolist())
    if unique != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0,1}}, got {sorted(unique)}"
        )

    return X, y, [str(c) for c in frame.columns]


# ============================================================
# Held-out design boundary
# ============================================================
class ConceptDesign:
    def __init__(
        self,
        scaler: StandardScaler,
        classifier: LogisticRegression,
        design_indices: np.ndarray,
        design_auroc: float,
        converged: bool,
        iterations: int,
    ) -> None:
        self.scaler = scaler
        self.classifier = classifier
        self.design_indices = design_indices
        self.design_auroc = design_auroc
        self.converged = converged
        self.iterations = iterations

    def signed_margin(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.decision_function(
            self.scaler.transform(X)
        ).astype(float)

    def absolute_margin(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_margin(X))


def build_concept_design(
    X: np.ndarray,
    y: np.ndarray,
    design_per_class: int,
) -> ConceptDesign:
    rng = np.random.default_rng(DESIGN_SEED)

    design_parts = []

    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls).copy()
        rng.shuffle(idx)

        if len(idx) < design_per_class:
            raise ValueError(
                f"Class {cls} has only {len(idx):,} rows, "
                f"need {design_per_class:,} design rows."
            )

        design_parts.append(
            idx[:design_per_class]
        )

    design_indices = np.concatenate(design_parts)
    rng.shuffle(design_indices)

    X_design = X[design_indices]
    y_design = y[design_indices]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X_design,
        y_design,
        test_size=0.30,
        stratify=y_design,
        random_state=DESIGN_SEED,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=DESIGN_SEED,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )
        clf.fit(X_train_s, y_train)

    convergence_warnings = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    probs = clf.predict_proba(X_test_s)[:, 1]
    auroc = float(
        roc_auc_score(
            y_test,
            probs,
        )
    )

    return ConceptDesign(
        scaler=scaler,
        classifier=clf,
        design_indices=design_indices,
        design_auroc=auroc,
        converged=(
            len(convergence_warnings) == 0
        ),
        iterations=int(
            np.max(clf.n_iter_)
        ),
    )


# ============================================================
# Disjoint fixed source/target split
# ============================================================
def build_seed_split(
    X: np.ndarray,
    y: np.ndarray,
    design: ConceptDesign,
    seed: int,
    source_train_per_class: int,
    source_val_per_class: int,
    source_test_per_class: int,
    target_adapt_per_class: int,
    target_test_per_class: int,
) -> dict[str, np.ndarray]:
    design_mask = np.zeros(
        len(y),
        dtype=bool,
    )
    design_mask[design.design_indices] = True

    required_per_class = (
        source_train_per_class
        + source_val_per_class
        + source_test_per_class
        + target_adapt_per_class
        + target_test_per_class
    )

    rng = np.random.default_rng(seed)

    parts: dict[str, list[np.ndarray]] = {
        "source_train": [],
        "source_val": [],
        "source_test": [],
        "target_adapt": [],
        "target_test": [],
    }

    for cls in [0, 1]:
        idx = np.flatnonzero(
            (y == cls)
            & (~design_mask)
        ).copy()

        rng.shuffle(idx)

        if len(idx) < required_per_class:
            raise ValueError(
                f"Class {cls}: need {required_per_class:,} non-design rows "
                f"but only {len(idx):,} available."
            )

        cursor = 0

        def take(n: int) -> np.ndarray:
            nonlocal cursor
            out = idx[
                cursor : cursor + n
            ]
            cursor += n
            return out

        parts["source_train"].append(
            take(source_train_per_class)
        )
        parts["source_val"].append(
            take(source_val_per_class)
        )
        parts["source_test"].append(
            take(source_test_per_class)
        )
        parts["target_adapt"].append(
            take(target_adapt_per_class)
        )
        parts["target_test"].append(
            take(target_test_per_class)
        )

    out = {}

    for name, arrays in parts.items():
        combined = np.concatenate(arrays)
        local_rng = np.random.default_rng(
            seed
            + {
                "source_train": 11,
                "source_val": 22,
                "source_test": 33,
                "target_adapt": 44,
                "target_test": 55,
            }[name]
        )
        local_rng.shuffle(combined)
        out[f"idx_{name}"] = combined

    all_groups = [
        out["idx_source_train"],
        out["idx_source_val"],
        out["idx_source_test"],
        out["idx_target_adapt"],
        out["idx_target_test"],
    ]

    for i in range(len(all_groups)):
        for j in range(i + 1, len(all_groups)):
            if np.intersect1d(
                all_groups[i],
                all_groups[j],
            ).size:
                raise RuntimeError(
                    "Split overlap detected."
                )

    for name in [
        "idx_source_train",
        "idx_source_val",
        "idx_source_test",
        "idx_target_adapt",
        "idx_target_test",
    ]:
        if not np.isclose(
            np.mean(y[out[name]]),
            0.5,
        ):
            raise RuntimeError(
                f"{name} is not class-balanced."
            )

    return out


# ============================================================
# Controlled concept-label intervention
# ============================================================
def apply_boundary_local_concept_shift(
    X_raw: np.ndarray,
    y_original: np.ndarray,
    design: ConceptDesign,
    severity: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Flip exactly severity fraction of each ORIGINAL class, choosing the rows
    with the smallest absolute held-out-design margin.

    X is never modified.
    """
    severity = float(severity)

    if severity < 0.0 or severity >= 0.5:
        raise ValueError(
            "Concept severity must be in [0, 0.5)."
        )

    y_original = np.asarray(
        y_original,
        dtype=np.int8,
    )

    if not np.isclose(
        np.mean(y_original),
        0.5,
    ):
        raise ValueError(
            "Concept-shift input must be 50/50 before flipping."
        )

    margins = design.absolute_margin(X_raw)

    y_shifted = y_original.copy()
    flip_mask = np.zeros(
        len(y_shifted),
        dtype=bool,
    )

    class_thresholds = {}
    class_flip_counts = {}

    for cls in [0, 1]:
        cls_positions = np.flatnonzero(
            y_original == cls
        )

        n_flip = int(
            round(
                severity
                * len(cls_positions)
            )
        )

        class_flip_counts[cls] = n_flip

        if n_flip == 0:
            class_thresholds[cls] = 0.0
            continue

        order = np.argsort(
            margins[cls_positions],
            kind="mergesort",
        )

        chosen = cls_positions[
            order[:n_flip]
        ]

        flip_mask[chosen] = True

        class_thresholds[cls] = float(
            np.max(
                margins[chosen]
            )
        )

    y_shifted[flip_mask] = (
        1 - y_shifted[flip_mask]
    )

    expected_flip_rate = float(
        np.mean(flip_mask)
    )

    if not np.isclose(
        np.mean(y_shifted),
        0.5,
    ):
        raise RuntimeError(
            "Shifted attack prior changed despite symmetric flips."
        )

    if not np.isclose(
        expected_flip_rate,
        severity,
        atol=1.0 / len(y_shifted) + 1e-12,
    ):
        raise RuntimeError(
            f"Flip rate {expected_flip_rate} does not match severity {severity}."
        )

    flipped_margins = margins[
        flip_mask
    ]
    unflipped_margins = margins[
        ~flip_mask
    ]

    return y_shifted, {
        "requested_concept_severity": severity,
        "actual_label_flip_rate": expected_flip_rate,
        "benign_to_attack_flips": int(
            np.sum(
                flip_mask
                & (y_original == 0)
            )
        ),
        "attack_to_benign_flips": int(
            np.sum(
                flip_mask
                & (y_original == 1)
            )
        ),
        "benign_boundary_margin_threshold": float(
            class_thresholds[0]
        ),
        "attack_boundary_margin_threshold": float(
            class_thresholds[1]
        ),
        "flipped_margin_mean": float(
            np.mean(flipped_margins)
            if len(flipped_margins)
            else 0.0
        ),
        "unflipped_margin_mean": float(
            np.mean(unflipped_margins)
            if len(unflipped_margins)
            else 0.0
        ),
        "flipped_margin_median": float(
            np.median(flipped_margins)
            if len(flipped_margins)
            else 0.0
        ),
        "unflipped_margin_median": float(
            np.median(unflipped_margins)
            if len(unflipped_margins)
            else 0.0
        ),
        "target_attack_prior_after": float(
            np.mean(y_shifted)
        ),
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
        previous = input_dim

        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        previous,
                        hidden,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden

        layers.append(
            nn.Linear(
                previous,
                1,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x).squeeze(1)


def sigmoid(
    logits: np.ndarray,
) -> np.ndarray:
    logits = np.clip(
        logits,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-logits)
    )


def predict_mlp_logits(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    outputs = []

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

            outputs.append(
                model(xb)
                .detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(outputs)


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


def make_unweighted_loader(
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


def train_source_mlp(
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

    loader = make_unweighted_loader(
        X_train,
        y_train,
        args.batch_size,
        True,
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

    for epoch in range(
        args.max_epochs
    ):
        model.train()

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

        val_probs = sigmoid(
            predict_mlp_logits(
                model,
                X_val,
                device,
            )
        )

        val_auroc = float(
            roc_auc_score(
                y_val,
                val_probs,
            )
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
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return model, {
        "best_epoch": int(best_epoch),
        "best_val_auroc": float(
            best_val_auroc
        ),
    }


def weighted_val_auroc(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
) -> float:
    probs = sigmoid(
        predict_mlp_logits(
            model,
            X,
            device,
        )
    )

    return float(
        roc_auc_score(
            y,
            probs,
            sample_weight=weights,
        )
    )


def adapt_mlp_on_source_weights(
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
) -> tuple[MLP, dict[str, Any]]:
    set_seed(seed)

    model = copy.deepcopy(
        source_model
    ).to(device)

    loader = make_weighted_loader(
        X_train,
        y_train,
        train_weights,
        args.batch_size,
        True,
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

    best_val = weighted_val_auroc(
        model,
        X_val,
        y_val,
        val_weights,
        device,
    )

    best_epoch = 0
    no_improve = 0

    for epoch in range(
        args.adapt_epochs
    ):
        model.train()

        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            losses = criterion(
                logits,
                yb,
            )

            loss = (
                losses * wb
            ).sum() / wb.sum().clamp_min(
                1e-8
            )

            loss.backward()
            optimizer.step()

        score = weighted_val_auroc(
            model,
            X_val,
            y_val,
            val_weights,
            device,
        )

        if score > best_val:
            best_val = score
            best_epoch = epoch + 1
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
        "best_epoch": int(best_epoch),
        "best_weighted_val_auroc": float(
            best_val
        ),
    }


def split_target_adaptation(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=seed,
    )


def adapt_mlp_target_labels(
    source_model: MLP,
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_val: np.ndarray,
    y_target_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    replay_X: np.ndarray | None = None,
    replay_y: np.ndarray | None = None,
) -> tuple[MLP, dict[str, Any]]:
    set_seed(seed)

    model = copy.deepcopy(
        source_model
    ).to(device)

    if (
        replay_X is None
        or replay_y is None
    ):
        X_train = X_target_train
        y_train = y_target_train
        replay_size = 0
    else:
        X_train = np.vstack(
            [
                X_target_train,
                replay_X,
            ]
        ).astype(np.float32)

        y_train = np.concatenate(
            [
                y_target_train,
                replay_y,
            ]
        ).astype(np.int8)

        replay_size = int(
            len(replay_y)
        )

    loader = make_unweighted_loader(
        X_train,
        y_train,
        args.batch_size,
        True,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.target_lr,
        weight_decay=args.weight_decay,
    )

    initial_probs = sigmoid(
        predict_mlp_logits(
            model,
            X_target_val,
            device,
        )
    )

    best_val = float(
        roc_auc_score(
            y_target_val,
            initial_probs,
        )
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    best_epoch = 0
    no_improve = 0

    for epoch in range(
        args.target_epochs
    ):
        model.train()

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

        val_probs = sigmoid(
            predict_mlp_logits(
                model,
                X_target_val,
                device,
            )
        )

        val_auroc = float(
            roc_auc_score(
                y_target_val,
                val_probs,
            )
        )

        if val_auroc > best_val:
            best_val = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.target_patience:
            break

    model.load_state_dict(
        best_state
    )

    return model, {
        "best_epoch": int(best_epoch),
        "best_target_val_auroc": float(
            best_val
        ),
        "target_train_size": int(
            len(y_target_train)
        ),
        "replay_size": replay_size,
        "replay_fraction_of_training": float(
            replay_size
            / (
                len(y_target_train)
                + replay_size
            )
            if (
                len(y_target_train)
                + replay_size
            ) > 0
            else 0.0
        ),
    }


def train_target_only_mlp(
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_val: np.ndarray,
    y_target_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[MLP, dict[str, Any]]:
    set_seed(seed)

    model = MLP(
        input_dim=X_target_train.shape[1],
        hidden_dims=tuple(
            args.hidden_dims
        ),
        dropout=args.dropout,
    ).to(device)

    loader = make_unweighted_loader(
        X_target_train,
        y_target_train,
        args.batch_size,
        True,
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(
        args.max_epochs
    ):
        model.train()

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

        val_probs = sigmoid(
            predict_mlp_logits(
                model,
                X_target_val,
                device,
            )
        )

        val_auroc = float(
            roc_auc_score(
                y_target_val,
                val_probs,
            )
        )

        if val_auroc > best_val:
            best_val = val_auroc
            best_epoch = epoch + 1
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
        "best_epoch": int(best_epoch),
        "best_target_val_auroc": float(
            best_val
        ),
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

    conv = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    return clf, {
        "converged": (
            len(conv) == 0
        ),
        "iterations": int(
            np.max(clf.n_iter_)
        ),
        "warnings": sorted(
            set(conv)
        ),
    }


# ============================================================
# Classifier prediction
# ============================================================
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
        ).astype(float)

    raise ValueError(
        classifier_name
    )


def classifier_probs(
    classifier_name: str,
    classifier: Any,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    return sigmoid(
        classifier_logits(
            classifier_name,
            classifier,
            X,
            device,
        )
    )


# ============================================================
# Prior correction
# ============================================================
def prior_correct_probabilities(
    probs: np.ndarray,
    source_prior: float,
    target_prior: float,
) -> np.ndarray:
    # Preserve the mathematical identity case EXACTLY.
    #
    # The controlled concept-shift benchmark deliberately fixes
    # source_prior == target_prior == 0.5. Clipping probabilities before
    # checking this case can perturb extremely confident predictions by
    # ~1e-7, which would make the identity guardrail fail even though the
    # prior-correction factor is exactly one.
    original_probs = np.asarray(
        probs,
        dtype=float,
    )

    source_prior = float(
        source_prior
    )
    target_prior = float(
        target_prior
    )

    if not (
        0.0 < source_prior < 1.0
        and 0.0 < target_prior < 1.0
    ):
        raise ValueError(
            "Priors must lie strictly between 0 and 1."
        )

    if np.isclose(
        source_prior,
        target_prior,
        rtol=0.0,
        atol=1e-15,
    ):
        return original_probs.copy()

    probs = np.clip(
        original_probs,
        EPS,
        1.0 - EPS,
    )

    source_odds = (
        source_prior
        / (1.0 - source_prior)
    )

    target_odds = (
        target_prior
        / (1.0 - target_prior)
    )

    likelihood_ratio = (
        probs
        / (1.0 - probs)
    ) / source_odds

    corrected_odds = (
        likelihood_ratio
        * target_odds
    )

    corrected = (
        corrected_odds
        / (1.0 + corrected_odds)
    )

    return np.clip(
        corrected,
        EPS,
        1.0 - EPS,
    )


# ============================================================
# Estimated unconditional IW
# ============================================================
def effective_sample_size(
    weights: np.ndarray,
) -> float:
    w = np.asarray(
        weights,
        dtype=float,
    )

    denom = float(
        np.sum(
            w ** 2
        )
    )

    if denom <= 0:
        return 0.0

    return float(
        (
            np.sum(w) ** 2
        )
        / denom
    )


def fit_domain_ratio_model(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
) -> tuple[
    StandardScaler,
    LogisticRegression,
    dict[str, Any],
]:
    n = min(
        len(X_source),
        len(X_target),
    )

    if n < 100:
        raise ValueError(
            "Too few samples for domain classifier."
        )

    rng = np.random.default_rng(seed)

    src_idx = rng.choice(
        len(X_source),
        size=n,
        replace=False,
    )

    tgt_idx = rng.choice(
        len(X_target),
        size=n,
        replace=False,
    )

    X_domain = np.vstack(
        [
            X_source[src_idx],
            X_target[tgt_idx],
        ]
    )

    y_domain = np.concatenate(
        [
            np.zeros(
                n,
                dtype=np.int8,
            ),
            np.ones(
                n,
                dtype=np.int8,
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

    X_train_s = scaler.fit_transform(
        X_train
    )

    X_test_s = scaler.transform(
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
            X_train_s,
            y_train,
        )

    conv = [
        str(w.message)
        for w in caught
        if issubclass(
            w.category,
            ConvergenceWarning,
        )
    ]

    test_probs = clf.predict_proba(
        X_test_s
    )[:, 1]

    return scaler, clf, {
        "domain_auroc": float(
            roc_auc_score(
                y_test,
                test_probs,
            )
        ),
        "converged": (
            len(conv) == 0
        ),
        "iterations": int(
            np.max(clf.n_iter_)
        ),
        "warnings": sorted(
            set(conv)
        ),
    }


def domain_ratio_weights(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X: np.ndarray,
    clip: float,
) -> np.ndarray:
    p_target = clf.predict_proba(
        scaler.transform(X)
    )[:, 1]

    p_target = np.clip(
        p_target,
        EPS,
        1.0 - EPS,
    )

    odds = (
        p_target
        / (1.0 - p_target)
    )

    odds = np.clip(
        odds,
        0.0,
        clip,
    )

    return odds


def estimate_unconditional_iw(
    X_source_train: np.ndarray,
    X_source_val: np.ndarray,
    X_target_adapt: np.ndarray,
    seed: int,
    clip: float,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    scaler, clf, info = (
        fit_domain_ratio_model(
            X_source_train,
            X_target_adapt,
            seed,
        )
    )

    raw_train = domain_ratio_weights(
        scaler,
        clf,
        X_source_train,
        clip,
    )

    raw_val = domain_ratio_weights(
        scaler,
        clf,
        X_source_val,
        clip,
    )

    mean_train = float(
        np.mean(raw_train)
    )

    if mean_train <= 0:
        raise RuntimeError(
            "Estimated IW mean is non-positive."
        )

    train_weights = (
        raw_train / mean_train
    )

    val_weights = (
        raw_val / mean_train
    )

    info = dict(info)

    info.update(
        {
            "train_weight_ess": effective_sample_size(
                train_weights
            ),
            "train_weight_ess_ratio": float(
                effective_sample_size(
                    train_weights
                )
                / len(train_weights)
            ),
            "train_weight_mean": float(
                np.mean(train_weights)
            ),
            "train_weight_std": float(
                np.std(train_weights)
            ),
            "train_weight_max": float(
                np.max(train_weights)
            ),
        }
    )

    return {
        "train": train_weights,
        "val": val_weights,
    }, info


# ============================================================
# Marginal X diagnostics
# ============================================================
def marginal_domain_diagnostics(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    _, _, info = fit_domain_ratio_model(
        X_source,
        X_target,
        seed,
    )

    source_mean = np.mean(
        X_source,
        axis=0,
    )

    target_mean = np.mean(
        X_target,
        axis=0,
    )

    pooled_std = np.std(
        np.vstack(
            [
                X_source,
                X_target,
            ]
        ),
        axis=0,
    )

    pooled_std = np.maximum(
        pooled_std,
        EPS,
    )

    smd = (
        target_mean
        - source_mean
    ) / pooled_std

    return {
        "marginal_domain_auroc": info[
            "domain_auroc"
        ],
        "marginal_domain_classifier_converged": info[
            "converged"
        ],
        "marginal_domain_classifier_iterations": info[
            "iterations"
        ],
        "mean_absolute_feature_smd": float(
            np.mean(
                np.abs(smd)
            )
        ),
        "max_absolute_feature_smd": float(
            np.max(
                np.abs(smd)
            )
        ),
    }


# ============================================================
# Metrics
# ============================================================
def expected_calibration_error(
    y: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = np.asarray(
        y,
        dtype=int,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = 0.0

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
            np.mean(
                probs[mask]
            )
        )

        observed = float(
            np.mean(
                y[mask]
            )
        )

        total += (
            count / len(y)
        ) * abs(
            confidence - observed
        )

    return float(total)


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
        "fpr": float(fpr),
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
    }


# ============================================================
# Replay selection
# ============================================================
def select_random_replay(
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    replay_per_class: int,
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    rng = np.random.default_rng(seed)

    positions = []

    for cls in [0, 1]:
        cls_positions = np.flatnonzero(
            y_source_train == cls
        )

        if len(cls_positions) < replay_per_class:
            raise ValueError(
                f"Class {cls} has too few source-train examples for replay."
            )

        selected = rng.choice(
            cls_positions,
            size=replay_per_class,
            replace=False,
        )

        positions.append(selected)

    selected_positions = np.concatenate(
        positions
    )

    rng.shuffle(
        selected_positions
    )

    return (
        X_source_train[
            selected_positions
        ],
        y_source_train[
            selected_positions
        ],
        {
            "replay_per_class": int(
                replay_per_class
            ),
            "replay_total": int(
                len(selected_positions)
            ),
            "replay_attack_prior": float(
                np.mean(
                    y_source_train[
                        selected_positions
                    ]
                )
            ),
        },
    )


# ============================================================
# Aggregation / paired statistics
# ============================================================
def mean_std(
    values: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    mean = float(
        np.mean(values)
    )

    std = float(
        np.std(
            values,
            ddof=1,
        )
        if len(values) > 1
        else 0.0
    )

    return mean, std


def aggregate_results(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "concept_severity",
        "supervision_regime",
    ]

    metric_cols = [
        "target_attack_prior",
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
        "source_after_auroc",
        "source_auroc_drop",
        "target_auroc_gain_vs_source_only",
        "average_source_target_auroc",
        "worst_source_target_auroc",
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
            mean, std = mean_std(
                group[metric].to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "concept_severity",
            "method",
        ]
    )


def aggregate_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [
        "source_attack_prior",
        "target_attack_prior",
        "attack_prior_change",
        "actual_adapt_label_flip_rate",
        "actual_test_label_flip_rate",
        "marginal_domain_auroc",
        "mean_absolute_feature_smd",
        "max_absolute_feature_smd",
        "estimated_iw_ess_ratio",
        "estimated_iw_weight_std",
        "estimated_iw_weight_max",
        "adapt_flipped_margin_mean",
        "adapt_unflipped_margin_mean",
        "test_flipped_margin_mean",
        "test_unflipped_margin_mean",
    ]

    rows = []

    for (
        dataset,
        severity,
    ), group in diagnostics.groupby(
        [
            "dataset",
            "concept_severity",
        ]
    ):
        row = {
            "dataset": dataset,
            "concept_severity": severity,
            "n_seeds": int(
                group["seed"].nunique()
            ),
            "all_marginal_domain_classifiers_converged": bool(
                group[
                    "marginal_domain_classifier_converged"
                ].all()
            ),
            "all_density_ratio_classifiers_converged": bool(
                group[
                    "estimated_iw_domain_classifier_converged"
                ].all()
            ),
        }

        for metric in metric_cols:
            mean, std = mean_std(
                group[metric].to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        "concept_severity"
    )


def paired_ci(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(
        np.mean(values)
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
            / math.sqrt(n)
        )

        low = mean - half
        high = mean + half
    else:
        low = mean
        high = mean

    return {
        "mean": mean,
        "std": std,
        "ci95_low": low,
        "ci95_high": high,
    }


def build_method_vs_source_paired(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "concept_severity",
    ]

    metrics = [
        "auroc",
        "auprc",
        "f1",
        "brier",
        "log_loss",
        "ece",
        "source_after_auroc",
        "source_auroc_drop",
        "average_source_target_auroc",
        "worst_source_target_auroc",
    ]

    source = (
        per_seed[
            per_seed["method"]
            == "source_only"
        ][
            keys + metrics
        ]
        .copy()
        .rename(
            columns={
                m: f"source_{m}"
                for m in metrics
            }
        )
    )

    adapted = per_seed[
        per_seed["method"]
        != "source_only"
    ].copy()

    merged = adapted.merge(
        source,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    for metric in [
        "auroc",
        "auprc",
        "f1",
        "source_after_auroc",
        "average_source_target_auroc",
        "worst_source_target_auroc",
    ]:
        merged[
            f"{metric}_improvement_vs_source"
        ] = (
            merged[metric]
            - merged[
                f"source_{metric}"
            ]
        )

    for metric in [
        "brier",
        "log_loss",
        "ece",
        "source_auroc_drop",
    ]:
        merged[
            f"{metric}_improvement_vs_source"
        ] = (
            merged[
                f"source_{metric}"
            ]
            - merged[metric]
        )

    return merged


def summarize_method_vs_source(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "concept_severity",
        "supervision_regime",
    ]

    diff_cols = [
        "auroc_improvement_vs_source",
        "auprc_improvement_vs_source",
        "f1_improvement_vs_source",
        "brier_improvement_vs_source",
        "log_loss_improvement_vs_source",
        "ece_improvement_vs_source",
        "source_after_auroc_improvement_vs_source",
        "source_auroc_drop_improvement_vs_source",
        "average_source_target_auroc_improvement_vs_source",
        "worst_source_target_auroc_improvement_vs_source",
    ]

    rows = []

    for keys, group in paired.groupby(
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

        for col in diff_cols:
            stats = paired_ci(
                group[col].to_numpy(
                    dtype=float
                )
            )

            for stat_name, value in stats.items():
                row[
                    f"{col}_{stat_name}"
                ] = value

        rows.append(row)

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "classifier",
            "concept_severity",
            "method",
        ]
    )


def build_mlp_iw_vs_uniform(
    per_seed: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "concept_severity",
    ]

    metrics = [
        "auroc",
        "auprc",
        "f1",
        "brier",
        "log_loss",
        "ece",
    ]

    uniform = (
        per_seed[
            (
                per_seed["classifier"]
                == "mlp"
            )
            & (
                per_seed["method"]
                == "uniform_extra_training"
            )
        ][
            keys + metrics
        ]
        .copy()
        .rename(
            columns={
                m: f"uniform_{m}"
                for m in metrics
            }
        )
    )

    compared = per_seed[
        (
            per_seed["classifier"]
            == "mlp"
        )
        & (
            per_seed["method"]
            == "unconditional_estimated_iw"
        )
    ].copy()

    merged = compared.merge(
        uniform,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    for metric in [
        "auroc",
        "auprc",
        "f1",
    ]:
        merged[
            f"{metric}_improvement_vs_uniform"
        ] = (
            merged[metric]
            - merged[
                f"uniform_{metric}"
            ]
        )

    for metric in [
        "brier",
        "log_loss",
        "ece",
    ]:
        merged[
            f"{metric}_improvement_vs_uniform"
        ] = (
            merged[
                f"uniform_{metric}"
            ]
            - merged[metric]
        )

    rows = []

    diff_cols = [
        "auroc_improvement_vs_uniform",
        "auprc_improvement_vs_uniform",
        "f1_improvement_vs_uniform",
        "brier_improvement_vs_uniform",
        "log_loss_improvement_vs_uniform",
        "ece_improvement_vs_uniform",
    ]

    for (
        dataset,
        severity,
    ), group in merged.groupby(
        [
            "dataset",
            "concept_severity",
        ]
    ):
        row = {
            "dataset": dataset,
            "classifier": "mlp",
            "method": "unconditional_estimated_iw",
            "concept_severity": severity,
            "n_seeds": int(
                group["seed"].nunique()
            ),
        }

        for col in diff_cols:
            stats = paired_ci(
                group[col].to_numpy(
                    dtype=float
                )
            )

            for stat_name, value in stats.items():
                row[
                    f"{col}_{stat_name}"
                ] = value

        rows.append(row)

    return (
        merged,
        pd.DataFrame(
            rows
        ).sort_values(
            "concept_severity"
        ),
    )


def build_ft_vs_replay(
    per_seed: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "concept_severity",
    ]

    metrics = [
        "auroc",
        "auprc",
        "f1",
        "source_after_auroc",
        "source_auroc_drop",
        "average_source_target_auroc",
        "worst_source_target_auroc",
    ]

    ft = (
        per_seed[
            (
                per_seed["classifier"]
                == "mlp"
            )
            & (
                per_seed["method"]
                == "naive_target_finetuning"
            )
        ][
            keys + metrics
        ]
        .copy()
        .rename(
            columns={
                m: f"ft_{m}"
                for m in metrics
            }
        )
    )

    replay = per_seed[
        (
            per_seed["classifier"]
            == "mlp"
        )
        & (
            per_seed["method"]
            == "finetuning_plus_replay"
        )
    ].copy()

    merged = replay.merge(
        ft,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Positive = replay is better.
    for metric in [
        "auroc",
        "auprc",
        "f1",
        "source_after_auroc",
        "average_source_target_auroc",
        "worst_source_target_auroc",
    ]:
        merged[
            f"{metric}_improvement_replay_vs_ft"
        ] = (
            merged[metric]
            - merged[
                f"ft_{metric}"
            ]
        )

    # Lower source drop is better.
    merged[
        "source_auroc_drop_improvement_replay_vs_ft"
    ] = (
        merged[
            "ft_source_auroc_drop"
        ]
        - merged[
            "source_auroc_drop"
        ]
    )

    diff_cols = [
        "auroc_improvement_replay_vs_ft",
        "auprc_improvement_replay_vs_ft",
        "f1_improvement_replay_vs_ft",
        "source_after_auroc_improvement_replay_vs_ft",
        "source_auroc_drop_improvement_replay_vs_ft",
        "average_source_target_auroc_improvement_replay_vs_ft",
        "worst_source_target_auroc_improvement_replay_vs_ft",
    ]

    rows = []

    for (
        dataset,
        severity,
    ), group in merged.groupby(
        [
            "dataset",
            "concept_severity",
        ]
    ):
        row = {
            "dataset": dataset,
            "classifier": "mlp",
            "comparison": "finetuning_plus_replay_minus_naive_target_finetuning",
            "concept_severity": severity,
            "n_seeds": int(
                group["seed"].nunique()
            ),
        }

        for col in diff_cols:
            stats = paired_ci(
                group[col].to_numpy(
                    dtype=float
                )
            )

            for stat_name, value in stats.items():
                row[
                    f"{col}_{stat_name}"
                ] = value

        rows.append(row)

    return (
        merged,
        pd.DataFrame(
            rows
        ).sort_values(
            "concept_severity"
        ),
    )


# ============================================================
# Plotting
# ============================================================
def plot_target_auroc(
    summary: pd.DataFrame,
    classifier_name: str,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.8, 5.4)
    )

    subset = summary[
        summary["classifier"]
        == classifier_name
    ]

    for method in (
        subset["method"]
        .drop_duplicates()
        .tolist()
    ):
        m = subset[
            subset["method"]
            == method
        ].sort_values(
            "concept_severity"
        )

        plt.errorbar(
            m["concept_severity"],
            m["auroc_mean"],
            yerr=m["auroc_std"],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel(
        "Boundary-localized concept severity"
    )
    plt.ylabel(
        "Target AUROC"
    )
    plt.legend()
    savefig(output_path)


def plot_stability_plasticity(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(
        figsize=(8.8, 5.4)
    )

    subset = summary[
        (
            summary["classifier"]
            == "mlp"
        )
        & (
            summary["method"].isin(
                [
                    "naive_target_finetuning",
                    "finetuning_plus_replay",
                ]
            )
        )
    ]

    for method in [
        "naive_target_finetuning",
        "finetuning_plus_replay",
    ]:
        m = subset[
            subset["method"]
            == method
        ].sort_values(
            "concept_severity"
        )

        plt.errorbar(
            m["concept_severity"],
            m["source_auroc_drop_mean"],
            yerr=m["source_auroc_drop_std"],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel(
        "Boundary-localized concept severity"
    )
    plt.ylabel(
        "Source AUROC drop after adaptation"
    )
    plt.legend()
    savefig(output_path)


# ============================================================
# One seed
# ============================================================
def run_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    design: ConceptDesign,
    severities: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    print(
        "\n"
        + "=" * 120
    )

    print(
        "CONTROLLED CONCEPT SHIFT"
        f" | dataset={args.dataset}"
        f" | seed={seed}"
    )

    print(
        "=" * 120
    )

    split = build_seed_split(
        X=X,
        y=y,
        design=design,
        seed=seed,
        source_train_per_class=args.source_train_per_class,
        source_val_per_class=args.source_val_per_class,
        source_test_per_class=args.source_test_per_class,
        target_adapt_per_class=args.target_adapt_per_class,
        target_test_per_class=args.target_test_per_class,
    )

    idx_source_train = split[
        "idx_source_train"
    ]
    idx_source_val = split[
        "idx_source_val"
    ]
    idx_source_test = split[
        "idx_source_test"
    ]
    idx_target_adapt = split[
        "idx_target_adapt"
    ]
    idx_target_test = split[
        "idx_target_test"
    ]

    X_source_train_raw = X[
        idx_source_train
    ]
    X_source_val_raw = X[
        idx_source_val
    ]
    X_source_test_raw = X[
        idx_source_test
    ]

    X_target_adapt_raw = X[
        idx_target_adapt
    ]
    X_target_test_raw = X[
        idx_target_test
    ]

    y_source_train = y[
        idx_source_train
    ]
    y_source_val = y[
        idx_source_val
    ]
    y_source_test = y[
        idx_source_test
    ]

    y_target_adapt_original = y[
        idx_target_adapt
    ]
    y_target_test_original = y[
        idx_target_test
    ]

    source_prior = float(
        np.mean(y_source_train)
    )

    if not np.isclose(
        source_prior,
        0.5,
    ):
        raise RuntimeError(
            "Source prior is not 0.5."
        )

    # Source-only model preprocessing.
    model_scaler = StandardScaler()

    X_source_train = (
        model_scaler
        .fit_transform(
            X_source_train_raw
        )
        .astype(np.float32)
    )

    X_source_val = (
        model_scaler
        .transform(
            X_source_val_raw
        )
        .astype(np.float32)
    )

    X_source_test = (
        model_scaler
        .transform(
            X_source_test_raw
        )
        .astype(np.float32)
    )

    X_target_adapt = (
        model_scaler
        .transform(
            X_target_adapt_raw
        )
        .astype(np.float32)
    )

    X_target_test = (
        model_scaler
        .transform(
            X_target_test_raw
        )
        .astype(np.float32)
    )

    # X-only diagnostics are identical across severities within this seed.
    x_diagnostics = marginal_domain_diagnostics(
        X_source_test,
        X_target_test,
        seed=seed + 101,
    )

    iw_sets, iw_info = estimate_unconditional_iw(
        X_source_train=X_source_train,
        X_source_val=X_source_val,
        X_target_adapt=X_target_adapt,
        seed=seed + 201,
        clip=args.iw_clip,
    )

    # Source models.
    source_mlp, source_mlp_info = train_source_mlp(
        X_source_train,
        y_source_train,
        X_source_val,
        y_source_val,
        args,
        device,
        seed=seed + 301,
    )

    source_logistic, source_logistic_info = fit_logistic(
        X_source_train,
        y_source_train,
        seed=seed + 302,
    )

    source_models = {
        "mlp": source_mlp,
        "logistic_regression": source_logistic,
    }

    # Source concept reference metrics before any adaptation.
    source_before_metrics = {}

    for classifier_name, classifier in source_models.items():
        source_probs = classifier_probs(
            classifier_name,
            classifier,
            X_source_test,
            device,
        )

        source_before_metrics[
            classifier_name
        ] = probability_metrics(
            y_source_test,
            source_probs,
        )

    source_rows = [
        {
            "seed": seed,
            "dataset": args.dataset,
            "classifier": "mlp",
            "source_attack_prior": source_prior,
            "source_train_size": int(
                len(y_source_train)
            ),
            "source_val_size": int(
                len(y_source_val)
            ),
            "source_test_size": int(
                len(y_source_test)
            ),
            "source_test_auroc": source_before_metrics[
                "mlp"
            ]["auroc"],
            "source_test_auprc": source_before_metrics[
                "mlp"
            ]["auprc"],
            "source_test_f1": source_before_metrics[
                "mlp"
            ]["f1"],
            "best_epoch": source_mlp_info[
                "best_epoch"
            ],
            "best_val_auroc": source_mlp_info[
                "best_val_auroc"
            ],
            "training_converged": True,
            "training_iterations": np.nan,
        },
        {
            "seed": seed,
            "dataset": args.dataset,
            "classifier": "logistic_regression",
            "source_attack_prior": source_prior,
            "source_train_size": int(
                len(y_source_train)
            ),
            "source_val_size": int(
                len(y_source_val)
            ),
            "source_test_size": int(
                len(y_source_test)
            ),
            "source_test_auroc": source_before_metrics[
                "logistic_regression"
            ]["auroc"],
            "source_test_auprc": source_before_metrics[
                "logistic_regression"
            ]["auprc"],
            "source_test_f1": source_before_metrics[
                "logistic_regression"
            ]["f1"],
            "best_epoch": np.nan,
            "best_val_auroc": np.nan,
            "training_converged": source_logistic_info[
                "converged"
            ],
            "training_iterations": source_logistic_info[
                "iterations"
            ],
        },
    ]

    # Fixed replay buffer for every severity in this seed.
    replay_X, replay_y, replay_info = select_random_replay(
        X_source_train,
        y_source_train,
        args.replay_per_class,
        seed=seed + 401,
    )

    # Source-weighted MLP controls can be fit once per severity with a matched
    # seed; IW weights themselves are the same across severities because X is.
    results = []
    diagnostic_rows = []

    for condition_index, severity in enumerate(
        severities
    ):
        severity = float(
            severity
        )

        (
            y_target_adapt,
            adapt_shift_info,
        ) = apply_boundary_local_concept_shift(
            X_target_adapt_raw,
            y_target_adapt_original,
            design,
            severity,
        )

        (
            y_target_test,
            test_shift_info,
        ) = apply_boundary_local_concept_shift(
            X_target_test_raw,
            y_target_test_original,
            design,
            severity,
        )

        target_prior = float(
            np.mean(
                y_target_test
            )
        )

        attack_prior_change = (
            target_prior
            - source_prior
        )

        if not np.isclose(
            attack_prior_change,
            0.0,
        ):
            raise RuntimeError(
                "Target prior changed under concept-shift construction."
            )

        # Same target X across all severities.
        # Generate one deterministic target train/val membership per severity
        # using original X membership and shifted labels for stratification.
        (
            X_target_train,
            X_target_val,
            y_target_train,
            y_target_val,
        ) = split_target_adaptation(
            X_target_adapt,
            y_target_adapt,
            seed=(
                seed
                + condition_index * 1000
                + 501
            ),
        )

        # ----------------------------------------------------
        # MLP mismatched IW vs matched uniform control.
        # ----------------------------------------------------
        matched_source_adapt_seed = (
            seed
            + condition_index * 1000
            + 601
        )

        uniform_mlp, _ = adapt_mlp_on_source_weights(
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
            seed=matched_source_adapt_seed,
        )

        iw_mlp, _ = adapt_mlp_on_source_weights(
            source_model=source_mlp,
            X_train=X_source_train,
            y_train=y_source_train,
            train_weights=iw_sets[
                "train"
            ],
            X_val=X_source_val,
            y_val=y_source_val,
            val_weights=iw_sets[
                "val"
            ],
            args=args,
            device=device,
            seed=matched_source_adapt_seed,
        )

        # ----------------------------------------------------
        # Labelled target adaptation.
        # Same target split for naive FT and FT+replay.
        # ----------------------------------------------------
        target_adapt_seed = (
            seed
            + condition_index * 1000
            + 701
        )

        naive_ft_mlp, naive_ft_info = adapt_mlp_target_labels(
            source_model=source_mlp,
            X_target_train=X_target_train,
            y_target_train=y_target_train,
            X_target_val=X_target_val,
            y_target_val=y_target_val,
            args=args,
            device=device,
            seed=target_adapt_seed,
            replay_X=None,
            replay_y=None,
        )

        replay_ft_mlp, replay_ft_info = adapt_mlp_target_labels(
            source_model=source_mlp,
            X_target_train=X_target_train,
            y_target_train=y_target_train,
            X_target_val=X_target_val,
            y_target_val=y_target_val,
            args=args,
            device=device,
            seed=target_adapt_seed,
            replay_X=replay_X,
            replay_y=replay_y,
        )

        target_only_mlp, target_only_mlp_info = train_target_only_mlp(
            X_target_train,
            y_target_train,
            X_target_val,
            y_target_val,
            args,
            device,
            seed=(
                seed
                + condition_index * 1000
                + 702
            ),
        )

        # ----------------------------------------------------
        # Logistic mismatched IW and target-only reference.
        # ----------------------------------------------------
        iw_logistic, iw_logistic_info = fit_logistic(
            X_source_train,
            y_source_train,
            seed=(
                seed
                + condition_index * 1000
                + 801
            ),
            sample_weight=iw_sets[
                "train"
            ],
        )

        target_only_logistic, target_only_logistic_info = fit_logistic(
            X_target_train,
            y_target_train,
            seed=(
                seed
                + condition_index * 1000
                + 802
            ),
        )

        # ----------------------------------------------------
        # Evaluate source-only predictions and identity prior correction.
        # ----------------------------------------------------
        for classifier_name, source_classifier in source_models.items():
            source_target_probs = classifier_probs(
                classifier_name,
                source_classifier,
                X_target_test,
                device,
            )

            corrected_probs = prior_correct_probabilities(
                source_target_probs,
                source_prior=source_prior,
                target_prior=target_prior,
            )

            max_prior_identity_error = float(
                np.max(
                    np.abs(
                        corrected_probs
                        - source_target_probs
                    )
                )
            )

            if max_prior_identity_error > 1e-10:
                raise RuntimeError(
                    "Prior correction was not identity despite equal priors."
                )

            for method, probs, source_after_auroc, supervision in [
                (
                    "source_only",
                    source_target_probs,
                    source_before_metrics[
                        classifier_name
                    ]["auroc"],
                    "no_target_supervision",
                ),
                (
                    "oracle_prior_correction",
                    corrected_probs,
                    source_before_metrics[
                        classifier_name
                    ]["auroc"],
                    "oracle_prior_only",
                ),
            ]:
                metrics = probability_metrics(
                    y_target_test,
                    probs,
                )

                source_before_auroc = source_before_metrics[
                    classifier_name
                ]["auroc"]

                source_drop = (
                    source_before_auroc
                    - source_after_auroc
                )

                results.append(
                    {
                        "seed": seed,
                        "dataset": args.dataset,
                        "classifier": classifier_name,
                        "method": method,
                        "concept_severity": severity,
                        "supervision_regime": supervision,
                        "target_attack_prior": target_prior,
                        "source_before_auroc": source_before_auroc,
                        "source_after_auroc": source_after_auroc,
                        "source_auroc_drop": source_drop,
                        "target_auroc_gain_vs_source_only": 0.0,
                        "average_source_target_auroc": float(
                            (
                                source_after_auroc
                                + metrics["auroc"]
                            )
                            / 2.0
                        ),
                        "worst_source_target_auroc": float(
                            min(
                                source_after_auroc,
                                metrics["auroc"],
                            )
                        ),
                        "replay_size": 0,
                        "replay_fraction_of_training": 0.0,
                        "prior_identity_max_abs_error": max_prior_identity_error,
                        **metrics,
                    }
                )

        # ----------------------------------------------------
        # Evaluate trained models.
        # ----------------------------------------------------
        trained_models = {
            "mlp": {
                "uniform_extra_training": (
                    uniform_mlp,
                    "no_target_supervision",
                    0,
                    0.0,
                ),
                "unconditional_estimated_iw": (
                    iw_mlp,
                    "unlabelled_target_X",
                    0,
                    0.0,
                ),
                "naive_target_finetuning": (
                    naive_ft_mlp,
                    "labelled_target",
                    0,
                    0.0,
                ),
                "finetuning_plus_replay": (
                    replay_ft_mlp,
                    "labelled_target_plus_source_replay",
                    replay_info[
                        "replay_total"
                    ],
                    replay_ft_info[
                        "replay_fraction_of_training"
                    ],
                ),
                "target_only_reference": (
                    target_only_mlp,
                    "labelled_target_reference",
                    0,
                    0.0,
                ),
            },
            "logistic_regression": {
                "unconditional_estimated_iw": (
                    iw_logistic,
                    "unlabelled_target_X",
                    0,
                    0.0,
                ),
                "target_only_reference": (
                    target_only_logistic,
                    "labelled_target_reference",
                    0,
                    0.0,
                ),
            },
        }

        # Need source-only target AUROC for gain field.
        source_only_target_auroc = {}

        for classifier_name, source_classifier in source_models.items():
            source_only_target_auroc[
                classifier_name
            ] = probability_metrics(
                y_target_test,
                classifier_probs(
                    classifier_name,
                    source_classifier,
                    X_target_test,
                    device,
                ),
            )["auroc"]

        for classifier_name in CLASSIFIERS:
            for (
                method,
                (
                    classifier,
                    supervision,
                    replay_size,
                    replay_fraction,
                ),
            ) in trained_models[
                classifier_name
            ].items():
                target_probs = classifier_probs(
                    classifier_name,
                    classifier,
                    X_target_test,
                    device,
                )

                source_probs_after = classifier_probs(
                    classifier_name,
                    classifier,
                    X_source_test,
                    device,
                )

                target_metrics = probability_metrics(
                    y_target_test,
                    target_probs,
                )

                source_metrics_after = probability_metrics(
                    y_source_test,
                    source_probs_after,
                )

                source_before_auroc = source_before_metrics[
                    classifier_name
                ]["auroc"]

                source_after_auroc = source_metrics_after[
                    "auroc"
                ]

                source_drop = (
                    source_before_auroc
                    - source_after_auroc
                )

                target_gain = (
                    target_metrics["auroc"]
                    - source_only_target_auroc[
                        classifier_name
                    ]
                )

                results.append(
                    {
                        "seed": seed,
                        "dataset": args.dataset,
                        "classifier": classifier_name,
                        "method": method,
                        "concept_severity": severity,
                        "supervision_regime": supervision,
                        "target_attack_prior": target_prior,
                        "source_before_auroc": source_before_auroc,
                        "source_after_auroc": source_after_auroc,
                        "source_auroc_drop": source_drop,
                        "target_auroc_gain_vs_source_only": target_gain,
                        "average_source_target_auroc": float(
                            (
                                source_after_auroc
                                + target_metrics[
                                    "auroc"
                                ]
                            )
                            / 2.0
                        ),
                        "worst_source_target_auroc": float(
                            min(
                                source_after_auroc,
                                target_metrics[
                                    "auroc"
                                ],
                            )
                        ),
                        "replay_size": int(
                            replay_size
                        ),
                        "replay_fraction_of_training": float(
                            replay_fraction
                        ),
                        "prior_identity_max_abs_error": 0.0,
                        **target_metrics,
                    }
                )

        diagnostic_rows.append(
            {
                "seed": seed,
                "dataset": args.dataset,
                "concept_severity": severity,
                "source_attack_prior": source_prior,
                "target_attack_prior": target_prior,
                "attack_prior_change": attack_prior_change,
                "actual_adapt_label_flip_rate": adapt_shift_info[
                    "actual_label_flip_rate"
                ],
                "actual_test_label_flip_rate": test_shift_info[
                    "actual_label_flip_rate"
                ],
                "adapt_benign_to_attack_flips": adapt_shift_info[
                    "benign_to_attack_flips"
                ],
                "adapt_attack_to_benign_flips": adapt_shift_info[
                    "attack_to_benign_flips"
                ],
                "test_benign_to_attack_flips": test_shift_info[
                    "benign_to_attack_flips"
                ],
                "test_attack_to_benign_flips": test_shift_info[
                    "attack_to_benign_flips"
                ],
                "adapt_flipped_margin_mean": adapt_shift_info[
                    "flipped_margin_mean"
                ],
                "adapt_unflipped_margin_mean": adapt_shift_info[
                    "unflipped_margin_mean"
                ],
                "test_flipped_margin_mean": test_shift_info[
                    "flipped_margin_mean"
                ],
                "test_unflipped_margin_mean": test_shift_info[
                    "unflipped_margin_mean"
                ],
                "adapt_benign_margin_threshold": adapt_shift_info[
                    "benign_boundary_margin_threshold"
                ],
                "adapt_attack_margin_threshold": adapt_shift_info[
                    "attack_boundary_margin_threshold"
                ],
                "test_benign_margin_threshold": test_shift_info[
                    "benign_boundary_margin_threshold"
                ],
                "test_attack_margin_threshold": test_shift_info[
                    "attack_boundary_margin_threshold"
                ],
                "marginal_domain_auroc": x_diagnostics[
                    "marginal_domain_auroc"
                ],
                "marginal_domain_classifier_converged": x_diagnostics[
                    "marginal_domain_classifier_converged"
                ],
                "marginal_domain_classifier_iterations": x_diagnostics[
                    "marginal_domain_classifier_iterations"
                ],
                "mean_absolute_feature_smd": x_diagnostics[
                    "mean_absolute_feature_smd"
                ],
                "max_absolute_feature_smd": x_diagnostics[
                    "max_absolute_feature_smd"
                ],
                "estimated_iw_domain_auroc": iw_info[
                    "domain_auroc"
                ],
                "estimated_iw_domain_classifier_converged": iw_info[
                    "converged"
                ],
                "estimated_iw_domain_classifier_iterations": iw_info[
                    "iterations"
                ],
                "estimated_iw_ess_ratio": iw_info[
                    "train_weight_ess_ratio"
                ],
                "estimated_iw_weight_std": iw_info[
                    "train_weight_std"
                ],
                "estimated_iw_weight_max": iw_info[
                    "train_weight_max"
                ],
                "replay_per_class": replay_info[
                    "replay_per_class"
                ],
                "replay_total": replay_info[
                    "replay_total"
                ],
                "replay_attack_prior": replay_info[
                    "replay_attack_prior"
                ],
                "naive_ft_best_target_val_auroc": naive_ft_info[
                    "best_target_val_auroc"
                ],
                "replay_ft_best_target_val_auroc": replay_ft_info[
                    "best_target_val_auroc"
                ],
                "target_only_mlp_best_target_val_auroc": target_only_mlp_info[
                    "best_target_val_auroc"
                ],
                "iw_logistic_converged": iw_logistic_info[
                    "converged"
                ],
                "target_only_logistic_converged": target_only_logistic_info[
                    "converged"
                ],
            }
        )

        print(
            f"\nseverity={severity:.2f}"
            f" | flip={test_shift_info['actual_label_flip_rate']:.3f}"
            f" | prior={target_prior:.3f}"
            f" | X-domain AUROC={x_diagnostics['marginal_domain_auroc']:.3f}"
            f" | IW-domain AUROC={iw_info['domain_auroc']:.3f}"
        )

    return (
        results,
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
        "--concept_severities",
        type=float,
        nargs="+",
        default=DEFAULT_CONCEPT_SEVERITIES,
    )

    parser.add_argument(
        "--design_per_class",
        type=int,
        default=DEFAULT_DESIGN_PER_CLASS,
    )

    parser.add_argument(
        "--source_train_per_class",
        type=int,
        default=DEFAULT_SOURCE_TRAIN_PER_CLASS,
    )

    parser.add_argument(
        "--source_val_per_class",
        type=int,
        default=DEFAULT_SOURCE_VAL_PER_CLASS,
    )

    parser.add_argument(
        "--source_test_per_class",
        type=int,
        default=DEFAULT_SOURCE_TEST_PER_CLASS,
    )

    parser.add_argument(
        "--target_adapt_per_class",
        type=int,
        default=DEFAULT_TARGET_ADAPT_PER_CLASS,
    )

    parser.add_argument(
        "--target_test_per_class",
        type=int,
        default=DEFAULT_TARGET_TEST_PER_CLASS,
    )

    parser.add_argument(
        "--replay_per_class",
        type=int,
        default=DEFAULT_REPLAY_PER_CLASS,
    )

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
        "--target_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--target_lr",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--target_patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--iw_clip",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    severities = sorted(
        set(
            float(s)
            for s in args.concept_severities
        )
    )

    if any(
        s < 0.0
        or s >= 0.5
        for s in severities
    ):
        raise ValueError(
            "concept_severities must be in [0, 0.5)."
        )

    if 0.0 not in severities:
        raise ValueError(
            "concept_severities must include 0.0 control."
        )

    required_per_class = (
        args.design_per_class
        + args.source_train_per_class
        + args.source_val_per_class
        + args.source_test_per_class
        + args.target_adapt_per_class
        + args.target_test_per_class
    )

    if required_per_class > 100_000:
        raise ValueError(
            f"Default balanced_100k reservoir supports 100k/class, "
            f"but requested allocation is {required_per_class:,}/class."
        )

    print(
        "=" * 120
    )
    print(
        "CONTROLLED BOUNDARY-LOCALIZED CONCEPT SHIFT"
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
        f"Severities: {severities}"
    )
    print(
        f"Seeds: {args.seeds}"
    )
    print(
        f"Allocation per class: {required_per_class:,}"
    )

    X, y, feature_names = load_dataset_pool(
        args.dataset,
        args.pool_variant,
    )

    print(
        f"Loaded {len(y):,} rows"
        f" | benign={(y == 0).sum():,}"
        f" | attack={(y == 1).sum():,}"
        f" | features={X.shape[1]}"
    )

    design = build_concept_design(
        X,
        y,
        args.design_per_class,
    )

    print(
        "Design boundary logistic:"
        f" AUROC={design.design_auroc:.6f}"
        f" | converged={design.converged}"
        f" | iterations={design.iterations}"
    )

    if not design.converged:
        raise RuntimeError(
            "Held-out design boundary classifier did not converge."
        )

    output_dir = (
        STAGE1_DIR
        / "controlled_concept_shift"
        / args.dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save design boundary coefficients for reproducibility.
    coef = design.classifier.coef_[0]

    direction_df = pd.DataFrame(
        {
            "feature": feature_names,
            "design_logistic_coefficient": coef,
            "absolute_design_logistic_coefficient": np.abs(coef),
        }
    ).sort_values(
        "absolute_design_logistic_coefficient",
        ascending=False,
    )

    direction_path = (
        output_dir
        / "controlled_concept_shift_design_boundary.csv"
    )

    direction_df.to_csv(
        direction_path,
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
    all_source_rows = []

    for seed in args.seeds:
        (
            result_rows,
            diagnostic_rows,
            source_rows,
        ) = run_seed(
            seed=int(seed),
            X=X,
            y=y,
            design=design,
            severities=severities,
            args=args,
            device=device,
        )

        all_results.extend(
            result_rows
        )
        all_diagnostics.extend(
            diagnostic_rows
        )
        all_source_rows.extend(
            source_rows
        )

    per_seed_df = pd.DataFrame(
        all_results
    )

    diagnostics_df = pd.DataFrame(
        all_diagnostics
    )

    source_df = pd.DataFrame(
        all_source_rows
    )

    summary_df = aggregate_results(
        per_seed_df
    )

    diagnostics_summary_df = aggregate_diagnostics(
        diagnostics_df
    )

    paired_df = build_method_vs_source_paired(
        per_seed_df
    )

    paired_summary_df = summarize_method_vs_source(
        paired_df
    )

    (
        mlp_iw_vs_uniform_df,
        mlp_iw_vs_uniform_summary_df,
    ) = build_mlp_iw_vs_uniform(
        per_seed_df
    )

    (
        ft_vs_replay_df,
        ft_vs_replay_summary_df,
    ) = build_ft_vs_replay(
        per_seed_df
    )

    paths = {
        "per_seed": (
            output_dir
            / "controlled_concept_shift_per_seed.csv"
        ),
        "summary": (
            output_dir
            / "controlled_concept_shift_summary.csv"
        ),
        "source": (
            output_dir
            / "controlled_concept_shift_source_metrics.csv"
        ),
        "diagnostics": (
            output_dir
            / "controlled_concept_shift_diagnostics_per_seed.csv"
        ),
        "diagnostics_summary": (
            output_dir
            / "controlled_concept_shift_diagnostics_summary.csv"
        ),
        "paired": (
            output_dir
            / "controlled_concept_shift_paired_differences.csv"
        ),
        "paired_summary": (
            output_dir
            / "controlled_concept_shift_paired_difference_summary.csv"
        ),
        "mlp_iw_uniform": (
            output_dir
            / "controlled_concept_shift_mlp_iw_vs_uniform.csv"
        ),
        "mlp_iw_uniform_summary": (
            output_dir
            / "controlled_concept_shift_mlp_iw_vs_uniform_summary.csv"
        ),
        "ft_replay": (
            output_dir
            / "controlled_concept_shift_ft_vs_replay.csv"
        ),
        "ft_replay_summary": (
            output_dir
            / "controlled_concept_shift_ft_vs_replay_summary.csv"
        ),
    }

    per_seed_df.to_csv(
        paths["per_seed"],
        index=False,
    )

    summary_df.to_csv(
        paths["summary"],
        index=False,
    )

    source_df.to_csv(
        paths["source"],
        index=False,
    )

    diagnostics_df.to_csv(
        paths["diagnostics"],
        index=False,
    )

    diagnostics_summary_df.to_csv(
        paths["diagnostics_summary"],
        index=False,
    )

    paired_df.to_csv(
        paths["paired"],
        index=False,
    )

    paired_summary_df.to_csv(
        paths["paired_summary"],
        index=False,
    )

    mlp_iw_vs_uniform_df.to_csv(
        paths["mlp_iw_uniform"],
        index=False,
    )

    mlp_iw_vs_uniform_summary_df.to_csv(
        paths["mlp_iw_uniform_summary"],
        index=False,
    )

    ft_vs_replay_df.to_csv(
        paths["ft_replay"],
        index=False,
    )

    ft_vs_replay_summary_df.to_csv(
        paths["ft_replay_summary"],
        index=False,
    )

    # ========================================================
    # Protocol
    # ========================================================
    protocol = {
        "research_question": (
            "How do mismatched reweighting methods, labelled target "
            "fine-tuning, and replay behave under controlled boundary-localized "
            "concept shift with fixed feature distribution and fixed class prior?"
        ),
        "dataset": args.dataset,
        "pool_variant": args.pool_variant,
        "seeds": [
            int(s)
            for s in args.seeds
        ],
        "concept_severities": severities,
        "construction": {
            "type": "boundary_localized_concept_shift",
            "source_attack_prior": 0.5,
            "target_attack_prior": 0.5,
            "design_seed": DESIGN_SEED,
            "design_per_class": args.design_per_class,
            "design_boundary_model": (
                "LogisticRegression on held-out design subset only"
            ),
            "boundary_score": (
                "absolute held-out-design logistic decision margin"
            ),
            "intervention": (
                "Within each original target class, flip exactly the requested "
                "fraction of labels for rows nearest the fixed design boundary."
            ),
            "target_X_changed_across_severities": False,
            "source_target_rows_overlap": False,
            "important_note": (
                "Source and target are disjoint finite samples from the same "
                "balanced reservoir. P_T(X) is exactly fixed across target "
                "severity conditions; source-vs-target domain AUROC validates "
                "that no material X-distribution shift was introduced."
            ),
        },
        "sizes_per_class": {
            "design": args.design_per_class,
            "source_train": args.source_train_per_class,
            "source_validation": args.source_val_per_class,
            "source_test": args.source_test_per_class,
            "target_adaptation": args.target_adapt_per_class,
            "target_test": args.target_test_per_class,
        },
        "replay": {
            "selection": "random source-train replay",
            "replay_per_class": args.replay_per_class,
            "replay_total": (
                2
                * args.replay_per_class
            ),
            "expected_default_replay_fraction": (
                (
                    2
                    * args.replay_per_class
                )
                / (
                    int(
                        0.8
                        * (
                            2
                            * args.target_adapt_per_class
                        )
                    )
                    + (
                        2
                        * args.replay_per_class
                    )
                )
            ),
        },
        "methods": {
            "source_only": {
                "target_labels_used": False,
            },
            "oracle_prior_correction": {
                "target_labels_used": False,
                "oracle_information": "known equal source/target prior",
                "expected_behavior": (
                    "exact identity transform; equal-prior branch returns "
                    "the original probability array without clipping"
                ),
            },
            "unconditional_estimated_iw": {
                "target_labels_used": False,
                "target_X_used": True,
                "purpose": (
                    "mismatched covariate-shift baseline"
                ),
            },
            "uniform_extra_training": {
                "classifier": "mlp_only",
                "target_labels_used": False,
                "purpose": (
                    "matched extra-optimization control for MLP IW"
                ),
            },
            "naive_target_finetuning": {
                "classifier": "mlp_only",
                "target_labels_used": True,
            },
            "finetuning_plus_replay": {
                "classifier": "mlp_only",
                "target_labels_used": True,
                "source_replay_labels_used": True,
            },
            "target_only_reference": {
                "target_labels_used": True,
                "comparison_note": (
                    "target-only reference, not upper bound"
                ),
            },
        },
        "hypotheses": {
            "H1": (
                "Feature-domain AUROC remains near 0.5 and target class prior "
                "remains exactly 0.5 across severity."
            ),
            "H2": (
                "Source-only target performance degrades as structured "
                "boundary-local concept changes increase."
            ),
            "H3": (
                "Prior correction does not repair concept shift when priors "
                "remain unchanged."
            ),
            "H4": (
                "Unconditional IW provides little benefit when P(X) has not "
                "deliberately changed."
            ),
            "H5": (
                "Labelled target fine-tuning repairs target performance by "
                "updating the classifier decision function."
            ),
            "H6": (
                "Replay reduces source forgetting relative to naive fine-tuning "
                "with limited target-performance cost."
            ),
            "H7": (
                "Sensitivity and adaptation benefit are model-dependent."
            ),
        },
        "leakage_controls": [
            "Design rows excluded from every model split.",
            "Design boundary chosen once without target-test selection.",
            "Source and target row sets are disjoint.",
            "Target X rows fixed across all severities within seed.",
            "Only target labels change across severities.",
            "Target prior exactly 0.5 after symmetric class-specific flips.",
            "Model scaler fits source train only.",
            "Target test labels used for evaluation only.",
            "Unconditional IW uses target X only.",
            "Target fine-tuning uses target-adaptation labels only.",
            "Replay uses source-train rows only.",
            "Replay membership fixed across severities within seed.",
            "Naive FT and replay use identical target train/validation membership.",
        ],
        "statistics": (
            "Five default seeds. Report mean±SD and paired descriptive "
            "t-based 95% CIs. Do not claim formal significance from five "
            "seeds without an explicit inferential plan."
        ),
    }

    protocol_path = (
        output_dir
        / "controlled_concept_shift_protocol.json"
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

    # ========================================================
    # Figures
    # ========================================================
    plot_target_auroc(
        summary_df,
        "mlp",
        output_dir
        / "controlled_concept_shift_mlp_target_auroc.png",
    )

    plot_target_auroc(
        summary_df,
        "logistic_regression",
        output_dir
        / "controlled_concept_shift_logistic_target_auroc.png",
    )

    plot_stability_plasticity(
        summary_df,
        output_dir
        / "controlled_concept_shift_stability_plasticity.png",
    )

    # ========================================================
    # Console summary
    # ========================================================
    print(
        "\n"
        + "=" * 140
    )

    print(
        "CONTROLLED CONCEPT SHIFT COMPLETE"
    )

    print(
        "=" * 140
    )

    print(
        "\nCONSTRUCTION DIAGNOSTICS"
    )

    print(
        diagnostics_summary_df[
            [
                "concept_severity",
                "target_attack_prior_mean",
                "actual_test_label_flip_rate_mean",
                "marginal_domain_auroc_mean",
                "mean_absolute_feature_smd_mean",
                "estimated_iw_ess_ratio_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nTARGET / RETENTION SUMMARY"
    )

    print(
        summary_df[
            [
                "classifier",
                "method",
                "concept_severity",
                "auroc_mean",
                "source_after_auroc_mean",
                "source_auroc_drop_mean",
                "average_source_target_auroc_mean",
                "worst_source_target_auroc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nGUARDRAILS"
    )

    print(
        "- target_attack_prior_mean must remain exactly 0.5."
    )

    print(
        "- actual_test_label_flip_rate_mean should match requested severity."
    )

    print(
        "- marginal_domain_auroc_mean should remain near 0.5 across severity."
    )

    print(
        "- marginal X diagnostics should be identical/nearly identical across severity because target X rows do not change."
    )

    print(
        "- oracle_prior_correction must be numerically identical to source_only."
    )

    print(
        "- For MLP IW, interpret against uniform_extra_training, not only source_only."
    )

    print(
        "- For stability/plasticity, compare finetuning_plus_replay directly against naive_target_finetuning."
    )

    print(
        "\nOUTPUTS"
    )

    for path in [
        paths["summary"],
        paths["diagnostics_summary"],
        paths["paired_summary"],
        paths["mlp_iw_uniform_summary"],
        paths["ft_replay_summary"],
        paths["source"],
        paths["per_seed"],
        paths["diagnostics"],
        paths["paired"],
        paths["mlp_iw_uniform"],
        paths["ft_replay"],
        direction_path,
        protocol_path,
    ]:
        print(
            f"- {path}"
        )


if __name__ == "__main__":
    main()
