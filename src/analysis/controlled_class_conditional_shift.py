"""
controlled_class_conditional_shift_GAP_CONTROLLED_FINAL.py

Controlled GAP-CONTROLLED class-conditional shift experiment for DANIDS.

Research purpose
----------------
Natural NIDS cross-domain shifts are mixed. This benchmark deliberately
changes class-conditional feature distributions while keeping the class prior
fixed:

    P_T(Y) = P_S(Y) = 0.5
    P_T(X | Y=0) != P_S(X | Y=0)
    P_T(X | Y=1) != P_S(X | Y=1)

Because Bayes' rule links P(X|Y), P(Y), and P(Y|X), changing P(X|Y) can also
change P(Y|X). Therefore this script calls the construction "controlled
class-conditional shift" rather than claiming a textbook-pure shift in which
every other distributional component is invariant.

Construction
------------
The Stage-1 balanced_100k pool contains 100k benign and 100k attack rows.

A fixed, label-free design subset is reserved and excluded from all modelling.
A StandardScaler + PCA(1) fitted on DESIGN X ONLY defines a scalar covariate
score g(X).

For each severity, the target is selected differently WITHIN each class,
but the direction is chosen ONCE from the held-out design subset so that the
two classes move TOWARD each other along g(X).

If mean_design(g(X)|benign) < mean_design(g(X)|attack):
    benign target: upper tail of g(X)
    attack target: lower tail of g(X)

If the ordering is reversed, the directions are reversed automatically.

The exact same number of benign and attack rows is selected, so the target
attack prior remains exactly 0.5.

Default retained tail fractions:
    1.00 -> control / no conditional selection
    0.70 -> mild shift
    0.40 -> moderate shift
    0.20 -> strong shift

The target remains composed of real observed rows from the dataset; no
synthetic feature values are fabricated.

Methods
-------
1. source_only
   No adaptation.

2. uniform_extra_training
   MLP only. Same warm-start / optimizer / adaptation schedule as weighted
   MLP methods, but all source weights are 1. This removes the "extra epochs"
   confound.

3. unconditional_estimated_iw
   Estimate p_T(X)/p_S(X) with a balanced source-vs-target domain classifier
   using unlabeled target-adaptation X. This deliberately ignores the class
   structure and serves as a mismatched/general covariate-shift baseline.

4. oracle_conditional_iw
   Exact class-conditional source weights from the known synthetic selection
   mechanism:

       w(x,y=0) proportional to I[g(x) <= benign_threshold]
       w(x,y=1) proportional to I[g(x) >= attack_threshold]

   This is a mechanism/reference condition, not a deployable method.

5. pseudo_conditional_iw
   Practical class-aware correction:
   - predict pseudo-labels on unlabeled target adaptation rows with the source
     classifier;
   - within each pseudo-class, train a balanced source-vs-target domain
     classifier;
   - estimate class-specific density-ratio weights for labelled source rows.

6. target_labeled_adaptation
   A supervision-heavy reference:
   - MLP: warm-start fine-tuning on labelled target adaptation rows;
   - logistic regression: refit on labelled target adaptation train rows.
   It is NOT supervision-comparable to the unsupervised IW methods and must be
   reported as a labelled-target reference.

Hypotheses
----------
H1. The class prior remains fixed while benign and attack conditional domain
    separability increase with severity, and target class separation along the
    controlled score axis decreases.

H2. Source-only target performance should degrade as converging
    class-conditional shift reduces class separation along the controlled axis.

H3. Oracle conditional IW should outperform an unconditional IW strategy when
    the predictive model is sensitive to the changed class-conditional
    distributions.

H4. Pseudo-conditional IW should approach oracle conditional IW when
    pseudo-label quality and class-specific density-ratio estimation are good.

H5. A method matched only to marginal/covariate shift (unconditional IW) may be
    insufficient when the shift mechanism is class-conditional.

H6. Adaptation benefit may be model-dependent: a flexible MLP can remain robust
    where a lower-capacity logistic model degrades.

Leakage / rigor controls
------------------------
- Design subset is excluded from all source and target model data.
- PCA shift direction uses design X only.
- Labels are used ONLY to define the synthetic class-conditional selection
  mechanism and to train/evaluate methods whose supervision regime explicitly
  allows labels.
- Source train/val/test are mutually disjoint.
- Target adaptation/test candidate banks are disjoint.
- Every target row is disjoint from every source row.
- Target adaptation and target test are exactly 50/50 benign/attack.
- NIDS StandardScaler fits source train only.
- Unconditional IW sees target-adaptation X only, no target labels.
- Pseudo-conditional IW sees target-adaptation X and source-model pseudo-labels,
  never true target labels.
- Target test labels are evaluation only.
- Labelled-target adaptation uses target-adaptation labels but never test labels.
- All methods are evaluated on the exact same target test rows per seed/severity.
- MLP IW methods are compared against matched uniform extra training.
- Five default seeds and paired descriptive t-based 95% CIs.

Run
---
Default ToN:
    python -B -m src.analysis.controlled_class_conditional_shift_converging

Other datasets:
    python -B -m src.analysis.controlled_class_conditional_shift --dataset NF-CSE-CIC-IDS2018-v3
    python -B -m src.analysis.controlled_class_conditional_shift --dataset NF-UNSW-NB15-v3
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

from src.config import STAGE1_DIR


# ============================================================
# Defaults
# ============================================================
DEFAULT_DATASET = "NF-ToN-IoT-v3"
DEFAULT_POOL_VARIANT = "balanced_100k"
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]
DEFAULT_TARGET_GAP_RATIOS = [1.0, 0.70, 0.40, 0.10]

# Dense search grid for the retained within-class tail fraction.
# The design subset alone is used to choose these values.
DEFAULT_TAIL_SEARCH_MIN = 0.15
DEFAULT_TAIL_SEARCH_MAX = 1.0
DEFAULT_TAIL_SEARCH_STEP = 0.0025

DESIGN_SEED = 271828

# Exact per-class allocations. Sum = 100k/class.
DEFAULT_DESIGN_PER_CLASS = 10_000
DEFAULT_SOURCE_TRAIN_PER_CLASS = 20_000
DEFAULT_SOURCE_VAL_PER_CLASS = 5_000
DEFAULT_SOURCE_TEST_PER_CLASS = 5_000
DEFAULT_TARGET_ADAPT_CANDIDATE_PER_CLASS = 20_000
DEFAULT_TARGET_TEST_CANDIDATE_PER_CLASS = 40_000

# Selected target rows per class.
DEFAULT_TARGET_ADAPT_PER_CLASS = 2_000
DEFAULT_TARGET_TEST_PER_CLASS = 5_000

CLASSIFIERS = ["mlp", "logistic_regression"]

BASE_METHODS = [
    "source_only",
    "unconditional_estimated_iw",
    "oracle_conditional_iw",
    "pseudo_conditional_iw",
    "target_labeled_adaptation",
]

MLP_EXTRA_METHOD = "uniform_extra_training"

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
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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
        raise FileNotFoundError(f"Missing labels: {y_path}")

    frame = pd.read_parquet(x_path)
    feature_names = [str(c) for c in frame.columns]

    X = frame.to_numpy(dtype=np.float32, copy=True)
    y = np.load(y_path).astype(np.int8)

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")

    unique = set(np.unique(y))
    if unique != {0, 1}:
        raise ValueError(f"Expected labels {{0,1}}, got {sorted(unique)}")

    return X, y, feature_names


# ============================================================
# Fixed label-free score direction
# ============================================================
class ShiftDesign:
    def __init__(
        self,
        scaler: StandardScaler,
        pca: PCA,
        benign_thresholds: dict[float, float],
        attack_thresholds: dict[float, float],
        design_indices: np.ndarray,
        design_scores: np.ndarray,
        benign_direction: str,
        attack_direction: str,
        benign_design_score_mean: float,
        attack_design_score_mean: float,
        selected_tail_fractions: dict[float, float],
        achieved_gap_ratios: dict[float, float],
        achieved_signed_gaps: dict[float, float],
        source_design_signed_gap: float,
    ) -> None:
        self.scaler = scaler
        self.pca = pca
        self.benign_thresholds = benign_thresholds
        self.attack_thresholds = attack_thresholds
        self.design_indices = design_indices
        self.design_scores = design_scores
        self.benign_direction = benign_direction
        self.attack_direction = attack_direction
        self.benign_design_score_mean = benign_design_score_mean
        self.attack_design_score_mean = attack_design_score_mean
        self.selected_tail_fractions = selected_tail_fractions
        self.achieved_gap_ratios = achieved_gap_ratios
        self.achieved_signed_gaps = achieved_signed_gaps
        self.source_design_signed_gap = source_design_signed_gap

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))[:, 0]


def build_shift_design(
    X: np.ndarray,
    y: np.ndarray,
    target_gap_ratios: list[float],
    design_per_class: int,
    tail_search_min: float,
    tail_search_max: float,
    tail_search_step: float,
) -> ShiftDesign:
    """
    Reserve an equal-size design subset from each class, but fit the score using
    X only. The labels are used solely to reserve equal counts so that the
    design subset itself does not accidentally become class-prior-skewed.
    """
    rng = np.random.default_rng(DESIGN_SEED)

    class_indices = {}
    design_parts = []

    for cls in [0, 1]:
        idx = np.flatnonzero(y == cls).copy()
        rng.shuffle(idx)

        if len(idx) < design_per_class:
            raise ValueError(
                f"Class {cls} has {len(idx):,} rows; "
                f"need at least {design_per_class:,} for design."
            )

        design_parts.append(idx[:design_per_class])
        class_indices[cls] = idx

    design_indices = np.concatenate(design_parts)
    rng.shuffle(design_indices)

    X_design = X[design_indices]

    scaler = StandardScaler()
    X_design_scaled = scaler.fit_transform(X_design)

    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(X_design_scaled)

    # Deterministic sign.
    component = pca.components_[0].copy()
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0:
        pca.components_[0] *= -1.0

    design_scores = pca.transform(X_design_scaled)[:, 0]
    y_design = y[design_indices]

    # PCA itself is label-free. Labels are used only here to define the
    # controlled class-conditional selection mechanism.
    #
    # The orientation and ALL severity values are chosen from this held-out
    # design subset only. Target adaptation/test performance is never used.
    y_design = y[design_indices]
    benign_design_scores = design_scores[y_design == 0]
    attack_design_scores = design_scores[y_design == 1]

    benign_design_score_mean = float(np.mean(benign_design_scores))
    attack_design_score_mean = float(np.mean(attack_design_scores))

    source_design_signed_gap = float(
        attack_design_score_mean - benign_design_score_mean
    )

    if abs(source_design_signed_gap) <= EPS:
        raise RuntimeError(
            "Design benign/attack score means are effectively equal; "
            "cannot define a stable class-gap experiment."
        )

    # Move the lower-mean class upward and the higher-mean class downward.
    if benign_design_score_mean < attack_design_score_mean:
        benign_direction = "upper"
        attack_direction = "lower"
    else:
        benign_direction = "lower"
        attack_direction = "upper"

    source_sign = float(np.sign(source_design_signed_gap))
    source_abs_gap = abs(source_design_signed_gap)

    def threshold_for_direction(
        class_scores: np.ndarray,
        direction: str,
        tail_fraction: float,
    ) -> float:
        tf = float(tail_fraction)

        if tf >= 1.0:
            return -np.inf if direction == "upper" else np.inf

        if direction == "upper":
            return float(np.quantile(class_scores, 1.0 - tf))

        if direction == "lower":
            return float(np.quantile(class_scores, tf))

        raise ValueError(direction)

    def selected_scores(
        class_scores: np.ndarray,
        threshold: float,
        direction: str,
    ) -> np.ndarray:
        if direction == "upper":
            return class_scores[class_scores >= threshold]

        if direction == "lower":
            return class_scores[class_scores <= threshold]

        raise ValueError(direction)

    # Search candidate tail fractions using DESIGN data only.
    candidates = np.arange(
        tail_search_min,
        tail_search_max + tail_search_step / 2.0,
        tail_search_step,
        dtype=float,
    )
    candidates = np.clip(candidates, tail_search_min, tail_search_max)
    candidates = np.unique(
        np.concatenate([candidates, np.array([1.0], dtype=float)])
    )

    candidate_rows = []

    for tf in candidates:
        benign_threshold = threshold_for_direction(
            benign_design_scores,
            benign_direction,
            float(tf),
        )
        attack_threshold = threshold_for_direction(
            attack_design_scores,
            attack_direction,
            float(tf),
        )

        benign_selected = selected_scores(
            benign_design_scores,
            benign_threshold,
            benign_direction,
        )
        attack_selected = selected_scores(
            attack_design_scores,
            attack_threshold,
            attack_direction,
        )

        if len(benign_selected) == 0 or len(attack_selected) == 0:
            continue

        signed_gap = float(
            np.mean(attack_selected) - np.mean(benign_selected)
        )

        # Critical guardrail: class ordering may shrink but may NEVER reverse.
        same_sign = (
            np.sign(signed_gap) == np.sign(source_design_signed_gap)
            or abs(signed_gap) <= EPS
        )

        if not same_sign:
            continue

        gap_ratio = float(
            abs(signed_gap) / max(source_abs_gap, EPS)
        )

        candidate_rows.append(
            {
                "tail_fraction": float(tf),
                "benign_threshold": float(benign_threshold),
                "attack_threshold": float(attack_threshold),
                "signed_gap": signed_gap,
                "gap_ratio": gap_ratio,
            }
        )

    if not candidate_rows:
        raise RuntimeError(
            "No non-crossing tail fractions were found on the design subset."
        )

    # Choose one non-crossing tail fraction per desired class-gap ratio.
    # Selection is greedy from largest requested ratio to smallest while
    # enforcing progressively smaller achieved gap ratios and progressively
    # smaller retained tail fractions.
    requested = sorted(
        set(float(r) for r in target_gap_ratios),
        reverse=True,
    )

    if any(r <= 0.0 or r > 1.0 for r in requested):
        raise ValueError("target_gap_ratios must lie in (0, 1].")

    benign_thresholds: dict[float, float] = {}
    attack_thresholds: dict[float, float] = {}
    selected_tail_fractions: dict[float, float] = {}
    achieved_gap_ratios: dict[float, float] = {}
    achieved_signed_gaps: dict[float, float] = {}

    previous_gap_ratio = float("inf")
    previous_tail_fraction = float("inf")

    for target_ratio in requested:
        feasible = [
            row
            for row in candidate_rows
            if row["gap_ratio"] <= previous_gap_ratio + 1e-12
            and row["tail_fraction"] <= previous_tail_fraction + 1e-12
        ]

        if target_ratio >= 0.999999:
            # Exact control condition.
            feasible_control = [
                row
                for row in feasible
                if abs(row["tail_fraction"] - 1.0) <= 1e-12
            ]
            if not feasible_control:
                raise RuntimeError("Could not construct the no-shift control.")
            chosen = feasible_control[0]
        else:
            chosen = min(
                feasible,
                key=lambda row: (
                    abs(row["gap_ratio"] - target_ratio),
                    -row["tail_fraction"],
                ),
            )

        selected_tail_fractions[target_ratio] = float(
            chosen["tail_fraction"]
        )
        achieved_gap_ratios[target_ratio] = float(
            chosen["gap_ratio"]
        )
        achieved_signed_gaps[target_ratio] = float(
            chosen["signed_gap"]
        )
        benign_thresholds[target_ratio] = float(
            chosen["benign_threshold"]
        )
        attack_thresholds[target_ratio] = float(
            chosen["attack_threshold"]
        )

        previous_gap_ratio = float(chosen["gap_ratio"])
        previous_tail_fraction = float(chosen["tail_fraction"])

    # Final invariants.
    ordered_achieved = [
        achieved_gap_ratios[r]
        for r in requested
    ]

    for earlier, later in zip(
        ordered_achieved,
        ordered_achieved[1:],
    ):
        if later > earlier + 1e-9:
            raise RuntimeError(
                "Achieved class-gap ratios are not monotonically decreasing."
            )

    for ratio in requested:
        signed_gap = achieved_signed_gaps[ratio]
        if abs(signed_gap) > EPS and np.sign(signed_gap) != source_sign:
            raise RuntimeError(
                "Class ordering crossed despite the non-crossing guardrail."
            )

    return ShiftDesign(
        scaler=scaler,
        pca=pca,
        benign_thresholds=benign_thresholds,
        attack_thresholds=attack_thresholds,
        design_indices=design_indices,
        design_scores=design_scores,
        benign_direction=benign_direction,
        attack_direction=attack_direction,
        benign_design_score_mean=benign_design_score_mean,
        attack_design_score_mean=attack_design_score_mean,
        selected_tail_fractions=selected_tail_fractions,
        achieved_gap_ratios=achieved_gap_ratios,
        achieved_signed_gaps=achieved_signed_gaps,
        source_design_signed_gap=source_design_signed_gap,
    )


def shift_loading_table(
    design: ShiftDesign,
    feature_names: list[str],
) -> pd.DataFrame:
    loadings = design.pca.components_[0]

    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "pc1_loading": loadings,
                "absolute_pc1_loading": np.abs(loadings),
            }
        )
        .sort_values("absolute_pc1_loading", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# Disjoint, class-balanced split
# ============================================================
def build_seed_split(
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    seed: int,
    source_train_per_class: int,
    source_val_per_class: int,
    source_test_per_class: int,
    target_adapt_candidate_per_class: int,
    target_test_candidate_per_class: int,
) -> dict[str, Any]:
    design_set = set(int(i) for i in design.design_indices.tolist())
    rng = np.random.default_rng(seed)

    parts: dict[str, list[np.ndarray]] = {
        "source_train": [],
        "source_val": [],
        "source_test": [],
        "target_adapt_candidate": [],
        "target_test_candidate": [],
    }

    required_per_class = (
        source_train_per_class
        + source_val_per_class
        + source_test_per_class
        + target_adapt_candidate_per_class
        + target_test_candidate_per_class
    )

    for cls in [0, 1]:
        idx = np.array(
            [
                i
                for i in np.flatnonzero(y == cls)
                if int(i) not in design_set
            ],
            dtype=np.int64,
        )

        rng.shuffle(idx)

        if len(idx) < required_per_class:
            raise ValueError(
                f"Class {cls}: need {required_per_class:,} non-design rows "
                f"but only {len(idx):,} available."
            )

        cursor = 0

        def take(n: int) -> np.ndarray:
            nonlocal cursor
            out = idx[cursor : cursor + n]
            cursor += n
            return out

        parts["source_train"].append(take(source_train_per_class))
        parts["source_val"].append(take(source_val_per_class))
        parts["source_test"].append(take(source_test_per_class))
        parts["target_adapt_candidate"].append(
            take(target_adapt_candidate_per_class)
        )
        parts["target_test_candidate"].append(
            take(target_test_candidate_per_class)
        )

    # Keep class-specific candidate arrays for conditional selection.
    return {
        "idx_source_train": np.concatenate(parts["source_train"]),
        "idx_source_val": np.concatenate(parts["source_val"]),
        "idx_source_test": np.concatenate(parts["source_test"]),
        "idx_target_adapt_candidate_by_class": {
            0: parts["target_adapt_candidate"][0],
            1: parts["target_adapt_candidate"][1],
        },
        "idx_target_test_candidate_by_class": {
            0: parts["target_test_candidate"][0],
            1: parts["target_test_candidate"][1],
        },
    }


def shuffled_eligible_prefix(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    eligible_mask: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    eligible_indices = candidate_indices[eligible_mask]

    rng = np.random.default_rng(seed)
    eligible_indices = eligible_indices.copy()
    rng.shuffle(eligible_indices)

    if len(eligible_indices) < n:
        raise ValueError(
            f"Only {len(eligible_indices):,} eligible rows; need {n:,}."
        )

    return eligible_indices[:n]


def select_target_condition(
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    split: dict[str, Any],
    target_gap_ratio: float,
    target_adapt_per_class: int,
    target_test_per_class: int,
    seed: int,
) -> dict[str, Any]:
    """
    GAP-CONTROLLED non-crossing class-conditional selection.

    The held-out design subset determines which class is lower/higher on the
    fixed PC1 score. The lower class is moved toward the upper class and the
    upper class toward the lower class. No target test performance is used to
    choose this direction.

    Exact equal counts are selected from each class, so P_T(Y)=0.5 by
    construction at every severity.
    """
    target_gap_ratio = float(target_gap_ratio)
    tf = float(
        design.selected_tail_fractions[target_gap_ratio]
    )
    benign_threshold = design.benign_thresholds[target_gap_ratio]
    attack_threshold = design.attack_thresholds[target_gap_ratio]

    adapt_selected = []
    test_selected = []

    class_stats = {}

    for cls in [0, 1]:
        adapt_candidates = split["idx_target_adapt_candidate_by_class"][cls]
        test_candidates = split["idx_target_test_candidate_by_class"][cls]

        adapt_scores = design.score(X[adapt_candidates])
        test_scores = design.score(X[test_candidates])

        if tf >= 1.0:
            adapt_mask = np.ones(len(adapt_candidates), dtype=bool)
            test_mask = np.ones(len(test_candidates), dtype=bool)
        else:
            direction = (
                design.benign_direction
                if cls == 0
                else design.attack_direction
            )
            threshold = (
                benign_threshold
                if cls == 0
                else attack_threshold
            )

            if direction == "upper":
                adapt_mask = adapt_scores >= threshold
                test_mask = test_scores >= threshold
            elif direction == "lower":
                adapt_mask = adapt_scores <= threshold
                test_mask = test_scores <= threshold
            else:
                raise ValueError(direction)

        adapt_idx = shuffled_eligible_prefix(
            adapt_candidates,
            adapt_scores,
            adapt_mask,
            target_adapt_per_class,
            seed=seed + cls * 10_000 + int(round(tf * 1000)) + 11,
        )

        test_idx = shuffled_eligible_prefix(
            test_candidates,
            test_scores,
            test_mask,
            target_test_per_class,
            seed=seed + cls * 10_000 + int(round(tf * 1000)) + 29,
        )

        adapt_selected.append(adapt_idx)
        test_selected.append(test_idx)

        class_stats[cls] = {
            "adapt_eligible_fraction": float(np.mean(adapt_mask)),
            "test_eligible_fraction": float(np.mean(test_mask)),
            "target_adapt_score_mean": float(
                np.mean(design.score(X[adapt_idx]))
            ),
            "target_test_score_mean": float(
                np.mean(design.score(X[test_idx]))
            ),
        }

    adapt_idx = np.concatenate(adapt_selected)
    test_idx = np.concatenate(test_selected)

    # Shuffle mixed-class target rows without changing membership.
    rng = np.random.default_rng(seed + int(round(tf * 10000)) + 777)
    rng.shuffle(adapt_idx)
    rng.shuffle(test_idx)

    if np.intersect1d(adapt_idx, test_idx).size:
        raise RuntimeError("Target adaptation/test overlap.")

    y_adapt = y[adapt_idx]
    y_test = y[test_idx]

    if not np.isclose(np.mean(y_adapt), 0.5):
        raise RuntimeError("Target adaptation prior is not exactly 0.5.")

    if not np.isclose(np.mean(y_test), 0.5):
        raise RuntimeError("Target test prior is not exactly 0.5.")

    return {
        "adapt_indices": adapt_idx,
        "test_indices": test_idx,
        "X_adapt_raw": X[adapt_idx],
        "y_adapt": y_adapt,
        "X_test_raw": X[test_idx],
        "y_test": y_test,
        "benign_threshold": float(benign_threshold),
        "attack_threshold": float(attack_threshold),
        "target_gap_ratio_requested": target_gap_ratio,
        "design_gap_ratio_achieved": float(
            design.achieved_gap_ratios[target_gap_ratio]
        ),
        "design_signed_gap_achieved": float(
            design.achieved_signed_gaps[target_gap_ratio]
        ),
        "selected_tail_fraction": tf,
        "class_stats": class_stats,
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
                    nn.Linear(previous, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden

        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_mlp_logits(
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
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            chunks.append(model(xb).detach().cpu().numpy())

    return np.concatenate(chunks)


def make_weighted_loader(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
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
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    loader = make_weighted_loader(
        X_train,
        y_train,
        np.ones(len(y_train), dtype=np.float32),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_auroc = -np.inf
    best_epoch = -1
    no_improve = 0

    for epoch in range(args.max_epochs):
        model.train()

        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()

        val_probs = sigmoid(
            predict_mlp_logits(model, X_val, device)
        )

        val_auroc = float(roc_auc_score(y_val, val_probs))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auroc": float(best_val_auroc),
        "best_epoch": int(best_epoch),
    }


def weighted_val_auroc(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    device: torch.device,
) -> float:
    probs = sigmoid(predict_mlp_logits(model, X, device))
    return float(
        roc_auc_score(y, probs, sample_weight=weights)
    )


def adapt_mlp_with_source_weights(
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
    """
    Same optimizer path can be used for uniform and IW conditions.
    """
    set_seed(seed)

    model = copy.deepcopy(source_model).to(device)

    loader = make_weighted_loader(
        X_train,
        y_train,
        train_weights.astype(np.float32),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.adapt_lr,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = weighted_val_auroc(
        model, X_val, y_val, val_weights, device
    )
    best_epoch = 0
    no_improve = 0

    for epoch in range(args.adapt_epochs):
        model.train()

        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            losses = criterion(logits, yb)
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()

        val_score = weighted_val_auroc(
            model, X_val, y_val, val_weights, device
        )

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.adapt_patience:
            break

    model.load_state_dict(best_state)

    return model, {
        "best_weighted_val_auroc": float(best_val),
        "best_epoch": int(best_epoch),
    }


def adapt_mlp_with_target_labels(
    source_model: MLP,
    X_target: np.ndarray,
    y_target: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[MLP, dict[str, Any]]:
    """
    Labelled-target reference. Target labels are explicitly allowed here.
    """
    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_target,
        y_target,
        test_size=0.20,
        stratify=y_target,
        random_state=seed,
    )

    set_seed(seed)
    model = copy.deepcopy(source_model).to(device)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.target_lr,
        weight_decay=args.weight_decay,
    )

    initial_probs = sigmoid(
        predict_mlp_logits(model, X_val, device)
    )
    best_val = float(roc_auc_score(y_val, initial_probs))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0

    for epoch in range(args.target_epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_probs = sigmoid(
            predict_mlp_logits(model, X_val, device)
        )
        val_score = float(roc_auc_score(y_val, val_probs))

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.target_patience:
            break

    model.load_state_dict(best_state)

    return model, {
        "best_target_val_auroc": float(best_val),
        "best_epoch": int(best_epoch),
        "target_train_size": int(len(y_train)),
        "target_val_size": int(len(y_val)),
    }


# ============================================================
# Logistic regression
# ============================================================
def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[LogisticRegression, dict[str, Any]]:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)

        clf.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
        )

    conv = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]

    return clf, {
        "converged": len(conv) == 0,
        "iterations": int(np.max(clf.n_iter_)),
        "warnings": sorted(set(conv)),
    }


def classifier_logits(
    classifier_name: str,
    classifier: Any,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if classifier_name == "mlp":
        return predict_mlp_logits(classifier, X, device)

    if classifier_name == "logistic_regression":
        return classifier.decision_function(X).astype(float)

    raise ValueError(classifier_name)


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
# Importance weights
# ============================================================
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    mean = float(np.mean(weights))

    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Invalid importance-weight mean.")

    return weights / mean


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w ** 2))

    if denom <= 0:
        return 0.0

    return float((np.sum(w) ** 2) / denom)


def oracle_conditional_weights(
    scores: np.ndarray,
    labels: np.ndarray,
    benign_threshold: float,
    attack_threshold: float,
    tail_fraction: float,
    benign_direction: str,
    attack_direction: str,
) -> np.ndarray:
    if tail_fraction >= 1.0:
        return np.ones(len(labels), dtype=float)

    def keep_mask(
        class_label: int,
        threshold: float,
        direction: str,
    ) -> np.ndarray:
        class_mask = labels == class_label

        if direction == "upper":
            return class_mask & (scores >= threshold)

        if direction == "lower":
            return class_mask & (scores <= threshold)

        raise ValueError(direction)

    benign_keep = keep_mask(
        0,
        benign_threshold,
        benign_direction,
    )
    attack_keep = keep_mask(
        1,
        attack_threshold,
        attack_direction,
    )

    indicator = (benign_keep | attack_keep).astype(float)

    if np.sum(indicator) == 0:
        raise ValueError("No source rows receive non-zero oracle weights.")

    # With balanced source classes and equal retained class fractions, a common
    # normalization is valid. Empirical mean normalization absorbs finite
    # threshold-sampling differences.
    return normalize_weights(indicator)


def fit_domain_ratio_model(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, dict[str, Any]]:
    n = min(len(X_source), len(X_target))

    if n < 100:
        raise ValueError(
            f"Too few rows for density-ratio model: n={n}."
        )

    rng = np.random.default_rng(seed)

    src_idx = rng.choice(len(X_source), size=n, replace=False)
    tgt_idx = rng.choice(len(X_target), size=n, replace=False)

    X_domain = np.vstack(
        [
            X_source[src_idx],
            X_target[tgt_idx],
        ]
    )
    y_domain = np.concatenate(
        [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
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
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train_s, y_train)

    conv = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]

    test_probs = clf.predict_proba(X_test_s)[:, 1]

    return scaler, clf, {
        "domain_auroc": float(
            roc_auc_score(y_test, test_probs)
        ),
        "converged": len(conv) == 0,
        "iterations": int(np.max(clf.n_iter_)),
        "warnings": sorted(set(conv)),
    }


def ratio_from_domain_model(
    scaler: StandardScaler,
    clf: LogisticRegression,
    X: np.ndarray,
    clip: float,
) -> np.ndarray:
    p_t = clf.predict_proba(
        scaler.transform(X)
    )[:, 1]

    p_t = np.clip(p_t, EPS, 1.0 - EPS)

    # Balanced domain training -> prior odds = 1.
    odds = p_t / (1.0 - p_t)
    odds = np.clip(odds, 0.0, clip)

    return odds


def estimate_unconditional_iw(
    X_source_train: np.ndarray,
    X_target_adapt: np.ndarray,
    X_source_eval_sets: dict[str, np.ndarray],
    seed: int,
    clip: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scaler, clf, info = fit_domain_ratio_model(
        X_source_train,
        X_target_adapt,
        seed,
    )

    raw = {
        name: ratio_from_domain_model(
            scaler, clf, X_eval, clip
        )
        for name, X_eval in X_source_eval_sets.items()
    }

    train_mean = float(np.mean(raw["train"]))
    if train_mean <= 0:
        raise ValueError("Unconditional IW train mean is non-positive.")

    weights = {
        name: values / train_mean
        for name, values in raw.items()
    }

    info = dict(info)
    info.update(
        {
            "train_weight_ess": effective_sample_size(weights["train"]),
            "train_weight_ess_ratio": float(
                effective_sample_size(weights["train"])
                / len(weights["train"])
            ),
            "train_weight_max": float(np.max(weights["train"])),
        }
    )

    return weights, info


def estimate_pseudo_conditional_iw(
    classifier_name: str,
    source_classifier: Any,
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    X_source_val: np.ndarray,
    y_source_val: np.ndarray,
    X_target_adapt: np.ndarray,
    y_target_adapt_hidden: np.ndarray,
    device: torch.device,
    seed: int,
    clip: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    True target labels are supplied ONLY to calculate hidden diagnostics.
    They are never used in the estimator.
    """
    target_probs = classifier_probs(
        classifier_name,
        source_classifier,
        X_target_adapt,
        device,
    )

    pseudo = (target_probs >= 0.5).astype(int)

    pseudo_accuracy = float(
        accuracy_score(
            y_target_adapt_hidden,
            pseudo,
        )
    )

    train_weights = np.zeros(
        len(y_source_train),
        dtype=float,
    )

    val_weights = np.zeros(
        len(y_source_val),
        dtype=float,
    )

    class_info = {}

    for cls in [0, 1]:
        X_source_cls = X_source_train[
            y_source_train == cls
        ]

        X_target_pseudo_cls = X_target_adapt[
            pseudo == cls
        ]

        if len(X_target_pseudo_cls) < 100:
            # Conservative fallback: no class-specific adaptation for this
            # class if pseudo-label support collapses.
            train_weights[y_source_train == cls] = 1.0
            val_weights[y_source_val == cls] = 1.0

            class_info[cls] = {
                "fallback": True,
                "pseudo_target_count": int(len(X_target_pseudo_cls)),
                "domain_auroc": np.nan,
                "converged": False,
                "iterations": np.nan,
            }
            continue

        scaler, clf, info = fit_domain_ratio_model(
            X_source_cls,
            X_target_pseudo_cls,
            seed + cls * 1000,
        )

        train_mask = y_source_train == cls
        val_mask = y_source_val == cls

        train_weights[train_mask] = ratio_from_domain_model(
            scaler,
            clf,
            X_source_train[train_mask],
            clip,
        )

        val_weights[val_mask] = ratio_from_domain_model(
            scaler,
            clf,
            X_source_val[val_mask],
            clip,
        )

        class_info[cls] = {
            "fallback": False,
            "pseudo_target_count": int(len(X_target_pseudo_cls)),
            "domain_auroc": info["domain_auroc"],
            "converged": info["converged"],
            "iterations": info["iterations"],
        }

    # Normalize using source-train mean; apply same factor to validation.
    train_mean = float(np.mean(train_weights))

    if train_mean <= 0:
        raise ValueError("Pseudo-conditional IW train mean is non-positive.")

    train_weights = train_weights / train_mean
    val_weights = val_weights / train_mean

    return {
        "train": train_weights,
        "val": val_weights,
    }, {
        "pseudo_label_accuracy_hidden": pseudo_accuracy,
        "pseudo_predicted_attack_rate": float(np.mean(pseudo)),
        "benign_domain_auroc": class_info[0]["domain_auroc"],
        "attack_domain_auroc": class_info[1]["domain_auroc"],
        "all_domain_classifiers_converged": bool(
            class_info[0]["converged"]
            and class_info[1]["converged"]
        ),
        "benign_fallback": bool(class_info[0]["fallback"]),
        "attack_fallback": bool(class_info[1]["fallback"]),
        "train_weight_ess": effective_sample_size(train_weights),
        "train_weight_ess_ratio": float(
            effective_sample_size(train_weights)
            / len(train_weights)
        ),
        "train_weight_max": float(np.max(train_weights)),
    }


# ============================================================
# Shift diagnostics
# ============================================================
def domain_auroc(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    scaler, clf, info = fit_domain_ratio_model(
        X_source,
        X_target,
        seed,
    )

    return {
        "auroc": info["domain_auroc"],
        "converged": info["converged"],
        "iterations": info["iterations"],
    }


def conditional_shift_diagnostics(
    X_source_test: np.ndarray,
    y_source_test: np.ndarray,
    X_target_test: np.ndarray,
    y_target_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    marginal = domain_auroc(
        X_source_test,
        X_target_test,
        seed + 101,
    )

    benign = domain_auroc(
        X_source_test[y_source_test == 0],
        X_target_test[y_target_test == 0],
        seed + 201,
    )

    attack = domain_auroc(
        X_source_test[y_source_test == 1],
        X_target_test[y_target_test == 1],
        seed + 301,
    )

    return {
        "marginal_domain_auroc": marginal["auroc"],
        "marginal_domain_classifier_converged": marginal["converged"],
        "benign_conditional_domain_auroc": benign["auroc"],
        "benign_conditional_domain_classifier_converged": benign["converged"],
        "attack_conditional_domain_auroc": attack["auroc"],
        "attack_conditional_domain_classifier_converged": attack["converged"],
        "mean_conditional_domain_auroc": float(
            np.mean([benign["auroc"], attack["auroc"]])
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
    y = np.asarray(y)
    probs = np.asarray(probs)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            mask = (probs >= edges[i]) & (probs < edges[i + 1])

        count = int(np.sum(mask))
        if count == 0:
            continue

        confidence = float(np.mean(probs[mask]))
        observed = float(np.mean(y[mask]))

        ece += (count / len(y)) * abs(confidence - observed)

    return float(ece)


def probability_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probs = np.clip(
        np.asarray(probs, dtype=float),
        EPS,
        1.0 - EPS,
    )

    pred = (probs >= threshold).astype(int)

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
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "fpr": float(fpr),
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(
            average_precision_score(y, probs)
        ),
        "brier": float(
            brier_score_loss(y, probs)
        ),
        "log_loss": float(
            log_loss(y, probs, labels=[0, 1])
        ),
        "ece": float(
            expected_calibration_error(y, probs)
        ),
        "true_attack_rate": float(np.mean(y)),
        "predicted_attack_probability_mean": float(
            np.mean(probs)
        ),
        "predicted_attack_rate_at_0_5": float(
            np.mean(pred)
        ),
    }


# ============================================================
# Aggregation
# ============================================================
def summarize_metric_group(
    values: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))
    std = float(
        np.std(values, ddof=1)
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
        "tail_fraction",
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "shift_severity",
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
        "predicted_attack_probability_mean",
        "predicted_attack_rate_at_0_5",
    ]

    rows = []

    for keys, group in per_seed.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in metric_cols:
            mean, std = summarize_metric_group(
                group[metric].to_numpy(dtype=float)
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["classifier", "shift_severity", "method"]
    )


def aggregate_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "tail_fraction",
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "shift_severity",
    ]

    metric_cols = [
        "source_attack_prior",
        "target_attack_prior",
        "attack_prior_change",
        "marginal_domain_auroc",
        "benign_conditional_domain_auroc",
        "attack_conditional_domain_auroc",
        "mean_conditional_domain_auroc",
        "benign_shift_score_smd",
        "attack_shift_score_smd",
        "source_signed_class_score_gap",
        "target_signed_class_score_gap",
        "source_absolute_class_score_gap",
        "target_absolute_class_score_gap",
        "target_to_source_absolute_gap_ratio",
        "oracle_weight_ess_ratio",
        "unconditional_weight_ess_ratio",
    ]

    optional_metric_cols = [
        "mlp_pseudo_label_accuracy_hidden",
        "logistic_regression_pseudo_label_accuracy_hidden",
        "mlp_pseudo_conditional_weight_ess_ratio",
        "logistic_regression_pseudo_conditional_weight_ess_ratio",
        "mlp_pseudo_vs_oracle_weight_correlation",
        "logistic_regression_pseudo_vs_oracle_weight_correlation",
        "unconditional_vs_oracle_weight_correlation",
    ]

    rows = []

    for keys, group in diagnostics.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())

        bool_cols = [
            "marginal_domain_classifier_converged",
            "benign_conditional_domain_classifier_converged",
            "attack_conditional_domain_classifier_converged",
            "unconditional_domain_classifier_converged",
        ]

        for col in bool_cols:
            if col in group.columns:
                row[f"all_{col}"] = bool(group[col].all())

        for metric in metric_cols + optional_metric_cols:
            if metric not in group.columns:
                continue

            mean, std = summarize_metric_group(
                group[metric].to_numpy(dtype=float)
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std

        rows.append(row)

    return pd.DataFrame(rows).sort_values("shift_severity")


def build_paired_differences(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "tail_fraction",
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "shift_severity",
    ]

    metric_cols = [
        "auroc",
        "auprc",
        "f1",
        "brier",
        "log_loss",
        "ece",
    ]

    source = (
        per_seed[
            per_seed["method"] == "source_only"
        ][keys + metric_cols]
        .copy()
        .rename(
            columns={
                m: f"source_{m}"
                for m in metric_cols
            }
        )
    )

    adapted = per_seed[
        per_seed["method"] != "source_only"
    ].copy()

    merged = adapted.merge(
        source,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    for metric in ["auroc", "auprc", "f1"]:
        merged[
            f"{metric}_improvement_vs_source"
        ] = (
            merged[metric]
            - merged[f"source_{metric}"]
        )

    for metric in ["brier", "log_loss", "ece"]:
        merged[
            f"{metric}_improvement_vs_source"
        ] = (
            merged[f"source_{metric}"]
            - merged[metric]
        )

    return merged


def summarize_paired(
    paired: pd.DataFrame,
    reference_name: str,
) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "classifier",
        "method",
        "tail_fraction",
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "shift_severity",
        "supervision_regime",
    ]

    metrics = [
        f"auroc_improvement_vs_{reference_name}",
        f"auprc_improvement_vs_{reference_name}",
        f"f1_improvement_vs_{reference_name}",
        f"brier_improvement_vs_{reference_name}",
        f"log_loss_improvement_vs_{reference_name}",
        f"ece_improvement_vs_{reference_name}",
    ]

    rows = []

    for keys, group in paired.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())

        for metric in metrics:
            if metric not in group.columns:
                continue

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
                    low = high = mean

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["classifier", "shift_severity", "method"]
    )


def build_mlp_iw_vs_uniform(
    per_seed: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "seed",
        "dataset",
        "classifier",
        "tail_fraction",
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "shift_severity",
    ]

    metric_cols = [
        "auroc",
        "auprc",
        "f1",
        "brier",
        "log_loss",
        "ece",
    ]

    uniform = (
        per_seed[
            (per_seed["classifier"] == "mlp")
            & (
                per_seed["method"]
                == "uniform_extra_training"
            )
        ][keys + metric_cols]
        .copy()
        .rename(
            columns={
                m: f"uniform_{m}"
                for m in metric_cols
            }
        )
    )

    compared = per_seed[
        (per_seed["classifier"] == "mlp")
        & (
            per_seed["method"].isin(
                [
                    "unconditional_estimated_iw",
                    "oracle_conditional_iw",
                    "pseudo_conditional_iw",
                ]
            )
        )
    ].copy()

    merged = compared.merge(
        uniform,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    for metric in ["auroc", "auprc", "f1"]:
        merged[
            f"{metric}_improvement_vs_uniform"
        ] = (
            merged[metric]
            - merged[f"uniform_{metric}"]
        )

    for metric in ["brier", "log_loss", "ece"]:
        merged[
            f"{metric}_improvement_vs_uniform"
        ] = (
            merged[f"uniform_{metric}"]
            - merged[metric]
        )

    return merged


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
    plt.figure(figsize=(8.5, 5.2))

    subset = summary[
        summary["classifier"] == classifier_name
    ]

    methods = subset["method"].drop_duplicates().tolist()

    for method in methods:
        m = subset[
            subset["method"] == method
        ].sort_values("shift_severity")

        plt.errorbar(
            m["shift_severity"],
            m[f"{metric}_mean"],
            yerr=m[f"{metric}_std"],
            marker="o",
            capsize=3,
            label=method,
        )

    plt.xlabel("Gap-controlled conditional-shift severity (1 - requested gap ratio)")
    plt.ylabel(ylabel)
    plt.legend()
    savefig(output_path)


def plot_conditional_domain_auroc(
    diagnostics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 5.2))

    for prefix, label in [
        ("benign_conditional_domain_auroc", "Benign conditional"),
        ("attack_conditional_domain_auroc", "Attack conditional"),
        ("marginal_domain_auroc", "Marginal"),
    ]:
        plt.errorbar(
            diagnostics["shift_severity"],
            diagnostics[f"{prefix}_mean"],
            yerr=diagnostics[f"{prefix}_std"],
            marker="o",
            capsize=3,
            label=label,
        )

    plt.axhline(0.5, linestyle="--", label="No separability")
    plt.xlabel("Gap-controlled conditional-shift severity (1 - requested gap ratio)")
    plt.ylabel("Source-vs-target domain AUROC")
    plt.legend()
    savefig(output_path)


# ============================================================
# One seed
# ============================================================
def run_seed(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    design: ShiftDesign,
    target_gap_ratios: list[float],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    print("\n" + "=" * 120)
    print(
        f"CONTROLLED GAP-CONTROLLED CLASS-CONDITIONAL SHIFT | "
        f"dataset={args.dataset} | seed={seed}"
    )
    print("=" * 120)

    split = build_seed_split(
        X=X,
        y=y,
        design=design,
        seed=seed,
        source_train_per_class=args.source_train_per_class,
        source_val_per_class=args.source_val_per_class,
        source_test_per_class=args.source_test_per_class,
        target_adapt_candidate_per_class=args.target_adapt_candidate_per_class,
        target_test_candidate_per_class=args.target_test_candidate_per_class,
    )

    idx_source_train = split["idx_source_train"]
    idx_source_val = split["idx_source_val"]
    idx_source_test = split["idx_source_test"]

    # Shuffle source arrays deterministically.
    rng = np.random.default_rng(seed + 555)

    for key in [
        "idx_source_train",
        "idx_source_val",
        "idx_source_test",
    ]:
        arr = split[key].copy()
        rng.shuffle(arr)
        split[key] = arr

    idx_source_train = split["idx_source_train"]
    idx_source_val = split["idx_source_val"]
    idx_source_test = split["idx_source_test"]

    y_source_train = y[idx_source_train]
    y_source_val = y[idx_source_val]
    y_source_test = y[idx_source_test]

    source_prior = float(np.mean(y_source_train))

    if not np.isclose(source_prior, 0.5):
        raise RuntimeError(
            f"Expected source prior 0.5, got {source_prior}."
        )

    model_scaler = StandardScaler()

    X_source_train = model_scaler.fit_transform(
        X[idx_source_train]
    ).astype(np.float32)

    X_source_val = model_scaler.transform(
        X[idx_source_val]
    ).astype(np.float32)

    X_source_test = model_scaler.transform(
        X[idx_source_test]
    ).astype(np.float32)

    source_train_scores = design.score(
        X[idx_source_train]
    )
    source_val_scores = design.score(
        X[idx_source_val]
    )
    source_test_scores = design.score(
        X[idx_source_test]
    )

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

    source_classifiers = {
        "mlp": source_mlp,
        "logistic_regression": source_logistic,
    }

    source_rows = []

    for classifier_name, classifier in source_classifiers.items():
        probs = classifier_probs(
            classifier_name,
            classifier,
            X_source_test,
            device,
        )

        metrics = probability_metrics(
            y_source_test,
            probs,
        )

        source_rows.append(
            {
                "seed": seed,
                "dataset": args.dataset,
                "classifier": classifier_name,
                "source_attack_prior": source_prior,
                "source_train_size": len(y_source_train),
                "source_val_size": len(y_source_val),
                "source_test_size": len(y_source_test),
                "training_best_epoch": (
                    source_mlp_info["best_epoch"]
                    if classifier_name == "mlp"
                    else np.nan
                ),
                "training_best_val_auroc": (
                    source_mlp_info["best_val_auroc"]
                    if classifier_name == "mlp"
                    else np.nan
                ),
                "training_converged": (
                    True
                    if classifier_name == "mlp"
                    else source_logistic_info["converged"]
                ),
                "training_iterations": (
                    np.nan
                    if classifier_name == "mlp"
                    else source_logistic_info["iterations"]
                ),
                **{
                    f"source_test_{k}": v
                    for k, v in metrics.items()
                },
            }
        )

    result_rows = []
    diagnostic_rows = []

    for condition_index, target_gap_ratio in enumerate(target_gap_ratios):
        requested_gap_ratio = float(target_gap_ratio)
        tf = float(
            design.selected_tail_fractions[requested_gap_ratio]
        )
        achieved_design_gap_ratio = float(
            design.achieved_gap_ratios[requested_gap_ratio]
        )

        # Report severity from the requested gap contraction rather than from
        # the hidden tail fraction used to realize it.
        severity = float(1.0 - requested_gap_ratio)

        target = select_target_condition(
            X=X,
            y=y,
            design=design,
            split=split,
            target_gap_ratio=requested_gap_ratio,
            target_adapt_per_class=args.target_adapt_per_class,
            target_test_per_class=args.target_test_per_class,
            seed=seed,
        )

        X_target_adapt = model_scaler.transform(
            target["X_adapt_raw"]
        ).astype(np.float32)

        X_target_test = model_scaler.transform(
            target["X_test_raw"]
        ).astype(np.float32)

        y_target_adapt = target["y_adapt"]
        y_target_test = target["y_test"]

        target_prior = float(np.mean(y_target_test))
        prior_change = target_prior - source_prior

        if not np.isclose(prior_change, 0.0):
            raise RuntimeError(
                f"Class prior changed unexpectedly: {prior_change}"
            )

        diagnostics = conditional_shift_diagnostics(
            X_source_test,
            y_source_test,
            X_target_test,
            y_target_test,
            seed=seed + condition_index * 1000,
        )

        # Class-specific score SMD.
        target_scores = design.score(
            target["X_test_raw"]
        )

        benign_source_scores = source_test_scores[
            y_source_test == 0
        ]
        attack_source_scores = source_test_scores[
            y_source_test == 1
        ]

        benign_target_scores = target_scores[
            y_target_test == 0
        ]
        attack_target_scores = target_scores[
            y_target_test == 1
        ]

        def smd(a: np.ndarray, b: np.ndarray) -> float:
            pooled = np.std(
                np.concatenate([a, b])
            )
            return float(
                (np.mean(b) - np.mean(a))
                / max(float(pooled), EPS)
            )

        benign_smd = smd(
            benign_source_scores,
            benign_target_scores,
        )
        attack_smd = smd(
            attack_source_scores,
            attack_target_scores,
        )

        source_signed_class_score_gap = float(
            np.mean(attack_source_scores)
            - np.mean(benign_source_scores)
        )
        target_signed_class_score_gap = float(
            np.mean(attack_target_scores)
            - np.mean(benign_target_scores)
        )

        source_absolute_class_score_gap = abs(
            source_signed_class_score_gap
        )
        target_absolute_class_score_gap = abs(
            target_signed_class_score_gap
        )

        target_to_source_absolute_gap_ratio = float(
            target_absolute_class_score_gap
            / max(source_absolute_class_score_gap, EPS)
        )

        # ----------------------------------------------------
        # Oracle conditional IW.
        # ----------------------------------------------------
        oracle_train = oracle_conditional_weights(
            source_train_scores,
            y_source_train,
            target["benign_threshold"],
            target["attack_threshold"],
            tf,
            design.benign_direction,
            design.attack_direction,
        )

        oracle_val = oracle_conditional_weights(
            source_val_scores,
            y_source_val,
            target["benign_threshold"],
            target["attack_threshold"],
            tf,
            design.benign_direction,
            design.attack_direction,
        )

        # ----------------------------------------------------
        # Unconditional estimated IW.
        # ----------------------------------------------------
        unconditional_sets, unconditional_info = estimate_unconditional_iw(
            X_source_train,
            X_target_adapt,
            {
                "train": X_source_train,
                "val": X_source_val,
            },
            seed=seed + condition_index * 1000 + 501,
            clip=args.iw_clip,
        )

        # ----------------------------------------------------
        # Pseudo-conditional IW, separately for each classifier.
        # ----------------------------------------------------
        pseudo_sets_by_classifier = {}
        pseudo_info_by_classifier = {}

        for classifier_name, source_classifier in source_classifiers.items():
            p_sets, p_info = estimate_pseudo_conditional_iw(
                classifier_name=classifier_name,
                source_classifier=source_classifier,
                X_source_train=X_source_train,
                y_source_train=y_source_train,
                X_source_val=X_source_val,
                y_source_val=y_source_val,
                X_target_adapt=X_target_adapt,
                y_target_adapt_hidden=y_target_adapt,
                device=device,
                seed=seed + condition_index * 1000 + 601,
                clip=args.iw_clip,
            )

            pseudo_sets_by_classifier[classifier_name] = p_sets
            pseudo_info_by_classifier[classifier_name] = p_info

        # Weight correlations.
        unconditional_corr = (
            float(
                np.corrcoef(
                    oracle_train,
                    unconditional_sets["train"],
                )[0, 1]
            )
            if (
                np.std(oracle_train) > 0
                and np.std(unconditional_sets["train"]) > 0
            )
            else (1.0 if tf >= 1.0 else np.nan)
        )

        pseudo_corr = {}
        for classifier_name in CLASSIFIERS:
            p_train = pseudo_sets_by_classifier[
                classifier_name
            ]["train"]

            pseudo_corr[classifier_name] = (
                float(
                    np.corrcoef(
                        oracle_train,
                        p_train,
                    )[0, 1]
                )
                if (
                    np.std(oracle_train) > 0
                    and np.std(p_train) > 0
                )
                else (1.0 if tf >= 1.0 else np.nan)
            )

        diagnostic_rows.append(
            {
                "seed": seed,
                "dataset": args.dataset,
                "tail_fraction": tf,
                "requested_gap_ratio": requested_gap_ratio,
                "achieved_design_gap_ratio": achieved_design_gap_ratio,
                "shift_severity": severity,
                "source_attack_prior": source_prior,
                "target_attack_prior": target_prior,
                "attack_prior_change": prior_change,
                "benign_threshold": target["benign_threshold"],
                "attack_threshold": target["attack_threshold"],
                "benign_target_adapt_eligible_fraction": target[
                    "class_stats"
                ][0]["adapt_eligible_fraction"],
                "attack_target_adapt_eligible_fraction": target[
                    "class_stats"
                ][1]["adapt_eligible_fraction"],
                "benign_target_test_eligible_fraction": target[
                    "class_stats"
                ][0]["test_eligible_fraction"],
                "attack_target_test_eligible_fraction": target[
                    "class_stats"
                ][1]["test_eligible_fraction"],
                "benign_shift_score_smd": benign_smd,
                "attack_shift_score_smd": attack_smd,
                "source_signed_class_score_gap": source_signed_class_score_gap,
                "target_signed_class_score_gap": target_signed_class_score_gap,
                "source_absolute_class_score_gap": source_absolute_class_score_gap,
                "target_absolute_class_score_gap": target_absolute_class_score_gap,
                "target_to_source_absolute_gap_ratio": (
                    target_to_source_absolute_gap_ratio
                ),
                "benign_selection_direction": design.benign_direction,
                "attack_selection_direction": design.attack_direction,
                "oracle_weight_ess": effective_sample_size(
                    oracle_train
                ),
                "oracle_weight_ess_ratio": float(
                    effective_sample_size(oracle_train)
                    / len(oracle_train)
                ),
                "unconditional_weight_ess": effective_sample_size(
                    unconditional_sets["train"]
                ),
                "unconditional_weight_ess_ratio": float(
                    effective_sample_size(
                        unconditional_sets["train"]
                    )
                    / len(unconditional_sets["train"])
                ),
                "unconditional_vs_oracle_weight_correlation": unconditional_corr,
                "unconditional_domain_auroc": unconditional_info[
                    "domain_auroc"
                ],
                "unconditional_domain_classifier_converged": unconditional_info[
                    "converged"
                ],
                "mlp_pseudo_label_accuracy_hidden": pseudo_info_by_classifier[
                    "mlp"
                ]["pseudo_label_accuracy_hidden"],
                "logistic_regression_pseudo_label_accuracy_hidden": (
                    pseudo_info_by_classifier[
                        "logistic_regression"
                    ]["pseudo_label_accuracy_hidden"]
                ),
                "mlp_pseudo_conditional_weight_ess_ratio": float(
                    pseudo_info_by_classifier[
                        "mlp"
                    ]["train_weight_ess_ratio"]
                ),
                "logistic_regression_pseudo_conditional_weight_ess_ratio": float(
                    pseudo_info_by_classifier[
                        "logistic_regression"
                    ]["train_weight_ess_ratio"]
                ),
                "mlp_pseudo_vs_oracle_weight_correlation": pseudo_corr[
                    "mlp"
                ],
                "logistic_regression_pseudo_vs_oracle_weight_correlation": (
                    pseudo_corr["logistic_regression"]
                ),
                **diagnostics,
            }
        )

        print(
            f"\nRequested gap ratio={requested_gap_ratio:.2f} "
            f"| achieved(design)={achieved_design_gap_ratio:.3f} "
            f"| selected tail={tf:.3f} "
            f"| severity={severity:.2f} "
            f"| prior={target_prior:.3f} "
            f"| benign domain AUROC="
            f"{diagnostics['benign_conditional_domain_auroc']:.3f} "
            f"| attack domain AUROC="
            f"{diagnostics['attack_conditional_domain_auroc']:.3f} "
            f"| class-gap ratio="
            f"{target_to_source_absolute_gap_ratio:.3f}"
        )

        print(
            "  pseudo accuracy hidden | "
            f"MLP={pseudo_info_by_classifier['mlp']['pseudo_label_accuracy_hidden']:.3f} "
            f"| logistic="
            f"{pseudo_info_by_classifier['logistic_regression']['pseudo_label_accuracy_hidden']:.3f}"
        )

        # ----------------------------------------------------
        # Build adapted MLPs.
        # ----------------------------------------------------
        matched_seed = (
            seed
            + condition_index * 1000
            + 701
        )

        uniform_mlp, _ = adapt_mlp_with_source_weights(
            source_model=source_mlp,
            X_train=X_source_train,
            y_train=y_source_train,
            train_weights=np.ones(len(y_source_train), dtype=float),
            X_val=X_source_val,
            y_val=y_source_val,
            val_weights=np.ones(len(y_source_val), dtype=float),
            args=args,
            device=device,
            seed=matched_seed,
        )

        unconditional_mlp, _ = adapt_mlp_with_source_weights(
            source_model=source_mlp,
            X_train=X_source_train,
            y_train=y_source_train,
            train_weights=unconditional_sets["train"],
            X_val=X_source_val,
            y_val=y_source_val,
            val_weights=unconditional_sets["val"],
            args=args,
            device=device,
            seed=matched_seed,
        )

        oracle_mlp, _ = adapt_mlp_with_source_weights(
            source_model=source_mlp,
            X_train=X_source_train,
            y_train=y_source_train,
            train_weights=oracle_train,
            X_val=X_source_val,
            y_val=y_source_val,
            val_weights=oracle_val,
            args=args,
            device=device,
            seed=matched_seed,
        )

        pseudo_mlp, _ = adapt_mlp_with_source_weights(
            source_model=source_mlp,
            X_train=X_source_train,
            y_train=y_source_train,
            train_weights=pseudo_sets_by_classifier["mlp"]["train"],
            X_val=X_source_val,
            y_val=y_source_val,
            val_weights=pseudo_sets_by_classifier["mlp"]["val"],
            args=args,
            device=device,
            seed=matched_seed,
        )

        target_labeled_mlp, _ = adapt_mlp_with_target_labels(
            source_model=source_mlp,
            X_target=X_target_adapt,
            y_target=y_target_adapt,
            args=args,
            device=device,
            seed=seed + condition_index * 1000 + 801,
        )

        # ----------------------------------------------------
        # Logistic adaptations.
        # ----------------------------------------------------
        unconditional_logistic, unconditional_logistic_info = fit_logistic(
            X_source_train,
            y_source_train,
            seed=seed + condition_index * 1000 + 901,
            sample_weight=unconditional_sets["train"],
        )

        oracle_logistic, oracle_logistic_info = fit_logistic(
            X_source_train,
            y_source_train,
            seed=seed + condition_index * 1000 + 902,
            sample_weight=oracle_train,
        )

        pseudo_logistic, pseudo_logistic_info = fit_logistic(
            X_source_train,
            y_source_train,
            seed=seed + condition_index * 1000 + 903,
            sample_weight=pseudo_sets_by_classifier[
                "logistic_regression"
            ]["train"],
        )

        (
            X_target_label_train,
            X_target_label_val,
            y_target_label_train,
            y_target_label_val,
        ) = train_test_split(
            X_target_adapt,
            y_target_adapt,
            test_size=0.20,
            stratify=y_target_adapt,
            random_state=seed + condition_index * 1000 + 904,
        )

        target_labeled_logistic, target_labeled_logistic_info = fit_logistic(
            X_target_label_train,
            y_target_label_train,
            seed=seed + condition_index * 1000 + 905,
        )

        # ----------------------------------------------------
        # Evaluate.
        # ----------------------------------------------------
        models_by_classifier = {
            "mlp": {
                "source_only": source_mlp,
                "uniform_extra_training": uniform_mlp,
                "unconditional_estimated_iw": unconditional_mlp,
                "oracle_conditional_iw": oracle_mlp,
                "pseudo_conditional_iw": pseudo_mlp,
                "target_labeled_adaptation": target_labeled_mlp,
            },
            "logistic_regression": {
                "source_only": source_logistic,
                "unconditional_estimated_iw": unconditional_logistic,
                "oracle_conditional_iw": oracle_logistic,
                "pseudo_conditional_iw": pseudo_logistic,
                "target_labeled_adaptation": target_labeled_logistic,
            },
        }

        for classifier_name in CLASSIFIERS:
            for method, classifier in models_by_classifier[
                classifier_name
            ].items():
                probs = classifier_probs(
                    classifier_name,
                    classifier,
                    X_target_test,
                    device,
                )

                metrics = probability_metrics(
                    y_target_test,
                    probs,
                )

                if method == "target_labeled_adaptation":
                    supervision = "labelled_target"
                elif method == "oracle_conditional_iw":
                    supervision = "oracle_synthetic_reference"
                else:
                    supervision = "unlabelled_target"

                result_rows.append(
                    {
                        "seed": seed,
                        "dataset": args.dataset,
                        "classifier": classifier_name,
                        "method": method,
                        "supervision_regime": supervision,
                        "tail_fraction": tf,
                        "requested_gap_ratio": requested_gap_ratio,
                        "achieved_design_gap_ratio": achieved_design_gap_ratio,
                        "shift_severity": severity,
                        "source_attack_prior": source_prior,
                        "target_attack_prior": target_prior,
                        "attack_prior_change": prior_change,
                        "marginal_domain_auroc": diagnostics[
                            "marginal_domain_auroc"
                        ],
                        "benign_conditional_domain_auroc": diagnostics[
                            "benign_conditional_domain_auroc"
                        ],
                        "attack_conditional_domain_auroc": diagnostics[
                            "attack_conditional_domain_auroc"
                        ],
                        "oracle_weight_ess_ratio": float(
                            effective_sample_size(oracle_train)
                            / len(oracle_train)
                        ),
                        "unconditional_weight_ess_ratio": float(
                            effective_sample_size(
                                unconditional_sets["train"]
                            )
                            / len(unconditional_sets["train"])
                        ),
                        "pseudo_conditional_weight_ess_ratio": float(
                            pseudo_info_by_classifier[
                                classifier_name
                            ]["train_weight_ess_ratio"]
                        ),
                        "pseudo_label_accuracy_hidden": float(
                            pseudo_info_by_classifier[
                                classifier_name
                            ]["pseudo_label_accuracy_hidden"]
                        ),
                        **metrics,
                    }
                )

    return result_rows, diagnostic_rows, source_rows


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
        "--target_gap_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_GAP_RATIOS,
        help=(
            "Desired retained class-separation ratios on held-out design data. "
            "Example: 1.0 0.7 0.4 0.1."
        ),
    )

    parser.add_argument(
        "--tail_search_min",
        type=float,
        default=DEFAULT_TAIL_SEARCH_MIN,
    )

    parser.add_argument(
        "--tail_search_max",
        type=float,
        default=DEFAULT_TAIL_SEARCH_MAX,
    )

    parser.add_argument(
        "--tail_search_step",
        type=float,
        default=DEFAULT_TAIL_SEARCH_STEP,
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
        "--target_adapt_candidate_per_class",
        type=int,
        default=DEFAULT_TARGET_ADAPT_CANDIDATE_PER_CLASS,
    )

    parser.add_argument(
        "--target_test_candidate_per_class",
        type=int,
        default=DEFAULT_TARGET_TEST_CANDIDATE_PER_CLASS,
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

    target_gap_ratios = sorted(
        set(float(x) for x in args.target_gap_ratios),
        reverse=True,
    )

    if any(
        x <= 0.0 or x > 1.0
        for x in target_gap_ratios
    ):
        raise ValueError(
            "target_gap_ratios must lie in (0,1]."
        )

    if 1.0 not in target_gap_ratios:
        raise ValueError(
            "target_gap_ratios must include 1.0 as the no-shift control."
        )

    if not (
        0.0 < args.tail_search_min
        <= args.tail_search_max
        <= 1.0
    ):
        raise ValueError(
            "Require 0 < tail_search_min <= tail_search_max <= 1."
        )

    if args.tail_search_step <= 0:
        raise ValueError(
            "tail_search_step must be positive."
        )

    print("=" * 120)
    print("CONTROLLED GAP-CONTROLLED CLASS-CONDITIONAL SHIFT EXPERIMENT")
    print("=" * 120)
    print(f"Dataset: {args.dataset}")
    print(f"Pool: {args.pool_variant}")
    print(f"Requested design gap ratios: {target_gap_ratios}")
    print(f"Seeds: {args.seeds}")

    X, y, feature_names = load_dataset_pool(
        args.dataset,
        args.pool_variant,
    )

    print(
        f"Loaded: {len(y):,} rows "
        f"| benign={(y == 0).sum():,} "
        f"| attack={(y == 1).sum():,} "
        f"| features={X.shape[1]}"
    )

    design = build_shift_design(
        X=X,
        y=y,
        target_gap_ratios=target_gap_ratios,
        design_per_class=args.design_per_class,
        tail_search_min=args.tail_search_min,
        tail_search_max=args.tail_search_max,
        tail_search_step=args.tail_search_step,
    )

    print(
        "Shift-design PC1 explained variance ratio: "
        f"{design.pca.explained_variance_ratio_[0]:.6f}"
    )
    print(
        "Design class score means | "
        f"benign={design.benign_design_score_mean:.6f} "
        f"| attack={design.attack_design_score_mean:.6f}"
    )
    print(
        "Gap-reduction orientation | "
        f"benign={design.benign_direction} "
        f"| attack={design.attack_direction}"
    )
    print("Design-selected conditions:")
    for requested_ratio in target_gap_ratios:
        print(
            f"  requested gap={requested_ratio:.2f} "
            f"| achieved={design.achieved_gap_ratios[requested_ratio]:.4f} "
            f"| tail={design.selected_tail_fractions[requested_ratio]:.4f} "
            f"| signed_gap={design.achieved_signed_gaps[requested_ratio]:.4f}"
        )

    output_dir = (
        STAGE1_DIR
        / "controlled_class_conditional_shift"
        / args.dataset
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    direction_path = (
        output_dir
        / "controlled_class_conditional_shift_direction.csv"
    )

    shift_loading_table(
        design,
        feature_names,
    ).to_csv(
        direction_path,
        index=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    all_results = []
    all_diagnostics = []
    all_source = []

    for seed in args.seeds:
        results, diagnostics, source = run_seed(
            seed=int(seed),
            X=X,
            y=y,
            design=design,
            target_gap_ratios=target_gap_ratios,
            args=args,
            device=device,
        )

        all_results.extend(results)
        all_diagnostics.extend(diagnostics)
        all_source.extend(source)

    per_seed_df = pd.DataFrame(all_results)
    diagnostics_df = pd.DataFrame(all_diagnostics)
    source_df = pd.DataFrame(all_source)

    summary_df = aggregate_results(per_seed_df)
    diagnostics_summary_df = aggregate_diagnostics(
        diagnostics_df
    )

    paired_df = build_paired_differences(per_seed_df)

    paired_summary_df = summarize_paired(
        paired_df,
        reference_name="source",
    )

    mlp_iw_vs_uniform_df = build_mlp_iw_vs_uniform(
        per_seed_df
    )

    # Reuse paired summarizer by renaming reference suffix.
    mlp_uniform_for_summary = mlp_iw_vs_uniform_df.copy()

    mlp_iw_vs_uniform_summary_df = summarize_paired(
        mlp_uniform_for_summary,
        reference_name="uniform",
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------
    paths = {
        "per_seed": output_dir
        / "controlled_class_conditional_shift_per_seed.csv",
        "summary": output_dir
        / "controlled_class_conditional_shift_summary.csv",
        "source": output_dir
        / "controlled_class_conditional_shift_source_metrics.csv",
        "diagnostics": output_dir
        / "controlled_class_conditional_shift_diagnostics_per_seed.csv",
        "diagnostics_summary": output_dir
        / "controlled_class_conditional_shift_diagnostics_summary.csv",
        "paired": output_dir
        / "controlled_class_conditional_shift_paired_differences.csv",
        "paired_summary": output_dir
        / "controlled_class_conditional_shift_paired_difference_summary.csv",
        "mlp_uniform": output_dir
        / "controlled_class_conditional_shift_mlp_iw_vs_uniform.csv",
        "mlp_uniform_summary": output_dir
        / "controlled_class_conditional_shift_mlp_iw_vs_uniform_summary.csv",
    }

    per_seed_df.to_csv(paths["per_seed"], index=False)
    summary_df.to_csv(paths["summary"], index=False)
    source_df.to_csv(paths["source"], index=False)
    diagnostics_df.to_csv(paths["diagnostics"], index=False)
    diagnostics_summary_df.to_csv(
        paths["diagnostics_summary"],
        index=False,
    )
    paired_df.to_csv(paths["paired"], index=False)
    paired_summary_df.to_csv(
        paths["paired_summary"],
        index=False,
    )
    mlp_iw_vs_uniform_df.to_csv(
        paths["mlp_uniform"],
        index=False,
    )
    mlp_iw_vs_uniform_summary_df.to_csv(
        paths["mlp_uniform_summary"],
        index=False,
    )

    # --------------------------------------------------------
    # Protocol.
    # --------------------------------------------------------
    protocol = {
        "research_question": (
            "How do unconditional and class-aware adaptation strategies behave "
            "under controlled class-conditional shift with fixed class prior?"
        ),
        "hypotheses": {
            "H1": (
                "Class prior remains fixed while within-class domain separability "
                "increases and class separation is reduced without crossing."
            ),
            "H2": (
                "Source-only target performance should degrade as gap-controlled "
                "conditional shift reduces class separation along the "
                "controlled score axis without reversing class order."
            ),
            "H3": (
                "Oracle conditional IW can outperform unconditional IW when "
                "the model is sensitive to class-conditional redistribution."
            ),
            "H4": (
                "Pseudo-conditional IW approaches oracle conditional IW when "
                "pseudo-label quality and density-ratio estimation are good."
            ),
            "H5": (
                "Unconditional IW can be insufficient under class-conditional "
                "shift because it ignores class-specific redistribution."
            ),
            "H6": (
                "Adaptation benefit is model-dependent."
            ),
        },
        "dataset": args.dataset,
        "pool_variant": args.pool_variant,
        "seeds": [int(s) for s in args.seeds],
        "requested_gap_ratios": target_gap_ratios,
        "selected_tail_fractions": {
            str(r): float(design.selected_tail_fractions[r])
            for r in target_gap_ratios
        },
        "achieved_design_gap_ratios": {
            str(r): float(design.achieved_gap_ratios[r])
            for r in target_gap_ratios
        },
        "achieved_design_signed_gaps": {
            str(r): float(design.achieved_signed_gaps[r])
            for r in target_gap_ratios
        },
        "shift_severities": [
            float(1.0 - x)
            for x in target_gap_ratios
        ],
        "construction": {
            "source_prior": 0.5,
            "target_prior": 0.5,
            "score": (
                "PC1 of StandardScaler-transformed fixed design X"
            ),
            "design_seed": DESIGN_SEED,
            "design_per_class": args.design_per_class,
            "pca_uses_labels": False,
            "thresholds_use_labels_for_controlled_construction": True,
            "selection_mode": "gap_controlled_non_crossing",
            "design_benign_score_mean": float(
                design.benign_design_score_mean
            ),
            "design_attack_score_mean": float(
                design.attack_design_score_mean
            ),
            "selection": {
                "benign": (
                    f"{design.benign_direction} tail of its within-class "
                    "score distribution"
                ),
                "attack": (
                    f"{design.attack_direction} tail of its within-class "
                    "score distribution"
                ),
            },
            "orientation_rule": (
                "The lower-mean class on design PC1 is shifted toward the "
                "higher-mean class and vice versa. Tail fractions are selected "
                "on held-out design data to approximate requested class-gap "
                "ratios while preserving the original signed class ordering."
            ),
            "non_crossing_guardrail": (
                "Candidate conditions that reverse the sign of the design-set "
                "class gap are rejected."
            ),
            "important_note": (
                "Changing P(X|Y) with P(Y) fixed generally also changes "
                "P(Y|X); this is called controlled class-conditional shift "
                "rather than pure isolated class-conditional shift."
            ),
        },
        "sizes_per_class": {
            "source_train": args.source_train_per_class,
            "source_validation": args.source_val_per_class,
            "source_test": args.source_test_per_class,
            "target_adaptation_candidate": (
                args.target_adapt_candidate_per_class
            ),
            "target_test_candidate": (
                args.target_test_candidate_per_class
            ),
            "target_adaptation_selected": (
                args.target_adapt_per_class
            ),
            "target_test_selected": (
                args.target_test_per_class
            ),
        },
        "methods": {
            "source_only": {
                "target_labels_used": False,
            },
            "uniform_extra_training": {
                "classifier": "mlp_only",
                "target_labels_used": False,
                "purpose": "matched extra-optimization control",
            },
            "unconditional_estimated_iw": {
                "target_labels_used": False,
                "purpose": (
                    "marginal/covariate-shift baseline that ignores classes"
                ),
            },
            "oracle_conditional_iw": {
                "target_labels_used": False,
                "oracle_information": (
                    "known synthetic class-specific selection rule"
                ),
            },
            "pseudo_conditional_iw": {
                "target_labels_used": False,
                "uses": (
                    "source-model pseudo-labels on target adaptation X"
                ),
            },
            "target_labeled_adaptation": {
                "target_labels_used": True,
                "comparison_note": (
                    "labelled-target reference; not supervision-comparable "
                    "to unsupervised methods"
                ),
            },
        },
        "mlp": {
            "architecture": [
                int(X.shape[1]),
                *[int(h) for h in args.hidden_dims],
                1,
            ],
            "dropout": args.dropout,
            "source_lr": args.lr,
            "adapt_lr": args.adapt_lr,
            "target_lr": args.target_lr,
        },
        "leakage_controls": [
            "Design rows excluded from all model data.",
            "Converging orientation is selected once from held-out design data only.",
            "Source and target rows are disjoint.",
            "Target adaptation and test banks are disjoint.",
            "Target prior is exactly 0.5 at every severity.",
            "NIDS scaler fits source train only.",
            "Pseudo-conditional IW never uses true target labels.",
            "Hidden target-adapt labels are used only to score pseudo-label quality.",
            "Target test labels are evaluation only.",
            "Labelled-target reference never accesses target test labels.",
            "MLP weighted methods use matched adaptation seeds.",
        ],
        "statistics": (
            "Five default seeds; paired method-vs-source and MLP IW-vs-uniform "
            "differences with mean, SD and descriptive t-based 95% CI."
        ),
    }

    protocol_path = (
        output_dir
        / "controlled_class_conditional_shift_protocol.json"
    )

    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    # --------------------------------------------------------
    # Plots.
    # --------------------------------------------------------
    for classifier_name in CLASSIFIERS:
        for metric, ylabel in [
            ("auroc", "Target AUROC"),
            ("auprc", "Target AUPRC"),
            ("f1", "Target F1 @ 0.5"),
            ("brier", "Brier score (lower is better)"),
        ]:
            plot_metric(
                summary_df,
                classifier_name,
                metric,
                ylabel,
                output_dir
                / (
                    f"class_conditional_shift_"
                    f"{classifier_name}_{metric}.png"
                ),
            )

    plot_conditional_domain_auroc(
        diagnostics_summary_df,
        output_dir
        / "class_conditional_shift_domain_separability.png",
    )

    # --------------------------------------------------------
    # Console.
    # --------------------------------------------------------
    print("\n" + "=" * 135)
    print("CONTROLLED GAP-CONTROLLED CLASS-CONDITIONAL SHIFT COMPLETE")
    print("=" * 135)

    print("\nSHIFT DIAGNOSTICS")

    diagnostic_cols = [
        "requested_gap_ratio",
        "achieved_design_gap_ratio",
        "tail_fraction",
        "shift_severity",
        "target_attack_prior_mean",
        "marginal_domain_auroc_mean",
        "benign_conditional_domain_auroc_mean",
        "attack_conditional_domain_auroc_mean",
        "benign_shift_score_smd_mean",
        "attack_shift_score_smd_mean",
        "source_absolute_class_score_gap_mean",
        "target_absolute_class_score_gap_mean",
        "target_to_source_absolute_gap_ratio_mean",
        "oracle_weight_ess_ratio_mean",
        "unconditional_weight_ess_ratio_mean",
    ]

    optional_cols = [
        "mlp_pseudo_label_accuracy_hidden_mean",
        "logistic_regression_pseudo_label_accuracy_hidden_mean",
    ]

    diagnostic_cols += [
        c
        for c in optional_cols
        if c in diagnostics_summary_df.columns
    ]

    print(
        diagnostics_summary_df[
            diagnostic_cols
        ].to_string(index=False)
    )

    print("\nTARGET PERFORMANCE")

    print(
        summary_df[
            [
                "classifier",
                "method",
                "supervision_regime",
                "tail_fraction",
                "shift_severity",
                "auroc_mean",
                "auprc_mean",
                "f1_mean",
                "brier_mean",
            ]
        ].to_string(index=False)
    )

    print("\nGUARDRAILS")
    print(
        "- Target prior must remain exactly 0.5 at every severity."
    )
    print(
        "- Benign/attack conditional domain AUROCs should rise with severity."
    )
    print(
        "- Do not compare labelled-target adaptation as if it had the same supervision budget as IW."
    )
    print(
        "- For MLP, interpret weighted adaptation relative to uniform_extra_training."
    )
    print(
        "- Pseudo-label accuracy is a hidden diagnostic only; the pseudo-IW method does not use true target labels."
    )

    print("\nOUTPUTS")
    for path in [
        paths["summary"],
        paths["source"],
        paths["diagnostics_summary"],
        paths["paired_summary"],
        paths["mlp_uniform_summary"],
        paths["per_seed"],
        paths["diagnostics"],
        paths["paired"],
        paths["mlp_uniform"],
        direction_path,
        protocol_path,
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
