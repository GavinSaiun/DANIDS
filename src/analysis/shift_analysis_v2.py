"""
Research-grade shift characterisation for DANIDS.

Uses STAGE1_DIR/full so original class priors are preserved. It reports
class-prior shift separately from P(X) and P(X|Y) shift, and deliberately
does not call a prior change "pure label shift" unless the class-conditional
distributions are approximately invariant.

Run:
    python -B -m src.analysis.shift_analysis_v2
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, STAGE1_DIR

DEFAULT_SAMPLE_PER_DOMAIN = 20_000
DEFAULT_SAMPLE_PER_CLASS = 10_000
DEFAULT_BOOTSTRAP_REPS = 50
DEFAULT_BOOTSTRAP_N = 3_000
DEFAULT_DOMAIN_REPEATS = 10
DEFAULT_MMD_SAMPLES = 1_500
DEFAULT_TOP_K_FEATURES = 8
LOWER_CLIP_Q = 0.01
UPPER_CLIP_Q = 0.99

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def short_dataset(name: str) -> str:
    return (
        name.replace("NF-", "")
        .replace("-v3", "")
        .replace("UNSW-NB15", "UNSW")
        .replace("ToN-IoT", "ToN")
        .replace("CSE-CIC-IDS2018", "CIC")
    )


def short_pair(source: str, target: str) -> str:
    return f"{short_dataset(source)}→{short_dataset(target)}"


def load_stage1_summary() -> dict[str, Any]:
    path = STAGE1_DIR / "stage1_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Stage 1 first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_names() -> list[str]:
    path = STAGE1_DIR / "common_features.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run Stage 1 first.")
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f)["common_feature_columns"])


# ------------------------------------------------------------------
# Exact P(Y) statistics from the FULL labels
# ------------------------------------------------------------------
def wilson_interval(positives: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return np.nan, np.nan
    p = positives / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = z * math.sqrt(
        p * (1.0 - p) / total + z2 / (4.0 * total * total)
    ) / denom
    return centre - half, centre + half


def prior_difference_stats(ys: np.ndarray, yt: np.ndarray) -> dict[str, float]:
    ns, nt = int(len(ys)), int(len(yt))
    source_attack, target_attack = int(np.sum(ys == 1)), int(np.sum(yt == 1))
    ps, pt = source_attack / ns, target_attack / nt
    ps_lo, ps_hi = wilson_interval(source_attack, ns)
    pt_lo, pt_hi = wilson_interval(target_attack, nt)
    signed_delta = pt - ps
    se = math.sqrt(ps * (1.0 - ps) / ns + pt * (1.0 - pt) / nt)
    z = 1.959963984540054
    return {
        "source_rows_full": ns,
        "target_rows_full": nt,
        "source_attack_count": source_attack,
        "target_attack_count": target_attack,
        "source_attack_prior": float(ps),
        "source_attack_prior_ci95_low": float(ps_lo),
        "source_attack_prior_ci95_high": float(ps_hi),
        "target_attack_prior": float(pt),
        "target_attack_prior_ci95_low": float(pt_lo),
        "target_attack_prior_ci95_high": float(pt_hi),
        "signed_attack_prior_change_target_minus_source": float(signed_delta),
        "absolute_attack_prior_difference": float(abs(signed_delta)),
        "signed_prior_change_ci95_low": float(signed_delta - z * se),
        "signed_prior_change_ci95_high": float(signed_delta + z * se),
    }


# ------------------------------------------------------------------
# Bounded-memory uniform priority reservoir sampling
# ------------------------------------------------------------------
class PriorityReservoir:
    def __init__(self, capacity: int, n_features: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.keys = np.empty(0, dtype=np.float64)
        self.indices = np.empty(0, dtype=np.int64)
        self.X = np.empty((0, n_features), dtype=np.float32)

    def update(self, X_batch: np.ndarray, global_indices: np.ndarray) -> None:
        if len(X_batch) == 0 or self.capacity <= 0:
            return
        keys = self.rng.random(len(X_batch))
        take = min(self.capacity, len(X_batch))
        if len(X_batch) > take:
            local = np.argpartition(keys, -take)[-take:]
            keys = keys[local]
            global_indices = global_indices[local]
            X_batch = X_batch[local]

        merged_keys = np.concatenate([self.keys, keys])
        merged_indices = np.concatenate([self.indices, global_indices])
        merged_X = np.vstack([self.X, X_batch])
        keep = min(self.capacity, len(merged_keys))
        chosen = (
            np.argpartition(merged_keys, -keep)[-keep:]
            if len(merged_keys) > keep
            else np.arange(len(merged_keys))
        )
        self.keys = merged_keys[chosen]
        self.indices = merged_indices[chosen]
        self.X = merged_X[chosen]

    def result(self):
        order = np.argsort(self.indices)
        return self.X[order], self.indices[order]


def build_or_load_dataset_sample(
    dataset_name: str,
    feature_names: list[str],
    cache_dir: Path,
    sample_per_domain: int,
    sample_per_class: int,
    batch_size: int,
    rebuild_cache: bool,
) -> dict[str, np.ndarray]:
    cache_path = cache_dir / f"{dataset_name}_shift_sample.npz"
    required = {
        "X_marginal", "y_marginal", "X_benign", "X_attack",
        "idx_marginal", "idx_benign", "idx_attack",
    }

    if cache_path.exists() and not rebuild_cache:
        cached = np.load(cache_path)
        if required.issubset(cached.files):
            print(f"Loading cached full-data sample: {cache_path}")
            return {k: cached[k] for k in cached.files}

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required for full-data bounded-memory sampling.") from exc

    full_dir = STAGE1_DIR / "full"
    x_path = full_dir / f"{dataset_name}_X.parquet"
    y_path = full_dir / f"{dataset_name}_y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing full Stage-1 files for {dataset_name}")

    y = np.load(y_path, mmap_mode="r")
    parquet = pq.ParquetFile(x_path)
    if parquet.metadata.num_rows != len(y):
        raise ValueError(
            f"Row mismatch for {dataset_name}: Parquet={parquet.metadata.num_rows}, y={len(y)}"
        )

    p = len(feature_names)
    marginal = PriorityReservoir(sample_per_domain, p, RANDOM_STATE + 11)
    benign = PriorityReservoir(sample_per_class, p, RANDOM_STATE + 23)
    attack = PriorityReservoir(sample_per_class, p, RANDOM_STATE + 37)

    print(f"Scanning full Stage-1 data for {dataset_name} ({len(y):,} rows) ...")
    offset = 0
    for batch_number, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=feature_names, use_threads=True),
        start=1,
    ):
        Xb = batch.to_pandas().to_numpy(dtype=np.float32, copy=False)
        n = len(Xb)
        idx = np.arange(offset, offset + n, dtype=np.int64)
        yb = np.asarray(y[offset : offset + n])
        marginal.update(Xb, idx)
        m0, m1 = yb == 0, yb == 1
        if np.any(m0):
            benign.update(Xb[m0], idx[m0])
        if np.any(m1):
            attack.update(Xb[m1], idx[m1])
        offset += n
        if batch_number % 25 == 0:
            print(f"  processed {offset:,}/{len(y):,} rows")

    X_m, idx_m = marginal.result()
    X_b, idx_b = benign.result()
    X_a, idx_a = attack.result()
    result = {
        "X_marginal": X_m.astype(np.float32),
        "y_marginal": np.asarray(y[idx_m], dtype=np.int8),
        "X_benign": X_b.astype(np.float32),
        "X_attack": X_a.astype(np.float32),
        "idx_marginal": idx_m,
        "idx_benign": idx_b,
        "idx_attack": idx_a,
    }
    np.savez_compressed(cache_path, **result)
    print(f"Saved sample cache: {cache_path}")
    return result


# ------------------------------------------------------------------
# Symmetric pooled transform for DESCRIPTIVE shift measurement
# ------------------------------------------------------------------
class PairStandardTransform:
    """
    Symmetric transform used only for descriptive shift measurement.

    Pool source + target marginal samples, clip each feature to pooled
    q01/q99 limits, then standardise using pooled mean/std.

    StandardScaler is used instead of IQR scaling because sparse NetFlow
    features can have an IQR near zero, producing extreme transformed values.

    This is NOT the source-only scaler used by the NIDS training pipeline.
    """

    def __init__(self):
        self.lo = None
        self.hi = None
        self.scaler = StandardScaler()

    def fit(self, Xs: np.ndarray, Xt: np.ndarray):
        pooled = np.vstack([Xs, Xt]).astype(np.float64)
        self.lo = np.quantile(pooled, LOWER_CLIP_Q, axis=0)
        self.hi = np.quantile(pooled, UPPER_CLIP_Q, axis=0)
        clipped = np.clip(pooled, self.lo, self.hi)
        self.scaler.fit(clipped)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        clipped = np.clip(X.astype(np.float64), self.lo, self.hi)
        return self.scaler.transform(clipped).astype(np.float32)


# ------------------------------------------------------------------
# Shift metrics
# ------------------------------------------------------------------
def feature_wasserstein(Xs, Xt, feature_names, top_k):
    d = np.array(
        [wasserstein_distance(Xs[:, j], Xt[:, j]) for j in range(Xs.shape[1])],
        dtype=float,
    )
    top = np.argsort(d)[-top_k:][::-1]
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
        "per_feature": d.tolist(),
        "top_features": [
            {
                "feature_index": int(j),
                "feature_name": feature_names[j],
                "wasserstein": float(d[j]),
            }
            for j in top
        ],
    }


def bootstrap_wasserstein_ci(Xs, Xt, reps, bootstrap_n, seed):
    """Bootstrap 95% intervals for mean and median feature-wise Wasserstein."""
    if reps <= 0:
        return {
            "mean_ci95_low": np.nan,
            "mean_ci95_high": np.nan,
            "median_ci95_low": np.nan,
            "median_ci95_high": np.nan,
            "bootstrap_reps": 0,
            "bootstrap_n_per_domain": 0,
        }

    rng = np.random.default_rng(seed)
    n = min(bootstrap_n, len(Xs), len(Xt))
    mean_scores = []
    median_scores = []

    for _ in range(reps):
        xs = Xs[rng.integers(0, len(Xs), size=n)]
        xt = Xt[rng.integers(0, len(Xt), size=n)]
        per_feature = np.asarray(
            [
                wasserstein_distance(xs[:, j], xt[:, j])
                for j in range(xs.shape[1])
            ],
            dtype=float,
        )
        mean_scores.append(float(np.mean(per_feature)))
        median_scores.append(float(np.median(per_feature)))

    return {
        "mean_ci95_low": float(np.percentile(mean_scores, 2.5)),
        "mean_ci95_high": float(np.percentile(mean_scores, 97.5)),
        "median_ci95_low": float(np.percentile(median_scores, 2.5)),
        "median_ci95_high": float(np.percentile(median_scores, 97.5)),
        "bootstrap_reps": int(reps),
        "bootstrap_n_per_domain": int(n),
    }


def covariance_discrepancy(Xs, Xt):
    cs, ct = np.cov(Xs, rowvar=False), np.cov(Xt, rowvar=False)
    diff = np.linalg.norm(cs - ct, ord="fro")
    ns, nt = np.linalg.norm(cs, ord="fro"), np.linalg.norm(ct, ord="fro")
    return {
        "frobenius_difference": float(diff),
        "relative_frobenius_difference": float(diff / (ns + nt + 1e-12)),
        "source_frobenius": float(ns),
        "target_frobenius": float(nt),
    }


def repeated_domain_classifier(Xs, Xt, repeats, seed):
    """
    Repeated classifier two-sample test with explicit convergence diagnostics.

    A StandardScaler is fit on each classifier TRAIN split only, then applied
    to its held-out test split. This is an optimisation safeguard.
    """
    n = min(len(Xs), len(Xt))
    rng = np.random.default_rng(seed)

    si = rng.choice(len(Xs), size=n, replace=False)
    ti = rng.choice(len(Xt), size=n, replace=False)

    X = np.vstack([Xs[si], Xt[ti]])
    d = np.concatenate(
        [np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)]
    )

    scores = []
    iterations = []
    converged_flags = []
    warning_messages = []
    max_iter = 5000

    for r in range(repeats):
        split_seed = seed + r

        Xtr, Xte, dtr, dte = train_test_split(
            X,
            d,
            test_size=0.30,
            stratify=d,
            random_state=split_seed,
        )

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=split_seed,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Xtr, dtr)

        this_warnings = [
            str(w.message)
            for w in caught
            if issubclass(w.category, ConvergenceWarning)
        ]

        n_iter = int(np.max(clf.n_iter_))
        converged = len(this_warnings) == 0 and n_iter < max_iter

        scores.append(
            float(roc_auc_score(dte, clf.predict_proba(Xte)[:, 1]))
        )
        iterations.append(n_iter)
        converged_flags.append(bool(converged))
        warning_messages.extend(this_warnings)

    return {
        "mean_auroc": float(np.mean(scores)),
        "std_auroc": float(
            np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        ),
        "empirical_95_low": float(np.percentile(scores, 2.5)),
        "empirical_95_high": float(np.percentile(scores, 97.5)),
        "repeats": int(repeats),
        "samples_per_domain": int(n),
        "all_aurocs": scores,
        "all_converged": bool(all(converged_flags)),
        "convergence_rate": float(np.mean(converged_flags)),
        "converged_flags": converged_flags,
        "iterations": iterations,
        "max_iterations_observed": int(max(iterations)),
        "configured_max_iter": int(max_iter),
        "num_convergence_warnings": int(len(warning_messages)),
        "warning_messages_unique": sorted(set(warning_messages)),
    }


def rbf_mmd_squared(Xs, Xt, max_samples, seed):
    rng = np.random.default_rng(seed)
    ns, nt = min(max_samples, len(Xs)), min(max_samples, len(Xt))
    xs = Xs[rng.choice(len(Xs), size=ns, replace=False)].astype(np.float64)
    xt = Xt[rng.choice(len(Xt), size=nt, replace=False)].astype(np.float64)
    pooled = np.vstack([xs, xt])
    hn = min(800, len(pooled))
    heuristic = pooled[rng.choice(len(pooled), size=hn, replace=False)]
    distances = pairwise_distances(heuristic, metric="euclidean")
    tri = distances[np.triu_indices_from(distances, k=1)]
    positive = tri[tri > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / (2.0 * max(sigma * sigma, 1e-12))
    mmd2 = (
        float(rbf_kernel(xs, xs, gamma=gamma).mean())
        + float(rbf_kernel(xt, xt, gamma=gamma).mean())
        - 2.0 * float(rbf_kernel(xs, xt, gamma=gamma).mean())
    )
    return {
        "mmd2_rbf": float(max(mmd2, 0.0)),
        "rbf_sigma_median_heuristic": sigma,
        "rbf_gamma": float(gamma),
        "source_samples": int(ns),
        "target_samples": int(nt),
    }


def analyse_distribution(Xs, Xt, feature_names, args, seed):
    w = feature_wasserstein(Xs, Xt, feature_names, args.top_k_features)
    w.update(bootstrap_wasserstein_ci(
        Xs, Xt, args.bootstrap_reps, args.bootstrap_n, seed + 101
    ))
    return {
        "source_samples": int(len(Xs)),
        "target_samples": int(len(Xt)),
        "robust_wasserstein": w,
        "domain_classifier": repeated_domain_classifier(
            Xs, Xt, args.domain_repeats, seed + 211
        ),
        "covariance": covariance_discrepancy(Xs, Xt),
        "mmd": rbf_mmd_squared(Xs, Xt, args.mmd_samples, seed + 307),
    }


def analyse_unordered_pair(a, b, samples, full_labels, feature_names, args, seed):
    print("\n" + "=" * 100)
    print(f"SHIFT ANALYSIS: {short_dataset(a)} ↔ {short_dataset(b)}")
    print("=" * 100)
    sa, sb = samples[a], samples[b]
    transform = PairStandardTransform().fit(sa["X_marginal"], sb["X_marginal"])
    Xa_m, Xb_m = transform.transform(sa["X_marginal"]), transform.transform(sb["X_marginal"])
    Xa_b, Xb_b = transform.transform(sa["X_benign"]), transform.transform(sb["X_benign"])
    Xa_a, Xb_a = transform.transform(sa["X_attack"]), transform.transform(sb["X_attack"])

    prior = prior_difference_stats(full_labels[a], full_labels[b])
    marginal = analyse_distribution(Xa_m, Xb_m, feature_names, args, seed + 1000)
    benign = analyse_distribution(Xa_b, Xb_b, feature_names, args, seed + 2000)
    attack = analyse_distribution(Xa_a, Xb_a, feature_names, args, seed + 3000)

    print(
        f"Attack priors: {short_dataset(a)}={prior['source_attack_prior']:.4f}, "
        f"{short_dataset(b)}={prior['target_attack_prior']:.4f}, "
        f"|Δ|={prior['absolute_attack_prior_difference']:.4f}"
    )
    print(
        f"Marginal domain AUROC: {marginal['domain_classifier']['mean_auroc']:.4f} "
        f"| converged={marginal['domain_classifier']['all_converged']}"
    )
    print(
        f"Benign conditional AUROC: {benign['domain_classifier']['mean_auroc']:.4f} "
        f"| converged={benign['domain_classifier']['all_converged']}"
    )
    print(
        f"Attack conditional AUROC: {attack['domain_classifier']['mean_auroc']:.4f} "
        f"| converged={attack['domain_classifier']['all_converged']}"
    )

    return {
        "dataset_a": a,
        "dataset_b": b,
        "pair_unordered": f"{a}__VS__{b}",
        "transform": {
            "type": "pooled_q01_q99_clip_then_StandardScaler",
            "lower_quantile": LOWER_CLIP_Q,
            "upper_quantile": UPPER_CLIP_Q,
            "note": (
                "Descriptive analysis only; not the source-only training scaler. "
                "No categorical shift label is assigned automatically."
            ),
        },
        "prior_shift_a_to_b": prior,
        "marginal_PX": marginal,
        "benign_PX_given_Y0": benign,
        "attack_PX_given_Y1": attack,
        "shift_profile": {
            "absolute_attack_prior_difference": prior[
                "absolute_attack_prior_difference"
            ],
            "marginal_domain_auroc": marginal[
                "domain_classifier"
            ]["mean_auroc"],
            "benign_conditional_domain_auroc": benign[
                "domain_classifier"
            ]["mean_auroc"],
            "attack_conditional_domain_auroc": attack[
                "domain_classifier"
            ]["mean_auroc"],
            "marginal_mmd2": marginal["mmd"]["mmd2_rbf"],
            "benign_conditional_mmd2": benign["mmd"]["mmd2_rbf"],
            "attack_conditional_mmd2": attack["mmd"]["mmd2_rbf"],
            "guardrail": (
                "Observed P(Y) change is class-prior shift. Do not call it "
                "pure label shift unless P(X|Y) is approximately invariant. "
                "Interpret the continuous component profile before assigning "
                "a descriptive shift category."
            ),
        },
    }


def directed_row(source, target, result, full_labels):
    prior = prior_difference_stats(full_labels[source], full_labels[target])
    m = result["marginal_PX"]
    b = result["benign_PX_given_Y0"]
    a = result["attack_PX_given_Y1"]

    return {
        "pair": f"{source}__TO__{target}",
        "source_dataset": source,
        "target_dataset": target,

        "source_attack_prior": prior["source_attack_prior"],
        "target_attack_prior": prior["target_attack_prior"],
        "signed_attack_prior_change": prior[
            "signed_attack_prior_change_target_minus_source"
        ],
        "absolute_attack_prior_difference": prior[
            "absolute_attack_prior_difference"
        ],
        "prior_change_ci95_low": prior["signed_prior_change_ci95_low"],
        "prior_change_ci95_high": prior["signed_prior_change_ci95_high"],

        "marginal_domain_auroc": m["domain_classifier"]["mean_auroc"],
        "marginal_domain_auroc_std": m["domain_classifier"]["std_auroc"],
        "marginal_domain_auroc_95_low": m["domain_classifier"]["empirical_95_low"],
        "marginal_domain_auroc_95_high": m["domain_classifier"]["empirical_95_high"],
        "marginal_domain_classifier_all_converged": m[
            "domain_classifier"
        ]["all_converged"],
        "marginal_domain_classifier_convergence_rate": m[
            "domain_classifier"
        ]["convergence_rate"],
        "marginal_domain_classifier_max_iterations": m[
            "domain_classifier"
        ]["max_iterations_observed"],
        "marginal_wasserstein_mean": m["robust_wasserstein"]["mean"],
        "marginal_wasserstein_mean_ci95_low": m[
            "robust_wasserstein"
        ]["mean_ci95_low"],
        "marginal_wasserstein_mean_ci95_high": m[
            "robust_wasserstein"
        ]["mean_ci95_high"],
        "marginal_wasserstein_median": m["robust_wasserstein"]["median"],
        "marginal_wasserstein_median_ci95_low": m[
            "robust_wasserstein"
        ]["median_ci95_low"],
        "marginal_wasserstein_median_ci95_high": m[
            "robust_wasserstein"
        ]["median_ci95_high"],
        "marginal_mmd2": m["mmd"]["mmd2_rbf"],
        "marginal_covariance_relative_frobenius": m[
            "covariance"
        ]["relative_frobenius_difference"],

        "benign_domain_auroc": b["domain_classifier"]["mean_auroc"],
        "benign_domain_auroc_std": b["domain_classifier"]["std_auroc"],
        "benign_domain_classifier_all_converged": b[
            "domain_classifier"
        ]["all_converged"],
        "benign_domain_classifier_convergence_rate": b[
            "domain_classifier"
        ]["convergence_rate"],
        "benign_domain_classifier_max_iterations": b[
            "domain_classifier"
        ]["max_iterations_observed"],
        "benign_wasserstein_mean": b["robust_wasserstein"]["mean"],
        "benign_wasserstein_mean_ci95_low": b[
            "robust_wasserstein"
        ]["mean_ci95_low"],
        "benign_wasserstein_mean_ci95_high": b[
            "robust_wasserstein"
        ]["mean_ci95_high"],
        "benign_wasserstein_median": b["robust_wasserstein"]["median"],
        "benign_wasserstein_median_ci95_low": b[
            "robust_wasserstein"
        ]["median_ci95_low"],
        "benign_wasserstein_median_ci95_high": b[
            "robust_wasserstein"
        ]["median_ci95_high"],
        "benign_mmd2": b["mmd"]["mmd2_rbf"],
        "benign_covariance_relative_frobenius": b[
            "covariance"
        ]["relative_frobenius_difference"],

        "attack_domain_auroc": a["domain_classifier"]["mean_auroc"],
        "attack_domain_auroc_std": a["domain_classifier"]["std_auroc"],
        "attack_domain_classifier_all_converged": a[
            "domain_classifier"
        ]["all_converged"],
        "attack_domain_classifier_convergence_rate": a[
            "domain_classifier"
        ]["convergence_rate"],
        "attack_domain_classifier_max_iterations": a[
            "domain_classifier"
        ]["max_iterations_observed"],
        "attack_wasserstein_mean": a["robust_wasserstein"]["mean"],
        "attack_wasserstein_mean_ci95_low": a[
            "robust_wasserstein"
        ]["mean_ci95_low"],
        "attack_wasserstein_mean_ci95_high": a[
            "robust_wasserstein"
        ]["mean_ci95_high"],
        "attack_wasserstein_median": a["robust_wasserstein"]["median"],
        "attack_wasserstein_median_ci95_low": a[
            "robust_wasserstein"
        ]["median_ci95_low"],
        "attack_wasserstein_median_ci95_high": a[
            "robust_wasserstein"
        ]["median_ci95_high"],
        "attack_mmd2": a["mmd"]["mmd2_rbf"],
        "attack_covariance_relative_frobenius": a[
            "covariance"
        ]["relative_frobenius_difference"],

        "conditional_wasserstein_mean": float(
            np.mean(
                [
                    b["robust_wasserstein"]["mean"],
                    a["robust_wasserstein"]["mean"],
                ]
            )
        ),
        "conditional_wasserstein_median_mean": float(
            np.mean(
                [
                    b["robust_wasserstein"]["median"],
                    a["robust_wasserstein"]["median"],
                ]
            )
        ),
        "conditional_domain_auroc_mean": float(
            np.mean(
                [
                    b["domain_classifier"]["mean_auroc"],
                    a["domain_classifier"]["mean_auroc"],
                ]
            )
        ),
    }


def feature_detail_rows(result):
    rows = []
    for key, label in [
        ("marginal_PX", "marginal"),
        ("benign_PX_given_Y0", "benign_conditional"),
        ("attack_PX_given_Y1", "attack_conditional"),
    ]:
        for item in result[key]["robust_wasserstein"]["top_features"]:
            rows.append({
                "pair_unordered": result["pair_unordered"],
                "component": label,
                "feature_index": item["feature_index"],
                "feature_name": item["feature_name"],
                "robust_wasserstein": item["wasserstein"],
            })
    return rows


def plot_attack_priors(summary, output_dir):
    datasets = list(summary["datasets"].keys())
    priors = []
    for dataset in datasets:
        full = summary["datasets"][dataset]["variants"]["full"]
        priors.append(full["attack"] / full["rows"])
    plt.figure(figsize=(7.5, 5.0))
    plt.bar([short_dataset(d) for d in datasets], priors)
    plt.ylabel("Attack prevalence P(Y=1)")
    plt.ylim(0.0, max(priors) * 1.2)
    savefig(output_dir / "full_attack_priors.png")


def plot_shift_component_matrix(df, output_dir):
    seen, rows = set(), []
    for _, row in df.iterrows():
        key = frozenset([row["source_dataset"], row["target_dataset"]])
        if key not in seen:
            seen.add(key)
            rows.append(row)
    d = pd.DataFrame(rows)
    labels = [
        f"{short_dataset(r['source_dataset'])}↔{short_dataset(r['target_dataset'])}"
        for _, r in d.iterrows()
    ]
    matrix = np.column_stack([
        d["absolute_attack_prior_difference"].to_numpy(),
        np.clip(2.0 * (d["marginal_domain_auroc"].to_numpy() - 0.5), 0.0, 1.0),
        np.clip(2.0 * (d["benign_domain_auroc"].to_numpy() - 0.5), 0.0, 1.0),
        np.clip(2.0 * (d["attack_domain_auroc"].to_numpy() - 0.5), 0.0, 1.0),
    ])
    plt.figure(figsize=(8.5, 4.8))
    im = plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, label="Descriptive shift strength (0–1)")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(
        np.arange(4),
        ["|Δ attack prior|", "Marginal P(X)", "Benign P(X|Y=0)", "Attack P(X|Y=1)"],
        rotation=20,
        ha="right",
    )
    savefig(output_dir / "shift_component_matrix.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_per_domain", type=int, default=DEFAULT_SAMPLE_PER_DOMAIN)
    parser.add_argument("--sample_per_class", type=int, default=DEFAULT_SAMPLE_PER_CLASS)
    parser.add_argument("--parquet_batch_size", type=int, default=100_000)
    parser.add_argument("--bootstrap_reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap_n", type=int, default=DEFAULT_BOOTSTRAP_N)
    parser.add_argument("--domain_repeats", type=int, default=DEFAULT_DOMAIN_REPEATS)
    parser.add_argument("--mmd_samples", type=int, default=DEFAULT_MMD_SAMPLES)
    parser.add_argument("--top_k_features", type=int, default=DEFAULT_TOP_K_FEATURES)
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()

    output_dir = STAGE1_DIR / "shift_analysis_v2"
    cache_dir = output_dir / "sample_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary = load_stage1_summary()
    feature_names = load_feature_names()
    datasets = list(summary["datasets"].keys())

    print("Datasets:")
    for dataset in datasets:
        print(f"  - {dataset}")

    full_dir = STAGE1_DIR / "full"
    full_labels, samples = {}, {}
    for dataset in datasets:
        y_path = full_dir / f"{dataset}_y.npy"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing {y_path}")
        full_labels[dataset] = np.load(y_path, mmap_mode="r")
        samples[dataset] = build_or_load_dataset_sample(
            dataset, feature_names, cache_dir,
            args.sample_per_domain, args.sample_per_class,
            args.parquet_batch_size, args.rebuild_cache,
        )

    unordered_results = {}
    feature_rows = []
    for pair_index, (a, b) in enumerate(combinations(datasets, 2), start=1):
        result = analyse_unordered_pair(
            a, b, samples, full_labels, feature_names, args,
            RANDOM_STATE + pair_index * 10_000,
        )
        unordered_results[result["pair_unordered"]] = result
        feature_rows.extend(feature_detail_rows(result))

    directed_rows, unordered_rows = [], []
    for result in unordered_results.values():
        a, b = result["dataset_a"], result["dataset_b"]
        directed_rows.extend([
            directed_row(a, b, result, full_labels),
            directed_row(b, a, result, full_labels),
        ])
        u = directed_row(a, b, result, full_labels)
        u["pair"] = f"{a}__VS__{b}"
        unordered_rows.append(u)

    directed_df = pd.DataFrame(directed_rows).sort_values("pair")
    unordered_df = pd.DataFrame(unordered_rows).sort_values("pair")
    feature_df = pd.DataFrame(feature_rows)

    directed_path = output_dir / "shift_profiles_all_directed_pairs.csv"
    unordered_path = output_dir / "shift_profiles_unordered_pairs.csv"
    feature_path = output_dir / "shift_top_features.csv"
    json_path = output_dir / "shift_analysis_v2_full.json"
    protocol_path = output_dir / "shift_analysis_v2_protocol.json"

    directed_df.to_csv(directed_path, index=False)
    unordered_df.to_csv(unordered_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unordered_results, f, indent=2)

    protocol = {
        "research_goal": "Characterise shift components before choosing an adaptation method.",
        "data_variant_for_shift_diagnosis": "full",
        "controlled_adaptation_variant": "Keep balanced_100k experiments unchanged.",
        "sample_per_domain": args.sample_per_domain,
        "sample_per_class": args.sample_per_class,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_n": args.bootstrap_n,
        "domain_classifier_repeats": args.domain_repeats,
        "mmd_samples": args.mmd_samples,
        "random_state": RANDOM_STATE,
        "transform": "Pooled q01/q99 clipping + pooled StandardScaler; descriptive analysis only.",
        "label_shift_guardrail": (
            "P(Y) difference is class-prior shift evidence. Pure label shift additionally assumes "
            "P(X|Y) invariance and is not inferred from priors alone."
        ),
        "symmetry_guardrail": (
            "Feature divergence is computed once per unordered pair. Directed rows retain the "
            "signed target-minus-source prior change for joining to adaptation results."
        ),
        "classification_policy": (
            "No final covariate/label/class-conditional/mixed category is assigned automatically. "
            "Continuous P(Y), P(X), P(X|Y=0), and P(X|Y=1) components are interpreted first."
        ),
        "domain_classifier_policy": (
            "Each repeated logistic domain classifier fits a StandardScaler on its training split "
            "only and records convergence diagnostics."
        ),
    }
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    plot_attack_priors(summary, output_dir)
    plot_shift_component_matrix(directed_df, output_dir)

    display = directed_df[[
        "pair",
        "source_attack_prior",
        "target_attack_prior",
        "absolute_attack_prior_difference",
        "marginal_domain_auroc",
        "marginal_wasserstein_mean",
        "marginal_wasserstein_median",
        "benign_domain_auroc",
        "benign_wasserstein_median",
        "attack_domain_auroc",
        "attack_wasserstein_median",
        "marginal_mmd2",
        "marginal_domain_classifier_all_converged",
    ]].copy()
    display["pair"] = [
        short_pair(s, t)
        for s, t in zip(directed_df["source_dataset"], directed_df["target_dataset"])
    ]

    print("\n" + "=" * 120)
    print("SHIFT ANALYSIS V2 COMPLETE")
    print("=" * 120)
    print(display.to_string(index=False))
    print("\nGuardrails:")
    print("- P(Y) differences are class-prior shift evidence.")
    print("- Do not claim pure label shift unless P(X|Y) is approximately stable.")
    print("- Interpret continuous components before assigning a descriptive shift category.")
    print("- Check the *_domain_classifier_all_converged columns before using AUROC as a headline result.")
    print("\nOutputs:")
    for p in [directed_path, unordered_path, feature_path, json_path, protocol_path]:
        print(f"- {p}")
    print(f"- {output_dir / 'full_attack_priors.png'}")
    print(f"- {output_dir / 'shift_component_matrix.png'}")


if __name__ == "__main__":
    main()
