import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import PAIR_DIR, RANDOM_STATE, STAGE1_DIR


def load_feature_names():
    with open(STAGE1_DIR / "common_features.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["common_feature_columns"]


def load_pair(pair_path: Path):
    Xs = np.load(pair_path / "Xs_train.npy")
    ys = np.load(pair_path / "ys_train.npy")
    Xt = np.load(pair_path / "Xt_test.npy")
    yt = np.load(pair_path / "yt_test.npy")
    return Xs, ys, Xt, yt


def clipped_wasserstein_1d(xs, xt, lower_q=0.01, upper_q=0.99):
    combined = np.concatenate([xs, xt])
    lo = np.quantile(combined, lower_q)
    hi = np.quantile(combined, upper_q)

    xs_clip = np.clip(xs, lo, hi)
    xt_clip = np.clip(xt, lo, hi)

    return wasserstein_distance(xs_clip, xt_clip)


def compute_wasserstein_stats(Xs, Xt, feature_names):
    raw_distances = []
    clipped_distances = []

    for i in range(Xs.shape[1]):
        raw_d = wasserstein_distance(Xs[:, i], Xt[:, i])
        clip_d = clipped_wasserstein_1d(Xs[:, i], Xt[:, i])

        raw_distances.append(raw_d)
        clipped_distances.append(clip_d)

    raw_distances = np.array(raw_distances)
    clipped_distances = np.array(clipped_distances)

    top_idx = np.argsort(clipped_distances)[-5:][::-1]

    top_features = [
        {
            "feature_index": int(i),
            "feature_name": feature_names[i],
            "raw_wasserstein": float(raw_distances[i]),
            "clipped_wasserstein": float(clipped_distances[i]),
        }
        for i in top_idx
    ]

    return {
        "raw_mean": float(np.mean(raw_distances)),
        "raw_median": float(np.median(raw_distances)),
        "raw_max": float(np.max(raw_distances)),
        "clipped_mean": float(np.mean(clipped_distances)),
        "clipped_median": float(np.median(clipped_distances)),
        "clipped_max": float(np.max(clipped_distances)),
        "top_5_features_by_clipped_wasserstein": top_features,
    }


def covariance_shift_stats(Xs, Xt):
    cov_s = np.cov(Xs, rowvar=False)
    cov_t = np.cov(Xt, rowvar=False)

    diff = cov_s - cov_t
    fro_norm = np.linalg.norm(diff, ord="fro")
    mean_abs_diff = np.mean(np.abs(diff))

    var_s = np.var(Xs, axis=0)
    var_t = np.var(Xt, axis=0)

    # More stable than raw ratio means
    log_var_ratio = np.log10((var_t + 1e-8) / (var_s + 1e-8))

    return {
        "frobenius_norm": float(fro_norm),
        "mean_absolute_covariance_difference": float(mean_abs_diff),
        "mean_absolute_log10_variance_ratio": float(np.mean(np.abs(log_var_ratio))),
        "median_absolute_log10_variance_ratio": float(np.median(np.abs(log_var_ratio))),
    }


def label_shift_stats(ys, yt):
    p_source_attack = float(np.mean(ys == 1))
    p_target_attack = float(np.mean(yt == 1))

    p_source_benign = 1.0 - p_source_attack
    p_target_benign = 1.0 - p_target_attack

    # For binary class priors, total variation distance = 0.5 * sum |p - q|
    tv_distance = 0.5 * (
        abs(p_source_benign - p_target_benign) +
        abs(p_source_attack - p_target_attack)
    )

    return {
        "source_attack_prior": p_source_attack,
        "target_attack_prior": p_target_attack,
        "source_benign_prior": p_source_benign,
        "target_benign_prior": p_target_benign,
        "absolute_attack_prior_difference": float(abs(p_source_attack - p_target_attack)),
        "total_variation_distance": float(tv_distance),
    }


def domain_classifier_score(Xs, Xt):
    X = np.vstack([Xs, Xt])
    y = np.concatenate([
        np.zeros(len(Xs), dtype=int),
        np.ones(len(Xt), dtype=int)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    clf = LogisticRegression(
        max_iter=3000,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
    }


def class_conditional_shift_stats(Xs, ys, Xt, yt, feature_names):
    result = {}

    for class_value, class_name in [(0, "benign"), (1, "attack")]:
        Xs_c = Xs[ys == class_value]
        Xt_c = Xt[yt == class_value]

        if len(Xs_c) < 2 or len(Xt_c) < 2:
            result[class_name] = {
                "num_source": int(len(Xs_c)),
                "num_target": int(len(Xt_c)),
                "wasserstein": None,
                "covariance_shift": None,
                "domain_classifier": None,
            }
            continue

        w_stats = compute_wasserstein_stats(Xs_c, Xt_c, feature_names)
        cov_stats = covariance_shift_stats(Xs_c, Xt_c)

        # only run domain classifier if both classes have enough samples
        if len(Xs_c) >= 50 and len(Xt_c) >= 50:
            domain_stats = domain_classifier_score(Xs_c, Xt_c)
        else:
            domain_stats = None

        result[class_name] = {
            "num_source": int(len(Xs_c)),
            "num_target": int(len(Xt_c)),
            "wasserstein": w_stats,
            "covariance_shift": cov_stats,
            "domain_classifier": domain_stats,
        }

    benign_mean = (
        result["benign"]["wasserstein"]["clipped_mean"]
        if result["benign"]["wasserstein"] is not None
        else None
    )
    attack_mean = (
        result["attack"]["wasserstein"]["clipped_mean"]
        if result["attack"]["wasserstein"] is not None
        else None
    )

    valid_vals = [v for v in [benign_mean, attack_mean] if v is not None]
    overall_mean = float(np.mean(valid_vals)) if valid_vals else None

    result["summary"] = {
        "benign_clipped_wasserstein_mean": benign_mean,
        "attack_clipped_wasserstein_mean": attack_mean,
        "overall_class_conditional_shift_mean": overall_mean,
    }

    return result


def plot_pca(Xs, Xt, save_path, max_points=5000):
    rng = np.random.default_rng(RANDOM_STATE)

    if len(Xs) > max_points:
        idx_s = rng.choice(len(Xs), size=max_points, replace=False)
        Xs_plot = Xs[idx_s]
    else:
        Xs_plot = Xs

    if len(Xt) > max_points:
        idx_t = rng.choice(len(Xt), size=max_points, replace=False)
        Xt_plot = Xt[idx_t]
    else:
        Xt_plot = Xt

    X = np.vstack([Xs_plot, Xt_plot])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    Xs_pca = X_pca[:len(Xs_plot)]
    Xt_pca = X_pca[len(Xs_plot):]

    plt.figure(figsize=(7, 6))
    plt.scatter(Xs_pca[:, 0], Xs_pca[:, 1], s=8, alpha=0.35, label="Source")
    plt.scatter(Xt_pca[:, 0], Xt_pca[:, 1], s=8, alpha=0.35, label="Target")
    plt.title("PCA: Source vs Target")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def assign_tertiles(values_dict, atol=1e-12):
    if len(values_dict) == 0:
        return {}

    values = np.array(list(values_dict.values()), dtype=float)

    # If all values are effectively identical, do not force artificial low/medium/high splits
    if np.allclose(values, values[0], atol=atol, rtol=0.0):
        return {name: "low" for name in values_dict}

    items = sorted(values_dict.items(), key=lambda x: x[1])
    n = len(items)

    low_cut = max(1, n // 3)
    med_cut = max(low_cut + 1, (2 * n) // 3)

    levels = {}
    for idx, (name, _) in enumerate(items):
        if idx < low_cut:
            levels[name] = "low"
        elif idx < med_cut:
            levels[name] = "medium"
        else:
            levels[name] = "high"

    return levels


def level_from_thresholds(value: float, low_cut: float, high_cut: float) -> str:
    if value < low_cut:
        return "low"
    if value < high_cut:
        return "medium"
    return "high"


def classify_shift_absolute(pair_metrics):
    """
    Absolute rule-based shift classification.

    These thresholds are practical thresholds for this NetFlow setup.
    They are not universal laws. They make the classification reproducible
    and independent of which other dataset pairs are included.

    Definitions:
    - Covariate shift: P(X) changes strongly, especially feature/covariance structure.
    - Label shift: P(Y) changes.
    - Class-conditional shift: P(X|Y) changes, suggesting class structure / boundary changes.
    - Mixed shift: multiple shift signals are present.
    - Mild shift: all shift signals are low.
    """
    global_w = pair_metrics["wasserstein"]["clipped_mean"]
    domain_auroc = pair_metrics["domain_classifier"]["auroc"]
    cov_log_ratio = pair_metrics["covariance_shift"]["mean_absolute_log10_variance_ratio"]
    label_diff = pair_metrics["label_shift"]["absolute_attack_prior_difference"]

    class_cond = pair_metrics["class_conditional_shift"]["summary"][
        "overall_class_conditional_shift_mean"
    ]
    if class_cond is None:
        class_cond = 0.0

    levels = {
        "global_feature_shift": level_from_thresholds(
            global_w,
            low_cut=0.50,
            high_cut=1.50,
        ),
        "domain_separability": level_from_thresholds(
            domain_auroc,
            low_cut=0.70,
            high_cut=0.90,
        ),
        "covariance_shift": level_from_thresholds(
            cov_log_ratio,
            low_cut=1.00,
            high_cut=1.50,
        ),
        "label_shift": level_from_thresholds(
            label_diff,
            low_cut=0.05,
            high_cut=0.20,
        ),
        "class_conditional_shift": level_from_thresholds(
            class_cond,
            low_cut=0.50,
            high_cut=1.50,
        ),
    }

    has_label_shift = label_diff >= 0.05

    global_high = levels["global_feature_shift"] == "high"
    global_med_or_high = levels["global_feature_shift"] in {"medium", "high"}

    cov_high = levels["covariance_shift"] == "high"
    cov_med_or_high = levels["covariance_shift"] in {"medium", "high"}

    class_cond_high = levels["class_conditional_shift"] == "high"
    class_cond_med_or_high = levels["class_conditional_shift"] in {"medium", "high"}

    domain_high = levels["domain_separability"] == "high"

    # Mild only when all key signals are genuinely low.
    if (
        levels["global_feature_shift"] == "low"
        and levels["class_conditional_shift"] == "low"
        and levels["label_shift"] == "low"
        and levels["domain_separability"] in {"low", "medium"}
    ):
        shift_type = "mild_shift"
        recommendation = [
            "source-only baseline may transfer reasonably",
            "small fine-tuning budgets may be sufficient",
        ]

    # Label shift requires actual class-prior difference.
    elif has_label_shift and levels["label_shift"] == "high":
        shift_type = "label_shift"
        recommendation = [
            "class prior correction",
            "loss reweighting",
            "feature alignment alone may be insufficient",
        ]

    # Covariate shift: strong feature/covariance shift, but not strong class-conditional shift.
    elif (global_high or cov_high or domain_high) and not class_cond_high:
        shift_type = "covariate_shift"
        recommendation = [
            "Deep CORAL",
            "MMD",
            "DANN",
        ]

    # Class-conditional shift: class-specific distributions change strongly.
    elif class_cond_high and not (global_high and cov_high):
        shift_type = "class_conditional_shift"
        recommendation = [
            "fine-tuning with small target budgets",
            "replay-based adaptation",
            "boundary-critical replay",
        ]

    # Mixed shift: multiple shift signals are present.
    elif (
        (global_med_or_high and class_cond_med_or_high)
        or (cov_med_or_high and class_cond_med_or_high)
        or (domain_high and class_cond_med_or_high)
    ):
        shift_type = "mixed_shift"
        recommendation = [
            "DANN",
            "Deep CORAL",
            "replay-based continual learning",
        ]

    else:
        shift_type = "mild_shift"
        recommendation = [
            "source-only baseline may transfer reasonably",
            "small fine-tuning budgets may be sufficient",
        ]

    return {
        "shift_type": shift_type,
        "level_summary": levels,
        "absolute_thresholds": {
            "global_feature_shift_clipped_wasserstein": {
                "low": "< 0.50",
                "medium": "0.50-1.50",
                "high": ">= 1.50",
            },
            "domain_classifier_auroc": {
                "low": "< 0.70",
                "medium": "0.70-0.90",
                "high": ">= 0.90",
            },
            "covariance_mean_abs_log10_variance_ratio": {
                "low": "< 1.00",
                "medium": "1.00-1.50",
                "high": ">= 1.50",
            },
            "label_prior_difference": {
                "low": "< 0.05",
                "medium": "0.05-0.20",
                "high": ">= 0.20",
            },
            "class_conditional_wasserstein": {
                "low": "< 0.50",
                "medium": "0.50-1.50",
                "high": ">= 1.50",
            },
        },
        "recommended_adaptation_methods": recommendation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage1_variant",
        default="balanced_100k",
        type=str,
        help="Example: balanced_10k, balanced_25k, balanced_50k, balanced_100k, full",
    )
    args = parser.parse_args()

    base_dir = PAIR_DIR / args.stage1_variant
    feature_names = load_feature_names()

    if not base_dir.exists():
        raise FileNotFoundError(f"Pair directory not found: {base_dir}")

    results = {}

    for pair_path in sorted(base_dir.iterdir()):
        if not pair_path.is_dir():
            continue

        pair_name = pair_path.name
        print(f"\nAnalyzing: {pair_name}")

        Xs, ys, Xt, yt = load_pair(pair_path)

        w_stats = compute_wasserstein_stats(Xs, Xt, feature_names)
        cov_stats = covariance_shift_stats(Xs, Xt)
        domain_stats = domain_classifier_score(Xs, Xt)
        label_stats = label_shift_stats(ys, yt)
        conditional_stats = class_conditional_shift_stats(Xs, ys, Xt, yt, feature_names)

        print("Global clipped Wasserstein mean:", w_stats["clipped_mean"])
        print("Domain classifier AUROC:", domain_stats["auroc"])
        print("Covariance log-var ratio:", cov_stats["mean_absolute_log10_variance_ratio"])
        print("Attack prior abs difference:", label_stats["absolute_attack_prior_difference"])
        print(
            "Class-conditional shift mean:",
            conditional_stats["summary"]["overall_class_conditional_shift_mean"]
        )

        plot_path = pair_path / "pca_plot.png"
        plot_pca(Xs, Xt, plot_path)

        results[pair_name] = {
            "pair_name": pair_name,
            "wasserstein": w_stats,
            "domain_classifier": domain_stats,
            "covariance_shift": cov_stats,
            "label_shift": label_stats,
            "class_conditional_shift": conditional_stats,
        }

    for pair_name, pair_metrics in results.items():
        pair_metrics["shift_classification"] = classify_shift_absolute(pair_metrics)

    summary_path = base_dir / "shift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    import argparse
    main()