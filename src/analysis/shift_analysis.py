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


def classify_shift_type(metric_levels, pair_metrics, label_eps=1e-6):
    global_level = metric_levels["global_feature_shift"]
    cov_level = metric_levels["covariance_shift"]
    label_level = metric_levels["label_shift"]
    cond_level = metric_levels["class_conditional_shift"]

    pair_name = pair_metrics["pair_name"]

    g = global_level[pair_name]
    c = cov_level[pair_name]
    l = label_level[pair_name]
    cc = cond_level[pair_name]

    # Use the actual label-shift metric as a guard.
    # In your balanced_100k setup this will usually be zero.
    label_abs_diff = pair_metrics["label_shift"]["absolute_attack_prior_difference"]
    has_real_label_shift = label_abs_diff > label_eps

    if has_real_label_shift and l == "high" and g in {"low", "medium"} and cc in {"low", "medium"}:
        shift_type = "label_shift"
        recommendation = [
            "prior correction / reweighting",
            "feature alignment alone may be insufficient",
        ]
    elif g == "high" and c == "high" and (not has_real_label_shift) and cc in {"medium", "high"}:
        shift_type = "covariate_shift"
        recommendation = [
            "Deep CORAL",
            "MMD",
            "DANN",
        ]
    elif cc == "high" and (not has_real_label_shift):
        shift_type = "class_conditional_shift"
        recommendation = [
            "DANN",
            "replay-based adaptation",
            "boundary-critical replay",
        ]
    elif g == "high" or c == "high" or cc == "high" or (has_real_label_shift and l == "high"):
        shift_type = "mixed_shift"
        recommendation = [
            "DANN",
            "Deep CORAL",
            "boundary-critical replay",
        ]
    else:
        shift_type = "mild_shift"
        recommendation = [
            "source-only baseline may already transfer reasonably",
            "naive fine-tuning",
        ]

    return {
        "shift_type": shift_type,
        "level_summary": {
            "global_feature_shift": g,
            "covariance_shift": c,
            "label_shift": "low" if not has_real_label_shift else l,
            "class_conditional_shift": cc,
        },
        "recommended_adaptation_methods": recommendation,
    }

def main():
    base_dir = PAIR_DIR / "balanced_100k"
    feature_names = load_feature_names()

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
        print("Covariance Frobenius norm:", cov_stats["frobenius_norm"])
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

    # second pass: classify all pairs relative to one another
    global_feature_values = {
        pair_name: pair_metrics["wasserstein"]["clipped_mean"]
        for pair_name, pair_metrics in results.items()
    }
    covariance_values = {
        pair_name: pair_metrics["covariance_shift"]["frobenius_norm"]
        for pair_name, pair_metrics in results.items()
    }
    label_shift_values = {
        pair_name: pair_metrics["label_shift"]["absolute_attack_prior_difference"]
        for pair_name, pair_metrics in results.items()
    }
    class_conditional_values = {
        pair_name: (
            pair_metrics["class_conditional_shift"]["summary"]["overall_class_conditional_shift_mean"]
            if pair_metrics["class_conditional_shift"]["summary"]["overall_class_conditional_shift_mean"] is not None
            else 0.0
        )
        for pair_name, pair_metrics in results.items()
    }

    metric_levels = {
        "global_feature_shift": assign_tertiles(global_feature_values),
        "covariance_shift": assign_tertiles(covariance_values),
        "label_shift": assign_tertiles(label_shift_values),
        "class_conditional_shift": assign_tertiles(class_conditional_values),
    }

    for pair_name, pair_metrics in results.items():
        classification = classify_shift_type(metric_levels, pair_metrics)
        pair_metrics["shift_classification"] = classification

    summary_path = base_dir / "shift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()