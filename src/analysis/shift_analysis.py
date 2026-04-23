import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import PAIR_DIR, STAGE1_DIR, RANDOM_STATE


def load_feature_names():
    with open(STAGE1_DIR / "common_features.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["common_feature_columns"]


def load_pair(pair_path: Path):
    Xs = np.load(pair_path / "Xs_train.npy")
    Xt = np.load(pair_path / "Xt_test.npy")
    return Xs, Xt


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


def main():
    base_dir = PAIR_DIR / "balanced_100k"
    feature_names = load_feature_names()

    results = {}

    for pair_path in sorted(base_dir.iterdir()):
        if not pair_path.is_dir():
            continue

        pair_name = pair_path.name
        print(f"\nAnalyzing: {pair_name}")

        Xs, Xt = load_pair(pair_path)

        w_stats = compute_wasserstein_stats(Xs, Xt, feature_names)
        print("Clipped Wasserstein mean:", w_stats["clipped_mean"])
        print("Clipped Wasserstein median:", w_stats["clipped_median"])

        domain_stats = domain_classifier_score(Xs, Xt)
        print("Domain classifier acc:", domain_stats["accuracy"])
        print("Domain classifier AUROC:", domain_stats["auroc"])

        plot_path = pair_path / "pca_plot.png"
        plot_pca(Xs, Xt, plot_path)

        results[pair_name] = {
            "wasserstein": w_stats,
            "domain_classifier": domain_stats,
        }

    summary_path = base_dir / "shift_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()