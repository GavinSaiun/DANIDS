import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("artifacts")
OUT = Path("outputs/proposal_artifacts")
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Transfer pairs
# --------------------------------------------------
PAIRS = [
    "NF-CSE-CIC-IDS2018-v3__TO__NF-ToN-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3__TO__NF-UNSW-NB15-v3",
    "NF-ToN-IoT-v3__TO__NF-CSE-CIC-IDS2018-v3",
    "NF-ToN-IoT-v3__TO__NF-UNSW-NB15-v3",
    "NF-UNSW-NB15-v3__TO__NF-CSE-CIC-IDS2018-v3",
    "NF-UNSW-NB15-v3__TO__NF-ToN-IoT-v3",
]


# --------------------------------------------------
# Global presentation-quality matplotlib style
# --------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close()


def short_pair(pair: str) -> str:
    return (
        pair.replace("NF-", "")
        .replace("-v3", "")
        .replace("CSE-CIC-IDS2018", "CIC")
        .replace("UNSW-NB15", "UNSW")
        .replace("ToN-IoT", "ToN")
        .replace("__TO__", " → ")
    )


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Dataset and sampling tables
# --------------------------------------------------
def build_dataset_tables():
    data = load_json(BASE / "stage1" / "stage1_summary.json")

    dataset_rows = []
    sampling_rows = []

    for dataset, info in data["datasets"].items():
        full = info["variants"]["full"]
        attack_pct = 100 * full["attack"] / full["rows"]

        dataset_rows.append({
            "Dataset": dataset,
            "Full rows": full["rows"],
            "Benign": full["benign"],
            "Attack": full["attack"],
            "Attack %": round(attack_pct, 2),
            "Common features": data["common_feature_count"],
        })

        for variant, v in info["variants"].items():
            if variant == "full":
                continue

            sampling_rows.append({
                "Dataset": dataset,
                "Variant": variant,
                "Benign selected": v["benign"],
                "Attack selected": v["attack"],
                "Total selected": v["rows"],
                "Sampling method": "Stratified random, balanced binary",
            })

    df_dataset = pd.DataFrame(dataset_rows)
    df_sampling = pd.DataFrame(sampling_rows)

    df_dataset.to_csv(OUT / "table_1_dataset_summary.csv", index=False)
    df_sampling.to_csv(OUT / "table_2_sampling_strategy.csv", index=False)

    return df_dataset, df_sampling


# --------------------------------------------------
# Domain shift metrics
# --------------------------------------------------
def build_shift_table():
    data = load_json(BASE / "pairs" / "balanced_100k" / "shift_summary.json")

    rows = []

    for pair, info in data.items():
        rows.append({
            "Pair": short_pair(pair),
            "Wasserstein": round(info["wasserstein"]["clipped_mean"], 3),
            "Domain AUROC": round(info["domain_classifier"]["auroc"], 3),
            "Cov shift": round(info["covariance_shift"]["mean_absolute_log10_variance_ratio"], 3),
            "Label shift": round(info["label_shift"]["absolute_attack_prior_difference"], 3),
            "Class-cond shift": round(
                info["class_conditional_shift"]["summary"]["overall_class_conditional_shift_mean"],
                3,
            ),
            "Shift type": info["shift_classification"]["shift_type"].replace("_", " "),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_3_shift_metrics.csv", index=False)

    # --------------------------------------------------
    # Slide-ready shift metrics chart
    # Use this as backup, not main slide.
    # --------------------------------------------------
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(9.5, 5.5))

    plt.bar(
        x - width,
        plot_df["Wasserstein"],
        width,
        label="Wasserstein",
    )

    plt.bar(
        x,
        plot_df["Domain AUROC"],
        width,
        label="Domain AUROC",
    )

    plt.bar(
        x + width,
        plot_df["Class-cond shift"],
        width,
        label="Class-cond shift",
    )

    plt.xticks(x, plot_df["Pair"], rotation=30, ha="right")
    plt.ylabel("Metric value")
    plt.legend(frameon=False, loc="upper left")

    savefig(OUT / "slide_shift_metrics_bar.png")

    return df


# --------------------------------------------------
# Baseline model results
# --------------------------------------------------
def build_baseline_table():
    rows = []

    for pair in PAIRS:
        path = (
            BASE
            / "pairs"
            / "balanced_100k"
            / pair
            / "mlp_baseline_results"
            / "summary.json"
        )

        if not path.exists():
            print(f"Missing baseline summary, skipping: {path}")
            continue

        data = load_json(path)
        comp = data["comparison"]

        rows.append({
            "Pair": short_pair(pair),
            "Zero-shot AUROC": round(comp["target_auroc_zero_shot"], 3),
            "Fine-tuned AUROC": round(comp["target_auroc_after_finetune"], 3),
            "Target-only AUROC": round(comp["target_auroc_upper_bound"], 3),
            "Target gain": round(comp["target_auroc_gain_from_finetune"], 3),
            "Source AUROC before": round(comp["source_auroc_before_finetune"], 3),
            "Source AUROC after": round(comp["source_auroc_after_finetune"], 3),
            "Source drop": round(comp["source_auroc_drop_after_finetune"], 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_4_mlp_baseline_results.csv", index=False)

    if df.empty:
        print("No baseline results found. Skipping baseline plots.")
        return df

    # --------------------------------------------------
    # Slide 4: Zero-shot vs fine-tuned target AUROC
    # --------------------------------------------------
        # --------------------------------------------------
    # Slide 4: Zero-shot vs fine-tuned target AUROC
    # --------------------------------------------------
    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(9.5, 5.5))

    plt.bar(
        x - width / 2,
        df["Zero-shot AUROC"],
        width,
        label="Zero-shot",
    )

    plt.bar(
        x + width / 2,
        df["Fine-tuned AUROC"],
        width,
        label="Fine-tuned",
    )

    plt.xticks(x, df["Pair"], rotation=30, ha="right")
    plt.ylabel("Target AUROC")
    plt.ylim(0, 1.08)

    # Cleaner slide legend: outside plot, top centre
    plt.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.5,
    )

    # Add light horizontal gridlines for readability
    plt.grid(axis="y", alpha=0.18, linewidth=0.8)

    savefig(OUT / "slide_zero_shot_vs_finetune.png")

# --------------------------------------------------
# Fine-tuning budget results
# --------------------------------------------------
def build_budget_tables_and_plots():
    pair = "NF-UNSW-NB15-v3__TO__NF-CSE-CIC-IDS2018-v3"

    path = (
        BASE
        / "pairs"
        / "balanced_100k"
        / pair
        / "mlp_finetune_budget_results"
        / "budget_summary.csv"
    )

    if not path.exists():
        print(f"Missing budget CSV: {path}")
        return None

    df = pd.read_csv(path).sort_values("target_budget_per_class")

    out = df[[
        "target_budget_per_class",
        "target_budget_total",
        "target_auroc_after_finetune",
        "target_auroc_gain",
        "source_auroc_after_finetune",
        "absolute_source_auroc_drop",
        "relative_source_auroc_drop_pct",
    ]].copy()

    out.columns = [
        "Budget per class",
        "Total target samples",
        "Target AUROC",
        "Target gain",
        "Source AUROC",
        "Absolute source drop",
        "Relative source drop %",
    ]

    for col in [
        "Target AUROC",
        "Target gain",
        "Source AUROC",
        "Absolute source drop",
        "Relative source drop %",
    ]:
        out[col] = out[col].round(3)

    out.to_csv(OUT / "table_5_finetuning_budget_tradeoff.csv", index=False)

    # --------------------------------------------------
    # Slide 6: Fine-tuning budget trade-off
    # --------------------------------------------------
    plt.figure(figsize=(9.5, 5.5))

    plt.plot(
        df["target_budget_per_class"],
        df["target_auroc_after_finetune"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Target AUROC",
    )

    plt.plot(
        df["target_budget_per_class"],
        df["source_auroc_after_finetune"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Source AUROC",
    )

    plt.axhline(
        df["target_auroc_zero_shot"].iloc[0],
        linestyle="--",
        linewidth=1.5,
        label="Target zero-shot",
    )

    plt.axhline(
        df["source_auroc_before_finetune"].iloc[0],
        linestyle=":",
        linewidth=1.8,
        label="Source before fine-tuning",
    )

    plt.xscale("log")
    plt.xlabel("Target fine-tuning samples per class")
    plt.ylabel("AUROC")
    plt.ylim(0.55, 1.05)
    plt.legend(frameon=False, loc="center right")

    savefig(OUT / "slide_finetuning_budget_tradeoff.png")

    # --------------------------------------------------
    # Backup: Adaptation vs forgetting scatter
    # --------------------------------------------------
    plt.figure(figsize=(8.5, 5.5))

    plt.scatter(
        df["absolute_source_auroc_drop"],
        df["target_auroc_gain"],
        s=80,
        alpha=0.85,
    )

    x_min = df["absolute_source_auroc_drop"].min()
    x_max = df["absolute_source_auroc_drop"].max()
    y_min = df["target_auroc_gain"].min()
    y_max = df["target_auroc_gain"].max()

    x_pad = max((x_max - x_min) * 0.08, 0.01)
    y_pad = max((y_max - y_min) * 0.10, 0.02)

    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    for _, row in df.iterrows():
        budget = int(row["target_budget_per_class"])

        plt.annotate(
            str(budget),
            (
                row["absolute_source_auroc_drop"],
                row["target_auroc_gain"],
            ),
            fontsize=11,
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.xlabel("Source AUROC drop")
    plt.ylabel("Target AUROC gain")

    savefig(OUT / "slide_adaptation_vs_forgetting.png")

    # --------------------------------------------------
    # Backup: Source forgetting vs budget
    # --------------------------------------------------
    plt.figure(figsize=(9.5, 5.5))

    plt.plot(
        df["target_budget_per_class"],
        df["absolute_source_auroc_drop"],
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Source AUROC drop",
    )

    plt.xscale("log")
    plt.xlabel("Target fine-tuning samples per class")
    plt.ylabel("Source AUROC drop")

    max_idx = df["absolute_source_auroc_drop"].idxmax()
    max_budget = df.loc[max_idx, "target_budget_per_class"]
    max_drop = df.loc[max_idx, "absolute_source_auroc_drop"]

    plt.ylim(0, max_drop * 1.35)

    plt.annotate(
        f"Largest drop = {max_drop:.3f}",
        xy=(max_budget, max_drop),
        xytext=(max_budget / 3, max_drop * 1.18),
        arrowprops=dict(arrowstyle="->", lw=1.5),
        fontsize=13,
        ha="center",
    )

    plt.legend(frameon=False, loc="upper left")

    savefig(OUT / "slide_budget_forgetting.png")

    return out


# --------------------------------------------------
# Robust PCA for one source-target pair
# Backup or optional visual
# --------------------------------------------------
def robust_pca_for_pair(pair: str, max_points: int = 4000):
    pair_dir = BASE / "pairs" / "balanced_100k" / pair

    Xs_path = pair_dir / "Xs_train.npy"
    Xt_path = pair_dir / "Xt_test.npy"

    if not Xs_path.exists() or not Xt_path.exists():
        print(f"Missing PCA inputs for {pair}")
        return

    Xs = np.load(Xs_path)
    Xt = np.load(Xt_path)

    rng = np.random.default_rng(42)

    Xs = Xs[
        rng.choice(
            len(Xs),
            size=min(max_points, len(Xs)),
            replace=False,
        )
    ]

    Xt = Xt[
        rng.choice(
            len(Xt),
            size=min(max_points, len(Xt)),
            replace=False,
        )
    ]

    X = np.vstack([Xs, Xt])

    # Robust clipping to reduce outlier dominance.
    lo = np.quantile(X, 0.01, axis=0)
    hi = np.quantile(X, 0.99, axis=0)
    X = np.clip(X, lo, hi)

    X = RobustScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)

    labels = np.array(["Source"] * len(Xs) + ["Target"] * len(Xt))

    plt.figure(figsize=(8.5, 5.5))

    plt.scatter(
        Xp[labels == "Source", 0],
        Xp[labels == "Source", 1],
        s=14,
        alpha=0.45,
        label="Source",
    )

    plt.scatter(
        Xp[labels == "Target", 0],
        Xp[labels == "Target", 1],
        s=14,
        alpha=0.45,
        label="Target",
    )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.legend(frameon=False, loc="best")

    savefig(OUT / "slide_pca_UNSW_to_CIC.png")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Generating slide-ready proposal artifacts...")

    df_dataset, df_sampling = build_dataset_tables()
    df_shift = build_shift_table()
    df_baseline = build_baseline_table()
    df_budget = build_budget_tables_and_plots()

    robust_pca_for_pair("NF-UNSW-NB15-v3__TO__NF-CSE-CIC-IDS2018-v3")

    print(f"\nSaved all outputs to: {OUT.resolve()}")
    print("\nSlide-ready visuals generated:")
    print("- slide_shift_metrics_bar.png")
    print("- slide_zero_shot_vs_finetune.png")
    print("- slide_source_forgetting_by_pair.png")
    print("- slide_finetuning_budget_tradeoff.png")
    print("- slide_adaptation_vs_forgetting.png")
    print("- slide_budget_forgetting.png")
    print("- slide_pca_UNSW_to_CIC.png")


if __name__ == "__main__":
    main()