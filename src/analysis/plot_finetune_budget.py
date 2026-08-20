import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PAIR_DIR


# --------------------------------------------------
# Presentation-quality plot style
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
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, type=str)
    parser.add_argument("--stage1_variant", default="balanced_100k", type=str)
    args = parser.parse_args()

    results_dir = (
        PAIR_DIR
        / args.stage1_variant
        / args.pair
        / "mlp_finetune_budget_results"
    )

    csv_path = results_dir / "budget_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.sort_values("target_budget_per_class")

    pair_display = short_pair(args.pair)

    # --------------------------------------------------
    # Plot 1: Fine-tuning trade-off
    # Use this for main Slide 6.
    # --------------------------------------------------
    plt.figure(figsize=(9, 5.5))

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

    # Smaller legend than before. Put explanation in PowerPoint, not inside the plot.
    plt.legend(frameon=False, loc="center right")

    savefig(results_dir / "slide_finetuning_tradeoff.png")

    # --------------------------------------------------
    # Plot 2: Source forgetting only
    # Use as backup or detailed forgetting slide.
    # --------------------------------------------------
    plt.figure(figsize=(9, 5.5))

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

    # More space so the annotation does not hit the top border
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

    savefig(results_dir / "slide_budget_forgetting.png")

    # --------------------------------------------------
    # Plot 3: Adaptation vs forgetting scatter
    # Use as backup slide.
    # --------------------------------------------------
    plt.figure(figsize=(8, 5.5))

    plt.scatter(
        df["absolute_source_auroc_drop"],
        df["target_auroc_gain"],
        s=80,
        alpha=0.85,
    )

    # Add a little padding so labels are not clipped
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

        # Slightly smaller labels to reduce overlap
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

    # No internal sentence here.
    # Add “Each point shows a fine-tuning budget” in PowerPoint if needed.

    savefig(results_dir / "slide_adaptation_vs_forgetting.png")

    print(f"Saved presentation-quality plots to: {results_dir}")
    print(f"Pair: {pair_display}")
    print("Generated:")
    print("- slide_finetuning_tradeoff.png")
    print("- slide_budget_forgetting.png")
    print("- slide_adaptation_vs_forgetting.png")


if __name__ == "__main__":
    main()