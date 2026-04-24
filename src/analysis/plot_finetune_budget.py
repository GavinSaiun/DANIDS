import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PAIR_DIR


def main():
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

    # Plot 1: Target AUROC and source AUROC after fine-tuning
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["target_budget_per_class"],
        df["target_auroc_after_finetune"],
        marker="o",
        label="Target AUROC after fine-tuning",
    )
    plt.plot(
        df["target_budget_per_class"],
        df["source_auroc_after_finetune"],
        marker="o",
        label="Source AUROC after fine-tuning",
    )
    plt.axhline(
        df["target_auroc_zero_shot"].iloc[0],
        linestyle="--",
        label="Target zero-shot AUROC",
    )
    plt.axhline(
        df["source_auroc_before_finetune"].iloc[0],
        linestyle="--",
        label="Source AUROC before fine-tuning",
    )
    plt.xscale("log")
    plt.xlabel("Target fine-tuning samples per class")
    plt.ylabel("AUROC")
    plt.title(f"Fine-tuning Budget Trade-off\n{args.pair}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "budget_tradeoff_auroc.png", dpi=200)
    plt.close()

    # Plot 2: Source forgetting
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["target_budget_per_class"],
        df["absolute_source_auroc_drop"],
        marker="o",
        label="Absolute source AUROC drop",
    )
    plt.plot(
        df["target_budget_per_class"],
        df["relative_source_auroc_drop_pct"],
        marker="o",
        label="Relative source AUROC drop (%)",
    )
    plt.xscale("log")
    plt.xlabel("Target fine-tuning samples per class")
    plt.ylabel("Forgetting")
    plt.title(f"Source Forgetting vs Fine-tuning Budget\n{args.pair}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "budget_forgetting.png", dpi=200)
    plt.close()

    # Plot 3: Target gain vs source drop
    plt.figure(figsize=(7, 5))
    plt.scatter(
        df["absolute_source_auroc_drop"],
        df["target_auroc_gain"],
    )

    for _, row in df.iterrows():
        plt.annotate(
            str(int(row["target_budget_per_class"])),
            (
                row["absolute_source_auroc_drop"],
                row["target_auroc_gain"],
            ),
            fontsize=9,
        )

    plt.xlabel("Absolute source AUROC drop")
    plt.ylabel("Target AUROC gain")
    plt.title(f"Adaptation vs Forgetting\n{args.pair}")
    plt.tight_layout()
    plt.savefig(results_dir / "adaptation_vs_forgetting.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {results_dir}")


if __name__ == "__main__":
    main()