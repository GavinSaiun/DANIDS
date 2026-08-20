from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS = {
    "NF-UNSW-NB15-v3": BASE_DIR / "Datasets" / "NF-UNSW-NB15-v3" / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3.csv",
    "NF-ToN-IoT-v3": BASE_DIR / "Datasets" / "NF-ToN-IoT-v3" / "02934b58528a226b_NFV3DATA-A11964_A11964" / "data" / "NF-ToN-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": BASE_DIR / "Datasets" / "NF-CSE-CIC-IDS2018-v3" / "f78acbaa2afe1595_NFV3DATA-A11964_A11964" / "data" / "NF-CICIDS2018-v3.csv",
}

OUTPUT_DIR = BASE_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "Label",
    "Attack",
]

SAMPLE_PER_DATASET = 10000
RANDOM_STATE = 42


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


def short_dataset_name(name: str) -> str:
    return (
        name.replace("NF-", "")
        .replace("-v3", "")
        .replace("CSE-CIC-IDS2018", "CIC")
        .replace("UNSW-NB15", "UNSW")
        .replace("ToN-IoT", "ToN")
    )


def label_name(label) -> str:
    label = str(label)

    if label == "0":
        return "Benign"
    if label == "1":
        return "Attack"

    return label


def load_sample(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if len(df) > SAMPLE_PER_DATASET:
        df = df.sample(n=SAMPLE_PER_DATASET, random_state=RANDOM_STATE)

    df["dataset"] = name
    return df


def plot_class_distribution(dfs: dict[str, pd.DataFrame]) -> None:
    rows = []

    for name, df in dfs.items():
        counts = df["Label"].value_counts(dropna=False)

        for label, count in counts.items():
            rows.append({
                "dataset": short_dataset_name(name),
                "class": label_name(label),
                "count": count,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    pivot = summary.pivot(index="dataset", columns="class", values="count").fillna(0)

    # Keep column order consistent if both exist
    ordered_cols = [col for col in ["Benign", "Attack"] if col in pivot.columns]
    remaining_cols = [col for col in pivot.columns if col not in ordered_cols]
    pivot = pivot[ordered_cols + remaining_cols]

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5.5),
        width=0.72,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Number of samples")
    ax.legend(title="Class", frameon=False)

    plt.xticks(rotation=0)
    savefig(OUTPUT_DIR / "slide_class_distribution.png")


def plot_pca(dfs: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(dfs.values(), ignore_index=True)

    feature_cols = [
        col for col in combined.columns
        if col not in DROP_COLUMNS + ["dataset"]
    ]

    X = combined[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dataset = combined["dataset"]

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "dataset": y_dataset,
    })

    pca_df.to_csv(OUTPUT_DIR / "pca_projection.csv", index=False)

    plt.figure(figsize=(9, 5.5))

    # Force a nice legend order
    dataset_order = [
        "NF-UNSW-NB15-v3",
        "NF-ToN-IoT-v3",
        "NF-CSE-CIC-IDS2018-v3",
    ]

    for dataset_name in dataset_order:
        if dataset_name not in set(pca_df["dataset"]):
            continue

        subset = pca_df[pca_df["dataset"] == dataset_name]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            s=14,
            alpha=0.45,
            label=short_dataset_name(dataset_name),
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.legend(frameon=False, loc="best")

    # No title or message inside graph.
    # Add the message in PowerPoint instead.
    savefig(OUTPUT_DIR / "slide_pca_projection.png")


def main() -> None:
    dfs = {}

    for name, path in DATASETS.items():
        print(f"Loading {name}")
        dfs[name] = load_sample(name, path)

    plot_class_distribution(dfs)
    plot_pca(dfs)

    print(f"EDA outputs saved to: {OUTPUT_DIR}")
    print("Generated:")
    print("- slide_class_distribution.png")
    print("- slide_pca_projection.png")


if __name__ == "__main__":
    main()