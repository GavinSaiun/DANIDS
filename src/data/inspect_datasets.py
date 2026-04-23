from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

DATASETS = {
    "NF-UNSW-NB15-v3": BASE_DIR / "Datasets" / "NF-UNSW-NB15-v3" / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3.csv",
    "NF-ToN-IoT-v3": BASE_DIR / "Datasets" / "NF-ToN-IoT-v3" / "02934b58528a226b_NFV3DATA-A11964_A11964" / "data" / "NF-ToN-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": BASE_DIR / "Datasets" / "NF-CSE-CIC-IDS2018-v3" / "f78acbaa2afe1595_NFV3DATA-A11964_A11964" / "data" / "NF-CICIDS2018-v3.csv",
}

OUTPUT_DIR = BASE_DIR / "outputs" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_dataset(name: str, path: Path) -> dict:
    print(f"\n{'=' * 80}")
    print(f"Inspecting: {name}")
    print(f"Path: {path}")
    print(f"{'=' * 80}")

    df = pd.read_csv(path, nrows=10000)

    print(f"Shape (sample): {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    print("\nColumns with missing values:")
    if len(missing_counts) == 0:
        print("None")
    else:
        print(missing_counts)

    numeric_df = df.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum()
    inf_counts = inf_counts[inf_counts > 0].sort_values(ascending=False)

    print("\nNumeric columns with inf values:")
    if len(inf_counts) == 0:
        print("None")
    else:
        print(inf_counts)

    print("\nPotential label columns:")
    potential_label_cols = [col for col in df.columns if "label" in col.lower() or "attack" in col.lower()]
    print(potential_label_cols)

    for col in potential_label_cols:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False).head(20))

    summary = {
        "dataset": name,
        "path": str(path),
        "sample_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_columns": missing_counts.to_dict(),
        "inf_columns": inf_counts.to_dict(),
        "potential_label_columns": potential_label_cols,
    }

    return summary


def compare_columns(dataset_summaries: list[dict]) -> None:
    print(f"\n{'=' * 80}")
    print("Comparing column schemas")
    print(f"{'=' * 80}")

    column_sets = {
        summary["dataset"]: set(summary["columns"])
        for summary in dataset_summaries
    }

    dataset_names = list(column_sets.keys())
    base_name = dataset_names[0]
    base_cols = column_sets[base_name]

    for other_name in dataset_names[1:]:
        other_cols = column_sets[other_name]

        only_in_base = sorted(base_cols - other_cols)
        only_in_other = sorted(other_cols - base_cols)

        print(f"\nComparing {base_name} vs {other_name}")
        print(f"Only in {base_name}: {only_in_base if only_in_base else 'None'}")
        print(f"Only in {other_name}: {only_in_other if only_in_other else 'None'}")

        if not only_in_base and not only_in_other:
            print("Schemas match exactly.")


def main():
    summaries = []

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"File not found: {path}")
            continue

        summary = inspect_dataset(name, path)
        summaries.append(summary)

    if summaries:
        compare_columns(summaries)

        summary_df = pd.DataFrame(
            [
                {
                    "dataset": s["dataset"],
                    "sample_rows": s["sample_rows"],
                    "num_columns": s["num_columns"],
                    "potential_label_columns": ", ".join(s["potential_label_columns"]),
                }
                for s in summaries
            ]
        )

        summary_path = OUTPUT_DIR / "dataset_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()