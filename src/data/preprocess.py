import json
from pathlib import Path

import numpy as np
import pandas as pd


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def clean_dataframe(
    df: pd.DataFrame,
    drop_columns: list[str],
    label_column: str,
    attack_column: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    df = replace_inf_with_nan(df).copy()

    # Binary label
    y = df[label_column].astype(int).to_numpy()

    # Keep attack labels for later analysis only
    attack = df[attack_column].astype(str)

    # Features only
    X_df = df.drop(columns=drop_columns).copy()

    # Force numeric
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Fill NaN after coercion / inf replacement
    X_df = X_df.fillna(0.0)

    # Cast to float32 to reduce memory
    X_df = X_df.astype(np.float32)

    return X_df, y, attack


def get_common_feature_columns(
    dataset_paths: dict[str, Path],
    drop_columns: list[str],
    nrows: int = 5000,
) -> list[str]:
    feature_sets = []

    for name, path in dataset_paths.items():
        df = pd.read_csv(path, nrows=nrows)
        cols = [c for c in df.columns if c not in drop_columns]
        feature_sets.append(set(cols))

    common = set.intersection(*feature_sets)
    return sorted(common)


def save_stage1_dataset(
    output_dir: Path,
    dataset_name: str,
    X_df: pd.DataFrame,
    y: np.ndarray,
    attack: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    X_path = output_dir / f"{dataset_name}_X.parquet"
    y_path = output_dir / f"{dataset_name}_y.npy"
    attack_path = output_dir / f"{dataset_name}_attack.csv"
    meta_path = output_dir / f"{dataset_name}_meta.json"

    X_df.to_parquet(X_path, index=False)
    np.save(y_path, y)
    attack.to_frame(name="Attack").to_csv(attack_path, index=False)

    meta = {
        "dataset": dataset_name,
        "num_rows": int(len(X_df)),
        "num_features": int(X_df.shape[1]),
        "feature_columns": X_df.columns.tolist(),
        "label_distribution": {
            "benign_0": int((y == 0).sum()),
            "attack_1": int((y == 1).sum()),
        },
        "attack_distribution_top20": attack.value_counts().head(20).to_dict(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)