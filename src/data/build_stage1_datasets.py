import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ATTACK_COLUMN,
    DATASET_PATHS,
    DROP_COLUMNS,
    LABEL_COLUMN,
    RANDOM_STATE,
    STAGE1_DIR,
)
from src.data.preprocess import (
    clean_dataframe,
    get_common_feature_columns,
    save_stage1_dataset,
)


def make_balanced_subset(
    X_df: pd.DataFrame,
    y: np.ndarray,
    attack: pd.Series,
    max_per_class: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    if max_per_class is None:
        return X_df, y, attack

    y_series = pd.Series(y, name="Label")
    benign_idx = y_series[y_series == 0].index.to_numpy()
    attack_idx = y_series[y_series == 1].index.to_numpy()

    benign_take = min(len(benign_idx), max_per_class)
    attack_take = min(len(attack_idx), max_per_class)

    rng = np.random.default_rng(random_state)
    benign_sample = rng.choice(benign_idx, size=benign_take, replace=False)
    attack_sample = rng.choice(attack_idx, size=attack_take, replace=False)

    selected_idx = np.concatenate([benign_sample, attack_sample])
    rng.shuffle(selected_idx)

    return (
        X_df.iloc[selected_idx].reset_index(drop=True),
        y[selected_idx],
        attack.iloc[selected_idx].reset_index(drop=True),
    )


def main():
    # Shared schema from your inspection is already matched, but this rechecks in code.
    common_features = get_common_feature_columns(
        dataset_paths=DATASET_PATHS,
        drop_columns=DROP_COLUMNS,
        nrows=5000,
    )

    summary = {
        "common_feature_count": len(common_features),
        "common_feature_columns": common_features,
        "datasets": {},
    }

    feature_path = STAGE1_DIR / "common_features.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Set this to None if you want full cleaned datasets.
    # For practical first runs, 100_000 per class is a good starting point.
    max_per_class = 100_000

    for dataset_name, path in DATASET_PATHS.items():
        print(f"\nProcessing {dataset_name}")
        print(f"Reading: {path}")

        df = pd.read_csv(path)

        X_df, y, attack = clean_dataframe(
            df=df,
            drop_columns=DROP_COLUMNS,
            label_column=LABEL_COLUMN,
            attack_column=ATTACK_COLUMN,
        )

        # Keep only shared features and fixed order
        X_df = X_df[common_features].copy()

        # Save full cleaned dataset
        full_dir = STAGE1_DIR / "full"
        save_stage1_dataset(full_dir, dataset_name, X_df, y, attack)

        # Save balanced subset for quick experiments
        subset_X, subset_y, subset_attack = make_balanced_subset(
            X_df=X_df,
            y=y,
            attack=attack,
            max_per_class=max_per_class,
            random_state=RANDOM_STATE,
        )

        subset_dir = STAGE1_DIR / "balanced_100k"
        save_stage1_dataset(subset_dir, dataset_name, subset_X, subset_y, subset_attack)

        summary["datasets"][dataset_name] = {
            "full_rows": int(len(X_df)),
            "full_benign": int((y == 0).sum()),
            "full_attack": int((y == 1).sum()),
            "subset_rows": int(len(subset_X)),
            "subset_benign": int((subset_y == 0).sum()),
            "subset_attack": int((subset_y == 1).sum()),
        }

    with open(STAGE1_DIR / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nStage 1 dataset build complete.")
    print(f"Saved to: {STAGE1_DIR}")


if __name__ == "__main__":
    main()