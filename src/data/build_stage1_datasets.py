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


BALANCED_SAMPLE_SIZES = [10_000, 25_000, 50_000, 100_000]


def make_balanced_subset(
    X_df: pd.DataFrame,
    y: np.ndarray,
    attack: pd.Series,
    max_per_class: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, dict]:
    """
    Stratified random sampling by binary class.

    This creates a fair controlled subset by:
    - sampling benign and attack flows separately
    - using the same max_per_class rule for each dataset
    - using a fixed random seed for reproducibility
    - preserving a 50/50 benign/attack class balance where possible
    """
    if max_per_class is None:
        sampling_meta = {
            "sampling_strategy": "full_dataset",
            "max_per_class": None,
            "random_state": random_state,
            "benign_available": int((y == 0).sum()),
            "attack_available": int((y == 1).sum()),
            "benign_selected": int((y == 0).sum()),
            "attack_selected": int((y == 1).sum()),
        }
        return X_df, y, attack, sampling_meta

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

    sampling_meta = {
        "sampling_strategy": "stratified_random_balanced_binary",
        "max_per_class": int(max_per_class),
        "random_state": int(random_state),
        "benign_available": int(len(benign_idx)),
        "attack_available": int(len(attack_idx)),
        "benign_selected": int(benign_take),
        "attack_selected": int(attack_take),
        "total_selected": int(len(selected_idx)),
        "fairness_definition": (
            "Same feature space, same class-balance rule, same random sampling procedure, "
            "same random seed, and same downstream split protocol for every dataset."
        ),
    }

    return (
        X_df.iloc[selected_idx].reset_index(drop=True),
        y[selected_idx],
        attack.iloc[selected_idx].reset_index(drop=True),
        sampling_meta,
    )


def main():
    common_features = get_common_feature_columns(
        dataset_paths=DATASET_PATHS,
        drop_columns=DROP_COLUMNS,
        nrows=5000,
    )

    summary = {
        "common_feature_count": len(common_features),
        "common_feature_columns": common_features,
        "sampling_variants": {
            "full": {
                "description": "Full cleaned dataset. Keeps original class distribution.",
            },
            **{
                f"balanced_{n // 1000}k": {
                    "description": (
                        f"Stratified random balanced subset with up to {n} benign "
                        f"and {n} attack samples per dataset."
                    ),
                    "max_per_class": n,
                    "random_state": RANDOM_STATE,
                }
                for n in BALANCED_SAMPLE_SIZES
            },
        },
        "datasets": {},
    }

    feature_path = STAGE1_DIR / "common_features.json"
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

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

        X_df = X_df[common_features].copy()

        dataset_summary = {
            "full_rows": int(len(X_df)),
            "full_benign": int((y == 0).sum()),
            "full_attack": int((y == 1).sum()),
            "variants": {},
        }

        # Save full cleaned dataset
        full_dir = STAGE1_DIR / "full"
        save_stage1_dataset(full_dir, dataset_name, X_df, y, attack)

        dataset_summary["variants"]["full"] = {
            "rows": int(len(X_df)),
            "benign": int((y == 0).sum()),
            "attack": int((y == 1).sum()),
            "sampling_strategy": "full_dataset",
        }

        # Save multiple balanced subsets for sampling sensitivity
        for max_per_class in BALANCED_SAMPLE_SIZES:
            variant_name = f"balanced_{max_per_class // 1000}k"
            print(f"Creating {variant_name} for {dataset_name}")

            subset_X, subset_y, subset_attack, sampling_meta = make_balanced_subset(
                X_df=X_df,
                y=y,
                attack=attack,
                max_per_class=max_per_class,
                random_state=RANDOM_STATE,
            )

            subset_dir = STAGE1_DIR / variant_name
            save_stage1_dataset(
                subset_dir,
                dataset_name,
                subset_X,
                subset_y,
                subset_attack,
            )

            dataset_summary["variants"][variant_name] = {
                "rows": int(len(subset_X)),
                "benign": int((subset_y == 0).sum()),
                "attack": int((subset_y == 1).sum()),
                "sampling": sampling_meta,
            }

        summary["datasets"][dataset_name] = dataset_summary

    with open(STAGE1_DIR / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nStage 1 dataset build complete.")
    print(f"Saved to: {STAGE1_DIR}")


if __name__ == "__main__":
    main()