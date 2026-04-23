import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import PAIR_DIR, RANDOM_STATE, STAGE1_DIR


def load_stage1_dataset(stage1_subdir: Path, dataset_name: str):
    X = pd.read_parquet(stage1_subdir / f"{dataset_name}_X.parquet")
    y = np.load(stage1_subdir / f"{dataset_name}_y.npy")
    attack = pd.read_csv(stage1_subdir / f"{dataset_name}_attack.csv")["Attack"]
    return X, y, attack


def split_source_target(
    X_source,
    y_source,
    X_target,
    y_target,
    random_state: int = 42,
):
    # Source: train / val / test = 70 / 15 / 15
    Xs_train, Xs_temp, ys_train, ys_temp = train_test_split(
        X_source,
        y_source,
        test_size=0.30,
        stratify=y_source,
        random_state=random_state,
    )

    Xs_val, Xs_test, ys_val, ys_test = train_test_split(
        Xs_temp,
        ys_temp,
        test_size=0.50,
        stratify=ys_temp,
        random_state=random_state,
    )

    # Target: adapt / test = 20 / 80
    Xt_adapt, Xt_test, yt_adapt, yt_test = train_test_split(
        X_target,
        y_target,
        test_size=0.80,
        stratify=y_target,
        random_state=random_state,
    )

    return {
        "Xs_train": Xs_train,
        "ys_train": ys_train,
        "Xs_val": Xs_val,
        "ys_val": ys_val,
        "Xs_test": Xs_test,
        "ys_test": ys_test,
        "Xt_adapt": Xt_adapt,
        "yt_adapt": yt_adapt,
        "Xt_test": Xt_test,
        "yt_test": yt_test,
    }


def fit_source_only_scaler(splits: dict):
    scaler = StandardScaler()
    scaler.fit(splits["Xs_train"])

    scaled = {}
    for key, value in splits.items():
        if key.startswith("X"):
            scaled[key] = scaler.transform(value).astype(np.float32)
        else:
            scaled[key] = value

    return scaler, scaled


def save_pair_artifacts(pair_output_dir: Path, scaled: dict, scaler, metadata: dict):
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    for key, value in scaled.items():
        if key.startswith("X") or key.startswith("y"):
            np.save(pair_output_dir / f"{key}.npy", value)

    joblib.dump(scaler, pair_output_dir / "source_scaler.joblib")

    with open(pair_output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--target", required=True, type=str)
    parser.add_argument(
        "--stage1_variant",
        default="balanced_100k",
        type=str,
        help="Use 'balanced_100k' or 'full'",
    )
    args = parser.parse_args()

    if args.source == args.target:
        raise ValueError("Source and target must be different.")

    stage1_subdir = STAGE1_DIR / args.stage1_variant

    Xs, ys, _ = load_stage1_dataset(stage1_subdir, args.source)
    Xt, yt, _ = load_stage1_dataset(stage1_subdir, args.target)

    splits = split_source_target(
        X_source=Xs,
        y_source=ys,
        X_target=Xt,
        y_target=yt,
        random_state=RANDOM_STATE,
    )

    scaler, scaled = fit_source_only_scaler(splits)

    pair_name = f"{args.source}__TO__{args.target}"
    pair_output_dir = PAIR_DIR / args.stage1_variant / pair_name

    metadata = {
        "source": args.source,
        "target": args.target,
        "stage1_variant": args.stage1_variant,
        "random_state": RANDOM_STATE,
        "source_split_sizes": {
            "train": int(len(scaled["ys_train"])),
            "val": int(len(scaled["ys_val"])),
            "test": int(len(scaled["ys_test"])),
        },
        "target_split_sizes": {
            "adapt": int(len(scaled["yt_adapt"])),
            "test": int(len(scaled["yt_test"])),
        },
        "num_features": int(scaled["Xs_train"].shape[1]),
        "scaler": "StandardScaler fitted on source train only",
    }

    save_pair_artifacts(pair_output_dir, scaled, scaler, metadata)

    print(f"Saved pair artifacts to: {pair_output_dir}")


if __name__ == "__main__":
    main()