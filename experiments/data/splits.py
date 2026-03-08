"""Stratified train/val split generation and test set loading."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _load_bad_images(bad_images_path: str = "experiments/data/bad_images.json") -> set[str]:
    """Load the set of known corrupt image filenames."""
    if os.path.exists(bad_images_path):
        with open(bad_images_path) as f:
            return set(json.load(f))
    return set()


def load_and_filter_data(
    csv_path: str,
    image_dir: str,
    image_extension: str = ".jpg",
    drop_lens_status: list[str] | None = None,
    corneal_map: dict[str, str] | None = None,
    bad_images_path: str = "experiments/data/bad_images.json",
) -> pd.DataFrame:
    """Load Visilant CSV, filter to images on disk, apply label mappings."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Filter to images that exist on disk
    existing_images = set(os.listdir(image_dir))
    bad_images = _load_bad_images(bad_images_path)
    valid_images = existing_images - bad_images
    df["image_file"] = df["image_name"].apply(lambda x: f"{x}{image_extension}")
    df = df[df["image_file"].apply(lambda x: x in valid_images)].copy()
    if bad_images:
        print(f"Filtered out {len(bad_images)} known corrupt images")

    # Drop unwanted lens status values
    if drop_lens_status:
        df = df[~df["mapped_lens_status"].isin(drop_lens_status)].copy()

    # Map corneal abnormality to binned classes
    if corneal_map:
        df["corneal_binned"] = df["mapped_corneal_abnormality"].map(corneal_map)
        # Any unmapped values go to "Rare"
        df["corneal_binned"] = df["corneal_binned"].fillna("Rare")
    else:
        df["corneal_binned"] = df["mapped_corneal_abnormality"]

    return df


def get_test_image_names(test_csv_path: str) -> set[str]:
    """Extract all test image names (both eyes) from the test CSV."""
    test_df = pd.read_csv(test_csv_path)
    test_images = set()
    for col in ["image_name_right", "image_name_left"]:
        if col in test_df.columns:
            names = test_df[col].dropna().astype(str)
            # Strip .jpg if present
            test_images.update(names.apply(lambda x: x.replace(".jpg", "")))
    return test_images


def create_stratified_split(
    df: pd.DataFrame,
    test_csv_path: str,
    val_fraction: float = 0.1,
    seed: int = 42,
    save_path: str | None = None,
) -> dict:
    """Create stratified train/val split, excluding test set images.

    Stratification is on the composite key: mapped_lens_status|corneal_binned.
    Returns dict with 'train_indices' and 'val_indices'.
    """
    # Exclude test images
    test_images = get_test_image_names(test_csv_path)
    mask = ~df["image_name"].isin(test_images)
    df_trainval = df[mask].copy()

    # Create composite stratification key
    df_trainval["strat_key"] = (
        df_trainval["mapped_lens_status"] + "|" + df_trainval["corneal_binned"]
    )

    # For very rare combos, group them to avoid split errors
    combo_counts = df_trainval["strat_key"].value_counts()
    rare_combos = combo_counts[combo_counts < 2].index
    df_trainval.loc[df_trainval["strat_key"].isin(rare_combos), "strat_key"] = "RARE_COMBO"

    train_idx, val_idx = train_test_split(
        df_trainval.index.values,
        test_size=val_fraction,
        stratify=df_trainval["strat_key"],
        random_state=seed,
    )

    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_images": sorted(test_images),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_test_images": len(test_images),
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"Splits saved to {save_path}")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test images excluded: {len(test_images)}")

    return splits


def load_splits(path: str) -> dict:
    """Load previously saved splits."""
    with open(path) as f:
        return json.load(f)


def load_test_data(test_csv_path: str, image_dir: str) -> pd.DataFrame:
    """Load and unstack test CSV into per-image rows with labels.

    Maps binary columns to our taxonomy:
    - lens_status: direct from lens_status_{side} column
    - corneal_abnormality: derived from binary indicator columns
    """
    test_df = pd.read_csv(test_csv_path)
    rows = []

    lens_map = {
        "Clear crystalline lens": "clear_crystalline_lens",
        "Immature cataract": "immature_cataract",
        "Lens Changes": "early_lens_changes",
        "PCIOL": "PCIOL",
        "cannot visualize lens": "not_able_to_visualize_lens",
        "Mature cataract": "mature_cataract",
        "Aphakia": "aphakia",
    }

    for _, row in test_df.iterrows():
        for side in ["right", "left"]:
            img_name = row.get(f"image_name_{side}")
            if pd.isna(img_name):
                continue

            img_file = str(img_name)
            if not img_file.endswith(".jpg"):
                img_file += ".jpg"
            img_path = os.path.join(image_dir, img_file)
            if not os.path.exists(img_path):
                continue

            # Lens status
            raw_lens = row.get(f"lens_status_{side}", "")
            lens = lens_map.get(raw_lens, raw_lens)

            # Corneal abnormality from binary columns
            active = row.get(f"Active corneal infection_{side}", 0)
            inactive = row.get(f"Inactive corneal opacity_{side}", 0)
            if active == 1:
                cornea = "Active corneal infection"
            elif inactive == 1:
                cornea = "Inactive corneal opacity"
            else:
                cornea = "Normal"

            rows.append({
                "image_name": str(img_name).replace(".jpg", ""),
                "image_file": img_file,
                "mapped_lens_status": lens,
                "corneal_binned": cornea,
                "age": row.get("Age"),
                "gender": row.get("gender"),
            })

    return pd.DataFrame(rows)
