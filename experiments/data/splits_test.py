"""Tests for splits module."""

import json
import os

import pandas as pd
import pytest

from data.splits import (
    _load_bad_images,
    create_stratified_split,
    get_test_image_names,
    load_and_filter_data,
)


class TestLoadBadImages:
    def test_existing_file(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump(["img1.jpg", "img2.jpg"], f)
        result = _load_bad_images(path)
        assert result == {"img1.jpg", "img2.jpg"}

    def test_missing_file(self, tmp_path):
        result = _load_bad_images(str(tmp_path / "nonexistent.json"))
        assert result == set()


class TestLoadAndFilterData:
    @pytest.fixture
    def setup_data(self, tmp_path):
        """Create a minimal CSV and image directory."""
        image_dir = str(tmp_path / "images")
        os.makedirs(image_dir)
        for name in ["a", "b", "c"]:
            with open(os.path.join(image_dir, f"{name}.jpg"), "w") as f:
                f.write("fake")

        csv_path = str(tmp_path / "data.csv")
        df = pd.DataFrame({
            "image_name": ["a", "b", "c", "missing"],
            "mapped_lens_status": ["clear", "PCIOL", "clear", "clear"],
            "mapped_corneal_abnormality": ["Normal", "Normal", "Active", "Normal"],
        })
        df.to_csv(csv_path, index=False)

        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            json.dump([], f)

        return csv_path, image_dir, bad_path

    def test_filters_missing_images(self, setup_data):
        csv_path, image_dir, bad_path = setup_data
        result = load_and_filter_data(csv_path, image_dir, bad_images_path=bad_path)
        assert len(result) == 3  # "missing" filtered out

    def test_bad_images_excluded(self, setup_data, tmp_path):
        csv_path, image_dir, _ = setup_data
        bad_path = str(tmp_path / "bad2.json")
        with open(bad_path, "w") as f:
            json.dump(["a.jpg"], f)
        result = load_and_filter_data(csv_path, image_dir, bad_images_path=bad_path)
        assert len(result) == 2

    def test_drop_lens_status(self, setup_data):
        csv_path, image_dir, bad_path = setup_data
        result = load_and_filter_data(
            csv_path, image_dir, drop_lens_status=["PCIOL"], bad_images_path=bad_path
        )
        assert "PCIOL" not in result["mapped_lens_status"].values

    def test_corneal_map(self, setup_data):
        csv_path, image_dir, bad_path = setup_data
        corneal_map = {"Normal": "Normal", "Active": "Active corneal infection"}
        result = load_and_filter_data(
            csv_path, image_dir, corneal_map=corneal_map, bad_images_path=bad_path
        )
        assert "corneal_binned" in result.columns
        # Unmapped values should go to "Rare" (but Active is mapped here)
        assert set(result["corneal_binned"].unique()) <= {"Normal", "Active corneal infection", "Rare"}


class TestGetTestImageNames:
    def test_extracts_both_eyes(self, tmp_path):
        csv_path = str(tmp_path / "test.csv")
        df = pd.DataFrame({
            "image_name_right": ["img_r1.jpg", "img_r2"],
            "image_name_left": ["img_l1", "img_l2.jpg"],
        })
        df.to_csv(csv_path, index=False)
        result = get_test_image_names(csv_path)
        # .jpg should be stripped
        assert "img_r1" in result
        assert "img_r2" in result
        assert "img_l1" in result
        assert "img_l2" in result


class TestCreateStratifiedSplit:
    @pytest.fixture
    def setup_split_data(self, tmp_path):
        """Create a DataFrame and test CSV for split testing."""
        n = 100
        df = pd.DataFrame({
            "image_name": [f"img_{i}" for i in range(n)],
            "mapped_lens_status": ["clear"] * 50 + ["PCIOL"] * 50,
            "corneal_binned": ["Normal"] * 25 + ["Active"] * 25 + ["Normal"] * 25 + ["Active"] * 25,
        })

        test_csv_path = str(tmp_path / "test.csv")
        test_df = pd.DataFrame({
            "image_name_right": ["img_98"],
            "image_name_left": ["img_99"],
        })
        test_df.to_csv(test_csv_path, index=False)
        return df, test_csv_path

    def test_split_sizes(self, setup_split_data):
        df, test_csv_path = setup_split_data
        splits = create_stratified_split(df, test_csv_path, val_fraction=0.1)
        total = splits["num_train"] + splits["num_val"]
        # 2 test images excluded from 100
        assert total == 98

    def test_no_overlap(self, setup_split_data):
        df, test_csv_path = setup_split_data
        splits = create_stratified_split(df, test_csv_path)
        train = set(splits["train_indices"])
        val = set(splits["val_indices"])
        assert train.isdisjoint(val)

    def test_test_images_excluded(self, setup_split_data):
        df, test_csv_path = setup_split_data
        splits = create_stratified_split(df, test_csv_path)
        all_indices = set(splits["train_indices"] + splits["val_indices"])
        # img_98 and img_99 should not be in train or val
        assert 98 not in all_indices
        assert 99 not in all_indices

    def test_reproducible(self, setup_split_data):
        df, test_csv_path = setup_split_data
        s1 = create_stratified_split(df, test_csv_path, seed=42)
        s2 = create_stratified_split(df, test_csv_path, seed=42)
        assert s1["train_indices"] == s2["train_indices"]
        assert s1["val_indices"] == s2["val_indices"]
