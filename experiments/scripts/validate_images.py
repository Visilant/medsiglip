"""Preflight image validation script.

Scans all CSV-referenced images with strict decoding (LOAD_TRUNCATED_IMAGES=False)
to catch corrupt files before training. Updates bad_images.json with failures.

Usage:
    PYTHONPATH=experiments experiments/.venv/bin/python -m scripts.validate_images
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageFile

# Strict mode: reject truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = False


def validate_single_image(image_path: str) -> tuple[str, str | None]:
    """Validate a single image by forcing a full decode.

    Returns (filename, error_message) — error_message is None if valid.
    """
    try:
        with Image.open(image_path) as img:
            img.load()  # Force full decode
            img.convert("RGB")
        return (os.path.basename(image_path), None)
    except Exception as e:
        return (os.path.basename(image_path), str(e))


def main():
    # Import config loading here to keep the worker function picklable
    import yaml

    config_path = os.environ.get("CONFIG_PATH", "experiments/config/base.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["csv_path"]
    image_dir = config["data"]["image_dir"]
    image_extension = config["data"]["image_extension"]
    bad_images_path = "experiments/data/bad_images.json"

    # Load CSV to get referenced image names
    import pandas as pd

    df = pd.read_csv(csv_path, low_memory=False)
    image_files = [f"{name}{image_extension}" for name in df["image_name"]]

    # Filter to images that exist on disk
    existing = set(os.listdir(image_dir))
    image_files = [f for f in image_files if f in existing]
    image_files = sorted(set(image_files))

    print(f"Validating {len(image_files)} images from {csv_path}...")

    # Load existing bad images
    if os.path.exists(bad_images_path):
        with open(bad_images_path) as f:
            known_bad = set(json.load(f))
    else:
        known_bad = set()

    print(f"Known bad images: {len(known_bad)}")

    # Validate in parallel
    failures = {}
    num_workers = os.cpu_count() or 4
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validate_single_image, os.path.join(image_dir, f)): f
            for f in image_files
        }

        for future in as_completed(futures):
            completed += 1
            if completed % 5000 == 0:
                print(f"  Progress: {completed}/{len(image_files)}")

            filename, error = future.result()
            if error is not None:
                failures[filename] = error

    # Report results
    new_bad = set(failures.keys()) - known_bad
    print(f"\nValidation complete:")
    print(f"  Total scanned: {len(image_files)}")
    print(f"  Total failures: {len(failures)}")
    print(f"  Previously known: {len(known_bad & set(failures.keys()))}")
    print(f"  Newly discovered: {len(new_bad)}")

    if new_bad:
        print(f"\nNewly bad images:")
        for img in sorted(new_bad)[:20]:
            print(f"  {img}: {failures[img]}")
        if len(new_bad) > 20:
            print(f"  ... and {len(new_bad) - 20} more")

    # Merge into bad_images.json
    all_bad = sorted(known_bad | set(failures.keys()))
    Path(bad_images_path).parent.mkdir(parents=True, exist_ok=True)
    with open(bad_images_path, "w") as f:
        json.dump(all_bad, f, indent=2)
    print(f"\nUpdated {bad_images_path} ({len(all_bad)} total bad images)")

    # Save detailed validation log
    log_path = "experiments/data/validation_log.json"
    log = {
        "timestamp": datetime.now().isoformat(),
        "total_scanned": len(image_files),
        "total_failures": len(failures),
        "newly_discovered": len(new_bad),
        "failures": {k: failures[k] for k in sorted(failures.keys())},
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Detailed log saved to {log_path}")

    return 0 if not new_bad else 1


if __name__ == "__main__":
    sys.exit(main())
