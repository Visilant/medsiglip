#!/usr/bin/env python3
"""Classify and move bad images from visilant_data to visilant_data_bad."""
import json
import os
import subprocess
import collections
import shutil

BAD_LIST = "/home/adi/medsiglip/experiments/data/bad_images.json"
SRC = "/home/adi/visilant_data"
DST = "/home/adi/visilant_data_bad"

with open(BAD_LIST) as f:
    bad = json.load(f)

print(f"Total bad filenames: {len(bad)}")

# Check existence and sizes
missing = 0
zero = 0
exist = []
sizes = []

for fn in bad:
    fn = fn.strip()
    path = os.path.join(SRC, fn)
    if os.path.exists(path):
        sz = os.path.getsize(path)
        exist.append(fn)
        sizes.append(sz)
        if sz == 0:
            zero += 1
    else:
        missing += 1

print(f"Existing: {len(exist)}")
print(f"Missing: {missing}")
print(f"Zero-byte: {zero}")
if sizes:
    print(f"Min size: {min(sizes)} bytes")
    print(f"Max size: {max(sizes)} bytes")
    print(f"Avg size: {sum(sizes)//len(sizes)} bytes")

# Classify by file type using file command in batches
cats = collections.Counter()
batch_size = 500
for i in range(0, len(exist), batch_size):
    batch = [os.path.join(SRC, fn) for fn in exist[i:i+batch_size]]
    result = subprocess.run(["file"] + batch, capture_output=True, text=True)
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            ftype = line.split(":", 1)[1].strip()
            if ftype.startswith("PDF"):
                cats["PDF document (wrong format - actually PDF, not JPEG)"] += 1
            elif "DICOM" in ftype:
                cats["DICOM medical imaging data (wrong format)"] += 1
            elif "empty" in ftype.lower():
                cats["Empty/zero-byte file"] += 1
            elif ftype.startswith("JPEG"):
                cats["JPEG (truncated/corrupt)"] += 1
            elif ftype.startswith("PNG"):
                cats["PNG image"] += 1
            elif "HTML" in ftype:
                cats["HTML document"] += 1
            else:
                cats[ftype] += 1

print("\n=== FILE TYPE BREAKDOWN ===")
for ftype, count in cats.most_common():
    pct = 100.0 * count / len(exist)
    print(f"  {count:>6}  ({pct:5.1f}%)  {ftype}")

# Move files
print("\n=== MOVING FILES ===")
os.makedirs(DST, exist_ok=True)
moved = 0
skipped = 0
for fn in exist:
    src_path = os.path.join(SRC, fn)
    dst_path = os.path.join(DST, fn)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        moved += 1
    else:
        skipped += 1

print(f"Moved: {moved}")
print(f"Skipped: {skipped}")

remaining = len(os.listdir(SRC))
in_dst = len(os.listdir(DST))
print(f"\n=== SUMMARY ===")
print(f"Remaining in {SRC}: {remaining} files")
print(f"In {DST}: {in_dst} files")
print("Done.")
