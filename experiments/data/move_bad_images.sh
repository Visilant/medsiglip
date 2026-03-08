#!/bin/bash
# Script to analyze and move bad images from visilant_data to visilant_data_bad
# Run with: bash /home/adi/medsiglip/experiments/data/move_bad_images.sh

set -e

SRC="/home/adi/visilant_data"
DST="/home/adi/visilant_data_bad"
BAD_LIST="/home/adi/medsiglip/experiments/data/bad_images.json"

echo "=== Bad Image Analysis and Move Script ==="
echo ""

# Create destination directory
mkdir -p "$DST"

# Extract filenames
TMPFILE=$(mktemp)
tr -d '[]"' < "$BAD_LIST" | tr ',' '\n' | tr -d ' ' | grep -v '^$' > "$TMPFILE"
TOTAL=$(wc -l < "$TMPFILE")
echo "Total bad filenames: $TOTAL"

# Count missing and zero-byte
MISSING=0
ZERO=0
EXIST=0
while IFS= read -r fn; do
    path="$SRC/$fn"
    if [ ! -f "$path" ]; then
        MISSING=$((MISSING + 1))
    else
        EXIST=$((EXIST + 1))
        if [ ! -s "$path" ]; then
            ZERO=$((ZERO + 1))
        fi
    fi
done < "$TMPFILE"
echo "Existing: $EXIST"
echo "Missing: $MISSING"
echo "Zero-byte: $ZERO"

# Classify by file type
echo ""
echo "=== FILE TYPE BREAKDOWN ==="
cd "$SRC"
cat "$TMPFILE" | xargs file 2>/dev/null | sed 's/^[^:]*: //' | sort | uniq -c | sort -rn
cd -

# Move files
echo ""
echo "=== MOVING FILES ==="
MOVED=0
SKIP=0
while IFS= read -r fn; do
    src_path="$SRC/$fn"
    if [ -f "$src_path" ]; then
        mv "$src_path" "$DST/$fn"
        MOVED=$((MOVED + 1))
    else
        SKIP=$((SKIP + 1))
    fi
done < "$TMPFILE"

echo "Moved: $MOVED"
echo "Skipped (not found): $SKIP"

# Cleanup
rm -f "$TMPFILE"

echo ""
echo "=== SUMMARY ==="
echo "Bad images moved from $SRC to $DST"
echo "Remaining in $SRC: $(ls -1 "$SRC" | wc -l) files"
echo "In $DST: $(ls -1 "$DST" | wc -l) files"
echo "Done."
