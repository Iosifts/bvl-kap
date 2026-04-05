#!/usr/bin/env bash
set -euo pipefail

JSONL="./paintings/bde_train_mm.jsonl"
FEWSHOT="./files/bvl_fewshot_qwen.json"
LIMIT_FLAG="${1:-}"
LIMIT_VAL="${2:-}"

echo "Extracting image paths from $JSONL..."
if [ "$LIMIT_FLAG" = "--limit" ] && [ -n "$LIMIT_VAL" ]; then
    python extract_paths.py "$JSONL" --limit "$LIMIT_VAL" > /tmp/bvl_image_paths.txt
    echo "Using first $LIMIT_VAL images"
else
    python extract_paths.py "$JSONL" > /tmp/bvl_image_paths.txt
    echo "Using all $(wc -l < /tmp/bvl_image_paths.txt) images"
fi

mapfile -t IMAGES < /tmp/bvl_image_paths.txt

echo ""
echo "=== Step 1: Generate grounding data ==="
if [ -f groundings.csv ]; then
    echo "groundings.csv already exists, reusing it."
else
    python __init__.py -o groundings.csv "${IMAGES[@]}" ground
fi

echo ""
echo "=== Step 2: Condition A — BVL prompt only ==="
python __init__.py -o captions_condition_a.csv "${IMAGES[@]}" caption

echo ""
echo "=== Step 3: Condition B — BVL + grounded metadata ==="
python __init__.py -o captions_condition_b.csv "${IMAGES[@]}" caption \
    --grounding-data groundings.csv

echo ""
echo "=== Step 4: Condition C — BVL + metadata + few-shot ==="
python __init__.py -o captions_condition_c.csv "${IMAGES[@]}" caption \
    --grounding-data groundings.csv \
    --fewshot "$FEWSHOT"

echo ""
echo "=== All captions generated ==="
echo "Output files:"
echo "  groundings.csv              (grounding metadata)"
echo "  captions_condition_a.csv    (BVL only)"
echo "  captions_condition_b.csv    (BVL + metadata)"
echo "  captions_condition_c.csv    (BVL + metadata + few-shot)"