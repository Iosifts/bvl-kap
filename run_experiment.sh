#!/usr/bin/env bash
# BVL evaluation experiment — run all three conditions
#
# Usage:
#   source env.sh
#   bash run_experiment.sh                          # all 553 images
#   bash run_experiment.sh --limit 10               # first 10 images (test run)
#
set -euo pipefail

JSONL="./paintings/bde_train_mm.jsonl"
FEWSHOT="./files/bvl_fewshot_qwen.json"
LIMIT_FLAG="${1:-}"
LIMIT_VAL="${2:-}"

# Extract image paths from JSONL
echo "Extracting image paths from $JSONL..."
if [ "$LIMIT_FLAG" = "--limit" ] && [ -n "$LIMIT_VAL" ]; then
    python extract_paths.py "$JSONL" --limit "$LIMIT_VAL" > /tmp/bvl_image_paths.txt
    echo "Using first $LIMIT_VAL images"
else
    python extract_paths.py "$JSONL" > /tmp/bvl_image_paths.txt
    echo "Using all $(wc -l < /tmp/bvl_image_paths.txt) images"
fi

# Read paths into an array
mapfile -t IMAGES < /tmp/bvl_image_paths.txt

echo ""
echo "=== Step 1: Generate grounding data ==="
python __init__.py -o groundings.csv "${IMAGES[@]}" ground

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
echo ""
echo "To evaluate (requires OPENAI_API_KEY):"
echo "  python evaluate.py captions_condition_a.csv captions_condition_b.csv evaluation_a_vs_b.json"
echo "  python evaluate.py captions_condition_a.csv captions_condition_c.csv evaluation_a_vs_c.json"
