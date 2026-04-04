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
echo "=== Next: Classification + Evaluation ==="
echo "Requires OPENAI_API_KEY. Run these steps:"
echo ""
echo "  # Step 1: Classify each condition's captions"
echo "  python classify_captions.py captions_condition_a.csv results/classified_a.json"
echo "  python classify_captions.py captions_condition_b.csv results/classified_b.json"
echo "  python classify_captions.py captions_condition_c.csv results/classified_c.json"
echo ""
echo "  # Step 2: Compute all evaluation metrics"
echo "  python evaluate_metrics.py \\"
echo "      --classified-a results/classified_a.json \\"
echo "      --classified-b results/classified_b.json \\"
echo "      --classified-c results/classified_c.json \\"
echo "      --output results/analysis.json"
