"""
Evaluate generated BVL descriptions using GPT as a clause classifier.

Usage:
    python evaluate.py captions_bvl_only.csv captions_bvl_plus_metadata.csv

Requires: OPENAI_API_KEY environment variable set.
Reads the classifier prompt from files/bvl_classifier_prompt.json.
Outputs per-painting and aggregate clause profiles to stdout and a JSON file.
"""

import csv
import json
import os
import sys

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Install with: pip install openai")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PROMPT_FILE = os.path.join(
    SCRIPT_DIR, "files", "bvl_classifier_prompt.json")

CLASSES = ["Overview", "Spatial", "Objects", "Visual", "VisibleText",
           "Interpretation"]


def load_classifier_prompt():
    with open(CLASSIFIER_PROMPT_FILE, "r") as f:
        return json.load(f)


def read_captions_csv(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def classify_description(client, classifier_prompt, description):
    """Send a description to GPT and get clause classifications back."""
    messages = [{"role": "system", "content": classifier_prompt["system"]}]
    # add few-shot examples
    for example in classifier_prompt["few_shot_examples"]:
        messages.append({"role": example["role"], "content": example["content"]})
    # add the actual description
    user_turn = classifier_prompt["inference_user_turn_template"]["content"]
    user_turn = user_turn.replace("{GENERATED_DESCRIPTION}", description)
    messages.append({"role": "user", "content": user_turn})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    # strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return json.loads(content)


def compute_profile(clauses):
    """Compute clause class distribution and visible_fact ratio."""
    profile = {c: 0 for c in CLASSES}
    visible_facts = 0
    total = len(clauses)
    for clause in clauses:
        cls = clause.get("class", "Interpretation")
        if cls in profile:
            profile[cls] += 1
        visible_facts += int(clause.get("visible_fact", False))
    profile["total"] = total
    profile["visible_fact_ratio"] = (
        round(visible_facts / total, 3) if total > 0 else 0)
    return profile


def evaluate_file(client, classifier_prompt, caption_file, label):
    """Evaluate all captions in a CSV file."""
    captions = read_captions_csv(caption_file)
    results = []
    for row in captions:
        filename = row["file"]
        description = row["caption"]
        print(f"  classifying: {filename} ({label})...", flush=True)
        clauses = classify_description(
            client, classifier_prompt, description)
        profile = compute_profile(clauses)
        results.append({
            "file": filename,
            "condition": label,
            "profile": profile,
            "clauses": clauses,
        })
    return results


def aggregate_profiles(results):
    """Compute average profile across all paintings in a condition."""
    if not results:
        return {}
    agg = {c: 0 for c in CLASSES}
    agg["visible_fact_ratio"] = 0
    n = len(results)
    for r in results:
        p = r["profile"]
        for c in CLASSES:
            agg[c] += p[c]
        agg["visible_fact_ratio"] += p["visible_fact_ratio"]
    for c in CLASSES:
        agg[c] = round(agg[c] / n, 2)
    agg["visible_fact_ratio"] = round(agg["visible_fact_ratio"] / n, 3)
    return agg


def print_comparison(agg_a, label_a, agg_b, label_b):
    """Print a side-by-side comparison table."""
    print(f"\n{'Category':<20} {label_a:<20} {label_b:<20} {'Delta':<10}")
    print("-" * 70)
    for c in CLASSES:
        va = agg_a.get(c, 0)
        vb = agg_b.get(c, 0)
        delta = vb - va
        sign = "+" if delta > 0 else ""
        print(f"{c:<20} {va:<20.2f} {vb:<20.2f} {sign}{delta:.2f}")
    va = agg_a.get("visible_fact_ratio", 0)
    vb = agg_b.get("visible_fact_ratio", 0)
    delta = vb - va
    sign = "+" if delta > 0 else ""
    print(f"{'visible_fact_ratio':<20} {va:<20.3f} {vb:<20.3f} {sign}{delta:.3f}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <bvl_only.csv> <bvl_plus_metadata.csv> "
              "[output.json]")
        sys.exit(1)

    file_a = sys.argv[1]
    file_b = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "evaluation_results.json"

    client = OpenAI()
    classifier_prompt = load_classifier_prompt()

    print(f"Evaluating condition A: {file_a}")
    results_a = evaluate_file(client, classifier_prompt, file_a, "bvl_only")
    print(f"\nEvaluating condition B: {file_b}")
    results_b = evaluate_file(
        client, classifier_prompt, file_b, "bvl_plus_metadata")

    agg_a = aggregate_profiles(results_a)
    agg_b = aggregate_profiles(results_b)

    print_comparison(agg_a, "BVL only", agg_b, "BVL + metadata")

    # save full results
    output = {
        "conditions": {
            "bvl_only": {"file": file_a, "aggregate": agg_a,
                         "per_painting": results_a},
            "bvl_plus_metadata": {"file": file_b, "aggregate": agg_b,
                                  "per_painting": results_b},
        },
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
