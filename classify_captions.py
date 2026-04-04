"""
Classify generated BVL descriptions using GPT-4o as clause annotator.

Reads a caption CSV, sends each description to GPT for clause-level
classification, saves results incrementally to a JSON file.

Usage:
    python classify_captions.py captions_condition_a.csv classified_a.json
    python classify_captions.py captions_condition_b.csv classified_b.json
    python classify_captions.py captions_condition_c.csv classified_c.json

Supports resume: if the output JSON already exists, entries that were
previously classified are skipped. Safe to interrupt and restart.

Requires: OPENAI_API_KEY environment variable.
"""

import csv
import json
import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("openai package not installed. Install with: pip install openai")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PROMPT_FILE = os.path.join(
    SCRIPT_DIR, "files", "bvl_classifier_prompt.json")


def load_classifier_prompt():
    with open(CLASSIFIER_PROMPT_FILE) as f:
        return json.load(f)


def read_captions_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def classify_description(client, classifier_prompt, description, max_retries=3):
    """Send a description to GPT and get clause classifications back."""
    messages = [{"role": "system", "content": classifier_prompt["system"]}]
    for example in classifier_prompt["few_shot_examples"]:
        messages.append({"role": example["role"], "content": example["content"]})
    user_turn = classifier_prompt["inference_user_turn_template"]["content"]
    user_turn = user_turn.replace("{GENERATED_DESCRIPTION}", description)
    messages.append({"role": "user", "content": user_turn})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"    retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED after {max_retries} attempts: {e}")
                return []


def load_existing(output_path):
    """Load previously classified entries for resume support."""
    if os.path.exists(output_path):
        with open(output_path) as f:
            return json.load(f)
    return []


def save_results(output_path, results):
    """Atomically save results."""
    tmp = output_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, output_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: python classify_captions.py <captions.csv> <output.json>")
        sys.exit(1)

    caption_file = sys.argv[1]
    output_file = sys.argv[2]

    client = OpenAI()
    classifier_prompt = load_classifier_prompt()
    captions = read_captions_csv(caption_file)

    # resume support
    results = load_existing(output_file)
    done_files = {r["file"] for r in results}
    remaining = [row for row in captions if row["file"] not in done_files]

    print(f"Total: {len(captions)}, Already done: {len(done_files)}, "
          f"Remaining: {len(remaining)}")

    for i, row in enumerate(remaining):
        filename = row["file"]
        caption = row["caption"]
        print(f"[{len(done_files) + i + 1}/{len(captions)}] {filename}")

        clauses = classify_description(client, classifier_prompt, caption)
        results.append({
            "file": filename,
            "caption": caption,
            "clauses": clauses,
        })

        # save every entry for resume safety
        if (i + 1) % 5 == 0 or i == len(remaining) - 1:
            save_results(output_file, results)
            print(f"  saved ({len(results)}/{len(captions)})")

    save_results(output_file, results)
    print(f"\nDone. {len(results)} entries saved to {output_file}")


if __name__ == "__main__":
    main()
