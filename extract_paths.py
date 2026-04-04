"""Extract image paths from a JSONL file.

Usage:
    python extract_paths.py paintings/bde_train_mm.jsonl > image_paths.txt
    python extract_paths.py paintings/bde_train_mm.jsonl --limit 10 > image_paths.txt
"""

import json
import sys


def main():
    jsonl_file = sys.argv[1]
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    count = 0
    with open(jsonl_file, "r") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            entry = json.loads(line)
            print(entry["image"])
            count += 1


if __name__ == "__main__":
    main()
