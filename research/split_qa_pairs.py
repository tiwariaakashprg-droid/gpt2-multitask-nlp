"""
Splits qa_pairs.json into N roughly-equal, category-balanced batches so that
multiple small groups of annotators can each rate a manageable subset,
instead of every annotator rating every item.

Each batch is stratified by category so every batch still covers factual,
math, reasoning, writing, and instruction_following items in similar
proportions -- this matters for the per-category breakdown in
analyze_agreement.py.

Usage:
    python split_qa_pairs.py --input research/data/qa_pairs.json \
        --output_dir research/data/batches --num_batches 5

This writes batch_1.json ... batch_5.json into output_dir. Give each batch
file (together with annotate.html) to a small group of annotators (e.g. 3
people per batch for 5 batches = 15 annotators total). Every annotator in a
group rates the SAME batch file, so merge_annotations.py can still compute
inter-annotator agreement within each group.

No changes are needed to merge_annotations.py or analyze_agreement.py --
just pass ALL annotators' annotations_<name>.json files together when you
run merge_annotations.py at the end; it already averages over however many
annotators rated each item.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="research/data/qa_pairs.json")
    parser.add_argument("--output_dir", type=str, default="research/data/batches")
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        items = json.load(f)

    random.seed(args.seed)

    # group items by category so we can stratify
    by_category = defaultdict(list)
    for item in items:
        by_category[item.get("category", "unknown")].append(item)

    batches = [[] for _ in range(args.num_batches)]

    # round-robin each category's (shuffled) items across the batches
    for category, cat_items in by_category.items():
        shuffled = cat_items[:]
        random.shuffle(shuffled)
        for i, item in enumerate(shuffled):
            batches[i % args.num_batches].append(item)

    os.makedirs(args.output_dir, exist_ok=True)
    for i, batch in enumerate(batches, start=1):
        path = os.path.join(args.output_dir, f"batch_{i}.json")
        with open(path, "w") as f:
            json.dump(batch, f, indent=2)
        cat_counts = defaultdict(int)
        for item in batch:
            cat_counts[item.get("category", "unknown")] += 1
        print(f"{path}: {len(batch)} items  {dict(cat_counts)}")

    print(f"\nDone. Give each batch_N.json (with annotate.html) to a small "
          f"group of annotators (e.g. 3 people per batch).")


if __name__ == "__main__":
    main()
