"""
STEP 3 of the research pipeline.

Merges qa_pairs.json with one or more annotations_<name>.json files (one per
human annotator, produced by annotate.html), and:
    1. Computes each item's average human score across annotators.
    2. Computes inter-annotator agreement (Pearson correlation between every
       pair of annotators, plus overall average pairwise correlation).
    3. Writes a single combined CSV ready for the LLaMA-judge step.

Usage:
    python research/merge_annotations.py \
        --qa_pairs research/data/qa_pairs.json \
        --annotations research/data/annotations_alice.json research/data/annotations_bob.json \
        --output research/data/combined_human.csv
"""

import argparse
import json
import itertools
import pandas as pd
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_pairs", type=str, default="research/data/qa_pairs.json")
    parser.add_argument("--annotations", type=str, nargs="+", required=True,
                         help="One or more annotations_<name>.json files")
    parser.add_argument("--output", type=str, default="research/data/combined_human.csv")
    args = parser.parse_args()

    with open(args.qa_pairs) as f:
        qa_pairs = {item["id"]: item for item in json.load(f)}

    all_ratings = []
    for path in args.annotations:
        with open(path) as f:
            all_ratings.extend(json.load(f))

    ratings_df = pd.DataFrame(all_ratings)
    annotators = sorted(ratings_df["annotator"].unique())
    print(f"Found {len(annotators)} annotator(s): {annotators}")

    # pivot: rows = item id, columns = annotator, values = score
    pivot = ratings_df.pivot_table(index="id", columns="annotator", values="score", aggfunc="mean")

    # inter-annotator agreement (pairwise Pearson correlation)
    print("\n=== Inter-annotator agreement (Pearson r) ===")
    pairwise_corrs = []
    for a, b in itertools.combinations(annotators, 2):
        paired = pivot[[a, b]].dropna()
        if len(paired) >= 2:
            r, p = pearsonr(paired[a], paired[b])
            pairwise_corrs.append(r)
            print(f"  {a} vs {b}: r = {r:.3f}  (n={len(paired)}, p={p:.4f})")
    if pairwise_corrs:
        avg_r = sum(pairwise_corrs) / len(pairwise_corrs)
        print(f"\n  Average pairwise inter-annotator r = {avg_r:.3f}")
        if avg_r < 0.5:
            print("  WARNING: low inter-annotator agreement. Consider clarifying "
                  "the rating guidelines before trusting the human 'ground truth'.")

    pivot["human_avg_score"] = pivot.mean(axis=1)
    pivot["human_score_std"] = pivot[annotators].std(axis=1)

    # attach back to qa_pairs metadata
    rows = []
    for item_id, row in pivot.iterrows():
        base = qa_pairs.get(item_id, {})
        merged = {**base, **row.to_dict()}
        rows.append(merged)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved combined human-annotation dataset to {args.output} ({len(out_df)} rows)")
    print("Next step: python research/run_llama_judge.py --input", args.output)


if __name__ == "__main__":
    main()
