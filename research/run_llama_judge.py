"""
STEP 4 of the research pipeline.

Takes the combined human-annotation CSV and sends every model_output through
LLaMA-3 (via a locally running Ollama server) for automatic scoring.

To test the "is the judge actually deterministic?" question, each item is
scored `--repeats` times (default 3) with temperature=0 and a fixed seed, so
you can later measure how much the judge's own score varies across repeated
calls on the exact same input.

Requires Ollama running locally:
    ollama pull llama3
    ollama serve

Usage:
    python research/run_llama_judge.py \
        --input research/data/combined_human.csv \
        --output research/data/combined_with_judge.csv \
        --repeats 3
"""

import argparse
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluate_ollama import query_ollama, parse_score, JUDGE_PROMPT_TEMPLATE


def score_item(instruction, input_text, model_output, judge_model, repeats, seed_base):
    scores = []
    for i in range(repeats):
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_text=input_text or "(none)",
            response=model_output,
        )
        raw = query_ollama(prompt, model=judge_model, seed=seed_base + i, temperature=0.0)
        scores.append(parse_score(raw))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="research/data/combined_human.csv")
    parser.add_argument("--output", type=str, default="research/data/combined_with_judge.csv")
    parser.add_argument("--judge_model", type=str, default="llama3")
    parser.add_argument("--repeats", type=int, default=3,
                         help="How many times to score each item, to measure judge consistency")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Scoring {len(df)} items with judge model '{args.judge_model}', "
          f"{args.repeats} repeat(s) each ...")

    all_repeat_scores = []
    for idx, row in df.iterrows():
        scores = score_item(
            row["instruction"], row.get("input", ""), row["model_output"],
            judge_model=args.judge_model, repeats=args.repeats, seed_base=42,
        )
        all_repeat_scores.append(scores)
        print(f"[{idx+1}/{len(df)}] human_avg={row.get('human_avg_score', float('nan')):.1f}  "
              f"judge_scores={scores}")

    for r in range(args.repeats):
        df[f"judge_score_run{r+1}"] = [s[r] for s in all_repeat_scores]

    judge_cols = [f"judge_score_run{r+1}" for r in range(args.repeats)]
    df["judge_score_mean"] = df[judge_cols].mean(axis=1)
    df["judge_score_std"] = df[judge_cols].std(axis=1)  # 0 across repeats => judge is deterministic here
    df["response_length_chars"] = df["model_output"].astype(str).str.len()

    df.to_csv(args.output, index=False)
    print(f"\nSaved judge-scored dataset to {args.output}")
    print("Next step: python research/analyze_agreement.py --input", args.output)


if __name__ == "__main__":
    main()
