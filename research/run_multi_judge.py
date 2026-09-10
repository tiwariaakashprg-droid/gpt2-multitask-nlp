"""
Robust multi-judge experiment for the TMLR study.

Features:
- Multiple local Ollama judges
- Repeated evaluations with controlled seeds
- --limit for smoke tests
- --resume to continue interrupted experiments
- Incremental checkpoint saving after every item
- Retry on temporary Ollama failures
- Same v2 evaluation rubric for every judge
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.evaluate_ollama import query_ollama, parse_score


PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "judge_prompt_v2.txt"
)


def load_prompt():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def score_item(
    row,
    judge_model,
    repeats,
    seed_base,
    prompt_template,
    retries=3
):
    """
    Evaluate one response multiple times.

    Each repetition uses a different seed:
        seed_base, seed_base + 1, ...

    Retries are used only when an Ollama call fails or
    produces an invalid score.
    """

    scores = []

    for rep in range(repeats):

        prompt = prompt_template.format(
            instruction=row["instruction"],
            input_text=row.get("input", "") or "(none)",
            response=row["model_output"],
        )

        last_error = None

        for attempt in range(1, retries + 1):

            try:
                raw = query_ollama(
                    prompt,
                    model=judge_model,
                    seed=seed_base + rep,
                    temperature=0.0,
                )

                score = parse_score(raw, default=None)

                if score is not None and 0 <= score <= 100:
                    scores.append(score)
                    break

                last_error = ValueError(
                    f"Invalid score returned: {raw!r}"
                )

            except Exception as exc:
                last_error = exc

            if attempt < retries:
                print(
                    f"    Retry {attempt}/{retries - 1} "
                    f"for {judge_model}, repetition {rep + 1}..."
                )
                time.sleep(2)

        else:
            raise RuntimeError(
                f"Failed to obtain valid score from "
                f"{judge_model}, repetition {rep + 1}"
            ) from last_error

    return scores


def atomic_save(df, output_path):
    """
    Save dataframe safely.

    A temporary file is written first and then renamed,
    reducing the chance of losing the checkpoint if the
    process is interrupted during writing.
    """

    temp_path = output_path + ".tmp"

    df.to_csv(temp_path, index=False)

    os.replace(temp_path, output_path)


def save_metadata(metadata, output_path):
    meta_path = os.path.splitext(output_path)[0] + "_metadata.json"

    temp_path = meta_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    os.replace(temp_path, meta_path)

    return meta_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="research/data/combined_human.csv"
    )

    parser.add_argument(
        "--output",
        default="research/data/multi_judge_results.csv"
    )

    parser.add_argument(
        "--judge_models",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )

    parser.add_argument(
        "--seed_base",
        type=int,
        default=42
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of dataset rows to evaluate. "
             "Useful for smoke testing."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV."
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for each Ollama evaluation."
    )

    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    if args.retries < 1:
        raise ValueError("--retries must be >= 1")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(args.input)

    total_rows = len(df)

    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    print()
    print("=" * 70)
    print("TMLR MULTI-JUDGE EXPERIMENT")
    print("=" * 70)
    print(f"Input dataset rows      : {total_rows}")
    print(f"Rows to evaluate        : {len(df)}")
    print(f"Judge models            : {args.judge_models}")
    print(f"Repeats                 : {args.repeats}")
    print(f"Seed base               : {args.seed_base}")
    print(f"Retries                 : {args.retries}")
    print(f"Resume                  : {args.resume}")
    print("=" * 70)
    print()

    prompt_template = load_prompt()

    # ---------------------------------------------------------
    # Resume from existing checkpoint
    # ---------------------------------------------------------

    if args.resume and os.path.exists(args.output):

        print(f"Resuming from: {args.output}")

        existing = pd.read_csv(args.output)

        if len(existing) != len(df):
            raise ValueError(
                "Resume file row count does not match the "
                "requested dataset/limit."
            )

        out = existing.copy()

    else:

        out = df.copy()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "prompt_file": "research/judge_prompt_v2.txt",
        "temperature": 0.0,
        "seed_base": args.seed_base,
        "repeats": args.repeats,
        "retries": args.retries,
        "judge_models": args.judge_models,
        "n_items": len(df),
        "dataset_total_rows": total_rows,
        "limit": args.limit,
        "timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime()
        ),
    }

    # ---------------------------------------------------------
    # Evaluate each judge
    # ---------------------------------------------------------

    for model in args.judge_models:

        safe_model = (
            model
            .replace(":", "_")
            .replace("/", "_")
            .replace(".", "_")
        )

        run_cols = [
            f"{safe_model}_run{i + 1}"
            for i in range(args.repeats)
        ]

        mean_col = f"{safe_model}_mean"
        std_col = f"{safe_model}_std"

        # Create columns if they don't exist
        for col in run_cols:
            if col not in out.columns:
                out[col] = pd.NA

        if mean_col not in out.columns:
            out[mean_col] = pd.NA

        if std_col not in out.columns:
            out[std_col] = pd.NA

        print()
        print("-" * 70)
        print(f"JUDGE: {model}")
        print("-" * 70)

        for idx in range(len(df)):

            # -------------------------------------------------
            # Resume check
            # -------------------------------------------------

            completed = True

            for col in run_cols:
                value = out.loc[idx, col]

                if pd.isna(value):
                    completed = False
                    break

            if args.resume and completed:

                print(
                    f"[{model}] [{idx + 1}/{len(df)}] "
                    f"already completed -> skipping"
                )
                continue

            row = df.iloc[idx]

            # -------------------------------------------------
            # Query judge
            # -------------------------------------------------

            scores = score_item(
                row=row,
                judge_model=model,
                repeats=args.repeats,
                seed_base=args.seed_base,
                prompt_template=prompt_template,
                retries=args.retries,
            )

            # -------------------------------------------------
            # Store scores immediately
            # -------------------------------------------------

            for rep in range(args.repeats):
                out.loc[
                    idx,
                    run_cols[rep]
                ] = scores[rep]

            out.loc[idx, mean_col] = sum(scores) / len(scores)

            if len(scores) > 1:
                out.loc[idx, std_col] = pd.Series(
                    scores
                ).std(ddof=1)
            else:
                out.loc[idx, std_col] = 0.0

            # -------------------------------------------------
            # CHECKPOINT AFTER EVERY ITEM
            # -------------------------------------------------

            atomic_save(out, args.output)

            meta_path = save_metadata(
                metadata,
                args.output
            )

            print(
                f"[{model}] [{idx + 1}/{len(df)}] "
                f"human={row.get('human_avg_score', float('nan')):.2f} "
                f"scores={scores} "
                f"mean={out.loc[idx, mean_col]:.2f}"
            )

            print(
                f"    checkpoint saved"
            )

        print()
        print(f"Completed judge: {model}")

    # ---------------------------------------------------------
    # Final save
    # ---------------------------------------------------------

    atomic_save(out, args.output)

    meta_path = save_metadata(
        metadata,
        args.output
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"Results : {args.output}")
    print(f"Metadata: {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
