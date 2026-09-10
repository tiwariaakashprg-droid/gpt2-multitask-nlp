"""
STEP 5 (final) of the research pipeline.

Runs every statistical test needed for the paper's Results section on the
combined human+judge dataset, and writes:
    - research/results/plots/*.png    (scatter plots, bar charts)
    - research/results/REPORT.md      (auto-filled markdown with your real numbers)

Tests performed:
    1. Human vs judge correlation      -> Pearson r, Spearman rho, p-values
    2. Mean Absolute Error / bias       -> does the judge systematically over/under-score?
    3. Judge self-consistency           -> std dev across repeated judge calls per item
    4. Length bias                      -> does judge score correlate with response length,
                                            independent of human-perceived quality?
    5. Decoding-strategy effect         -> does judge agreement differ across
                                            low/mid/high temperature settings?
    6. Per-category breakdown           -> agreement broken down by question category

Usage:
    python research/analyze_agreement.py --input research/data/combined_with_judge.csv
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_corr(x, y):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    return pearsonr(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="research/data/combined_with_judge.csv")
    parser.add_argument("--out_dir", type=str, default="research/results")
    args = parser.parse_args()

    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    report_lines = []
    report_lines.append("# Results Report\n")
    report_lines.append(f"Dataset: `{args.input}` — {len(df)} items\n")

    # ---- 1. Human vs Judge correlation ----
    r, p = safe_corr(df["human_avg_score"], df["judge_score_mean"])
    rho, p_s = spearmanr(df["human_avg_score"], df["judge_score_mean"])
    report_lines.append("## 1. Human vs LLM-Judge Agreement\n")
    report_lines.append(f"- Pearson r = **{r:.3f}** (p = {p:.4g})")
    report_lines.append(f"- Spearman rho = **{rho:.3f}** (p = {p_s:.4g})")
    mae = (df["human_avg_score"] - df["judge_score_mean"]).abs().mean()
    bias = (df["judge_score_mean"] - df["human_avg_score"]).mean()
    report_lines.append(f"- Mean Absolute Error = **{mae:.2f}** points (on a 0-100 scale)")
    direction = "over-scores" if bias > 0 else "under-scores"
    report_lines.append(f"- Mean signed difference = **{bias:+.2f}** -> judge {direction} "
                         f"relative to humans on average\n")

    plt.figure(figsize=(5, 5))
    plt.scatter(df["human_avg_score"], df["judge_score_mean"], alpha=0.6)
    plt.plot([0, 100], [0, 100], "--", color="gray", label="perfect agreement")
    plt.xlabel("Human average score")
    plt.ylabel("LLaMA-3 judge score (mean of repeats)")
    plt.title(f"Human vs Judge (r = {r:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "human_vs_judge_scatter.png"), dpi=150)
    plt.close()

    # ---- 2. Judge self-consistency ----
    report_lines.append("## 2. Judge Self-Consistency (same input, repeated calls)\n")
    mean_std = df["judge_score_std"].mean()
    pct_perfectly_consistent = (df["judge_score_std"] == 0).mean() * 100
    report_lines.append(f"- Mean std-dev across repeated judge calls per item = **{mean_std:.2f}**")
    report_lines.append(f"- % of items with IDENTICAL score every repeat = **{pct_perfectly_consistent:.1f}%**")
    report_lines.append("  (If this is not ~100%, the judge is not fully deterministic "
                         "even at temperature=0 with a fixed seed — an interesting finding "
                         "worth discussing.)\n")

    # ---- 3. Length bias ----
    report_lines.append("## 3. Length Bias\n")
    r_len_judge, p_len_judge = safe_corr(df["response_length_chars"], df["judge_score_mean"])
    r_len_human, p_len_human = safe_corr(df["response_length_chars"], df["human_avg_score"])
    report_lines.append(f"- Correlation(length, judge score) = **{r_len_judge:.3f}** (p={p_len_judge:.4g})")
    report_lines.append(f"- Correlation(length, human score) = **{r_len_human:.3f}** (p={p_len_human:.4g})")
    if not np.isnan(r_len_judge) and not np.isnan(r_len_human) and abs(r_len_judge) > abs(r_len_human) + 0.1:
        report_lines.append("  -> Judge shows a stronger length bias than human annotators.\n")
    else:
        report_lines.append("  -> No strong evidence the judge favors longer responses more than humans do.\n")

    plt.figure(figsize=(5, 4))
    plt.scatter(df["response_length_chars"], df["judge_score_mean"], alpha=0.6, label="judge")
    plt.scatter(df["response_length_chars"], df["human_avg_score"], alpha=0.6, label="human")
    plt.xlabel("Response length (characters)")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Score vs response length")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "length_bias.png"), dpi=150)
    plt.close()

    # ---- 4. Effect of decoding strategy ----
    if "decoding_config" in df.columns:
        report_lines.append("## 4. Agreement by Decoding Strategy\n")
        report_lines.append("| Decoding config | n | Pearson r (human vs judge) | Mean judge score | Mean human score |")
        report_lines.append("|---|---|---|---|---|")
        for cfg, group in df.groupby("decoding_config"):
            r_cfg, _ = safe_corr(group["human_avg_score"], group["judge_score_mean"])
            report_lines.append(
                f"| {cfg} | {len(group)} | {r_cfg:.3f} | "
                f"{group['judge_score_mean'].mean():.1f} | {group['human_avg_score'].mean():.1f} |"
            )
        report_lines.append("")

        plt.figure(figsize=(6, 4))
        configs = df["decoding_config"].unique()
        x = np.arange(len(configs))
        judge_means = [df[df.decoding_config == c]["judge_score_mean"].mean() for c in configs]
        human_means = [df[df.decoding_config == c]["human_avg_score"].mean() for c in configs]
        width = 0.35
        plt.bar(x - width/2, human_means, width, label="human")
        plt.bar(x + width/2, judge_means, width, label="judge")
        plt.xticks(x, configs)
        plt.ylabel("Mean score")
        plt.title("Mean score by decoding strategy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "decoding_strategy_comparison.png"), dpi=150)
        plt.close()

    # ---- 5. Per-category breakdown ----
    if "category" in df.columns:
        report_lines.append("## 5. Agreement by Question Category\n")
        report_lines.append("| Category | n | Pearson r (human vs judge) |")
        report_lines.append("|---|---|---|")
        for cat, group in df.groupby("category"):
            r_cat, _ = safe_corr(group["human_avg_score"], group["judge_score_mean"])
            report_lines.append(f"| {cat} | {len(group)} | {r_cat:.3f} |")
        report_lines.append("")

    report_path = os.path.join(args.out_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Saved report to {report_path}")
    print(f"Saved plots to {plots_dir}/")
    print("\nCopy the numbers and plots from these files directly into the "
          "Results section of paper_template.md")


if __name__ == "__main__":
    main()
