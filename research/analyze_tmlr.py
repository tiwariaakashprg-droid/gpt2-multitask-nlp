"""
TMLR-oriented statistical analysis.

Uses the actual human annotations and judge outputs. No values are
hard-coded. Produces:

- Pearson/Spearman correlations with bootstrap 95% CIs
- MAE and mean signed bias with bootstrap 95% CIs
- Judge self-consistency
- Human-human agreement (pairwise Pearson + ICC(2,k))
- Response-length correlations
- Partial human-judge correlations controlling for response length
- Bootstrap 95% CIs for partial correlations
- Decoding/category breakdowns with bootstrap CIs
- Optional comparison of multiple judge columns

Usage:
    python research/analyze_tmlr.py \
        --input research/data/combined_with_judge.csv \
        --out_dir research/results/tmlr

For multi-judge output:
    python research/analyze_tmlr.py \
        --input research/data/multi_judge_results.csv \
        --judge_means llama3_mean qwen2_5_7b_mean
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DEFAULT_SEED = 20260908


# ============================================================
# BASIC CORRELATION
# ============================================================

def corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    if method == "spearman":
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)

    r = pearsonr(x, y)

    return float(r.statistic), float(r.pvalue)


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci(x, y, statistic, B=5000, seed=DEFAULT_SEED):
    """
    Bootstrap 95% CI for a statistic involving two variables.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    rng = np.random.default_rng(seed)

    n = len(x)

    if n < 3:
        return [np.nan, np.nan]

    vals = []

    for _ in range(B):

        idx = rng.integers(0, n, n)

        try:
            value = statistic(x[idx], y[idx])

            if np.isfinite(value):
                vals.append(value)

        except Exception:
            pass

    if not vals:
        return [np.nan, np.nan]

    return [
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5))
    ]


# ============================================================
# BOOTSTRAP CI FOR PARTIAL CORRELATION
# ============================================================

def bootstrap_partial_ci(
    x,
    y,
    control,
    B=5000,
    seed=DEFAULT_SEED
):
    """
    Bootstrap 95% CI for Pearson partial correlation.

    Measures correlation between x and y after removing the
    linear effect of control from both variables.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(control, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    x = x[mask]
    y = y[mask]
    z = z[mask]

    n = len(x)

    if n < 4:
        return [np.nan, np.nan]

    rng = np.random.default_rng(seed)

    values = []

    for _ in range(B):

        idx = rng.integers(0, n, n)

        try:

            value = partial_corr(
                x[idx],
                y[idx],
                z[idx]
            )

            if np.isfinite(value):
                values.append(value)

        except Exception:
            pass

    if not values:
        return [np.nan, np.nan]

    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5))
    ]


# ============================================================
# STATISTICS USED BY BOOTSTRAP
# ============================================================

def pearson_stat(x, y):
    return corr(x, y, "pearson")[0]


def spearman_stat(x, y):
    return corr(x, y, "spearman")[0]


def mae_stat(x, y):
    return float(np.mean(np.abs(x - y)))


def bias_stat(x, y):
    """
    Judge - Human
    """

    return float(np.mean(y - x))


# ============================================================
# ICC(2,k)
# ============================================================

def icc_2k(matrix):
    """
    Two-way random-effects, absolute-agreement ICC
    for average raters.
    """

    X = np.asarray(matrix, dtype=float)

    n, k = X.shape

    grand = X.mean()

    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)

    ss_subject = (
        k * np.sum((row_means - grand) ** 2)
    )

    ss_rater = (
        n * np.sum((col_means - grand) ** 2)
    )

    ss_error = np.sum(
        (
            X
            - row_means[:, None]
            - col_means[None, :]
            + grand
        ) ** 2
    )

    ms_subject = ss_subject / (n - 1)

    ms_rater = ss_rater / (k - 1)

    ms_error = ss_error / (
        (n - 1) * (k - 1)
    )

    return float(
        (ms_subject - ms_error)
        /
        (
            ms_subject
            + (ms_rater - ms_error) / n
        )
    )


# ============================================================
# PARTIAL CORRELATION
# ============================================================

def partial_corr(x, y, control):
    """
    Pearson partial correlation between x and y
    controlling for one variable.

    Example:

        partial_corr(
            human_score,
            judge_score,
            response_length
        )

    tells us how strongly human and judge scores are
    associated after removing the linear effect of response
    length from both scores.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(control, dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
    )

    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 4:
        return np.nan

    # Regress x on control variable z
    X = np.column_stack([
        np.ones(len(z)),
        z
    ])

    beta_x = np.linalg.lstsq(
        X,
        x,
        rcond=None
    )[0]

    residual_x = x - X @ beta_x

    # Regress y on control variable z
    beta_y = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )[0]

    residual_y = y - X @ beta_y

    if (
        np.std(residual_x) == 0
        or np.std(residual_y) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            residual_x,
            residual_y
        )[0, 1]
    )


# ============================================================
# PAIR SUMMARY
# ============================================================

def summarize_pair(
    df,
    human_col,
    judge_col,
    B,
    seed
):

    x = df[human_col].to_numpy(float)
    y = df[judge_col].to_numpy(float)

    r, p = corr(
        x,
        y,
        "pearson"
    )

    rho, ps = corr(
        x,
        y,
        "spearman"
    )

    return {

        "n": int(len(df)),

        "pearson_r": r,

        "pearson_p": p,

        "pearson_ci95": bootstrap_ci(
            x,
            y,
            pearson_stat,
            B,
            seed
        ),

        "spearman_rho": rho,

        "spearman_p": ps,

        "spearman_ci95": bootstrap_ci(
            x,
            y,
            spearman_stat,
            B,
            seed + 1
        ),

        "mae": mae_stat(
            x,
            y
        ),

        "mae_ci95": bootstrap_ci(
            x,
            y,
            mae_stat,
            B,
            seed + 2
        ),

        "signed_bias_judge_minus_human":
            bias_stat(x, y),

        "bias_ci95": bootstrap_ci(
            x,
            y,
            bias_stat,
            B,
            seed + 3
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="research/data/combined_with_judge.csv"
    )

    ap.add_argument(
        "--out_dir",
        default="research/results/tmlr"
    )

    ap.add_argument(
        "--human_col",
        default="human_avg_score"
    )

    ap.add_argument(
        "--judge_col",
        default="judge_score_mean"
    )

    ap.add_argument(
        "--judge_means",
        nargs="*",
        default=None,
        help=(
            "Optional judge mean columns, "
            "e.g. llama3_mean qwen2_5_7b_mean"
        )
    )

    ap.add_argument(
        "--bootstrap",
        type=int,
        default=5000
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED
    )

    args = ap.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    df = pd.read_csv(args.input)

    # ========================================================
    # RESPONSE LENGTH
    # ========================================================

    # If response_length_chars already exists, preserve it.
    #
    # Otherwise calculate it directly from model_output.

    if "response_length_chars" not in df.columns:

        if "model_output" not in df.columns:

            raise ValueError(
                "Neither 'response_length_chars' nor "
                "'model_output' exists in the input CSV."
            )

        df["response_length_chars"] = (
            df["model_output"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        print(
            "Calculated response_length_chars "
            "from model_output."
        )

    else:

        print(
            "Using existing response_length_chars column."
        )

    # ========================================================
    # JUDGE COLUMNS
    # ========================================================

    if args.judge_means:

        judge_cols = args.judge_means

    else:

        judge_cols = [args.judge_col]

    # ========================================================
    # REPORT OBJECT
    # ========================================================

    report = {

        "input": args.input,

        "n_items": len(df),

        "bootstrap_replicates": args.bootstrap,

        "response_length": {

            "source":
                "response_length_chars"
                if "response_length_chars" in df.columns
                else "model_output",

            "unit": "characters",

            "mean_chars":
                float(
                    df["response_length_chars"]
                    .mean()
                ),

            "median_chars":
                float(
                    df["response_length_chars"]
                    .median()
                ),

            "min_chars":
                int(
                    df["response_length_chars"]
                    .min()
                ),

            "max_chars":
                int(
                    df["response_length_chars"]
                    .max()
                ),
        }
    }

    # ========================================================
    # HUMAN-HUMAN AGREEMENT
    # ========================================================

    rater_cols = [
        c
        for c in df.columns
        if c.startswith("annotator_")
    ]

    if len(rater_cols) >= 2:

        pairs = []

        for a, b in itertools.combinations(
            sorted(rater_cols),
            2
        ):

            r, p = corr(
                df[a],
                df[b],
                "pearson"
            )

            pairs.append({

                "a": a,

                "b": b,

                "r": r,

                "p": p
            })

        report["human_human"] = {

            "raters":
                sorted(rater_cols),

            "n_raters":
                len(rater_cols),

            "mean_pairwise_pearson":
                float(
                    np.nanmean(
                        [q["r"] for q in pairs]
                    )
                ),

            "min_pairwise_pearson":
                float(
                    np.nanmin(
                        [q["r"] for q in pairs]
                    )
                ),

            "max_pairwise_pearson":
                float(
                    np.nanmax(
                        [q["r"] for q in pairs]
                    )
                ),

            "pairwise":
                pairs,

            "icc_2k":
                icc_2k(
                    df[
                        sorted(rater_cols)
                    ].to_numpy(float)
                )
        }

    # ========================================================
    # EACH JUDGE
    # ========================================================

    for judge_col in judge_cols:

        if judge_col not in df.columns:

            print(
                f"WARNING: {judge_col} "
                f"not found. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Human vs Judge
        # ----------------------------------------------------

        report[judge_col] = summarize_pair(
            df,
            args.human_col,
            judge_col,
            args.bootstrap,
            args.seed
        )

        # ----------------------------------------------------
        # RESPONSE LENGTH ANALYSIS
        # ----------------------------------------------------

        length = df[
            "response_length_chars"
        ].to_numpy(float)

        human = df[
            args.human_col
        ].to_numpy(float)

        judge = df[
            judge_col
        ].to_numpy(float)

        # Human vs length

        human_length_r, human_length_p = corr(
            human,
            length,
            "pearson"
        )

        # Judge vs length

        judge_length_r, judge_length_p = corr(
            judge,
            length,
            "pearson"
        )

        # Human vs Judge controlling for length

        partial_human_judge = partial_corr(
            human,
            judge,
            length
        )

        partial_ci = bootstrap_partial_ci(
            human,
            judge,
            length,
            args.bootstrap,
            args.seed + 10
        )

        report[judge_col]["length"] = {

            "unit": "characters",

            "mean_length_chars":
                float(np.mean(length)),

            "median_length_chars":
                float(np.median(length)),

            "human_vs_length": {

                "pearson_r":
                    human_length_r,

                "pearson_p":
                    human_length_p,

                "ci95":
                    bootstrap_ci(
                        human,
                        length,
                        pearson_stat,
                        args.bootstrap,
                        args.seed + 20
                    )
            },

            "judge_vs_length": {

                "pearson_r":
                    judge_length_r,

                "pearson_p":
                    judge_length_p,

                "ci95":
                    bootstrap_ci(
                        judge,
                        length,
                        pearson_stat,
                        args.bootstrap,
                        args.seed + 21
                    )
            },

            "human_vs_judge_partial_controlling_length": {

                "partial_r":
                    partial_human_judge,

                "ci95":
                    partial_ci
            }
        }

        # ----------------------------------------------------
        # JUDGE SELF-CONSISTENCY
        # ----------------------------------------------------

        prefix = judge_col.replace(
            "_mean",
            ""
        )

        run_cols = [
            c
            for c in df.columns
            if c.startswith(
                prefix + "_run"
            )
        ]

        if len(run_cols) >= 2:

            report[judge_col][
                "self_consistency"
            ] = {

                "run_columns":
                    run_cols,

                "mean_item_sd":
                    float(
                        df[run_cols]
                        .std(
                            axis=1,
                            ddof=1
                        )
                        .mean()
                    ),

                "exact_consistency_pct":
                    float(
                        (
                            df[run_cols]
                            .nunique(axis=1)
                            == 1
                        ).mean()
                        * 100
                    )
            }

        # ----------------------------------------------------
        # GROUP ANALYSES
        # ----------------------------------------------------

        for group_col in [
            "decoding_config",
            "category"
        ]:

            if group_col not in df.columns:
                continue

            groups = {}

            for value, g in df.groupby(
                group_col
            ):

                if len(g) < 3:
                    continue

                summary = summarize_pair(
                    g,
                    args.human_col,
                    judge_col,
                    args.bootstrap,
                    args.seed + 17
                )

                groups[str(value)] = summary

            report[judge_col][
                group_col
            ] = groups

    # ========================================================
    # JUDGE vs JUDGE
    # ========================================================

    if (
        len(judge_cols) >= 2
        and all(
            c in df.columns
            for c in judge_cols
        )
    ):

        vv = {}

        for a, b in itertools.combinations(
            judge_cols,
            2
        ):

            vv[
                f"{a}_vs_{b}"
            ] = {

                "pearson":
                    corr(
                        df[a],
                        df[b],
                        "pearson"
                    ),

                "spearman":
                    corr(
                        df[a],
                        df[b],
                        "spearman"
                    ),

                "mae":
                    mae_stat(
                        df[a],
                        df[b]
                    )
            }

        report[
            "judge_vs_judge"
        ] = vv

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        args.out_dir,
        "robustness_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            allow_nan=True
        )

    # ========================================================
    # HUMAN-READABLE REPORT
    # ========================================================

    lines = [

        "# TMLR Statistical Robustness Report",

        "",

        (
            "This report is generated from the supplied "
            "dataset; no result values are hard-coded."
        ),

        ""
    ]

    # ========================================================
    # HUMAN-HUMAN
    # ========================================================

    hh = report.get(
        "human_human"
    )

    if hh:

        lines += [

            "## Human-human agreement",

            "",

            f"- Annotators: "
            f"{hh['n_raters']}",

            (
                f"- Mean pairwise Pearson r: "
                f"**{hh['mean_pairwise_pearson']:.3f}**"
            ),

            (
                f"- Pairwise Pearson range: "
                f"{hh['min_pairwise_pearson']:.3f} "
                f"to "
                f"{hh['max_pairwise_pearson']:.3f}"
            ),

            (
                f"- ICC(2,k), absolute agreement: "
                f"**{hh['icc_2k']:.3f}**"
            ),

            ""
        ]

    # ========================================================
    # RESPONSE LENGTH OVERVIEW
    # ========================================================

    rl = report[
        "response_length"
    ]

    lines += [

        "## Response-length analysis",

        "",

        (
            f"- Length unit: "
            f"**{rl['unit']}**"
        ),

        (
            f"- Mean response length: "
            f"**{rl['mean_chars']:.2f} characters**"
        ),

        (
            f"- Median response length: "
            f"**{rl['median_chars']:.2f} characters**"
        ),

        (
            f"- Range: "
            f"{rl['min_chars']} to "
            f"{rl['max_chars']} characters"
        ),

        ""
    ]

    # ========================================================
    # EACH JUDGE REPORT
    # ========================================================

    for jc in judge_cols:

        if jc not in report:
            continue

        s = report[jc]

        lines += [

            f"## Human vs {jc}",

            "",

            f"- n = {s['n']}",

            (
                f"- Pearson r = "
                f"**{s['pearson_r']:.3f}**, "
                f"p = {s['pearson_p']:.4g}, "
                f"95% bootstrap CI = "
                f"[{s['pearson_ci95'][0]:.3f}, "
                f"{s['pearson_ci95'][1]:.3f}]"
            ),

            (
                f"- Spearman rho = "
                f"**{s['spearman_rho']:.3f}**, "
                f"p = {s['spearman_p']:.4g}, "
                f"95% bootstrap CI = "
                f"[{s['spearman_ci95'][0]:.3f}, "
                f"{s['spearman_ci95'][1]:.3f}]"
            ),

            (
                f"- MAE = "
                f"**{s['mae']:.2f}**, "
                f"95% bootstrap CI = "
                f"[{s['mae_ci95'][0]:.2f}, "
                f"{s['mae_ci95'][1]:.2f}]"
            ),

            (
                f"- Signed bias "
                f"(judge - human) = "
                f"**{s['signed_bias_judge_minus_human']:+.2f}**, "
                f"95% bootstrap CI = "
                f"[{s['bias_ci95'][0]:+.2f}, "
                f"{s['bias_ci95'][1]:+.2f}]"
            ),

            ""
        ]

        # ====================================================
        # LENGTH ANALYSIS
        # ====================================================

        if "length" in s:

            l = s["length"]

            human_l = l[
                "human_vs_length"
            ]

            judge_l = l[
                "judge_vs_length"
            ]

            partial = l[
                "human_vs_judge_partial_controlling_length"
            ]

            lines += [

                "### Response-length analysis",

                "",

                (
                    f"- Human score vs response length: "
                    f"r = "
                    f"{human_l['pearson_r']:.3f} "
                    f"(p={human_l['pearson_p']:.4g}), "
                    f"95% CI "
                    f"[{human_l['ci95'][0]:.3f}, "
                    f"{human_l['ci95'][1]:.3f}]"
                ),

                (
                    f"- {jc} score vs response length: "
                    f"r = "
                    f"{judge_l['pearson_r']:.3f} "
                    f"(p={judge_l['pearson_p']:.4g}), "
                    f"95% CI "
                    f"[{judge_l['ci95'][0]:.3f}, "
                    f"{judge_l['ci95'][1]:.3f}]"
                ),

                (
                    f"- Human vs {jc} partial Pearson r "
                    f"controlling for response length: "
                    f"**{partial['partial_r']:.3f}**, "
                    f"95% bootstrap CI = "
                    f"[{partial['ci95'][0]:.3f}, "
                    f"{partial['ci95'][1]:.3f}]"
                ),

                ""
            ]

        # ====================================================
        # SELF CONSISTENCY
        # ====================================================

        sc = s.get(
            "self_consistency"
        )

        if sc:

            lines += [

                "### Self-consistency",

                "",

                (
                    f"- Mean item SD across repeats: "
                    f"{sc['mean_item_sd']:.3f}"
                ),

                (
                    f"- Exact consistency across repeats: "
                    f"{sc['exact_consistency_pct']:.1f}%"
                ),

                ""
            ]

        # ====================================================
        # GROUP ANALYSIS
        # ====================================================

        for group_col in [
            "decoding_config",
            "category"
        ]:

            if group_col not in s:
                continue

            lines.append(
                f"### "
                f"{group_col.replace('_', ' ').title()}"
            )

            lines.append("")

            for value, g in s[
                group_col
            ].items():

                lines.append(

                    f"- **{value}** "
                    f"(n={g['n']}): "
                    f"r={g['pearson_r']:.3f}, "
                    f"95% CI "
                    f"[{g['pearson_ci95'][0]:.3f}, "
                    f"{g['pearson_ci95'][1]:.3f}], "
                    f"MAE={g['mae']:.2f}"
                )

            lines.append("")

    # ========================================================
    # JUDGE vs JUDGE
    # ========================================================

    if "judge_vs_judge" in report:

        lines += [

            "## Judge-vs-judge agreement",

            ""
        ]

        for name, value in report[
            "judge_vs_judge"
        ].items():

            lines.append(

                f"- **{name}**: "
                f"Pearson r="
                f"{value['pearson'][0]:.3f}, "
                f"Spearman rho="
                f"{value['spearman'][0]:.3f}, "
                f"MAE="
                f"{value['mae']:.2f}"
            )

        lines.append("")

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = os.path.join(
        args.out_dir,
        "REPORT.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    print(
        "Wrote",
        report_path
    )

    print(
        "Wrote",
        json_path
    )


if __name__ == "__main__":
    main()