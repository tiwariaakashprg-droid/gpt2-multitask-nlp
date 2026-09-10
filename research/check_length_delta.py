import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def partial_corr(x, y, z):
    """
    Pearson partial correlation between x and y
    controlling for z.
    """

    X = np.column_stack([
        np.ones(len(z)),
        z
    ])

    beta_x = np.linalg.lstsq(
        X, x, rcond=None
    )[0]

    beta_y = np.linalg.lstsq(
        X, y, rcond=None
    )[0]

    residual_x = x - X @ beta_x
    residual_y = y - X @ beta_y

    return np.corrcoef(
        residual_x,
        residual_y
    )[0, 1]


# ------------------------------------------------------------
# Load final dataset
# ------------------------------------------------------------

df = pd.read_csv(
    "research/data/multi_judge_results.csv"
)

human = df[
    "human_avg_score"
].to_numpy(float)

length = (
    df["model_output"]
    .fillna("")
    .astype(str)
    .str.len()
    .to_numpy(float)
)

B = 5000
rng = np.random.default_rng(
    20260908
)

n = len(df)

print("n =", n)
print()


# ------------------------------------------------------------
# LLaMA and Qwen
# ------------------------------------------------------------

judges = {
    "LLaMA": "llama3_mean",
    "Qwen": "qwen2_5_7b_mean"
}


for name, column in judges.items():

    judge = df[
        column
    ].to_numpy(float)

    # Raw human-judge correlation
    raw_r = pearsonr(
        human,
        judge
    ).statistic

    # Human-judge correlation after
    # controlling for response length
    partial_r = partial_corr(
        human,
        judge,
        length
    )

    # Difference
    delta_r = (
        raw_r - partial_r
    )

    # --------------------------------------------------------
    # Bootstrap Delta
    # --------------------------------------------------------

    bootstrap_values = []

    for _ in range(B):

        idx = rng.integers(
            0,
            n,
            n
        )

        h = human[idx]
        j = judge[idx]
        l = length[idx]

        try:

            raw_boot = pearsonr(
                h,
                j
            ).statistic

            partial_boot = partial_corr(
                h,
                j,
                l
            )

            value = (
                raw_boot
                - partial_boot
            )

            if np.isfinite(value):
                bootstrap_values.append(
                    value
                )

        except Exception:
            pass

    ci = np.percentile(
        bootstrap_values,
        [2.5, 97.5]
    )

    print(
        f"{name}"
    )

    print(
        f"Raw r = {raw_r:.6f}"
    )

    print(
        f"Partial r = {partial_r:.6f}"
    )

    print(
        f"Delta r = {delta_r:.6f}"
    )

    print(
        "Bootstrap 95% CI for Delta r = "
        f"[{ci[0]:.6f}, {ci[1]:.6f}]"
    )

    print()