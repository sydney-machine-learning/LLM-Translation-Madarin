import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

def paired_comparisons(df, metric, systems):
    """
    df: one row per aligned passage.
    Columns: passage_id, system, metric score.

    Returns pairwise comparisons for the specified metric.
    """

    scores = df.pivot(
        index="passage_id",
        columns="system",
        values=metric
    )

    comparisons = []

    for i, system_a in enumerate(systems):
        for system_b in systems[i + 1:]:

            paired = scores[[system_a, system_b]].dropna()

            a = paired[system_a].to_numpy()
            b = paired[system_b].to_numpy()
            differences = a - b

            if len(differences) < 2:
                continue

            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    a, b,
                    alternative="two-sided"
                )
                statistic = result.statistic
                p_value = result.pvalue

            comparisons.append({
                "Metric": metric,
                "Comparison": f"{system_a} vs {system_b}",
                "N": len(differences),
                "Mean difference": np.mean(differences),
                "Median difference": np.median(differences),
                "Test statistic": statistic,
                "Raw p-value": p_value
            })

    # Holm correction across pairwise comparisons
    if comparisons:
        adjusted = multipletests(
            [x["Raw p-value"] for x in comparisons],
            method="holm"
        )[1]

        for result, p_adjusted in zip(comparisons, adjusted):
            result["Adjusted p-value"] = p_adjusted

    return pd.DataFrame(comparisons)

