"""
Paired comparisons between translation systems on passage-level scores.

Reads evaluation/results/segment_scores.csv (written by compute_mt_metrics.py) and, for each metric
separately, compares every pair of systems with a two-sided Wilcoxon signed-rank test.

  * The unit of analysis is the aligned passage (passage_id = <doc>_<segment>); chapter or corpus
    averages are never used as observations.
  * A passage missing a score for either system of a pair is excluded from that pair only.
  * Mean / median difference = first system - second system, so a positive value means the first
    system scores higher (all metrics here are higher-is-better).
  * p-values are Holm-adjusted within each metric over its pairwise comparisons (6 for 4 systems),
    separately in each scope.
  * Effect size: matched-pairs rank-biserial correlation r = (R+ - R-) / (R+ + R-), where R+ and R-
    are the rank sums of the positive and negative differences, with zero differences dropped as in
    the Wilcoxon test.  r lies in [-1, 1] and has the same sign as the differences.

Scopes: 'all' (every passage pooled across corpora) and each corpus on its own.  Each scope is
written as its own CSV and LaTeX table to <results>/paired_comparisons/.

Usage
  python evaluation/paired_comparisons.py
  python evaluation/paired_comparisons.py --results /path/to/results --metrics MPNet comet
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Order used for the comparisons (and therefore for the sign of the differences).
SYSTEMS = ['DeepSeek', 'Google Translate', 'GPT-4', 'GPT-4o']
METRIC_LABELS = {'MPNet': 'MPNet', 'comet': 'COMET', 'cometkiwi': 'CometKiwi',
                 'bertscore_f1': 'BERTScore F1', 'bertscore_p': 'BERTScore P', 'bertscore_r': 'BERTScore R'}
DEFAULT_METRICS = ['MPNet', 'comet']
CORPUS_LABELS = {'all': 'all corpora pooled', 'news': 'News', 'red_mansions': 'A Dream of Red Mansions',
                 'red_sorghum': 'Red Sorghum'}
CSV_COLUMNS = ['Metric', 'Comparison', 'N', 'Mean difference', 'Median difference', 'Test statistic',
               'Raw p-value', 'Adjusted p-value', 'Rank-biserial r']


def rank_biserial(differences):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]                       # same zero handling as wilcoxon(zero_method='wilcox')
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    r_plus, r_minus = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (r_plus - r_minus) / (r_plus + r_minus)


def check_passages(df, metric, systems):
    """Report passages that do not have a score for every system (they drop out of some pairs)."""
    scores = df.pivot(index='passage_id', columns='system', values=metric).reindex(columns=systems)
    incomplete = scores[scores.isna().any(axis=1)]
    for passage_id, row in incomplete.iterrows():
        missing = ', '.join(row.index[row.isna()])
        print(f'  {metric}: passage {passage_id} has no score for {missing}; '
              f'excluded from comparisons involving that system')
    return incomplete


def paired_comparisons(df, metric, systems, label=None):
    """
    df: one row per aligned passage x system.
    Columns: passage_id, system, metric score.

    Returns pairwise comparisons for the specified metric, Holm-adjusted across those comparisons.
    """
    duplicated = df.duplicated(['passage_id', 'system'])
    if duplicated.any():
        raise ValueError(f'{duplicated.sum()} duplicate (passage_id, system) rows; passage_id must be unique')
    missing_systems = set(systems) - set(df.system)
    if missing_systems:
        raise ValueError(f'No scores for system(s): {sorted(missing_systems)}')

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

            row = {
                "Metric": label or metric,
                "Comparison": f"{system_a} vs. {system_b}",
                "N": len(differences),
                "Mean difference": np.nan,
                "Median difference": np.nan,
                "Test statistic": np.nan,
                "Raw p-value": np.nan,
                "Rank-biserial r": np.nan,
            }
            comparisons.append(row)

            if len(differences) < 2:
                warnings.warn(f'{metric}: {system_a} vs. {system_b} has only {len(differences)} paired '
                              f'passage(s); no test performed')
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

            row.update({
                "Mean difference": np.mean(differences),
                "Median difference": np.median(differences),
                "Test statistic": statistic,
                "Raw p-value": p_value,
                "Rank-biserial r": rank_biserial(differences),
            })

    # Holm correction across the pairwise comparisons of this metric
    results = pd.DataFrame(comparisons)
    tested = results["Raw p-value"].notna()
    results["Adjusted p-value"] = np.nan
    if tested.any():
        results.loc[tested, "Adjusted p-value"] = multipletests(
            results.loc[tested, "Raw p-value"],
            method="holm"
        )[1]

    return results[CSV_COLUMNS]


def run_scopes(df, metrics, systems=SYSTEMS):
    """Return {scope: comparisons DataFrame} for 'all' and for each corpus separately."""
    scopes = [('all', df)] + list(df.groupby('corpus', sort=False))
    out = {}
    for scope, group in scopes:
        print(f'[{scope}] {group.passage_id.nunique()} passages')
        for metric in metrics:
            check_passages(group, metric, systems)
        out[scope] = pd.concat([paired_comparisons(group, m, systems, METRIC_LABELS.get(m, m))
                                for m in metrics], ignore_index=True)
    return out


# ==================== LaTeX ====================
def _num(value, fmt):
    if pd.isna(value):
        return '--'
    text = format(value, fmt)
    return f'${text}$' if text.startswith('-') else text


def _p(value):
    if pd.isna(value):
        return '--'
    return '$<0.001$' if value < 0.001 else f'{value:.3f}'


def to_latex(results, scope):
    suffix = '' if scope == 'all' else f'_{scope}'
    scope_label = CORPUS_LABELS.get(scope, scope.replace('_', ' '))
    n_comparisons = len(results) // results.Metric.nunique()
    n_values = sorted(results.N.unique())
    n_text = (f'$N={n_values[0]}$ paired passages' if len(n_values) == 1
              else f'$N$ = number of paired passages ({n_values[0]}--{n_values[-1]})')
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        rf'\caption{{Paired comparisons between translation systems ({scope_label}). Two-sided Wilcoxon '
        r'signed-rank tests on passage-level scores; differences are the first system minus the second, '
        r'so positive values favour the first system. $p$-values are Holm-adjusted within metric '
        rf'({n_comparisons} comparisons). Effect size is the matched-pairs rank-biserial correlation. {n_text}.}}',
        rf'\label{{tab:paired_tests{suffix}}}',
        r'\small',
        r'\begin{tabular}{llrrrrrr}',
        r'\hline',
        r'\textbf{Metric} & \textbf{Comparison} & \textbf{$N$} & \textbf{Mean Difference} & '
        r'\textbf{Median Difference} & \textbf{Test statistic} & \textbf{Adjusted $p$} & '
        r'\textbf{Effect size} \\',
        r'\hline',
    ]
    for _, row in results.iterrows():
        cells = [row.Metric, row.Comparison, str(row.N), _num(row['Mean difference'], '.4f'),
                 _num(row['Median difference'], '.4f'), _num(row['Test statistic'], '.1f'),
                 _p(row['Adjusted p-value']), _num(row['Rank-biserial r'], '.3f')]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table*}', '']
    return '\n'.join(lines)


def write_outputs(scoped, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for scope, results in scoped.items():
        base = os.path.join(out_dir, f'paired_comparisons_{scope}')
        results.to_csv(base + '.csv', index=False, float_format='%.6g')
        with open(base + '.tex', 'w', encoding='utf-8') as fh:
            fh.write(to_latex(results, scope))
        paths += [base + '.csv', base + '.tex']
    return paths


def load_scores(results_dir, metrics):
    path = os.path.join(results_dir, 'segment_scores.csv')
    header = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
    missing = [m for m in metrics if m not in header]
    if missing:
        raise SystemExit(f'{path} has no column(s) {missing}; run compute_mt_metrics.py first '
                         f'(use --only-mpnet to add MPNet to existing results)')
    cols = [c for c in ['corpus', 'doc', 'segment', 'passage_id', 'system'] if c in header] + metrics
    df = pd.read_csv(path, usecols=cols, encoding='utf-8-sig')
    if 'passage_id' not in df.columns:
        df['passage_id'] = df.doc + '_' + df.segment.astype(str)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default=os.path.join(REPO_ROOT, 'evaluation', 'results'),
                        help='directory containing segment_scores.csv')
    parser.add_argument('--out', default=None, help='output directory (default <results>/paired_comparisons)')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    args = parser.parse_args()

    df = load_scores(args.results, args.metrics)
    scoped = run_scopes(df, args.metrics)
    paths = write_outputs(scoped, args.out or os.path.join(args.results, 'paired_comparisons'))

    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        for scope, results in scoped.items():
            print(f'\n=== {scope} ===')
            print(results.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\nWritten:\n  ' + '\n  '.join(paths))


if __name__ == '__main__':
    main()
