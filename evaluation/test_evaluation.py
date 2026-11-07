"""
Tests for the evaluation pipeline.

  python evaluation/test_evaluation.py           # fast synthetic checks, no models needed (seconds)
  python evaluation/test_evaluation.py --smoke   # + end-to-end run on the first document of each corpus
                                                 #   (needs the models; writes to a temp dir only)

The test_* functions can also be collected by pytest.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import compute_mt_metrics as cm  # noqa: E402
import paired_comparisons as pc  # noqa: E402


def synthetic_scores(seed=0, corpora=(('news', 30), ('red_sorghum', 40))):
    """Passage x system scores where DeepSeek > Google Translate > GPT-4 ~ GPT-4o on average."""
    rng = np.random.default_rng(seed)
    offsets = {'DeepSeek': 0.05, 'Google Translate': 0.0, 'GPT-4': -0.03, 'GPT-4o': -0.03}
    rows = []
    for corpus, n in corpora:
        for k in range(n):
            base = rng.uniform(0.6, 0.9)
            for system, offset in offsets.items():
                rows.append({'corpus': corpus, 'doc': f'{corpus}_1', 'segment': k,
                             'passage_id': f'{corpus}_1_{k}', 'system': system,
                             'MPNet': base + offset + rng.normal(0, 0.01),
                             'comet': base + offset / 2 + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


# ==================== MPNet cosine ====================
def test_mpnet_cosine():
    a = np.array([[1.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 5.0], [2.0, 4.0], [4.0, 3.0]])
    np.testing.assert_allclose(cm.mpnet_cosine(a, b), [0.0, 1.0, 24 / 25])


# ==================== masking ====================
def test_mask_failed_translations():
    df = pd.DataFrame({'passage_id': ['p0', 'p0', 'p1'], 'system': ['DeepSeek', 'GPT-4', 'DeepSeek'],
                       'corpus': 'news', 'mt': ['fine', 'fine', 'text [Translation Failed]'],
                       'comet': [0.8, 0.7, 0.1], 'MPNet': [0.9, 0.8, 0.2]})
    cm.mask_failed_translations(df, ['comet', 'MPNet'])
    assert df.loc[2, ['comet', 'MPNet']].isna().all()
    assert df.loc[:1, ['comet', 'MPNet']].notna().all().all()
    assert list(df.excluded_reason) == ['', '', 'translation_failed']


def test_summarise_skips_missing():
    df = synthetic_scores()
    df.loc[0, ['MPNet', 'comet']] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        cm.summarise(df, ['MPNet', 'comet'], tmp)
        system = pd.read_csv(os.path.join(tmp, 'system_scores.csv'))
        corr = pd.read_csv(os.path.join(tmp, 'metric_correlations.csv'))
    assert system.drop(columns=['corpus', 'system']).notna().all().all()
    row = system[(system.corpus == 'news') & (system.system == 'DeepSeek')].iloc[0]
    assert (row.n_segments, row.n_excluded) == (29, 1)
    assert corr.loc[corr.corpus == 'all', 'n'].iloc[0] == len(df) - 1


def test_segment_scores_roundtrip():
    df = synthetic_scores()
    df['src'], df['mt'], df['ref'] = '源', 'NA', 'None'   # texts that pandas would read as missing by default
    df['align_cosine'] = 0.9
    df['excluded_reason'] = ''
    df.loc[0, 'MPNet'] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        cm.write_segment_scores(df, ['comet', 'MPNet'], tmp)
        back = cm.read_segment_scores(os.path.join(tmp, 'segment_scores.csv'))
    assert (back.mt == 'NA').all() and (back.ref == 'None').all()
    assert back.MPNet.isna().sum() == 1 and back.comet.notna().all()


# ==================== paired comparisons ====================
def test_rank_biserial():
    assert pc.rank_biserial([0.1, 0.2, 0.3]) == 1.0
    assert pc.rank_biserial([-0.1, -0.2]) == -1.0
    assert pc.rank_biserial([0, 0]) == 0.0
    # |d| ranks: 0.1 -> 1, 0.2 -> 2, 0.3 -> 3; R+ = 1 + 3 = 4, R- = 2; zero dropped -> r = 2/6
    assert np.isclose(pc.rank_biserial([0.1, -0.2, 0.3, 0.0]), 2 / 6)


def test_paired_comparisons_structure_and_signs():
    df = synthetic_scores()
    res = pc.paired_comparisons(df, 'MPNet', pc.SYSTEMS, 'MPNet')
    assert list(res.columns) == pc.CSV_COLUMNS
    assert list(res.Comparison) == ['DeepSeek vs. Google Translate', 'DeepSeek vs. GPT-4', 'DeepSeek vs. GPT-4o',
                                    'Google Translate vs. GPT-4', 'Google Translate vs. GPT-4o', 'GPT-4 vs. GPT-4o']
    assert (res.N == 70).all()
    first = res.iloc[0]          # DeepSeek is better by construction
    assert first['Mean difference'] > 0 and first['Rank-biserial r'] > 0.9 and first['Adjusted p-value'] < 0.001
    assert (res['Adjusted p-value'] >= res['Raw p-value'] - 1e-12).all()
    assert res['Rank-biserial r'].between(-1, 1).all()
    # Holm over exactly these 6 p-values
    np.testing.assert_allclose(res['Adjusted p-value'], multipletests(res['Raw p-value'], method='holm')[1])


def test_missing_passage_dropped_from_its_pairs_only():
    df = synthetic_scores()
    df.loc[(df.passage_id == 'news_1_0') & (df.system == 'DeepSeek'), 'comet'] = np.nan
    res = pc.paired_comparisons(df, 'comet', pc.SYSTEMS).set_index('Comparison')
    assert (res.loc[[c for c in res.index if c.startswith('DeepSeek')], 'N'] == 69).all()
    assert (res.loc[[c for c in res.index if 'DeepSeek' not in c], 'N'] == 70).all()


def test_identical_scores():
    df = synthetic_scores()
    df.loc[df.system == 'GPT-4o', 'MPNet'] = df.loc[df.system == 'GPT-4', 'MPNet'].to_numpy()
    row = pc.paired_comparisons(df, 'MPNet', pc.SYSTEMS).set_index('Comparison').loc['GPT-4 vs. GPT-4o']
    assert row['Raw p-value'] == 1.0 and row['Rank-biserial r'] == 0.0 and row['Mean difference'] == 0.0


def test_duplicate_rows_rejected():
    df = synthetic_scores()
    try:
        pc.paired_comparisons(pd.concat([df, df.head(1)]), 'MPNet', pc.SYSTEMS)
    except ValueError:
        return
    raise AssertionError('duplicate (passage_id, system) rows were not rejected')


def test_scopes_and_outputs():
    df = synthetic_scores()
    scoped = pc.run_scopes(df, ['MPNet', 'comet'])
    assert list(scoped) == ['all', 'news', 'red_sorghum']
    assert (scoped['news'].N == 30).all() and (scoped['red_sorghum'].N == 40).all()
    assert (scoped['all'].N == scoped['news'].N + scoped['red_sorghum'].N).all()
    assert list(scoped['all'].Metric) == ['MPNet'] * 6 + ['COMET'] * 6
    with tempfile.TemporaryDirectory() as tmp:
        paths = pc.write_outputs(scoped, tmp)
        assert sorted(os.path.basename(p) for p in paths) == sorted(
            f'paired_comparisons_{s}.{ext}' for s in scoped for ext in ('csv', 'tex'))
        for scope in scoped:
            tex = open(os.path.join(tmp, f'paired_comparisons_{scope}.tex'), encoding='utf-8').read()
            check_latex(tex, scope)
            back = pd.read_csv(os.path.join(tmp, f'paired_comparisons_{scope}.csv'))
            assert list(back.columns) == pc.CSV_COLUMNS and len(back) == 12


def check_latex(tex, scope):
    label = 'tab:paired_tests' + ('' if scope == 'all' else f'_{scope}')
    assert tex.count('{') == tex.count('}'), 'unbalanced braces'
    assert rf'\label{{{label}}}' in tex
    assert r'\begin{tabular}{llrrrrrr}' in tex
    rows = [line for line in tex.splitlines() if line.startswith(('MPNet &', 'COMET &'))]
    assert len(rows) == 12 and all(line.count('&') == 7 for line in rows)


# ==================== smoke run ====================
def smoke_test():
    """End-to-end on the first document of each corpus; needs the models (LaBSE, roberta-large, COMET, MPNet)."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(HERE, 'compute_mt_metrics.py'), '--limit-docs', '1',
                        '--out', tmp], check=True)
        subprocess.run([sys.executable, os.path.join(HERE, 'paired_comparisons.py'), '--results', tmp], check=True)

        seg = cm.read_segment_scores(os.path.join(tmp, 'segment_scores.csv'))
        metrics = ['bertscore_p', 'bertscore_r', 'bertscore_f1', 'comet', 'MPNet']
        for column in ['passage_id', 'excluded_reason'] + metrics:
            assert column in seg.columns, f'missing column {column}'
        assert seg[metrics].notna().all().all(), 'unexpected missing scores'
        assert (seg.groupby('passage_id').system.nunique() == 4).all(), 'passage without all 4 systems'
        assert seg[['bertscore_p', 'bertscore_r', 'bertscore_f1']].stack().between(0.5, 1.0).all()
        assert seg.MPNet.between(-1, 1).all() and seg.comet.between(-0.5, 1.5).all()
        system = pd.read_csv(os.path.join(tmp, 'system_scores.csv'))
        assert {'MPNet', 'MPNet_ci_low', 'MPNet_ci_high', 'n_segments', 'n_excluded'} <= set(system.columns)
        corr = pd.read_csv(os.path.join(tmp, 'metric_correlations.csv'))
        assert ((corr.metric_a == 'MPNet') | (corr.metric_b == 'MPNet')).any()
        for scope in ['all', 'news', 'red_mansions', 'red_sorghum']:
            base = os.path.join(tmp, 'paired_comparisons', f'paired_comparisons_{scope}')
            res = pd.read_csv(base + '.csv')
            assert len(res) == 12 and set(res.Metric) == {'MPNet', 'COMET'}
            check_latex(open(base + '.tex', encoding='utf-8').read(), scope)
    print('smoke test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke', action='store_true', help='also run the end-to-end smoke test (needs models)')
    args = parser.parse_args()
    tests = [(name, fn) for name, fn in globals().items() if name.startswith('test_') and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as exc:  # report every failure, not just the first
            failed += 1
            print(f'FAIL {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - failed}/{len(tests)} synthetic tests passed')
    if failed:
        sys.exit(1)
    if args.smoke:
        smoke_test()


if __name__ == '__main__':
    main()
