# Evaluation update plan (issue #2)

Source: [issue #2, "Metric Evaluation and Comparison Scripts"](https://github.com/sydney-machine-learning/LLM-Translation-Madarin/issues/2),
branch `2-metric-evaluation-and-comparison-scripts`.

## What the issue asks for

1. **MPNet cosine similarity** between each machine-translated passage and its aligned expert passage,
   `cos(A, B) = A·B / (‖A‖‖B‖)` (A = expert embedding, B = machine embedding). It goes in every results
   table, with the column named `MPNet`.
2. **Paired statistical comparisons** between the four systems for MPNet and COMET. Use passage-level
   scores, run one test per metric, apply a multiplicity correction, and report the mean difference, median
   difference, test statistic, adjusted p-value, effect size and number of paired passages. Output goes to
   CSV first, then to a LaTeX table in the layout given in the issue.
3. **A small test script** that runs the pipeline on a subset of the data and checks that every metric and
   data structure is handled correctly.

## Current state

| File | What it does now | Gap |
|---|---|---|
| `compute_mt_metrics.py` | Aligns expert paragraphs to the source with LaBSE, then scores each aligned segment with BERTScore P/R/F1, COMET and optionally CometKiwi. Writes `segment_scores.csv`, `system_scores.csv` (mean + 95% bootstrap CI), `metric_correlations.csv` and `alignments/`. | No MPNet. |
| `paired_comparisons.py` | One function, `paired_comparisons(df, metric, systems)`. Pivots to passage × system, runs a two-sided Wilcoxon signed-rank test for each pair, and applies a Holm correction across the 6 pairs of one metric. | No effect size. Not wired to the results files. No CLI and no output. Comparison label uses `vs` but the table uses `vs.`. Drops pairs with fewer than 2 passages without warning. Needs `statsmodels`, which is not in the install instructions. |
| `results/` | Output of the October full run: 403 aligned passages (news 44, Red Mansions 105, Red Sorghum 254) × 4 systems = 1,612 rows. | No `MPNet` column. |
| `METRICS.md`, `../README.md` | Describe the metrics and the results. | Need MPNet, the paired tests and the new scripts. |
| `semantic similarity compare table/translation_similarity_analysis.py` | Older script. Computes `all-mpnet-base-v2` cosine **between systems**, not against the expert reference. | Shows which MPNet model the paper already uses. |

## Plan

### Step 1: Add MPNet to `compute_mt_metrics.py`

- Add `MPNET_MODEL = 'sentence-transformers/all-mpnet-base-v2'` (decided, U1). It is the same model the
  existing comparison tables use. Record the model name and the `sentence-transformers` version in
  `run.log` so the paper can cite them.
- Add `add_mpnet(df)`:
  - Encode `df.ref` (unique values only, then map back) and `df.mt` with `SentenceTransformer(MPNET_MODEL)`.
  - Compute the cosine with the issue's formula directly: `(A*B).sum(1) / (‖A‖·‖B‖)`. This is the same
    as normalising first, but the code then matches the equation in the paper.
  - Store the result in column `MPNet`.
  - Like the other metrics, print how many inputs exceed the model's `max_seq_length` (384 word pieces for
    all-mpnet-base-v2) and are therefore truncated.
- Call it in `main()` and add `'MPNet'` to `metrics`. The column then flows automatically into
  `segment_scores.csv`, `system_scores.csv` (`MPNet`, `MPNet_ci_low`, `MPNet_ci_high`),
  `metric_correlations.csv` and the console summary.
- Add `--only-mpnet` (working name). It reads an existing `segment_scores.csv`, adds the MPNet column and
  rewrites the three summary files without re-running LaBSE, BERTScore or COMET. This saves a long run and
  guarantees the passage set is the same one the existing COMET scores used. A full re-run stays the
  default and must reproduce identical alignments. (See U8.)
- **Set pipeline failures to missing (decided, U7).** Add `FAILED_MARKERS = ['[Translation Failed]']`.
  After all metrics are computed, any row whose `mt` contains a marker gets **every** metric column
  (`bertscore_*`, `comet`, `cometkiwi`, `MPNet`) set to NaN and a new column `excluded_reason` set to
  `translation_failed`. The row stays in `segment_scores.csv` for transparency.
  - Checked: the only occurrence in the data is DeepSeek, `news_2`, segment 6, in
    `data set/deepseek/deepseek_news/Chinese_version_news2_deepseek_chat_translated.csv`, line 8.
  - The same masking runs in `--only-mpnet` mode, so the existing BERTScore and COMET values for that row
    also become NaN.
  - Knock-on changes so NaNs are handled correctly:
    - `summarise()` drops NaNs per metric before the mean and bootstrap. Without this, `rng.choice` with a
      NaN gives a NaN CI.
    - `system_scores.csv` reports `n_segments` per system and metric. DeepSeek/news becomes 43 and
      DeepSeek/all becomes 402.
    - `metric_correlations.csv` uses only rows where both metrics are present, and reports that `n`.
    - The console log prints the excluded rows.
  - Genuine system behaviour (the GPT-4o refusal and the DeepSeek/Google repetition loops) is **not**
    masked.
  - Note: this changes the published October DeepSeek news means slightly, because one segment is
    removed. METRICS.md's results table must be updated to match.
- Add a stable `passage_id` column (`<doc>_<segment>`, e.g. `red_sorghum_4_120`) to `segment_scores.csv`
  so the paired-test script does not have to rebuild it.

### Step 2: Make `paired_comparisons.py` a library plus a results-processing script

Keep Rohit's function as the core. Add the following:

- **Effect size (decided: rank-biserial correlation).** Add the matched-pairs rank-biserial correlation
  for the Wilcoxon signed-rank test, `r = (R⁺ − R⁻) / (R⁺ + R⁻)`. R⁺ and R⁻ are the sums of the ranks of
  |A − B| for positive and negative differences, computed with the same zero handling as the Wilcoxon call
  (scipy's default `zero_method='wilcox'` drops zero differences). r ranges from −1 to 1, and its sign
  follows the mean difference convention: positive means the first system scores higher. If all
  differences are zero, r is 0. This is the only effect size reported, in both the CSV and the LaTeX
  table's "Effect size" column.
- **Validation:**
  - Assert there is exactly one row per (passage_id, system). `pivot` would otherwise fail obscurely.
  - Before any test, check that all four systems are present for the same passage set. List any passage
    missing a system score. Missing scores are excluded per comparison, as the issue requires.
  - Warn on pairs with too few passages instead of skipping them silently.
- **Label format.** Use `"{A} vs. {B}"` and use the issue's system order: DeepSeek, Google Translate,
  GPT-4, GPT-4o. This gives exactly the 6 rows per metric in the table. `Mean difference = A − B`, so a
  positive value means the first system scores higher. Both MPNet and COMET are higher-is-better.
- **Keep the raw p-value** in the CSV next to the adjusted one, and record which correction was used.
- **CLI** (`python evaluation/paired_comparisons.py [--results evaluation/results]`):
  1. Read `results/segment_scores.csv`.
  2. Run the tests for `MPNet` and `comet` separately, once for each **scope** (decided, see U4):
     - `all`: every passage, pooled across corpora (403).
     - `news`: 44 passages.
     - `red_mansions`: 105 passages.
     - `red_sorghum`: 254 passages.

     Each scope is analysed on its own. **Multiplicity correction (decided, U3):** Holm, applied within
     each metric over its 6 pairwise comparisons, separately in each scope. It is never pooled across
     metrics (12) or across scopes (48). This matches the supplied `paired_comparisons()` function, which
     already corrects per call.
  3. For each scope, write a CSV with these columns: Metric, Comparison, N, Mean difference, Median
     difference, Test statistic, Raw p, Adjusted p, Rank-biserial r.
  4. For each scope, write a `.tex` from that CSV using the issue's `table*` layout:
     - The caption names the corpus, and the label is unique per scope (e.g. `tab:paired_tests_news`),
       so the tables can coexist in one paper.
     - Fix the stray `}` in the issue's `\caption{...}}`.
     - Format p-values as `< 0.001` below that threshold, and print differences and effect sizes to 3–4 d.p.
     - Report N in the caption, or in an extra column if N differs between rows (see U5).

  Output files, one CSV + TeX pair per scope, all in `results/paired_comparisons/`:

  | Scope | CSV | LaTeX | Label |
  |---|---|---|---|
  | Pooled (all corpora) | `paired_comparisons_all.csv` | `paired_comparisons_all.tex` | `tab:paired_tests` |
  | News | `paired_comparisons_news.csv` | `paired_comparisons_news.tex` | `tab:paired_tests_news` |
  | A Dream of Red Mansions | `paired_comparisons_red_mansions.csv` | `paired_comparisons_red_mansions.tex` | `tab:paired_tests_red_mansions` |
  | Red Sorghum | `paired_comparisons_red_sorghum.csv` | `paired_comparisons_red_sorghum.tex` | `tab:paired_tests_red_sorghum` |

  The pooled table keeps the issue's original label `tab:paired_tests`. Scope names come from the
  `corpus` column, so a `--limit-docs` run or a future corpus produces matching files automatically.

The issue says this table "should be calculated after all other results", so this script only reads the
saved results files. It never calls a model.

### Step 3: Test script, `evaluation/test_evaluation.py`

Two layers, so the fast checks need no model downloads:

1. **Synthetic checks (seconds, no models):**
   - Build a small passage × system DataFrame with known differences and run `paired_comparisons`.
   - Check: 6 rows per metric; the sign convention (A better gives a positive mean difference); adjusted
     p ≥ raw p; Holm is applied over exactly 6 p-values per metric per scope (checked against
     `multipletests` on those 6 values directly); a passage missing for one system is dropped from that pair only; identical scores give
     p = 1 and r = 0.
   - Masking: a synthetic row with `mt = '[Translation Failed]'` ends up with all metrics NaN and
     `excluded_reason` set. `summarise()` still returns finite means and CIs. That passage's N drops by 1
     only in comparisons involving that system.
   - Check the rank-biserial r: it is in [−1, 1], and it is +1 when A beats B on every passage and −1 when
     B always wins. Also check it against a small example whose R⁺ and R⁻ are worked out by hand.
   - Check the LaTeX writer emits 12 data rows, balanced braces and the expected header.
   - With a synthetic two-corpus frame, check that one CSV + TeX pair is written for `all` and for each
     corpus. Each per-corpus file must contain only that corpus's passages, and N in the `all` file must
     equal the sum across corpora.
   - Check the MPNet cosine function against a hand-computed example, e.g. orthogonal vectors give 0 and
     identical vectors give 1.
2. **Smoke run (`--smoke`, needs the models):**
   - Run `compute_mt_metrics.py --limit-docs 1 --out <tmp dir>`, then the paired-comparison CLI on that
     output.
   - Assert: the expected columns exist (`bertscore_*`, `comet`, `MPNet`, `passage_id`); there are no NaNs;
     every passage has all 4 systems; scores are in plausible ranges; all four CSV + `.tex` pairs (`all`,
     `news`, `red_mansions`, `red_sorghum`) are written.
   - Writes to a temporary directory and never overwrites `results/`.

It runs with plain `python evaluation/test_evaluation.py`. Pytest is optional; the functions are
`test_*`-named so pytest can also collect them.

### Step 4: Regenerate results and update docs

- Run step 1 (MPNet only, or in full), then step 2, and commit the updated `results/*.csv` and the new
  `results/paired_comparisons/` files (four CSV + TeX pairs). Update `run.log`.
- `METRICS.md`:
  - Add MPNet to the summary table and the results table.
  - Add a "Paired comparisons" section covering the method, the passage as the unit of analysis,
    Wilcoxon + Holm, the effect size, and the issue's caveat that significance is not linguistic
    superiority.
  - Add `statsmodels` to the pip install line.
- `README.md`: add the new script and the test command.
- Add `evaluation/requirements.txt`, which pins the packages actually used. The local system Python
  currently has no pandas, so a venv is needed.

## Remaining uncertainties (need a decision)

| # | Question | Proposed default |
|---|---|---|
| U1 | ~~**Which MPNet model?**~~ **Resolved:** `sentence-transformers/all-mpnet-base-v2`. | Same model as the existing comparison tables (`semantic similarity compare table/translation_similarity_analysis.py`). |
| U2 | ~~**Which effect size?**~~ **Resolved:** matched-pairs rank-biserial correlation for the Wilcoxon test. | Implemented as described in Step 2. |
| U3 | ~~**Multiplicity family.**~~ **Resolved:** Holm correction within each metric, over its 6 pairwise comparisons, applied separately in each scope. | Each table is self-contained, and MPNet and COMET are corrected independently. Each caption states "Holm-adjusted within metric (6 comparisons)". |
| U4 | ~~**Pooled or per-corpus?**~~ **Resolved:** produce both, in separate files. | One CSV + TeX pair each for `all`, `news`, `red_mansions` and `red_sorghum` in `results/paired_comparisons/` (see Step 2). |
| U5 | **Where to report N.** The issue says "report the number of paired passages" but the table has no N column. Because of U7, N now **varies within the pooled and news tables**: DeepSeek pairs are 402 vs 403 pooled and 43 vs 44 for news. | Add an `N` column to every LaTeX table, which changes the layout from `llrrrrr` to `llrrrrrr`, for consistency across the four files. Alternative: keep the issue's layout and give the N values in a caption footnote. |
| U6 | **Definition of "passage".** The issue text says "paragraph". The pipeline's unit is the aligned segment, which can be 1–3 merged source/expert paragraphs. | Use the aligned segment as the passage, because it is the only unit with a matched expert reference. Say so in the paper. |
| U7 | ~~**Known bad outputs.**~~ **Resolved:** the `[Translation Failed]` row (DeepSeek, news_2 segment 6) is set to missing (NaN for all metrics). The refusal and the repetition loops are kept as genuine system output. | Implemented in Step 1. Through the per-pair exclusion rule, the passage drops out of the 3 DeepSeek comparisons only: N is 402 pooled and 43 for news, against 403 and 44 for the other pairs. |
| U8 | **Re-run or append?** A full re-run takes a long time and downloads about 5 GB. Recomputing could also shift COMET or BERTScore slightly (different library versions or hardware). | Append MPNet to the existing `segment_scores.csv` (`--only-mpnet`), so the reported COMET and BERTScore numbers stay unchanged. Check reproducibility with the smoke test. |
| U9 | **Truncation.** all-mpnet-base-v2 reads at most 384 word pieces, fewer than BERTScore and COMET (512). Long literary segments will be truncated more. | Report the count, as for the other metrics. Do not chunk, because chunk-averaging would change the metric's definition. Revisit if the count is large. |
| U10 | **Scope of "all the tables of results".** Does this mean only `evaluation/results` and METRICS.md, or also the paper's LaTeX tables and the older sentiment and "semantic similarity compare table" outputs? | Only the `evaluation/` pipeline outputs and docs. The older scripts measure different things (corpus-level MiniLM, system-vs-system MPNet) and are left as they are. |
| U11 | **Test statistic.** scipy's two-sided Wilcoxon reports W = min(R⁺, R⁻). Some papers report a z-value instead. | Report W. If needed, add z, computed from the normal approximation, to the CSV. |
| U12 | **CometKiwi.** Include it in the paired tests if the column is present? | No. The issue lists only MPNet and COMET. The script will accept `--metrics` so it can be added later. |
