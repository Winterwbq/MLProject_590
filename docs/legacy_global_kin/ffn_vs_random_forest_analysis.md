# Why FFN Is Worse Than Random Forest on This Dataset

This report explains why the two feed-forward-network baselines underperformed the main random-forest model on the locked random test split. The explanation is based on the saved experiment outputs rather than on architecture assumptions alone, so each conclusion below cites the exact CSVs and plots that support it.

## Scope

Models compared:

- main random forest: [`results/full_training_pipeline`](../../results/full_training_pipeline)
- FFN baseline 1: [`results/ffn_baselines/direct_all_inputs_end_to_end`](../../results/ffn_baselines/direct_all_inputs_end_to_end)
- FFN baseline 2: [`results/ffn_baselines/rf_replacement_composition_pca`](../../results/ffn_baselines/rf_replacement_composition_pca)

Primary comparison artifacts:

- overall comparison table: [`results/ffn_baselines/comparison_analysis/overall_model_comparison.csv`](../../results/ffn_baselines/comparison_analysis/overall_model_comparison.csv)
- per-case comparison table: [`results/ffn_baselines/comparison_analysis/per_case_log_rmse_comparison.csv`](../../results/ffn_baselines/comparison_analysis/per_case_log_rmse_comparison.csv)
- per-case summary table: [`results/ffn_baselines/comparison_analysis/per_case_log_rmse_distribution_summary.csv`](../../results/ffn_baselines/comparison_analysis/per_case_log_rmse_distribution_summary.csv)
- per-reaction comparison table: [`results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_comparison.csv`](../../results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_comparison.csv)
- per-reaction summary table: [`results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_distribution_summary.csv`](../../results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_distribution_summary.csv)
- magnitude-binned relative-error table: [`results/ffn_baselines/comparison_analysis/relative_error_by_magnitude_comparison.csv`](../../results/ffn_baselines/comparison_analysis/relative_error_by_magnitude_comparison.csv)
- overall comparison plot: [`results/ffn_baselines/comparison_figures/overall_log_rmse_comparison.png`](../../results/ffn_baselines/comparison_figures/overall_log_rmse_comparison.png)
- per-case histogram: [`results/ffn_baselines/comparison_figures/per_case_log_rmse_histogram.png`](../../results/ffn_baselines/comparison_figures/per_case_log_rmse_histogram.png)
- worst-reaction gap plot: [`results/ffn_baselines/comparison_figures/worst_reaction_gaps_ffn_all_vs_rf.png`](../../results/ffn_baselines/comparison_figures/worst_reaction_gaps_ffn_all_vs_rf.png)
- magnitude comparison plot: [`results/ffn_baselines/comparison_figures/relative_error_by_magnitude_comparison.png`](../../results/ffn_baselines/comparison_figures/relative_error_by_magnitude_comparison.png)

Dataset context used in the interpretation:

- parsed dataset summary: [`outputs/analysis/dataset_summary.csv`](../../outputs/analysis/dataset_summary.csv)
- advanced report: [`advanced_dataset_analysis_report.md`](./advanced_dataset_analysis_report.md)
- training report: [`training_experiment_report.md`](./training_experiment_report.md)

## Executive Answer

The random forest wins here because the problem is a small, highly structured, tabular regression task with only `609` cases arranged on a `21 x 29` grid, strong axis-aligned dependence on `log10(E/N)`, and many reaction-specific nonlinear regimes. In this regime, the forest captures local response-surface structure with much less broad smoothing than the FFNs. The FFNs do not merely fail on a few extreme cases; they carry noticeably larger error across a wide fraction of cases and reactions. The evidence points much more toward a tabular small-data / regime-sensitive mismatch than toward an isolated tuning failure.

## 1. The Data Regime Favors Tree Partitioning

The dataset has only `609` total cases, with `21` unique density groups and `29` local `E/N` positions per group, as recorded in [`outputs/analysis/dataset_summary.csv`](../../outputs/analysis/dataset_summary.csv). The earlier advanced analysis already showed that the dataset is highly structured rather than freely sampled, and that `E/N` is the dominant driver while composition changes across only `21` trajectories; see [`advanced_dataset_analysis_report.md`](./advanced_dataset_analysis_report.md) and its cited outputs.

This matters because the FFN is trying to learn a smooth global mapping from a small amount of tabular data, while the random forest can fit a set of local partitions that align naturally with discrete composition groups, sharp `E/N` thresholds, and reaction-specific changes in target behavior. That is an inference from the dataset structure and the observed error patterns below, not a direct theorem.

## 2. The Random Forest Is Better on Every Headline Metric That Matters

The locked-test comparison in [`results/ffn_baselines/comparison_analysis/overall_model_comparison.csv`](../../results/ffn_baselines/comparison_analysis/overall_model_comparison.csv) is decisive:

- random forest overall log-RMSE: `0.203350`
- FFN all-inputs overall log-RMSE: `0.246912`
- FFN PCA-feature overall log-RMSE: `0.263759`

The random forest is also much better on robust percent-style error summaries:

- median absolute relative error:
  - random forest: `0.000039`
  - FFN all-inputs: `0.032357`
  - FFN PCA-feature: `0.023671`
- within 10% relative error:
  - random forest: `0.972867`
  - FFN all-inputs: `0.750016`
  - FFN PCA-feature: `0.802166`
- median SMAPE:
  - random forest: `0.000028`
  - FFN all-inputs: `0.035322`
  - FFN PCA-feature: `0.025681`

These same numbers are also visible in the source experiment files:

- random forest overall metrics: [`results/full_training_pipeline/evaluation/test_overall_metrics.csv`](../../results/full_training_pipeline/evaluation/test_overall_metrics.csv)
- random forest relative-error summary: [`results/full_training_pipeline/evaluation/test_relative_error_overall_summary.csv`](../../results/full_training_pipeline/evaluation/test_relative_error_overall_summary.csv)
- random forest SMAPE summary: [`results/full_training_pipeline/evaluation/test_smape_overall_summary.csv`](../../results/full_training_pipeline/evaluation/test_smape_overall_summary.csv)
- FFN summary table: [`results/ffn_baselines/ffn_baseline_comparison_summary.csv`](../../results/ffn_baselines/ffn_baseline_comparison_summary.csv)

So the gap is not only in one metric or one output space. The forest is better in log space, better in relative calibration, and better in bounded percentage-like error.

## 3. FFN Error Is Broadly Distributed Across Cases, Not Just Confined to Outliers

The most important diagnostic is the per-case distribution, because it tells us whether the FFNs are failing only on a few pathological cases or are systematically less precise.

The per-case summary in [`results/ffn_baselines/comparison_analysis/per_case_log_rmse_distribution_summary.csv`](../../results/ffn_baselines/comparison_analysis/per_case_log_rmse_distribution_summary.csv) shows:

- random forest median case log-RMSE: `0.001029`
- FFN all-inputs median case log-RMSE: `0.127481`
- FFN PCA-feature median case log-RMSE: `0.090263`

The upper quantiles tell the same story:

- 75th percentile case log-RMSE:
  - random forest: `0.007628`
  - FFN all-inputs: `0.243085`
  - FFN PCA-feature: `0.142820`
- 90th percentile case log-RMSE:
  - random forest: `0.059185`
  - FFN all-inputs: `0.384660`
  - FFN PCA-feature: `0.323860`

These distributions are visualized in [`results/ffn_baselines/comparison_figures/per_case_log_rmse_histogram.png`](../../results/ffn_baselines/comparison_figures/per_case_log_rmse_histogram.png).

One nuance matters here: the random forest does have a slightly worse maximum case error than the all-inputs FFN (`1.330477` versus `0.814619`), while the PCA-feature FFN is worst overall at `1.412968`. That means the random forest is not uniformly better in the strict worst-case sense. But the medians and upper-middle quantiles are so much better that the forest is clearly more accurate on most of the test set. This pattern is consistent with a model that captures most local regimes very well but still misses a few hard corners, while the FFNs spread moderate error much more broadly.

The cases where the FFNs lag the random forest most are exported directly in:

- [`results/ffn_baselines/comparison_analysis/worst_20_cases_ffn_all_vs_rf.csv`](../../results/ffn_baselines/comparison_analysis/worst_20_cases_ffn_all_vs_rf.csv)
- [`results/ffn_baselines/comparison_analysis/worst_20_cases_ffn_pca_vs_rf.csv`](../../results/ffn_baselines/comparison_analysis/worst_20_cases_ffn_pca_vs_rf.csv)

Several of the largest FFN gaps occur at low local-case indices such as density group `11`, local cases `1` and `2`, and density group `2`, local cases `1` and `2`. That suggests the FFNs are especially weaker in early-`E/N` regimes where fewer reactions are active and the mapping may be more threshold-like.

## 4. FFNs Also Trail Across the Reaction Axis, Not Only Across Cases

The per-reaction summary in [`results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_distribution_summary.csv`](../../results/ffn_baselines/comparison_analysis/per_reaction_log_rmse_distribution_summary.csv) shows:

- random forest median reaction log-RMSE: `0.013282`
- FFN all-inputs median reaction log-RMSE: `0.091937`
- FFN PCA-feature median reaction log-RMSE: `0.093982`

This is a large gap. It says the FFN problem is not just that a few individual cases are wrong. Across the 204 targets themselves, the forest is much more accurate for a typical reaction channel.

The reaction-level gap is especially visible in the exported top-gap tables:

- [`results/ffn_baselines/comparison_analysis/worst_20_reactions_ffn_all_vs_rf.csv`](../../results/ffn_baselines/comparison_analysis/worst_20_reactions_ffn_all_vs_rf.csv)
- [`results/ffn_baselines/comparison_analysis/worst_20_reactions_ffn_pca_vs_rf.csv`](../../results/ffn_baselines/comparison_analysis/worst_20_reactions_ffn_pca_vs_rf.csv)

Representative examples:

- reaction `14`, `AR3S > AR^^`
  - random forest log-RMSE: `0.000043`
  - FFN all-inputs log-RMSE: `0.868219`
  - FFN PCA-feature log-RMSE: `0.454233`
- reaction `91`, `H2 > H*(ALPHA)`
  - random forest: `0.018408`
  - FFN all-inputs: `0.474835`
  - FFN PCA-feature: `0.318085`
- reaction `13`, `AR3S > AR^`
  - random forest: `0.016945`
  - FFN all-inputs: `0.423444`
  - FFN PCA-feature: `0.291665`

These are not tiny differences. They indicate that the forest is preserving some sharp or sparse channels much better than the FFNs. The visualization in [`results/ffn_baselines/comparison_figures/worst_reaction_gaps_ffn_all_vs_rf.png`](../../results/ffn_baselines/comparison_figures/worst_reaction_gaps_ffn_all_vs_rf.png) makes this easy to inspect.

## 5. The Relative-Error Gap Persists Across Almost Every Magnitude Range

The most convincing evidence that this is a real modeling gap, rather than just a metric artifact, is the magnitude-binned error comparison in [`results/ffn_baselines/comparison_analysis/relative_error_by_magnitude_comparison.csv`](../../results/ffn_baselines/comparison_analysis/relative_error_by_magnitude_comparison.csv), visualized in [`results/ffn_baselines/comparison_figures/relative_error_by_magnitude_comparison.png`](../../results/ffn_baselines/comparison_figures/relative_error_by_magnitude_comparison.png).

Examples:

- for true rates in `[1e-12, 1e-10)`:
  - random forest median absolute relative error: `6.291028e-05`
  - FFN all-inputs: `0.075896`
  - FFN PCA-feature: `0.049546`
- for `[1e-10, 1e-08)`:
  - random forest: `7.538296e-06`
  - FFN all-inputs: `0.038090`
  - FFN PCA-feature: `0.027573`
- for `[1e-08, 1e-06)`:
  - random forest: `2.256517e-13`
  - FFN all-inputs: `0.012782`
  - FFN PCA-feature: `0.008205`
- for `[1e-04, 1e-02)`:
  - random forest: `0.009065`
  - FFN all-inputs: `0.053140`
  - FFN PCA-feature: `0.049094`

So the forest is not only better on tiny near-zero channels where percentage metrics are fragile. It is also better for the mid-range and larger targets where relative error is more interpretable. That makes the FFN underperformance much more substantive.

## 6. PCA Features Help FFN Calibration Slightly, but They Do Not Solve the Core Problem

The two FFN baselines tell an important story by themselves, via [`results/ffn_baselines/ffn_baseline_comparison_summary.csv`](../../results/ffn_baselines/ffn_baseline_comparison_summary.csv):

- the all-inputs FFN is better on the main log-RMSE metric (`0.246912` vs `0.263759`)
- the PCA-feature FFN is slightly better on robust percent-style metrics such as median absolute relative error and median SMAPE

This suggests that feature compression can improve calibration and reduce some noise sensitivity, but it does not close the main accuracy gap to the random forest. In other words, the FFN weakness is not simply that the raw 49-input representation is too messy. Even when the FFN gets the same composition-PCA feature path that supported the best forest, it still remains materially worse.

That points away from a pure preprocessing problem and more toward a modeling mismatch for this data size and response geometry.

## 7. Most Likely Mechanism: FFN Smooths a Local, Regime-Structured Surface Too Aggressively

The combined evidence above suggests the following mechanism.

The dataset is a small, structured tabular grid with:

- only `21` unique composition trajectories
- a dominant ordered `E/N` axis
- reaction channels with different sparsity patterns and activation regimes
- heavy-tailed targets with many near-zero values

In that setting, the random forest can partition the feature space into local neighborhoods and fit response patterns that are piecewise and regime-specific. The FFN, by contrast, is forced to share a smooth parameterization across all cases and all regimes. With only `609` cases, the network likely does not have enough data to learn that global surface without smoothing over sharp local differences.

This is an inference from the error distributions, especially:

- low median case error for the forest but not for the FFNs
- low median reaction error for the forest but not for the FFNs
- large FFN degradation on specific channels that appear regime-sensitive
- persistent FFN underperformance across most magnitude bins

## 8. What This Means for Model Development

The current results do not imply that neural models are inappropriate in principle. They imply that a plain shared FFN is not yet the right neural baseline for this dataset.

If we want to keep exploring neural models, the next candidates should explicitly respect the dataset structure better than a plain FFN:

- separate towers for composition and `log10(E/N)`
- residual FFNs with monotonic or ordered conditioning on `E/N`
- mixture-of-experts or regime-aware models
- latent-output neural models with reaction-family structure

But for the current publishable benchmark, the evidence strongly supports keeping the random forest as the main baseline and reference model.

## Bottom Line

Based on the saved experiments, the FFNs are worse than the random forest mainly because this problem lives in the exact regime where forests are very strong: small-data, structured, tabular, regime-sensitive multi-output regression. The random forest does not merely edge out the FFNs. It is better across overall metrics, typical cases, typical reactions, and most target-magnitude ranges.
