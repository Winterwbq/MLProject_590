# Project Analysis and Experiment Memory

This file is a compact project memory note for the dataset-analysis and ML-experiment work completed so far. It is meant to make later continuation easier without re-reading every artifact from scratch.

## Dataset and Parsing

- Raw source: [`global_kin_boltz.out`](../../global_kin_boltz.out)
- Parsed outputs: [`outputs/parsed`](../../outputs/parsed)
- Core parser code: [`scripts/global_kin_dataset.py`](../../scripts/global_kin_dataset.py)
- Parser CLI: [`scripts/parse_global_kin_boltz.py`](../../scripts/parse_global_kin_boltz.py)

Key structure facts:

- `609` total cases
- `21` density groups
- `29` local cases per density group
- `49` input mole-fraction species
- `204` target rate constants

Primary notes:

- [`dataset_overview.md`](./dataset_overview.md)
- [`dataset_cleaning_notes.md`](./dataset_cleaning_notes.md)

## First-Pass and Advanced Analysis

First-pass structured summaries:

- [`dataset_analysis_findings.md`](./dataset_analysis_findings.md)
- [`outputs/analysis`](../../outputs/analysis)
- [`outputs/figures`](../../outputs/figures)

Advanced diagnostics:

- [`advanced_dataset_analysis_report.md`](./advanced_dataset_analysis_report.md)
- [`outputs/advanced_analysis`](../../outputs/advanced_analysis)
- [`outputs/advanced_figures`](../../outputs/advanced_figures)

Important analysis conclusions retained for modeling:

- `log10(E/N)` is the dominant predictor.
- Many input species are redundant or constant.
- The 204-target space is strongly low-rank.
- Random-case prediction is much easier than holdout by unseen `E/N` position.

## Production Training Pipeline

Code:

- [`src/global_kin_ml`](../../src/global_kin_ml)
- full training runner: [`run_global_kin_training.py`](../../scripts/run_global_kin_training.py)
- report export: [`export_global_kin_training_report.py`](../../scripts/export_global_kin_training_report.py)
- error-metric backfill: [`backfill_relative_error_metrics.py`](../../scripts/backfill_relative_error_metrics.py)

Strategy and protocol:

- [`model_strategy_and_reconstruction_protocol.md`](./model_strategy_and_reconstruction_protocol.md)

Main full experiment:

- results: [`results/full_training_pipeline`](../../results/full_training_pipeline)
- report: [`training_experiment_report.md`](./training_experiment_report.md)

Main retained result:

- best published model so far is a direct random forest on `composition_pca_plus_log_en`
- selected config recorded in [`selected_model.csv`](../../results/full_training_pipeline/tuning/selected_model.csv)

## Robust Error Analysis

Because raw relative error explodes for near-zero targets, the project now keeps:

- standard log-space and original-space metrics
- raw relative-error summaries for positive-ground-truth targets only
- magnitude-binned relative-error summaries
- bounded `SMAPE` summaries and plots

Main files:

- [`test_relative_error_overall_summary.csv`](../../results/full_training_pipeline/evaluation/test_relative_error_overall_summary.csv)
- [`test_relative_error_by_magnitude_bin.csv`](../../results/full_training_pipeline/evaluation/test_relative_error_by_magnitude_bin.csv)
- [`test_smape_overall_summary.csv`](../../results/full_training_pipeline/evaluation/test_smape_overall_summary.csv)
- [`test_smape_by_magnitude_bin.csv`](../../results/full_training_pipeline/evaluation/test_smape_by_magnitude_bin.csv)

## Dedicated FFN Baselines

Dedicated FFN baseline experiments are run separately from the main mixed-model benchmark.

Artifacts:

- runner: [`run_ffn_baseline_experiments.py`](../../scripts/run_ffn_baseline_experiments.py)
- results: [`results/ffn_baselines`](../../results/ffn_baselines)
- report: [`ffn_baseline_experiments.md`](./ffn_baseline_experiments.md)
- RF-vs-FFN deep dive: [`ffn_vs_random_forest_analysis.md`](./ffn_vs_random_forest_analysis.md)
- RF-vs-FFN comparison tables: [`results/ffn_baselines/comparison_analysis`](../../results/ffn_baselines/comparison_analysis)
- RF-vs-FFN comparison figures: [`results/ffn_baselines/comparison_figures`](../../results/ffn_baselines/comparison_figures)

Scenarios:

- `direct_all_inputs_end_to_end`: direct FFN on all `49` inputs plus `log10(E/N)`
- `rf_replacement_composition_pca`: direct FFN on the same PCA-compressed feature path used by the winning random-forest pipeline

Retained outcome:

- best FFN baseline by test log-RMSE is `direct_all_inputs_end_to_end`
- its result summary is stored in [`ffn_baseline_comparison_summary.csv`](../../results/ffn_baselines/ffn_baseline_comparison_summary.csv)
- both FFN baselines are clearly weaker than the main random-forest pipeline on the locked test split
- the deep-dive comparison attributes that gap mainly to the small, structured tabular regime and the FFN tendency to spread moderate error across many cases and reactions rather than only a few outliers
