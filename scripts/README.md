# Scripts Organization

This folder is organized by workflow stage rather than by dataset file format.

## Shared Utilities

- `_runner_utils.py`
  - Shared CLI runner helpers used by experiment entry scripts:
  - repository bootstrap (`REPO_ROOT`, `ensure_src_on_path`)
  - holdout-power helpers (`infer_power_label`, `resolve_holdout_power_labels`, `build_holdout_suffix`)
  - common logging/runner helpers (`format_seconds`, `run_subprocess_step`)

## Core Experiment Entry Points

- `run_global_kin_training.py`
  - Original end-to-end training pipeline for `global_kin_boltz.out`.
- `run_ners590_v02_training.py`
  - RATE CONST separate-task training for NERS590 datasets (v02 and v03-compatible inputs).
- `run_ners590_v02_super_rate_training.py`
  - SUPER RATE separate-task training for NERS590 datasets (v02 and v03-compatible inputs).
- `run_ners590_v03_multitask_training.py`
  - Joint multitask training (shared model for RATE CONST + SUPER RATE).
- `evaluate_ners590_v03_multitask_branches.py`
  - Final refit/test pass for the best multitask single-head and two-head branches.

## v03 Convenience Wrappers

- `run_ners590_v03_analysis.py`
  - Canonical parse + analysis + plotting workflow for v03.
- `run_ners590_v03_rate_const_training.py`
  - Wrapper around `run_ners590_v02_training.py` with v03 defaults.
- `run_ners590_v03_super_rate_training.py`
  - Wrapper around `run_ners590_v02_super_rate_training.py` with v03 defaults.
- `build_ners590_v03_joint_review.py`
  - Wrapper for generating combined v03 review assets.
- `build_ners590_v03_joint_review_assets.py`
  - Builds joint comparison tables/figures from RATE CONST, SUPER RATE, and multitask outputs.

## Parsing, Analysis, and Plotting

- `parse_global_kin_boltz.py`, `parse_ners590_v02.py`
- `analyze_global_kin_dataset.py`, `analyze_ners590_v02.py`, `analyze_ners590_v02_super_rate.py`
- `plot_global_kin_analysis.py`, `plot_ners590_v02_analysis.py`, `plot_ners590_v02_super_rate_analysis.py`
- `advanced_global_kin_analysis.py`

## Reporting and Maintenance

- `export_global_kin_training_report.py`
- `build_ners590_joint_experiment_report_assets.py`
- `run_ffn_baseline_experiments.py`
- `analyze_ffn_vs_random_forest.py`
- `backfill_relative_error_metrics.py`
