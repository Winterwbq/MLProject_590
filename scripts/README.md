# Scripts Organization (V03-focused)

This folder has been cleaned to keep only scripts relevant to the current `NERS590_data_V03` direction.

## Shared Utility

- `_runner_utils.py`
  - Shared runner helpers:
  - path/bootstrap (`REPO_ROOT`, `ensure_src_on_path`)
  - holdout-power helpers (`infer_power_label`, `resolve_holdout_power_labels`, `build_holdout_suffix`)
  - subprocess logging helpers (`format_seconds`, `run_subprocess_step`)

## V03 Data Parsing + Analysis

- `parse_ners590_v02.py`
  - Canonical parser for NERS590-format `.out` files (used by v03 despite legacy filename).
- `analyze_ners590_v02.py`
  - Core RATE CONST-oriented analysis tables for NERS590 datasets.
- `plot_ners590_v02_analysis.py`
  - Plots for the core RATE CONST-oriented analysis.
- `analyze_ners590_v02_super_rate.py`
  - SUPER RATE analysis tables.
- `plot_ners590_v02_super_rate_analysis.py`
  - SUPER RATE analysis plots.
- `run_ners590_v03_analysis.py`
  - Orchestrates v03 parsing + analysis + plotting pipeline.

## V03 Training

- `run_ners590_v02_training.py`
  - Separate-task RATE CONST training engine (v03-compatible input format).
- `run_ners590_v02_super_rate_training.py`
  - Separate-task SUPER RATE training engine (v03-compatible input format).
- `run_ners590_v03_rate_const_training.py`
  - v03 wrapper for RATE CONST training.
- `run_ners590_v03_super_rate_training.py`
  - v03 wrapper for SUPER RATE training.
- `run_ners590_v03_multitask_training.py`
  - Joint multitask training for RATE CONST + SUPER RATE.
- `evaluate_ners590_v03_multitask_branches.py`
  - Re-evaluates selected joint branch models.

## V03 Review and Post-processing

- `build_ners590_v03_joint_review.py`
  - Runner for combined v03 review asset generation.
- `build_ners590_v03_joint_review_assets.py`
  - Builds combined strategy comparison CSVs/figures.
- `plot_v03_io_space_for_slides.py`
  - Slide-oriented input/output distribution figures.
- `plot_v03_dependency_correlation_maps.py`
  - Correlation/dependence maps vs `log(E/N)` and `log(Power)`.
- `build_v03_thresholded_strategy_comparison.py`
  - Thresholded no-epsilon strategy comparison outputs.
- `plot_v03_thresholded_parity_no_epsilon.py`
  - Thresholded no-epsilon parity plots.
- `replot_v03_relative_error_by_magnitude_large_text.py`
  - Rebuilds relative-error-by-magnitude figures with larger text.
