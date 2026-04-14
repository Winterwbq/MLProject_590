# v03 Code Structure and Run Guide

This guide explains how `scripts/` and `src/` work together for the current `NERS590_data_V03` workflow.

## 1. High-Level Architecture

- `scripts/`
  - CLI entrypoints and orchestration.
  - Mostly responsible for “what to run” and “where to write outputs”.
- `src/global_kin_ml/`
  - Core library logic.
  - Responsible for parsing/loading, preprocessing, model training, evaluation, and plotting internals.

In short: **scripts call src**.

## 2. `src/global_kin_ml` (Core Logic)

- `data.py`
  - Parse/load dataset tables and validate expected shapes.
- `preprocessing.py`
  - Feature transforms, split generation, target transform/inverse transform, constant-column handling.
- `models.py`
  - Model config objects and model builders (Ridge / tree / MLP families).
- `evaluation.py`
  - Metric computation, prediction tables, relative/sMAPE outputs, plotting helpers.
- `pipeline.py`
  - Separate-task end-to-end pipeline (used for RATE CONST and SUPER RATE training workflows).
- `multitask_pipeline.py`
  - Joint training pipeline (shared model for RATE CONST + SUPER RATE).
- `experiment_configs.py`
  - Search spaces and canonical experiment config definitions.
- `reporting.py`
  - Markdown/report export helpers.
- `ffn_baselines.py`
  - FFN baseline experiment implementations.

## 3. Main `scripts/` for v03

### Data parsing and analysis

- `run_ners590_v03_analysis.py`
  - Canonical v03 analysis workflow runner.
  - Runs parser + analysis + plotting steps.
- `parse_ners590_v02.py`
  - Parser used by v03 (name kept for historical reasons).
- `analyze_ners590_v02.py`, `plot_ners590_v02_analysis.py`
  - Core dataset analysis + analysis figures (also used for v03).
- `analyze_ners590_v02_super_rate.py`, `plot_ners590_v02_super_rate_analysis.py`
  - SUPER RATE-focused analysis + figures.

### Training

- `run_ners590_v03_rate_const_training.py`
  - v03 wrapper for RATE CONST separate-task training.
- `run_ners590_v03_super_rate_training.py`
  - v03 wrapper for SUPER RATE separate-task training.
- `run_ners590_v03_multitask_training.py`
  - v03 joint multitask training.
- `evaluate_ners590_v03_multitask_branches.py`
  - Re-evaluate selected joint branches without full retuning.

### Result aggregation and review

- `build_ners590_v03_joint_review.py`
  - Canonical runner to build combined v03 review assets.
- `build_ners590_v03_joint_review_assets.py`
  - Produces combined CSV summaries and strategy comparison figures.

### Slide/support post-processing (v03 additions)

- `plot_v03_io_space_for_slides.py`
  - Input PCA and output distribution figures for presentation.
- `plot_v03_dependency_correlation_maps.py`
  - Reaction-level dependence maps vs `log(E/N)` and `log(Power)`.
- `build_v03_thresholded_strategy_comparison.py`
  - Thresholded (`true > 1e-20`) strategy comparison with no-epsilon evaluation metrics.
- `plot_v03_thresholded_parity_no_epsilon.py`
  - Thresholded no-epsilon parity plots.
- `replot_v03_relative_error_by_magnitude_large_text.py`
  - Rebuild relative-error-by-magnitude figures with larger fonts.

## 4. Canonical Run Order (v03)

1. Data analysis:

```bash
python scripts/run_ners590_v03_analysis.py
```

2. Separate-task training:

```bash
python scripts/run_ners590_v03_rate_const_training.py
python scripts/run_ners590_v03_super_rate_training.py
```

3. Joint training:

```bash
python scripts/run_ners590_v03_multitask_training.py
```

4. Combined review:

```bash
python scripts/build_ners590_v03_joint_review.py
```

5. Optional presentation/post-processing:

```bash
python scripts/plot_v03_io_space_for_slides.py
python scripts/plot_v03_dependency_correlation_maps.py
python scripts/build_v03_thresholded_strategy_comparison.py
python scripts/plot_v03_thresholded_parity_no_epsilon.py
python scripts/replot_v03_relative_error_by_magnitude_large_text.py
```

## 5. What to Share with a Collaborator

Minimum code to reproduce v03:

- `scripts/` (all v03-related runners listed above)
- `src/global_kin_ml/`
- dataset access (`NERS590_data_V03/`) or instructions to obtain it

Recommended to exclude from sharing:

- `results/` large artifacts (unless collaborator needs exact generated outputs)
- `__pycache__/`, `.DS_Store`, temporary office lock files
