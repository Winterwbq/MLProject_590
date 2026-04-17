# Scripts Guide for NERS590 V03

This folder contains the runnable command-line scripts for the current `dataset/NERS590_data_V03` workflow. The old `v01`/legacy global-kin scripts are no longer in the active script tree, and reusable code from the v02 phase now has generic `ners590` names.

Use the `run_ners590_v03_*` scripts for complete workflows. Use the lower-level `parse/analyze/plot/train_ners590_*` scripts when you want to rerun one stage manually.

## Folder Layout

- `utils/`
  - Shared helpers for repository paths, CSV/figure saving, numeric summaries, path normalization, subprocess logging, and holdout-power resolution.
- `data/`
  - Parsing plus RATE CONST and SUPER RATE dataset-analysis pipelines.
- `training/`
  - Separate-task and joint-task training runners.
- `plotting/`
  - Analysis figures, presentation figures, thresholded parity plots, and relative-error replots.
- `review/`
  - Combined experiment review tables and strategy-comparison assets.

## Canonical Run Order

1. Data analysis:

```bash
python scripts/data/run_ners590_v03_analysis.py
```

2. Separate-task training:

```bash
python scripts/training/run_ners590_v03_rate_const_training.py
python scripts/training/run_ners590_v03_super_rate_training.py
```

3. Joint-task training:

```bash
python scripts/training/run_ners590_v03_multitask_training.py
```

4. Combined review:

```bash
python scripts/review/build_ners590_v03_joint_review.py
```

5. Optional slide/post-processing assets:

```bash
python scripts/plotting/plot_v03_io_space_for_slides.py
python scripts/plotting/plot_v03_dependency_correlation_maps.py
python scripts/review/build_v03_thresholded_strategy_comparison.py
python scripts/plotting/plot_v03_thresholded_parity_no_epsilon.py
python scripts/plotting/replot_v03_relative_error_by_magnitude_large_text.py
```

## Script Groups

- `data/parse_ners590_dataset.py`
  - Parses multi-file NERS590 `.out` datasets into merged CSV tables.
- `data/analyze_ners590_rate_const.py`
  - Builds RATE CONST and shared input-analysis CSV tables.
- `data/analyze_ners590_super_rate.py`
  - Builds SUPER RATE-specific analysis CSV tables.
- `plotting/plot_ners590_rate_const_analysis.py`
  - Builds figures from RATE CONST/input analysis CSVs.
- `plotting/plot_ners590_super_rate_analysis.py`
  - Builds figures from SUPER RATE analysis CSVs.
- `training/train_ners590_rate_const.py`
  - Reusable separate-task RATE CONST training engine.
- `training/train_ners590_super_rate.py`
  - Reusable separate-task SUPER RATE training engine.
- `training/evaluate_ners590_v03_multitask_branches.py`
  - Re-evaluates selected joint-branch models after multitask training.
- `review/build_ners590_v03_joint_review_assets.py`
  - Aggregates separate-task and joint-task results into comparison CSVs and figures.

## Output Layout

- Data analysis outputs: `results/ners590_v03_analysis`.
- Separate RATE CONST training outputs: `results/ners590_v03_rate_const`.
- Separate SUPER RATE training outputs: `results/ners590_v03_super_rate`.
- Joint multitask training outputs: `results/ners590_v03_multitask`.
- Combined review outputs: `results/ners590_v03_joint_review`.
