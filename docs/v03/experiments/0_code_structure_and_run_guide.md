# v03 Code Structure and Run Guide

This guide explains how `scripts/` and `src/` work together for the current `dataset/NERS590_data_V03` workflow. It is the practical runbook for rerunning V03 parsing, analysis, separate-task training, joint multitask training, and result review.

## 1. High-Level Architecture

- `scripts/`
  - CLI entrypoints and orchestration.
  - Mostly responsible for “what to run” and “where to write outputs”.
- `src/global_kin_ml/`
  - Core library logic.
  - Responsible for parsing/loading, preprocessing, model training, evaluation, and plotting internals.

In short: **scripts call `src`**. The scripts are thin command-line wrappers; reusable parser, preprocessing, model, and evaluation logic lives in `src/global_kin_ml/`.

The current V03 data flow is:

```text
dataset/NERS590_data_V03/*.out
  -> scripts/data/parse_ners590_dataset.py
  -> src/global_kin_ml/data.py
  -> src/global_kin_ml/raw_parser.py
  -> results/.../data/parsed/*.csv or results/ners590_v03_analysis/parsed/*.csv
  -> analysis/training/evaluation scripts
```

## 2. `src/global_kin_ml` (Core Logic)

- `data.py`
  - Parse/load dataset tables and validate expected shapes.
- `raw_parser.py`
  - Low-level parser for the readable-but-raw `global_kin_boltz_*.out` files.
  - This parser used to live under `scripts/global_kin_dataset.py`; it now lives in `src` so analysis/training entrypoints do not depend on deleted script-side helpers.
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

- `data/run_ners590_v03_analysis.py`
  - Canonical v03 analysis workflow runner.
  - Runs parser + analysis + plotting steps.
- `data/parse_ners590_dataset.py`
  - Generic parser for multi-file NERS590 `.out` datasets.
  - Reads every `.out` file in `dataset/NERS590_data_V03`.
  - Writes merged parsed tables such as `training_inputs.csv`, `training_targets.csv`, `rate_constants_long.csv`, and `case_features.csv`.
- `data/analyze_ners590_rate_const.py`, `plotting/plot_ners590_rate_const_analysis.py`
  - RATE CONST and shared input analysis + figures.
- `data/analyze_ners590_super_rate.py`, `plotting/plot_ners590_super_rate_analysis.py`
  - SUPER RATE-focused analysis + figures.

### Training

- `training/run_ners590_v03_rate_const_training.py`
  - v03 wrapper for RATE CONST separate-task training.
- `training/run_ners590_v03_super_rate_training.py`
  - v03 wrapper for SUPER RATE separate-task training.
- `training/run_ners590_v03_multitask_training.py`
  - v03 joint multitask training.
- `training/train_ners590_rate_const.py`
  - Reusable separate-task RATE CONST training engine used by the v03 wrapper.
- `training/train_ners590_super_rate.py`
  - Reusable separate-task SUPER RATE training engine used by the v03 wrapper.
- `training/evaluate_ners590_v03_multitask_branches.py`
  - Re-evaluate selected joint branches without full retuning.

### Result aggregation and review

- `review/build_ners590_v03_joint_review.py`
  - Canonical runner to build combined v03 review assets.
- `review/build_ners590_v03_joint_review_assets.py`
  - Produces combined CSV summaries and strategy comparison figures.

### Slide/support post-processing (v03 additions)

- `plotting/plot_v03_io_space_for_slides.py`
  - Input PCA and output distribution figures for presentation.
- `plotting/plot_v03_dependency_correlation_maps.py`
  - Reaction-level dependence maps vs `log(E/N)` and `log(Power)`.
- `review/build_v03_thresholded_strategy_comparison.py`
  - Thresholded (`true > 1e-20`) strategy comparison with no-epsilon evaluation metrics.
- `plotting/plot_v03_thresholded_parity_no_epsilon.py`
  - Thresholded no-epsilon parity plots.
- `plotting/replot_v03_relative_error_by_magnitude_large_text.py`
  - Rebuild relative-error-by-magnitude figures with larger fonts.

## 4. Canonical Run Order (v03)

Run commands from the project root:

```bash
cd "/Users/bingqingwang/Desktop/UMich/590_Machine learning/project"
```

1. Data parsing, analysis, and plots:

```bash
python scripts/data/run_ners590_v03_analysis.py
```

2. Separate-task training:

```bash
python scripts/training/run_ners590_v03_rate_const_training.py
python scripts/training/run_ners590_v03_super_rate_training.py
```

3. Joint training:

```bash
python scripts/training/run_ners590_v03_multitask_training.py
```

4. Combined review:

```bash
python scripts/review/build_ners590_v03_joint_review.py
```

5. Optional presentation/post-processing:

```bash
python scripts/plotting/plot_v03_io_space_for_slides.py
python scripts/plotting/plot_v03_dependency_correlation_maps.py
python scripts/review/build_v03_thresholded_strategy_comparison.py
python scripts/plotting/plot_v03_thresholded_parity_no_epsilon.py
python scripts/plotting/replot_v03_relative_error_by_magnitude_large_text.py
```

## 5. Important Output Locations

### Data analysis outputs

- Parsed raw tables: `results/ners590_v03_analysis/parsed/`
- RATE CONST/input analysis CSVs: `results/ners590_v03_analysis/analysis/`
- RATE CONST/input figures: `results/ners590_v03_analysis/figures/`
- SUPER RATE analysis CSVs: `results/ners590_v03_analysis/super_rate_analysis/`
- SUPER RATE figures: `results/ners590_v03_analysis/super_rate_figures/`

### Separate-task training outputs

- RATE CONST random split: `results/ners590_v03_rate_const/training_random_case/`
- RATE CONST power holdout: `results/ners590_v03_rate_const/training_power_holdout_10mJ/`
- SUPER RATE random split: `results/ners590_v03_super_rate/training_random_case/`
- SUPER RATE power holdout: `results/ners590_v03_super_rate/training_power_holdout_10mJ/`

Each separate-task training folder contains:

- `data/parsed/`: parsed CSV tables regenerated from raw `.out` files.
- `data_snapshots/`: split assignments, transformed feature matrices, log-transformed target matrices, epsilon tables, and target-cleaning metadata.
- `tuning/`: fold-wise trials, leaderboards, selected model config, and split metadata.
- `pca/`: target-PCA oracle reconstruction and explained-variance outputs.
- `evaluation/`: test metrics, predictions, relative-error tables, sMAPE tables, and worst-case summaries.
- `figures/`: model leaderboard, parity plots, residual plots, relative-error plots, and PCA plots.

### Joint multitask outputs

- Joint random split: `results/ners590_v03_multitask/training_random_case/`
- Joint power holdout: `results/ners590_v03_multitask/training_power_holdout_10mJ/`
- Combined comparison/review: `results/ners590_v03_joint_review/`

## 6. Training Data Snapshots

The pipeline does not usually write one file literally named `training_set.csv`. Instead, it saves all-case snapshots plus split labels.

For example, RATE CONST random-case snapshots are here:

```text
results/ners590_v03_rate_const/training_random_case/data_snapshots/
```

Important files:

- `split_assignments.csv`
  - Contains `locked_split`.
  - Rows with `locked_split == "trainval"` are the final training/validation pool.
  - Rows with `locked_split == "test"` are the locked test set.
  - Columns `fold_1_role` through `fold_5_role` mark train/validation roles inside cross-validation.
- `composition_pca_plus_log_en_log_power_all_cases.csv`
  - PCA-compressed gas composition plus explicit `log10(E/N)` and `log10(power)` for all cases.
- `full_nonconstant_plus_log_en_log_power_all_cases.csv`
  - Nonconstant raw composition features plus explicit `log10(E/N)` and `log10(power)` for all cases.
- `log_transformed_rate_const_all_cases.csv` or `log_transformed_super_rate_all_cases.csv`
  - Log-transformed targets for all cases.

To reconstruct a fold-specific training matrix, join/filter by row order or `global_case_id`, then use the relevant fold role.

## 7. Quick Sanity Checks

Use these checks after moving files, cloning the repo, or changing scripts.

Compile all Python files:

```bash
python -m py_compile $(find src scripts -name "*.py" -print)
```

Check CLI imports without running heavy experiments:

```bash
python scripts/data/parse_ners590_dataset.py --help
python scripts/data/run_ners590_v03_analysis.py --help
python scripts/training/run_ners590_v03_rate_const_training.py --help
python scripts/training/run_ners590_v03_super_rate_training.py --help
python scripts/training/run_ners590_v03_multitask_training.py --help
python scripts/review/build_ners590_v03_joint_review.py --help
```

Run the full data-analysis workflow into a temporary folder without overwriting project results:

```bash
python scripts/data/run_ners590_v03_analysis.py --results-root /tmp/ners590_v03_analysis_check
rm -rf /tmp/ners590_v03_analysis_check
```

In the latest smoke test, this parsed 13 raw power files and 83,317 total cases, then completed RATE CONST and SUPER RATE analysis/plotting.

## 8. Common Errors

### `ModuleNotFoundError: No module named 'global_kin_ml'`

This means Python cannot see the `src/` package. Prefer running the provided scripts from the project root. If you are importing from a notebook or custom script, add:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/bingqingwang/Desktop/UMich/590_Machine learning/project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

or run shell commands with:

```bash
PYTHONPATH="$PWD/src" python your_script.py
```

### Missing `scripts/global_kin_dataset.py`

Older notes may mention `scripts/global_kin_dataset.py`. That file is no longer part of the active script tree. The parser is now packaged as:

```text
src/global_kin_ml/raw_parser.py
```

If you see a missing `scripts/global_kin_dataset.py` error, make sure `src/global_kin_ml/data.py` imports `raw_parser` from `src/global_kin_ml`.

## 9. What to Share with a Collaborator

Minimum code to reproduce v03:

- `scripts/` (all v03-related runners listed above)
- `src/global_kin_ml/`
- dataset access (`dataset/NERS590_data_V03/`) or instructions to obtain it

Recommended to exclude from sharing:

- `results/` large artifacts (unless collaborator needs exact generated outputs)
- `__pycache__/`, `.DS_Store`, temporary office lock files

If the collaborator only needs to rerun results and not inspect all historical notes, share this guide plus `scripts/README.md` as the entrypoint documentation.
