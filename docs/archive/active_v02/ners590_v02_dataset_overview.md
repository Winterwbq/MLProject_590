# NERS590 v02 Dataset Overview

This note summarizes the active dataset under [`NERS590_data_v02`](../../NERS590_data_v02), which replaces the older abandoned single-file dataset for current project work.

## Raw Files

The dataset consists of five power-specific solver outputs:

- [`global_kin_boltz_1mJ.out`](../../NERS590_data_v02/global_kin_boltz_1mJ.out)
- [`global_kin_boltz_2mJ.out`](../../NERS590_data_v02/global_kin_boltz_2mJ.out)
- [`global_kin_boltz_3mJ.out`](../../NERS590_data_v02/global_kin_boltz_3mJ.out)
- [`global_kin_boltz_4mJ.out`](../../NERS590_data_v02/global_kin_boltz_4mJ.out)
- [`global_kin_boltz_5mJ.out`](../../NERS590_data_v02/global_kin_boltz_5mJ.out)

Each file was confirmed to contain `2,204,696` lines, which matches exactly `6409 x 344` case-block lines.

## Parsed Structure

The merged parser outputs live in [`results/ners590_v02/parsed`](../../results/ners590_v02/parsed).

Primary parsed tables:

- [`parser_summary.csv`](../../results/ners590_v02/parsed/parser_summary.csv)
- [`parser_file_summary.csv`](../../results/ners590_v02/parsed/parser_file_summary.csv)
- [`case_features.csv`](../../results/ners590_v02/parsed/case_features.csv)
- [`species_map.csv`](../../results/ners590_v02/parsed/species_map.csv)
- [`reaction_map.csv`](../../results/ners590_v02/parsed/reaction_map.csv)
- [`training_inputs.csv`](../../results/ners590_v02/parsed/training_inputs.csv)
- [`training_targets.csv`](../../results/ners590_v02/parsed/training_targets.csv)

The merged dataset dimensions are:

- `32,045` total cases
- `5` power levels
- `1,105` total density groups
- `221` density groups per power level
- `29` local `E/N` points per density group
- `49` input species
- `204` target rate constants

These counts are recorded in:

- [`parser_summary.csv`](../../results/ners590_v02/parsed/parser_summary.csv)
- [`parser_file_summary.csv`](../../results/ners590_v02/parsed/parser_file_summary.csv)
- [`dataset_summary.csv`](../../results/ners590_v02/analysis/dataset_summary.csv)

## Case Metadata Added in v02

To support power-aware analysis and modeling, the merged parsed tables now carry these extra metadata fields:

- `source_file`
- `source_file_id`
- `power_label`
- `power_mj`
- `power_group_id`
- `file_case_id`
- `density_group_in_file_id`

This allows the pipeline to work with both dataset-global identifiers and file-local identifiers.

## Modeling Inputs and Targets

The current v02 modeling inputs are drawn from:

- the `49` `INPUT GAS MOLE FRACTIONS`
- `E/N` from `ELECTRIC FIELD/NUMBER DENSITY`
- power from the filename-derived `power_mj`

The target remains:

- the full `204`-dimensional `RATE CONST` vector

## Current Workflow

The v02 workflow is now:

1. parse the multi-file dataset into merged CSV tables
2. run power-aware dataset analysis and plotting
3. train baseline models on the merged dataset
4. evaluate both random-case generalization and power-held-out generalization

Main implementation entrypoints:

- parser: [`parse_ners590_dataset.py`](../../scripts/data/parse_ners590_dataset.py)
- analysis: [`analyze_ners590_rate_const.py`](../../scripts/data/analyze_ners590_rate_const.py)
- plotting: [`plot_ners590_rate_const_analysis.py`](../../scripts/plotting/plot_ners590_rate_const_analysis.py)
- training: [`train_ners590_rate_const.py`](../../scripts/training/train_ners590_rate_const.py)

Planning note:

- [`ners590_v02_experiment_plan.md`](./ners590_v02_experiment_plan.md)
