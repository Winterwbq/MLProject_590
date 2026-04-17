# NERS590 v02 Experiment Plan

This document records the plan for migrating the project from the abandoned single-file dataset to the new multi-file dataset under [`NERS590_data_v02`](../../NERS590_data_v02).

## Dataset Scope

The new dataset consists of five `.out` files:

- [`global_kin_boltz_1mJ.out`](../../NERS590_data_v02/global_kin_boltz_1mJ.out)
- [`global_kin_boltz_2mJ.out`](../../NERS590_data_v02/global_kin_boltz_2mJ.out)
- [`global_kin_boltz_3mJ.out`](../../NERS590_data_v02/global_kin_boltz_3mJ.out)
- [`global_kin_boltz_4mJ.out`](../../NERS590_data_v02/global_kin_boltz_4mJ.out)
- [`global_kin_boltz_5mJ.out`](../../NERS590_data_v02/global_kin_boltz_5mJ.out)

Initial inspection shows:

- each file contains `2,204,696` lines
- each file therefore contains exactly `6409` case blocks when divided by the known `344`-line block size
- `6409 = 221 x 29`, so the most likely structure is `221` density groups per power level, each with the same `29`-point `E/N` sweep

The old dataset and its reports are now legacy references only. All new analysis and modeling work should treat `NERS590_data_v02` as the active dataset.

## Objectives

The new workflow should:

1. parse all five power-specific files into one merged, power-aware dataset
2. redo data cleaning and dataset analysis from scratch on the new dataset
3. treat power as an explicit scientific input, not just a filename label
4. train and evaluate baseline ML models on the new merged dataset
5. generate CSV outputs, figures, and Markdown reports in the same spirit as the earlier pipeline

## Parsing and Data Layer Plan

The parser should be extended from a single-file workflow to a directory-based workflow.

Planned parser behavior:

- iterate through all `.out` files in [`NERS590_data_v02`](../../NERS590_data_v02)
- parse each file using the existing case-block logic
- extract `power_mj` from the filename
- assign dataset-global IDs across the merged dataset
- preserve file-local structure with explicit metadata columns

New per-case metadata to carry through the parsed tables:

- `source_file`
- `source_file_id`
- `power_label`
- `power_mj`
- `power_group_id`
- `file_case_id`
- `density_group_in_file_id`

The parsed outputs should still include:

- parser summary
- case features
- species map
- reaction map
- long input table
- long power-deposition table
- long rate-constant table
- training inputs
- training targets

But they should now be written for the merged multi-file dataset rather than a single raw file.

## Cleaning and Feature Plan

The new modeling inputs should be:

- the `49` input gas mole fractions
- `E/N`
- power

Power should be represented explicitly in the modeling pipeline. The planned feature encodings are:

- `log10(E/N)`
- `log10(power_mj)`
- full non-constant composition features plus the two log-scale scalar controls
- PCA-compressed composition features plus the two log-scale scalar controls
- all-input diagnostic feature sets when useful for neural baselines

Cleaning tasks to re-run on the new dataset:

- detect constant species across the merged dataset
- detect nearly constant or redundant species
- check whether species remain constant within each `29`-point sweep
- check whether the same `E/N` grid is reused across powers
- verify target sparsity and heavy-tail behavior on the new data

## Analysis Plan

The new analysis pass should reproduce the old style of dataset characterization and add power-aware diagnostics.

Core analysis outputs:

- overall dataset summary
- counts per power level
- counts per density group and per local case
- `E/N` distribution overall and per power
- input mole-fraction distributions overall and per power
- rate-constant distributions overall, per power, and per local case
- power-aware summaries of case features

New power-specific diagnostics:

- whether compositions repeat across power levels
- whether target profiles shift systematically with power at fixed composition and fixed `E/N`
- whether some reactions are especially power-sensitive
- whether power changes target sparsity or target magnitude ranges

Outputs should be saved primarily as CSV files, with a smaller number of high-value plots.

## Modeling Plan

The first round of v02 training should stay close to the proven earlier workflow while adapting to the larger merged dataset.

Planned model families:

- ridge
- random forest
- extra trees
- direct FFN
- latent PCA target variants for the above where practical

The first-pass feature sets should be:

- `full_nonconstant_plus_log_en_log_power`
- `composition_pca_plus_log_en_log_power`
- `all_inputs_plus_log_en_log_power` for selected neural baselines

The first-pass target treatment should remain:

- predict the `204` rate constants
- estimate per-reaction `epsilon` on the training split
- train in `log10(rate_const + epsilon)`
- always evaluate the reconstructed final `204` outputs in both log space and original space

## Split and Evaluation Plan

The new dataset introduces a second major axis, power, so evaluation should include more than one split style.

Planned split families:

- random-case split for the main benchmark
- power-held-out split to test extrapolation or transfer across power levels
- optional density-group-held-out split if useful after the structure is confirmed

Evaluation outputs should continue to include:

- overall log-space metrics
- overall original-space metrics
- per-reaction metrics
- per-case metrics
- raw relative-error summaries for positive targets only
- bounded `SMAPE` summaries
- magnitude-binned error summaries
- PCA oracle reconstruction diagnostics when latent models are tested

## Code Additions

Planned code changes:

- extend [`src/global_kin_ml/data.py`](../../../src/global_kin_ml/data.py) for directory-based parsing and generalized dataset validation
- extend [`src/global_kin_ml/preprocessing.py`](../../../src/global_kin_ml/preprocessing.py) with power-aware feature transformers and more flexible metadata handling
- extend the training pipeline so it can run on the merged dataset and optionally use different split strategies
- add new v02-specific CLI scripts for parsing, analysis, plotting, and training
- add new v02 Markdown reports under [`docs`](./)

## Result Layout

Planned output roots:

- parsed tables: `results/ners590_v02/parsed`
- analysis CSVs: `results/ners590_v02/analysis`
- analysis figures: `results/ners590_v02/figures`
- training experiments: `results/ners590_v02/training_*`

## Execution Order

Implementation and experimentation should proceed in this order:

1. build and validate the merged multi-file parser
2. generate parsed v02 tables
3. run full power-aware dataset analysis and plotting
4. decide final feature sets based on the new analysis
5. run baseline training experiments
6. write v02 dataset-analysis and training reports based on the actual outputs
