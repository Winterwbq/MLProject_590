# NERS590 v02 SUPER RATE Experiment Plan

## Goal

Build a dedicated training path for predicting `SUPER RATE (CC/S)` on the active `NERS590_data_v02` dataset, using the same merged multi-power dataset as the current `RATE CONST` work while adapting the target cleaning and evaluation to the much sparser `SUPER RATE` structure.

## Why This Needs a Separate Plan

The exploratory analysis shows that `SUPER RATE` is not just a relabeled copy of `RATE CONST`.

- `SUPER RATE` is much sparser, with about `60.65%` exact zeros overall.
- `110` of the `204` reactions are always zero and can be treated as trivial outputs.
- the strongest dataset-level variation follows the 29-point local-case / `E/N` sweep
- power changes the target only modestly at the aggregate level
- some reactions track `RATE CONST` very closely, but others are weakly or negatively correlated with it

So the first production-grade modeling step should be a dedicated `SUPER RATE` pipeline, not a naive reuse of the `RATE CONST` target path.

## Phase 1 Scope

This implementation phase will cover:

1. target-aware pipeline support so we can train on either `RATE CONST` or `SUPER RATE`
2. cleaned `SUPER RATE` targets with always-zero channels removed during training
3. full-output reconstruction back to the original 204 reaction outputs by zero-filling the dropped channels
4. baseline experiments for `SUPER RATE`
5. experiment reports and result exports

This phase will not yet implement a shared multi-task model with `RATE CONST`; that will remain a follow-up experiment after we establish strong single-task `SUPER RATE` baselines.

## Data and Cleaning Strategy

### Inputs

Use the same power-aware input set as the current v2 pipeline:

- 49 `INPUT GAS MOLE FRACTIONS`
- `E/N`
- `power_mj`

Candidate feature sets:

- `all_inputs_plus_log_en_log_power`
- `full_nonconstant_plus_log_en_log_power`
- `composition_pca_plus_log_en_log_power`

### Target

Use `SUPER RATE (CC/S)` as the prediction target.

Cleaning rules:

- derive the 204-wide `SUPER RATE` table from the parsed long reaction table
- identify always-zero `SUPER RATE` columns
- remove those always-zero channels from the train target matrix
- keep a dropped-column map so final predictions can be reconstructed to the full 204 outputs by inserting zeros in those channels

### Transform

For retained `SUPER RATE` channels:

- use per-reaction epsilon: `epsilon_j = max(min_positive_train / 10, 1e-30)`
- regress `log10(super_rate + epsilon_j)`

This keeps the target nonnegative after inverse transform and handles the long-tailed positive range more safely than raw-space regression.

## Model Baselines

### Baseline 1: Direct Tabular Regression

Train direct multi-output regressors on the cleaned log-transformed `SUPER RATE` targets.

Initial model families:

- Ridge
- ExtraTrees
- MLP
- latent PCA variants for a compact-output check

### Baseline 2: Sparse-Aware Two-Stage Tree Model

Implement a sparse-aware baseline for `SUPER RATE`:

- Stage A: predict whether each retained reaction is active or inactive
- Stage B: regress the log-transformed positive magnitude
- Final prediction: gate the regressed magnitude with the predicted activity mask

This is motivated directly by the heavy zero structure in `SUPER RATE`.

## Evaluation Protocol

### Split Sets

Run the same two experiment families used in the current v2 training workflow:

1. random-case split
   Purpose: interpolation within the merged dataset.

2. `5mJ` power holdout
   Purpose: test generalization to an unseen power level while still using a unified multi-power model.

### Metrics

Save both:

- retained-target metrics: performance on the nontrivial kept `SUPER RATE` channels
- full reconstructed metrics: performance after rebuilding the full 204 outputs by zero-filling the dropped channels

Required summaries:

- overall log-RMSE, log-MAE, log-R2
- overall original-space RMSE, MAE
- factor-of-2 / 5 / 10 accuracy on positive entries
- per-reaction and per-case metrics
- relative-error summaries
- SMAPE summaries

### Outputs

Save:

- cleaned-target metadata
- kept vs dropped channel lists
- selected model summaries
- full evaluation CSVs
- figures
- a cited experiment report

## Expected Result

The first strong `SUPER RATE` benchmark should likely be:

- one unified power-aware model trained on the merged multi-file dataset
- with cleaned `SUPER RATE` targets
- and either a direct ExtraTrees regressor or a sparse-aware two-stage tree model as the main baseline

If that works well, the next step after Phase 1 should be a shared-trunk / separate-head multi-task model that predicts both `RATE CONST` and `SUPER RATE`.

