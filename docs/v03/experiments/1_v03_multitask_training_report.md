# NERS590 v03 Multitask Training Report

This report compares the existing separate-task v03 baselines against two new multitask neural branches:

- `Joint Single Head`: one shared MLP predicts concatenated `RATE CONST + SUPER RATE` log-targets.
- `Joint Two Head`: one shared MLP backbone feeds separate RATE CONST and SUPER RATE output heads.

## Experiment Design

All multitask models use the same case inputs as the v03 separate-task experiments: composition features, `log10(E/N)`, and `log10(power_mJ)`. The RATE CONST task trains all `204` outputs. The SUPER RATE task drops `110` constant-zero channels from the learned target matrix, trains the remaining `94` active channels, then reconstructs the full `204`-channel target vector before evaluation.

The training target for each channel is `log10(y_j + epsilon_j)`, with `epsilon_j` estimated from the training pool only. The MLP inputs and transformed targets are standardized inside the model training loop. The multitask model-selection score is a balanced joint log-space RMSE:

`joint_validation_score = 0.5 * RMSE_log(RATE CONST) + 0.5 * RMSE_log(active SUPER RATE)`

This avoids letting RATE CONST dominate only because it has more active channels. The dual-head model uses the same principle in its loss:

`L = 0.5 * MSE(rate_head) + 0.5 * MSE(super_rate_head)`

Citations: [`model_leaderboard_summary.csv`](../../../results/ners590_v03_multitask/training_random_case/tuning/model_leaderboard_summary.csv), [`model_leaderboard_summary.csv`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv), [`multitask_target_summary.csv`](../../../results/ners590_v03_multitask/training_random_case/data_snapshots/multitask_target_summary.csv), [`multitask_target_summary.csv`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/data_snapshots/multitask_target_summary.csv).

## Model Branches

The single-head model can be written as:

`z = f_theta(x)`

`[rate_hat, super_hat] = W z + b`

This makes the final output layer fully shared across the concatenated output vector. It gives the model one common representation and one common output map.

The two-head model can be written as:

`z = f_theta(x)`

`rate_hat = W_rate z + b_rate`

`super_hat = W_super z + b_super`

This keeps a shared representation but allows task-specific output maps. It is the more physically motivated architecture if RATE CONST and SUPER RATE need shared energy-distribution features but different reaction-specific coefficient maps.

Citations: [`run_ners590_v03_multitask_training.py`](../../../scripts/training/run_ners590_v03_multitask_training.py), [`multitask_pipeline.py`](../../../src/global_kin_ml/multitask_pipeline.py), [`model_trials_foldwise.csv`](../../../results/ners590_v03_multitask/training_random_case/tuning/model_trials_foldwise.csv), [`model_trials_foldwise.csv`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/tuning/model_trials_foldwise.csv).

## Validation Results

In both splits, the best validation configuration selected by the balanced joint score was the `Joint Single Head` model with `all_inputs_plus_log_en_log_power`.

| Split | Best branch | Feature set | Mean joint validation log-RMSE | Mean RATE CONST validation log-RMSE | Mean active SUPER RATE validation log-RMSE |
|---|---|---:|---:|---:|---:|
| Random case | Joint Single Head | all inputs + log E/N + log power | 0.057847 | 0.095367 | 0.020327 |
| Power holdout 10mJ | Joint Single Head | all inputs + log E/N + log power | 0.054447 | 0.089946 | 0.018948 |

The two-head branch did reduce active SUPER RATE validation RMSE in some random-case comparisons, but it increased RATE CONST validation RMSE enough that the balanced joint score was worse. In the 10mJ holdout validation, the two-head branch remained worse overall.

Citations: [`model_leaderboard_summary.csv`](../../../results/ners590_v03_multitask/training_random_case/tuning/model_leaderboard_summary.csv), [`model_leaderboard_summary.csv`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv), [`multitask_model_leaderboard.png`](../../../results/ners590_v03_multitask/training_random_case/figures/multitask_model_leaderboard.png), [`multitask_model_leaderboard.png`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/figures/multitask_model_leaderboard.png).

## Locked Test Results

The final branch-level evaluation refit and tested the best saved validation config from each multitask family. The separate-task ExtraTrees baseline remains best in every task/split combination.

| Task | Split | Strategy | Test log-RMSE | Test log-R2 | Median abs relative error |
|---|---|---:|---:|---:|---:|
| RATE CONST | Random case | Separate Task | 0.068483 | 0.999839 | 0.000193 |
| RATE CONST | Random case | Joint Single Head | 0.098037 | 0.975080 | 0.012724 |
| RATE CONST | Random case | Joint Two Head | 0.098939 | 0.975058 | 0.008326 |
| RATE CONST | Power holdout 10mJ | Separate Task | 0.048740 | 0.999603 | 0.000545 |
| RATE CONST | Power holdout 10mJ | Joint Single Head | 0.141318 | 0.973822 | 0.012902 |
| RATE CONST | Power holdout 10mJ | Joint Two Head | 0.245245 | 0.969698 | 0.017221 |
| SUPER RATE | Random case | Separate Task | 0.003821 | 0.999997 | 0.000068 |
| SUPER RATE | Random case | Joint Single Head | 0.013810 | 0.999939 | 0.004797 |
| SUPER RATE | Random case | Joint Two Head | 0.011835 | 0.999955 | 0.002839 |
| SUPER RATE | Power holdout 10mJ | Separate Task | 0.003621 | 0.999941 | 0.000067 |
| SUPER RATE | Power holdout 10mJ | Joint Single Head | 0.025411 | 0.999544 | 0.004802 |
| SUPER RATE | Power holdout 10mJ | Joint Two Head | 0.043464 | 0.999076 | 0.007230 |

Citations: [`experiment_summary.csv`](../../../results/ners590_v03_joint_review/experiment_summary.csv), [`strategy_test_log_rmse_grid.png`](../../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png), [`strategy_relative_error_grid.png`](../../../results/ners590_v03_joint_review/figures/strategy_relative_error_grid.png), [`branch_test_summary.csv`](../../../results/ners590_v03_multitask/training_random_case/branch_evaluation/branch_test_summary.csv), [`branch_test_summary.csv`](../../../results/ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/branch_test_summary.csv).

## Interpretation

The results do not mean RATE CONST and SUPER RATE are physically unrelated. The v03 data analysis still shows strong per-reaction relationships between the two output families, and the two-head branch did slightly improve random-case SUPER RATE compared with the single-head branch. The more precise interpretation is that these MLP multitask architectures did not exploit the shared physics well enough to beat the separate-task ExtraTrees models.

There are three likely reasons:

- The separate-task baselines are strong nonlinear tabular models. ExtraTrees can model piecewise threshold-like relationships in composition, power, and `E/N` without neural optimization instability.
- Joint neural training creates a task tradeoff. Improving active SUPER RATE sometimes worsens RATE CONST, and the balanced score penalizes that tradeoff.
- Power-holdout generalization is stricter than random interpolation. The dual-head model especially overfits or under-generalizes in the 10mJ holdout case, where both RATE CONST and SUPER RATE get worse than the single-head branch.

Citations: [`super_rate_rate_const_relationship_summary.csv`](../../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv), [`strategy_delta_vs_separate.csv`](../../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv), [`strategy_delta_vs_separate.png`](../../../results/ners590_v03_joint_review/figures/strategy_delta_vs_separate.png), [`super_rate_rate_const_correlation_top.png`](../../../results/ners590_v03_analysis/super_rate_figures/super_rate_rate_const_correlation_top.png).

## Recommendation

For current v03 production-style prediction, keep the separate-task ExtraTrees baselines:

- RATE CONST: separate ExtraTrees with `composition_pca_plus_log_en_log_power`.
- SUPER RATE: separate ExtraTrees with `all_inputs_plus_log_en_log_power`.

The multitask branch should remain a research direction, not the current best baseline. If we revisit it, the next version should test stronger task-balancing or physics-informed coupling, such as predicting a shared latent EEDF representation before mapping to both target families, rather than only sharing a generic MLP trunk.

Citations: [`experiment_summary.csv`](../../../results/ners590_v03_joint_review/experiment_summary.csv), [`strategy_delta_vs_separate.csv`](../../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv), [`selected_model.csv`](../../../results/ners590_v03_rate_const/training_random_case/tuning/selected_model.csv), [`selected_model.csv`](../../../results/ners590_v03_super_rate/training_random_case/tuning/selected_model.csv), [`selected_model.csv`](../../../results/ners590_v03_multitask/training_random_case/branch_evaluation/joint_single_head_mlp/selected_model.csv), [`selected_model.csv`](../../../results/ners590_v03_multitask/training_random_case/branch_evaluation/joint_two_head_mlp/selected_model.csv).

