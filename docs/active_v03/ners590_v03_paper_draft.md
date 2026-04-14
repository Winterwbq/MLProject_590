# Paper Draft: v03 Surrogate Modeling for RATE CONST and SUPER RATE Prediction

Status: draft based on completed v03 artifacts available on 2026-04-10.

## Abstract

We studied the expanded `dataset/NERS590_data_V03` kinetic dataset as a surrogate-modeling problem: predict 204 RATE CONST outputs and 204 SUPER RATE outputs from gas composition, reduced electric field, and discharge power. The parsed v03 dataset contains 83,317 cases from 13 power levels, with each power file contributing 6,409 cases arranged as 221 number-density groups and 29 `E/N` points. The target distributions are highly non-Gaussian and sparse: RATE CONST has 16,996,668 scalar entries with 2,238,067 zeros, while SUPER RATE has the same number of scalar entries but 10,308,324 zeros.

We evaluated two split settings. The random-case split tests interpolation within the full observed power/composition/grid distribution. The 10mJ power-holdout split tests transfer to an unseen highest-power file. We trained separate-task models for RATE CONST and SUPER RATE, plus two multitask neural baselines: a shared single-head MLP that predicts RATE CONST and SUPER RATE together, and a shared-backbone/two-head MLP that predicts both tasks through separate heads. Across both splits and both target groups, direct ExtraTrees models were the strongest final choice. For RATE CONST, ExtraTrees achieved test log-RMSE 0.0685 on random cases and 0.0487 on the 10mJ holdout. For SUPER RATE, ExtraTrees achieved test log-RMSE 0.00382 on random cases and 0.00362 on the 10mJ holdout. Multitask MLPs learned physically correlated outputs, but they did not beat the separate ExtraTrees models in the present data regime.

Evidence: dataset counts and target distributions come from [`../../results/ners590_v03_analysis/analysis/dataset_summary.csv`](../../results/ners590_v03_analysis/analysis/dataset_summary.csv), [`../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv`](../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv), and [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv). Model comparisons come from [`../../results/ners590_v03_joint_review/experiment_summary.csv`](../../results/ners590_v03_joint_review/experiment_summary.csv), [`../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png`](../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png), and [`../../results/ners590_v03_joint_review/figures/strategy_relative_error_grid.png`](../../results/ners590_v03_joint_review/figures/strategy_relative_error_grid.png).

## 1. Dataset and Prediction Tasks

The v03 dataset is an expanded replacement for the earlier v02 and original `global_kin_boltz.out` datasets. The raw source is `dataset/NERS590_data_V03`, where each `.out` file corresponds to one deposited power value. The completed v03 parsing pass found 13 power levels: `0.1mJ`, `0.2mJ`, `0.5mJ`, and integer powers from `1mJ` to `10mJ`. Every power file has the same case structure: 221 density groups times 29 local `E/N` cases, giving 6,409 rows per power and 83,317 total cases.

Each model input row consists of:

| Input type | Description |
| --- | --- |
| Gas composition | 49 input mole-fraction columns parsed from `INPUT GAS MOLE FRACTIONS` |
| Reduced electric field | `ELECTRIC FIELD/NUMBER DENSITY`, represented as `log10(E/N)` |
| Power | Power inferred from the filename, represented as `log10(power_mJ)` |
| Tracking IDs | Case and group identifiers retained for analysis, never used as model features |

The targets are two 204-dimensional output groups:

| Target group | Meaning | Sparsity |
| --- | --- | --- |
| RATE CONST | 204 reaction rate constants | 14,758,601 positive entries and 2,238,067 zeros |
| SUPER RATE | 204 super-elastic rate entries | 6,688,344 positive entries and 10,308,324 zeros |

The target ranges justify log-space modeling. RATE CONST spans from 0 to 0.006326, with median 5.869e-10 and p99 0.002127. SUPER RATE is even more sparse: the median is exactly 0, the positive-only median is 2.193e-09, and the positive-only p95 is 1.945e-07.

Evidence: the parsed case grid is documented in [`../../results/ners590_v03_analysis/analysis/power_summary.csv`](../../results/ners590_v03_analysis/analysis/power_summary.csv) and visualized in [`../../results/ners590_v03_analysis/figures/case_count_by_power.png`](../../results/ners590_v03_analysis/figures/case_count_by_power.png) and [`../../results/ners590_v03_analysis/figures/e_over_n_grid_by_power.png`](../../results/ners590_v03_analysis/figures/e_over_n_grid_by_power.png). RATE CONST summaries come from [`../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv`](../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv) and [`../../results/ners590_v03_analysis/figures/top_reactions_by_mean_rate.png`](../../results/ners590_v03_analysis/figures/top_reactions_by_mean_rate.png). SUPER RATE summaries come from [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv), [`../../results/ners590_v03_analysis/super_rate_figures/top_super_rate_reactions_by_mean.png`](../../results/ners590_v03_analysis/super_rate_figures/top_super_rate_reactions_by_mean.png), and [`../../results/ners590_v03_analysis/super_rate_figures/super_rate_positive_fraction_by_power.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_positive_fraction_by_power.png).

## 2. Data Preprocessing

All modeling pipelines start from the raw v03 `.out` files, parse the readable Boltzmann output into structured tables, and then build training matrices. The parser writes case-level features, long target tables, species maps, reaction maps, and wide training target tables. The central parsed feature table is [`../../results/ners590_v03_analysis/parsed/training_inputs.csv`](../../results/ners590_v03_analysis/parsed/training_inputs.csv), while the wide RATE CONST table is [`../../results/ners590_v03_analysis/parsed/training_targets.csv`](../../results/ners590_v03_analysis/parsed/training_targets.csv). SUPER RATE is reconstructed from the long reaction table because the original long table stores both RATE CONST and SUPER RATE columns.

For features, three representations were used in v03:

| Feature set | Definition | Motivation |
| --- | --- | --- |
| `full_nonconstant_plus_log_en_log_power` | Train-only nonconstant mole fractions, `log10(E/N)`, `log10(power_mJ)` | Preserve raw composition columns while dropping features that are constant in the training pool |
| `all_inputs_plus_log_en_log_power` | All 49 mole fractions, `log10(E/N)`, `log10(power_mJ)` | Useful for MLPs where fixed input dimensionality is preferred |
| `composition_pca_plus_log_en_log_power` | Standardized train-only nonconstant composition compressed by PCA to 99 percent training variance, plus `log10(E/N)` and `log10(power_mJ)` | Reduce composition collinearity while keeping the physically dominant `E/N` and power coordinates explicit |

For targets, each output column is transformed independently:

$$
epsilon_j = max(min_positive_training_value_j / 10, 1e-30)

z_j = log10(y_j + epsilon_j)

y_hat_j = max(10^(z_hat_j) - epsilon_j, 0)
$$

During cross-validation, target epsilons, feature scaling, constant-feature detection, and PCA are fit only on the fold training split. During final locked-test evaluation, they are fit on the full train/validation pool and applied once to the locked test set. This prevents test leakage while giving the final model all non-test data. For SUPER RATE, constant all-zero outputs are dropped from the active training target set and restored as zeros in the full 204-output reconstruction. This is why final SUPER RATE rows report 94 active targets and 110 dropped constant targets.

Evidence: preprocessing implementation is in [`../../src/global_kin_ml/preprocessing.py`](../../src/global_kin_ml/preprocessing.py), separate-task pipeline logic is in [`../../src/global_kin_ml/pipeline.py`](../../src/global_kin_ml/pipeline.py), and multitask preprocessing is in [`../../src/global_kin_ml/multitask_pipeline.py`](../../src/global_kin_ml/multitask_pipeline.py). Saved preprocessing artifacts include [`../../results/ners590_v03_rate_const/training_random_case/data_snapshots/feature_set_final_metadata.csv`](../../results/ners590_v03_rate_const/training_random_case/data_snapshots/feature_set_final_metadata.csv), [`../../results/ners590_v03_rate_const/training_random_case/data_snapshots/rate_const_epsilons_trainval.csv`](../../results/ners590_v03_rate_const/training_random_case/data_snapshots/rate_const_epsilons_trainval.csv), [`../../results/ners590_v03_super_rate/training_random_case/data_snapshots/super_rate_epsilons_trainval.csv`](../../results/ners590_v03_super_rate/training_random_case/data_snapshots/super_rate_epsilons_trainval.csv), and [`../../results/ners590_v03_multitask/training_random_case/data_snapshots/multitask_target_summary.csv`](../../results/ners590_v03_multitask/training_random_case/data_snapshots/multitask_target_summary.csv).

## 3. Experimental Design

Two generalization settings were used:

| Split | Purpose | Test set |
| --- | --- | --- |
| Random Case | Estimate interpolation performance over the full v03 distribution | 15 percent randomly held-out cases, seed 42 |
| Power Holdout 10mJ | Estimate generalization to an unseen power file | All 6,409 cases from the 10mJ file |

Within the train/validation pool, five shuffled validation folds were used with seeds 7, 21, 42, 84, and 126. Model selection used mean validation log-space RMSE after reconstructing the final output space. For RATE CONST, this means all 204 outputs. For SUPER RATE, models are selected on active outputs and evaluated both on active targets and full 204-output reconstructions.

The evaluation metrics are:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Log RMSE | RMSE of `log10(y + epsilon)` | Primary metric, stable across heavy-tailed target magnitudes |
| Log MAE | MAE of `log10(y + epsilon)` | Robust average log error |
| Log R2 | Coefficient of determination in transformed target space | Near 1 indicates strong explained variance, but can be forgiving for high-variance targets |
| Factor-of-5 accuracy | Fraction of positive predictions within a factor of 5 of ground truth | Practical multiplicative accuracy |
| Absolute relative error | `abs(pred - truth) / truth`, positive ground-truth entries only | Useful as a percent-like measure but unstable for tiny denominators |
| SMAPE | `2*abs(pred - truth)/(abs(pred)+abs(truth))` | Safer bounded relative metric for sparse/tiny targets |

Evidence: split assignments are saved under [`../../results/ners590_v03_rate_const/training_random_case/data_snapshots/split_assignments.csv`](../../results/ners590_v03_rate_const/training_random_case/data_snapshots/split_assignments.csv), [`../../results/ners590_v03_rate_const/training_power_holdout_10mJ/data_snapshots/split_assignments.csv`](../../results/ners590_v03_rate_const/training_power_holdout_10mJ/data_snapshots/split_assignments.csv), and their SUPER RATE/multitask equivalents. Verification checks are saved in [`../../results/ners590_v03_rate_const/training_random_case/verification_checks.csv`](../../results/ners590_v03_rate_const/training_random_case/verification_checks.csv), [`../../results/ners590_v03_super_rate/training_random_case/verification_checks.csv`](../../results/ners590_v03_super_rate/training_random_case/verification_checks.csv), and [`../../results/ners590_v03_multitask/training_random_case/verification_checks.csv`](../../results/ners590_v03_multitask/training_random_case/verification_checks.csv).

## 4. Models

We tested both independent-target and joint-target ideas.

The direct Ridge baseline solves a regularized linear least-squares problem:

```text
min_W ||Z - XW||_2^2 + alpha ||W||_2^2
```

where `X` is the feature matrix and `Z` is the transformed target matrix.

RandomForest and ExtraTrees use tree ensembles:

```text
z_hat(x) = (1 / T) * sum_t h_t(x)
```

where each tree `h_t` partitions the input space and predicts a vector-valued target. RandomForest uses bootstrap/randomized split construction; ExtraTrees adds stronger split randomization and often reduces variance on small-to-medium tabular datasets.

The direct MLP baseline is:

```text
h_0 = x
h_l = ReLU(W_l h_(l-1) + b_l)
z_hat = W_out h_L + b_out
```

The v03 curated MLP used 3 hidden layers, width 256, dropout 0.0, weight decay 1e-5, learning rate 1e-3, batch size 256, max 120 epochs, and early stopping.

Latent-output PCA models first compress transformed targets:

```text
c = PCA_k(z)
c_hat = f(x)
z_hat = PCA_k^(-1)(c_hat)
```

These test whether the 204 reaction outputs lie on a lower-dimensional manifold. The model is still evaluated after reconstructing all target outputs, not merely in latent space.

For multitask learning, we tested:

| Multitask model | Definition | Intended benefit |
| --- | --- | --- |
| Joint single-head MLP | One MLP predicts concatenated RATE CONST and SUPER RATE outputs | Exploit shared signal between physically related targets |
| Joint two-head MLP | Shared MLP backbone with one RATE CONST head and one SUPER RATE head | Share representation while allowing task-specific output mappings |

The multitask objective balances the two target groups in log space. Conceptually:

```text
L = 0.5 * RMSE_log_rate + 0.5 * RMSE_log_super
```

The two-head model is attractive physically because RATE CONST and SUPER RATE are derived from the same electron energy distribution but with different reaction-specific coefficients. In this dataset, however, the multitask architecture adds optimization complexity and does not outperform separate ExtraTrees.

Evidence: separate-task curated model definitions are in [`../../scripts/run_ners590_v02_training.py`](../../scripts/run_ners590_v02_training.py), [`../../scripts/run_ners590_v02_super_rate_training.py`](../../scripts/run_ners590_v02_super_rate_training.py), [`../../scripts/run_ners590_v03_rate_const_training.py`](../../scripts/run_ners590_v03_rate_const_training.py), and [`../../scripts/run_ners590_v03_super_rate_training.py`](../../scripts/run_ners590_v03_super_rate_training.py). Multitask model definitions are in [`../../scripts/run_ners590_v03_multitask_training.py`](../../scripts/run_ners590_v03_multitask_training.py) and [`../../src/global_kin_ml/multitask_pipeline.py`](../../src/global_kin_ml/multitask_pipeline.py). Model implementation details are in [`../../src/global_kin_ml/models.py`](../../src/global_kin_ml/models.py).

## 5. Data Analysis Results

### 5.1 Power and E/N Structure

All 13 powers have identical case counts and identical `E/N` grid coverage. The `E/N` range is 1e-20 to 1e-14 at every power. This is useful experimentally: a power-holdout result is not confounded by missing `E/N` positions. The difference between the random-case and power-holdout tasks is therefore mainly extrapolation/interpolation in power-dependent kinetics, not a different grid topology.

Evidence: [`../../results/ners590_v03_analysis/analysis/power_summary.csv`](../../results/ners590_v03_analysis/analysis/power_summary.csv), [`../../results/ners590_v03_analysis/analysis/power_grid_consistency.csv`](../../results/ners590_v03_analysis/analysis/power_grid_consistency.csv), and [`../../results/ners590_v03_analysis/figures/e_over_n_grid_by_power.png`](../../results/ners590_v03_analysis/figures/e_over_n_grid_by_power.png).

### 5.2 RATE CONST Distribution

RATE CONST is heavy-tailed and sparse. The global median is 5.869e-10, while the maximum is 0.006326. The top power-sensitive reactions include vibrational oxygen channels and ionization/dissociation reactions such as `O2-VIB 4`, `O2-VIB 3`, `E + H2OV > O^ + 2H + 2E`, `E + H2OV > H2^ + O + E`, and `E + H2O > H^ + OH + 2E`. These are also among the reactions that appear later as difficult prediction targets, suggesting that model error is tied to sharp physical transitions rather than simply random noise.

Evidence: [`../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv`](../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv), [`../../results/ners590_v03_analysis/analysis/reaction_power_sensitivity_summary.csv`](../../results/ners590_v03_analysis/analysis/reaction_power_sensitivity_summary.csv), [`../../results/ners590_v03_analysis/figures/median_rate_by_power.png`](../../results/ners590_v03_analysis/figures/median_rate_by_power.png), and [`../../results/ners590_v03_analysis/figures/top_power_sensitive_reactions.png`](../../results/ners590_v03_analysis/figures/top_power_sensitive_reactions.png).

### 5.3 SUPER RATE Distribution

SUPER RATE is much sparser than RATE CONST: only 39.35 percent of entries are positive. The active output count increases strongly with local `E/N` case. At local case 1, the mean number of nonzero SUPER RATE reactions is 19; by local case 7 it rises to 83; by local case 8 it reaches 94 and then stays at 94 through later local cases. This activation threshold explains why early local cases become the hardest SUPER RATE region: the model must learn both a near-zero/active transition and the magnitude after activation.

Several SUPER RATE outputs are strongly correlated with their matching RATE CONST outputs. Among 94 reactions with non-null positive-only log correlations, the mean correlation is 0.783, the median is 0.938, and 55 are at least 0.9. This supports the physical intuition that RATE CONST and SUPER RATE are related through the electron energy distribution. However, 8 correlations are nonpositive and many reactions are always zero for SUPER RATE, so the relationship is not a uniform one-to-one mapping across all 204 reactions.

Evidence: [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv), [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_local_case_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_local_case_summary.csv), [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv), [`../../results/ners590_v03_analysis/super_rate_figures/super_rate_nonzero_count_by_local_case.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_nonzero_count_by_local_case.png), [`../../results/ners590_v03_analysis/super_rate_figures/super_rate_sum_by_local_case.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_sum_by_local_case.png), and [`../../results/ners590_v03_analysis/super_rate_figures/super_rate_rate_const_correlation_top.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_rate_const_correlation_top.png).

## 6. Model Performance Results

### 6.1 Final Test Performance

| Task | Split | Strategy | Best family | Feature set | Test log RMSE | Test log R2 | Median abs relative error | p95 abs relative error | Factor-5 acc. |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| RATE CONST | Random Case | Separate Task | ExtraTrees | compPCA+logEN+logP | 0.068483 | 0.999839 | 0.000193 | 0.042730 | 0.997896 |
| RATE CONST | Random Case | Joint Single Head | MLP | all+logEN+logP | 0.098037 | 0.975080 | 0.012724 | 0.210111 | 0.996552 |
| RATE CONST | Random Case | Joint Two Head | MLP | compPCA+logEN+logP | 0.098939 | 0.975058 | 0.008326 | 0.205029 | 0.996697 |
| RATE CONST | Power Holdout 10mJ | Separate Task | ExtraTrees | compPCA+logEN+logP | 0.048740 | 0.999603 | 0.000545 | 0.089635 | 0.999700 |
| RATE CONST | Power Holdout 10mJ | Joint Single Head | MLP | all+logEN+logP | 0.141318 | 0.973822 | 0.012902 | 0.239965 | 0.997707 |
| RATE CONST | Power Holdout 10mJ | Joint Two Head | MLP | all+logEN+logP | 0.245245 | 0.969698 | 0.017221 | 0.365859 | 0.993821 |
| SUPER RATE | Random Case | Separate Task | ExtraTrees | all+logEN+logP | 0.003821 | 0.999997 | 0.000068 | 0.001263 | 0.999974 |
| SUPER RATE | Random Case | Joint Single Head | MLP | all+logEN+logP | 0.013810 | 0.999939 | 0.004797 | 0.034845 | 0.999866 |
| SUPER RATE | Random Case | Joint Two Head | MLP | compPCA+logEN+logP | 0.011835 | 0.999955 | 0.002839 | 0.027624 | 0.999868 |
| SUPER RATE | Power Holdout 10mJ | Separate Task | ExtraTrees | all+logEN+logP | 0.003621 | 0.999941 | 0.000067 | 0.004343 | 1.000000 |
| SUPER RATE | Power Holdout 10mJ | Joint Single Head | MLP | all+logEN+logP | 0.025411 | 0.999544 | 0.004802 | 0.054381 | 0.999817 |
| SUPER RATE | Power Holdout 10mJ | Joint Two Head | MLP | all+logEN+logP | 0.043464 | 0.999076 | 0.007230 | 0.085214 | 0.999532 |

The main result is stable: separate-task ExtraTrees is best in all final test comparisons. The multitask models remain useful baselines because they demonstrate that shared representations can learn both outputs, but their errors are consistently higher than the separate tree ensembles.

Evidence: [`../../results/ners590_v03_joint_review/experiment_summary.csv`](../../results/ners590_v03_joint_review/experiment_summary.csv), [`../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png`](../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png), and [`../../results/ners590_v03_joint_review/figures/strategy_relative_error_grid.png`](../../results/ners590_v03_joint_review/figures/strategy_relative_error_grid.png).

### 6.2 Validation Leaderboards

In separate RATE CONST training, the best validation model was direct ExtraTrees on `composition_pca_plus_log_en_log_power`, with mean validation log-RMSE 0.06955 for random cases and 0.06999 for power holdout. Direct MLP on all inputs was second at 0.08541 and 0.08211. RandomForest and latent-output PCA models were worse, around 0.134 to 0.144. Ridge was much worse, around 1.46, indicating the mapping is strongly nonlinear.

For SUPER RATE, direct ExtraTrees on `all_inputs_plus_log_en_log_power` was best with mean validation log-RMSE 0.00656 for random cases and 0.00604 for power holdout. Two-stage ExtraTrees and composition-PCA ExtraTrees were close but not better. Direct MLP was notably worse around 0.0245 and 0.0235, while Ridge was around 0.411.

For multitask learning, the best validation model was usually the joint single-head MLP on `all_inputs_plus_log_en_log_power`, with mean joint validation log-RMSE 0.05785 for random cases and 0.05445 for power holdout. This ranking is interesting because validation joint RMSE is not directly comparable to separate-task single-output-group RMSE: the joint score averages task losses and can hide RATE CONST degradation. The locked-test branch evaluation shows that separate ExtraTrees remains better for both output groups.

Evidence: RATE CONST leaderboards are [`../../results/ners590_v03_rate_const/training_random_case/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_rate_const/training_random_case/tuning/model_leaderboard_summary.csv) and [`../../results/ners590_v03_rate_const/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_rate_const/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv). SUPER RATE leaderboards are [`../../results/ners590_v03_super_rate/training_random_case/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_super_rate/training_random_case/tuning/model_leaderboard_summary.csv) and [`../../results/ners590_v03_super_rate/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_super_rate/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv). Multitask leaderboards are [`../../results/ners590_v03_multitask/training_random_case/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_multitask/training_random_case/tuning/model_leaderboard_summary.csv) and [`../../results/ners590_v03_multitask/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv`](../../results/ners590_v03_multitask/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv).

## 7. Comparative Analysis

### 7.1 Why ExtraTrees Performs Best

ExtraTrees is well matched to this dataset. The inputs are tabular, the sample count is large enough for tree ensembles, and many relationships are piecewise-smooth across `E/N`, composition, and power. The target space also contains many near-zero plateaus and activation thresholds. Tree ensembles handle such regimes naturally because splits can isolate threshold-like regions without requiring global smoothness.

The RATE CONST best model uses composition PCA plus explicit `log10(E/N)` and `log10(power_mJ)`. This suggests that composition collinearity is real enough that PCA helps the tree ensemble avoid redundant mole-fraction dimensions, while `E/N` and power should remain uncompressed because they are physically dominant. The SUPER RATE best model instead uses all 49 composition inputs plus log features. A plausible explanation is that SUPER RATE has more all-zero or thresholded outputs, so preserving sparse, reaction-specific composition signals is more valuable than compressing composition variance.

Evidence: final model keys and selected features are in [`../../results/ners590_v03_joint_review/experiment_summary.csv`](../../results/ners590_v03_joint_review/experiment_summary.csv). Feature metadata is in [`../../results/ners590_v03_rate_const/training_random_case/data_snapshots/feature_set_final_metadata.csv`](../../results/ners590_v03_rate_const/training_random_case/data_snapshots/feature_set_final_metadata.csv) and [`../../results/ners590_v03_super_rate/training_random_case/data_snapshots/feature_set_final_metadata.csv`](../../results/ners590_v03_super_rate/training_random_case/data_snapshots/feature_set_final_metadata.csv). Model leaderboard plots are [`../../results/ners590_v03_rate_const/training_random_case/figures/model_leaderboard.png`](../../results/ners590_v03_rate_const/training_random_case/figures/model_leaderboard.png) and [`../../results/ners590_v03_super_rate/training_random_case/figures/model_leaderboard.png`](../../results/ners590_v03_super_rate/training_random_case/figures/model_leaderboard.png).

### 7.2 Why Multitask MLPs Do Not Beat Separate Models

The physical motivation for multitask learning is valid: RATE CONST and SUPER RATE are derived from the same electron energy distribution. The data supports this partially, with median positive-only log correlation 0.938 among active SUPER RATE reactions. However, the practical learning problem is not simply "predict two correlated vectors." SUPER RATE has 110 constant all-zero outputs, only 94 active outputs, and activation boundaries across local `E/N` cases. RATE CONST is dense compared with SUPER RATE and has different hardest reactions. A shared neural model must balance these two loss landscapes.

The joint single-head MLP predicts a concatenated target vector, which can exploit shared features but can also let the easier SUPER RATE target group dominate the joint validation score. The two-head MLP is more physically interpretable, but it still shares a backbone and used equal task weighting. The results show the two-head branch does not automatically fix negative transfer: in power holdout, RATE CONST log-RMSE rises to 0.245, much worse than the separate ExtraTrees value of 0.0487.

The model comparison therefore suggests that the current dataset is more favorable to high-capacity tabular ensembles than to the current small MLP multitask formulation. A future neural approach may need reaction-aware heads, uncertainty weighting, a larger hyperparameter search, monotonic/smoothness constraints over `E/N`, or pretraining on RATE CONST followed by SUPER RATE fine-tuning.

Evidence: task correlations are in [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv). Strategy deltas are in [`../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv`](../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv) and [`../../results/ners590_v03_joint_review/figures/strategy_delta_vs_separate.png`](../../results/ners590_v03_joint_review/figures/strategy_delta_vs_separate.png). Branch evaluations are in [`../../results/ners590_v03_multitask/training_random_case/branch_evaluation/branch_test_summary.csv`](../../results/ners590_v03_multitask/training_random_case/branch_evaluation/branch_test_summary.csv) and [`../../results/ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/branch_test_summary.csv`](../../results/ners590_v03_multitask/training_power_holdout_10mJ/branch_evaluation/branch_test_summary.csv).

### 7.3 Why Errors Are Larger for Small Magnitudes

Relative error is unstable when the true value is tiny. For RATE CONST random-case ExtraTrees, the median absolute relative error is only 0.000270 in the `[1e-20, 1e-18)` bin, but the p95 relative error is 1.13 because a very small absolute deviation divided by an extremely small denominator becomes a large relative error. The `[1e-18, 1e-16)` bin is worse, with median relative error 0.166 and only 42.5 percent within 10 percent. Once rates reach `[1e-12, 1e-10)`, the median relative error falls to 0.000464 and 98.7 percent are within 10 percent; from `[1e-10, 1e-06)`, at least 99.9 percent are within 10 percent.

SUPER RATE shows the same pattern even more strongly because very-low-magnitude positive samples are rare. In random-case ExtraTrees, the `[1e-20, 1e-18)` bin has only 81 positive samples, median relative error 0.787, and p95 relative error 5.82. By `[1e-12, 1e-10)`, the median relative error is 0.000264 and 99.98 percent are within 10 percent. For `[1e-10, 1e-06)`, the within-10-percent rate is effectively 100 percent.

This is why log RMSE and SMAPE are more reliable primary metrics than raw mean signed relative error. Relative error is still useful, but it must be interpreted by magnitude bin.

Evidence: RATE CONST magnitude bins are in [`../../results/ners590_v03_rate_const/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v03_rate_const/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv), [`../../results/ners590_v03_rate_const/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v03_rate_const/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv), and [`../../results/ners590_v03_rate_const/training_random_case/figures/relative_error_by_magnitude.png`](../../results/ners590_v03_rate_const/training_random_case/figures/relative_error_by_magnitude.png). SUPER RATE magnitude bins are in [`../../results/ners590_v03_super_rate/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v03_super_rate/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv), [`../../results/ners590_v03_super_rate/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v03_super_rate/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv), and [`../../results/ners590_v03_super_rate/training_random_case/figures/relative_error_by_magnitude.png`](../../results/ners590_v03_super_rate/training_random_case/figures/relative_error_by_magnitude.png). SMAPE checks are in [`../../results/ners590_v03_rate_const/training_random_case/evaluation/test_smape_by_magnitude_bin.csv`](../../results/ners590_v03_rate_const/training_random_case/evaluation/test_smape_by_magnitude_bin.csv) and [`../../results/ners590_v03_super_rate/training_random_case/evaluation/test_smape_by_magnitude_bin.csv`](../../results/ners590_v03_super_rate/training_random_case/evaluation/test_smape_by_magnitude_bin.csv).

### 7.4 Local-Case Error Structure

The worst local cases cluster in the early-to-mid `E/N` sweep. For separate RATE CONST, the worst random-case local index is 8 with mean log-RMSE 0.120, while the worst 10mJ holdout local index is 10 with mean log-RMSE 0.0858. For joint MLP strategies, the hardest local cases are also in this range, especially local cases 8 to 10.

For SUPER RATE, the hardest local cases are earlier. Separate ExtraTrees has worst local case 4 in both random-case and power-holdout settings. Joint MLPs often peak around local cases 2 to 3. This matches the SUPER RATE activation analysis: local cases 1 to 7 are where the active output count transitions from 19 to 83, before saturating at 94 by local case 8. The model performs best after the active reaction set stabilizes.

Evidence: local-case errors are in [`../../results/ners590_v03_joint_review/local_case_error_profiles.csv`](../../results/ners590_v03_joint_review/local_case_error_profiles.csv), [`../../results/ners590_v03_joint_review/figures/strategy_local_case_error_profiles.png`](../../results/ners590_v03_joint_review/figures/strategy_local_case_error_profiles.png), and [`../../results/ners590_v03_joint_review/figures/local_case_error_profiles.png`](../../results/ners590_v03_joint_review/figures/local_case_error_profiles.png). SUPER RATE activation by local case is in [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_local_case_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_local_case_summary.csv) and [`../../results/ners590_v03_analysis/super_rate_figures/super_rate_nonzero_count_by_local_case.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_nonzero_count_by_local_case.png).

### 7.5 Reaction-Level Error Structure

The hardest RATE CONST reactions are repeatedly ionization and dissociation channels with strong power or threshold sensitivity. For separate random-case ExtraTrees, the worst RATE CONST reaction is `rate_const_158` (`E + H2OV > O^ + 2H + 2E`) with log-RMSE 0.279 and factor-of-5 accuracy 0.945. In the 10mJ holdout, the same reaction remains worst for separate ExtraTrees, with log-RMSE 0.201 and factor-of-5 accuracy 0.984. Other recurrent difficult channels include `AR+ IONIZATION`, `E + H2O > O^ + 2H + 2E`, and `E + H2OV > H2^ + O + E`.

For SUPER RATE, the most difficult reaction across many strategies is `super_rate_086` (`H2 > H2(V=2)`). Even so, separate ExtraTrees remains highly accurate: random-case log-RMSE is 0.0533 with factor-of-5 accuracy 0.9978, and 10mJ holdout log-RMSE is 0.0324 with factor-of-5 accuracy 1.0. The joint MLPs degrade this reaction substantially, especially under power holdout where the two-head MLP reaches log-RMSE 0.527 for `super_rate_086`.

Evidence: reaction-level comparisons are in [`../../results/ners590_v03_joint_review/worst_reactions_combined.csv`](../../results/ners590_v03_joint_review/worst_reactions_combined.csv), [`../../results/ners590_v03_rate_const/training_random_case/evaluation/worst_10_reactions.csv`](../../results/ners590_v03_rate_const/training_random_case/evaluation/worst_10_reactions.csv), [`../../results/ners590_v03_super_rate/training_random_case/evaluation/worst_10_reactions.csv`](../../results/ners590_v03_super_rate/training_random_case/evaluation/worst_10_reactions.csv), [`../../results/ners590_v03_rate_const/training_random_case/figures/worst_reactions_log_rmse.png`](../../results/ners590_v03_rate_const/training_random_case/figures/worst_reactions_log_rmse.png), and [`../../results/ners590_v03_super_rate/training_random_case/figures/worst_reactions_log_rmse.png`](../../results/ners590_v03_super_rate/training_random_case/figures/worst_reactions_log_rmse.png).

## 8. Interpretation and Recommendations

The v03 experiments support three conclusions.

First, power must be modeled explicitly. The expanded dataset has 13 power levels with identical `E/N` grids, and power-sensitive reactions appear among the hardest RATE CONST targets. Using `log10(power_mJ)` is the right default representation because the power range is multiplicative and the target dynamics are heavy-tailed.

Second, separate ExtraTrees models are the best publishable baseline for the current pipeline. They are accurate, deterministic, CPU-friendly, and robust on tabular threshold-like data. For RATE CONST, use direct ExtraTrees with `composition_pca_plus_log_en_log_power`. For SUPER RATE, use direct ExtraTrees with `all_inputs_plus_log_en_log_power`, while preserving full 204-output reconstruction with constant-zero reactions restored.

Third, multitask learning remains scientifically interesting but is not yet the best production model. The RATE CONST and SUPER RATE outputs are physically related, and many active reactions have strong log correlations. But the current shared MLPs underperform because they must balance dense RATE CONST, sparse SUPER RATE, and activation thresholds with one small neural architecture. Future multitask work should test reaction-aware designs, per-task uncertainty loss weighting, larger neural hyperparameter sweeps, output masking, and possibly a two-stage physics-guided design where RATE CONST predictions inform SUPER RATE but do not force all outputs through a single shared loss.

Evidence: recommendations synthesize [`../../results/ners590_v03_joint_review/experiment_summary.csv`](../../results/ners590_v03_joint_review/experiment_summary.csv), [`../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv`](../../results/ners590_v03_joint_review/strategy_delta_vs_separate.csv), [`../../results/ners590_v03_joint_review/figures/strategy_delta_vs_separate.png`](../../results/ners590_v03_joint_review/figures/strategy_delta_vs_separate.png), [`../../results/ners590_v03_analysis/analysis/reaction_power_sensitivity_summary.csv`](../../results/ners590_v03_analysis/analysis/reaction_power_sensitivity_summary.csv), and [`../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv).

## 9. Limitations and Next Steps

The current paper draft should be treated as a strong internal analysis, not yet a final manuscript. The largest limitation is that the MLP baselines were intentionally CPU-friendly and not exhaustively tuned. The ExtraTrees advantage is real for the tested configurations, but a larger neural study could include residual architectures, reaction embeddings, mixture-of-experts heads, monotonic encodings over `E/N`, or explicit coupling between RATE CONST and SUPER RATE. The second limitation is that the power-holdout experiment used the highest-power file, 10mJ. That is scientifically useful, but a complete extrapolation study should hold out each power level in turn or at least low, mid, and high powers.

The next recommended experiments are:

| Priority | Experiment | Reason |
| --- | --- | --- |
| 1 | Keep separate ExtraTrees as the main v03 baseline | It is clearly best across completed tests |
| 2 | Add leave-one-power-out sweep | Test whether 10mJ is representative of power extrapolation |
| 3 | Add reaction-aware multitask MLP | Better test the physics-correlation hypothesis |
| 4 | Add calibration/uncertainty analysis | Identify unreliable low-magnitude and activation-boundary predictions |
| 5 | Report magnitude-binned errors in any manuscript | Prevent misleading conclusions from raw relative error on tiny denominators |

Evidence: current split limitations are visible in [`../../results/ners590_v03_rate_const/training_power_holdout_10mJ/tuning/split_metadata.csv`](../../results/ners590_v03_rate_const/training_power_holdout_10mJ/tuning/split_metadata.csv) and [`../../results/ners590_v03_super_rate/training_power_holdout_10mJ/tuning/split_metadata.csv`](../../results/ners590_v03_super_rate/training_power_holdout_10mJ/tuning/split_metadata.csv). Relative-error limitations are documented in the magnitude-bin files cited in Section 7.3.

## 10. Reproducibility Map

The completed v03 workflow is split into focused commands and result roots:

| Workflow | Command | Primary output |
| --- | --- | --- |
| Data analysis | `python scripts/run_ners590_v03_analysis.py` | [`../../results/ners590_v03_analysis`](../../results/ners590_v03_analysis) |
| RATE CONST separate training | `python scripts/run_ners590_v03_rate_const_training.py` | [`../../results/ners590_v03_rate_const`](../../results/ners590_v03_rate_const) |
| SUPER RATE separate training | `python scripts/run_ners590_v03_super_rate_training.py` | [`../../results/ners590_v03_super_rate`](../../results/ners590_v03_super_rate) |
| Multitask training | `python scripts/run_ners590_v03_multitask_training.py` | [`../../results/ners590_v03_multitask`](../../results/ners590_v03_multitask) |
| Multitask branch evaluation | `python scripts/evaluate_ners590_v03_multitask_branches.py` | `branch_evaluation/` folders under [`../../results/ners590_v03_multitask`](../../results/ners590_v03_multitask) |
| Joint review assets | `python scripts/build_ners590_v03_joint_review_assets.py` | [`../../results/ners590_v03_joint_review`](../../results/ners590_v03_joint_review) |

The most important single file for comparing final models is [`../../results/ners590_v03_joint_review/experiment_summary.csv`](../../results/ners590_v03_joint_review/experiment_summary.csv). The most important single figure is [`../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png`](../../results/ners590_v03_joint_review/figures/strategy_test_log_rmse_grid.png).

