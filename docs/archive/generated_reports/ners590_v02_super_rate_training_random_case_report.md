# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/ners590_v03_super_rate/training_random_case`
- Target family: `super_rate` (SUPER RATE)
- Selected model key: `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `all_inputs_plus_log_en_log_power`
- Selected latent dimension: `nan`
- Reconstructed output width: `204` total targets, `94` modeled directly, `110` constant-zero targets restored as zeros

Primary supporting outputs:

- [`selected_model.csv`](../results/training_random_case/tuning/selected_model.csv)
- [`model_leaderboard_summary.csv`](../results/training_random_case/tuning/model_leaderboard_summary.csv)
- [`search_stage_summary.csv`](../results/training_random_case/tuning/search_stage_summary.csv)
- [`test_overall_metrics.csv`](../results/training_random_case/evaluation/test_overall_metrics.csv)
- [`test_overall_metrics_active_targets.csv`](../results/training_random_case/evaluation/test_overall_metrics_active_targets.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- [`test_smape_overall_summary.csv`](../results/training_random_case/evaluation/test_smape_overall_summary.csv)
- [`oracle_test_overall_by_k.csv`](../results/training_random_case/pca/oracle_test_overall_by_k.csv)
- [`target_metadata.csv`](../results/training_random_case/data_snapshots/target_metadata.csv)
- [`verification_checks.csv`](../results/training_random_case/verification_checks.csv)

## Main Findings

The winning configuration was `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.006561` and mean validation log-R2 `0.999990`.

On the locked test split, the final model achieved reconstructed-full-output log-space RMSE `0.003821`, log-space MAE `0.000100`, and log-space R2 `0.999997`. In original-value space, the model achieved RMSE `1.430929e-11` and MAE `1.569737e-12`.
The retained nontrivial target subset achieved log-space RMSE `0.005629`, log-space MAE `0.000217`, and log-space R2 `0.999993`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.003336`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_random_case/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `6.798595e-05`, the 95th percentile was `1.262668e-03`, and `0.999615` of positive predictions were within 10% relative error. These summaries are based only on entries where the true target value is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `0.000000e+00`, the 95th percentile was `1.817443e-03`, and `0.965454` of predictions were within 10% SMAPE.

Supporting outputs:

- [`test_overall_metrics.csv`](../results/training_random_case/evaluation/test_overall_metrics.csv)
- [`test_overall_metrics_active_targets.csv`](../results/training_random_case/evaluation/test_overall_metrics_active_targets.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- [`test_relative_error_by_magnitude_bin.csv`](../results/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv)
- [`test_smape_overall_summary.csv`](../results/training_random_case/evaluation/test_smape_overall_summary.csv)
- [`test_smape_by_magnitude_bin.csv`](../results/training_random_case/evaluation/test_smape_by_magnitude_bin.csv)
- [`worst_10_reactions.csv`](../results/training_random_case/evaluation/worst_10_reactions.csv)
- [`worst_10_cases.csv`](../results/training_random_case/evaluation/worst_10_cases.csv)
- [`model_leaderboard.png`](../results/training_random_case/figures/model_leaderboard.png)
- [`oracle_reconstruction_error_by_k.png`](../results/training_random_case/figures/oracle_reconstruction_error_by_k.png)
- [`relative_error_abs_histogram.png`](../results/training_random_case/figures/relative_error_abs_histogram.png)
- [`smape_histogram.png`](../results/training_random_case/figures/smape_histogram.png)

## Top Validation Configurations

- `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.006561`, log-R2 `0.999990`, factor-5 accuracy `0.999976`
- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.008392`, log-R2 `0.999963`, factor-5 accuracy `0.999977`
- `two_stage__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.008814`, log-R2 `0.999945`, factor-5 accuracy `0.999969`
- `two_stage__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.010748`, log-R2 `0.999962`, factor-5 accuracy `0.999964`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.015590`, log-R2 `0.996910`, factor-5 accuracy `0.999974`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_random_case/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_random_case/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `H2 > H2(V=2)`: log-RMSE `0.053279`, log-MAE `0.006628`, factor-5 accuracy `0.997769`
- `O2-VIB 3`: log-RMSE `0.010425`, log-MAE `0.002289`, factor-5 accuracy `1.000000`
- `H2 > H2(V=1)`: log-RMSE `0.001973`, log-MAE `0.000639`, factor-5 accuracy `1.000000`
- `O2-VIB 2`: log-RMSE `0.001696`, log-MAE `0.000576`, factor-5 accuracy `1.000000`
- `O2-VIB 4`: log-RMSE `0.001621`, log-MAE `0.000576`, factor-5 accuracy `1.000000`
- `O-1D`: log-RMSE `0.001587`, log-MAE `0.000435`, factor-5 accuracy `1.000000`
- `O2-VIB 1`: log-RMSE `0.001392`, log-MAE `0.000503`, factor-5 accuracy `1.000000`
- `O2-ROTATIONAL`: log-RMSE `0.001319`, log-MAE `0.000431`, factor-5 accuracy `1.000000`
- `AR4P > AR4D,6S,RYD`: log-RMSE `0.001314`, log-MAE `0.000380`, factor-5 accuracy `1.000000`
- `H2-ROTATIONAL (1-3)`: log-RMSE `0.001290`, log-MAE `0.000407`, factor-5 accuracy `1.000000`

Worst test cases by log-RMSE:

- case `12821` (group `443`, local case `3`): log-RMSE `0.121722`, factor-5 accuracy `0.966667`
- case `526` (group `19`, local case `4`): log-RMSE `0.090557`, factor-5 accuracy `0.966667`
- case `525` (group `19`, local case `3`): log-RMSE `0.088095`, factor-5 accuracy `0.966667`
- case `468` (group `17`, local case `4`): log-RMSE `0.074945`, factor-5 accuracy `0.966667`
- case `2324` (group `81`, local case `4`): log-RMSE `0.064037`, factor-5 accuracy `0.966667`
- case `5833` (group `202`, local case `4`): log-RMSE `0.063821`, factor-5 accuracy `0.966667`
- case `5253` (group `182`, local case `4`): log-RMSE `0.062625`, factor-5 accuracy `0.966667`
- case `6992` (group `242`, local case `3`): log-RMSE `0.061744`, factor-5 accuracy `0.966667`
- case `6934` (group `240`, local case `3`): log-RMSE `0.060552`, factor-5 accuracy `0.966667`
- case `3426` (group `119`, local case `4`): log-RMSE `0.058363`, factor-5 accuracy `0.966667`

Worst reactions by median absolute relative error:

- `O2-VIB 1`: median abs relative error `2.911933e-04`, 95th percentile `6.029418e-03`, within 10% `1.000000`
- `O2-VIB 2`: median abs relative error `2.807019e-04`, 95th percentile `7.630787e-03`, within 10% `0.999920`
- `O2-ROTATIONAL`: median abs relative error `2.631273e-04`, 95th percentile `4.751513e-03`, within 10% `1.000000`
- `E + H2OV>O(3p5P)[777.4nm]+H2+E`: median abs relative error `2.521180e-04`, 95th percentile `8.278246e-04`, within 10% `1.000000`
- `E + H2O >O(3p5P)[777.4nm]+H2+E`: median abs relative error `2.498411e-04`, 95th percentile `8.303711e-04`, within 10% `1.000000`
- `E + H2O > H(2) [121.6nm] +OH+E`: median abs relative error `2.264750e-04`, 95th percentile `8.639212e-04`, within 10% `1.000000`
- `E + H2OV > H(2)[121.6nm] +OH+E`: median abs relative error `2.246255e-04`, 95th percentile `9.126096e-04`, within 10% `1.000000`
- `OH-E1PI`: median abs relative error `2.034121e-04`, 95th percentile `7.458637e-04`, within 10% `1.000000`
- `E + H2OV > H(4)[486.1nm] +OH+E`: median abs relative error `1.906103e-04`, 95th percentile `6.180273e-04`, within 10% `1.000000`
- `E + H2O > H(4) [486.1nm] +OH+E`: median abs relative error `1.894556e-04`, 95th percentile `6.273332e-04`, within 10% `1.000000`

Worst test cases by median absolute relative error:

- case `83059` (group `2865`, local case `3`): median abs relative error `5.056563e-02`, within 10% `0.900000`
- case `83147` (group `2868`, local case `4`): median abs relative error `3.799610e-02`, within 10% `0.933333`
- case `83234` (group `2871`, local case `4`): median abs relative error `3.565739e-02`, within 10% `0.933333`
- case `83058` (group `2865`, local case `2`): median abs relative error `2.801193e-02`, within 10% `0.965517`
- case `82857` (group `2858`, local case `4`): median abs relative error `2.695896e-02`, within 10% `0.966667`
- case `83204` (group `2870`, local case `3`): median abs relative error `2.683340e-02`, within 10% `0.933333`
- case `83291` (group `2873`, local case `3`): median abs relative error `2.652447e-02`, within 10% `0.933333`
- case `79813` (group `2753`, local case `5`): median abs relative error `2.074545e-02`, within 10% `1.000000`
- case `82074` (group `2831`, local case `4`): median abs relative error `1.986050e-02`, within 10% `0.933333`
- case `83057` (group `2865`, local case `1`): median abs relative error `1.867313e-02`, within 10% `1.000000`

Worst reactions by median SMAPE:

- `O2-VIB 1`: median SMAPE `2.911509e-04`, 95th percentile `6.032852e-03`, within 10% `1.000000`
- `O2-VIB 2`: median SMAPE `2.806625e-04`, 95th percentile `7.625074e-03`, within 10% `0.999920`
- `E + H2O >O(3p5P)[777.4nm]+H2+E`: median SMAPE `2.762026e-04`, 95th percentile `2.000000e+00`, within 10% `0.857577`
- `E + H2OV > H(2)[121.6nm] +OH+E`: median SMAPE `2.664109e-04`, 95th percentile `2.000000e+00`, within 10% `0.819091`
- `E + H2OV > OH + H + E`: median SMAPE `2.651303e-04`, 95th percentile `2.000000e+00`, within 10% `0.792767`
- `E + H2OV > H(4)[486.1nm] +OH+E`: median SMAPE `2.644756e-04`, 95th percentile `2.000000e+00`, within 10% `0.757321`
- `O2-ROTATIONAL`: median SMAPE `2.630927e-04`, 95th percentile `4.746113e-03`, within 10% `1.000000`
- `E + H2O > OH + H + E`: median SMAPE `2.560382e-04`, 95th percentile `2.000000e+00`, within 10% `0.792767`
- `E + H2O > O + 2H + E`: median SMAPE `2.422862e-04`, 95th percentile `2.000000e+00`, within 10% `0.792767`
- `02-130NM LINE EXCITE`: median SMAPE `2.179788e-04`, 95th percentile `2.000000e+00`, within 10% `0.792767`

Worst test cases by median SMAPE:

- case `25` (group `1`, local case `25`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `40` (group `2`, local case `11`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `89` (group `4`, local case `2`): median SMAPE `0.000000e+00`, within 10% `0.843137`
- case `104` (group `4`, local case `17`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `107` (group `4`, local case `20`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `111` (group `4`, local case `24`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `115` (group `4`, local case `28`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `125` (group `5`, local case `9`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `128` (group `5`, local case `12`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `129` (group `5`, local case `13`): median SMAPE `0.000000e+00`, within 10% `1.000000`

Supporting outputs:

- [`test_per_reaction_metrics.csv`](../results/training_random_case/evaluation/test_per_reaction_metrics.csv)
- [`test_per_reaction_metrics_active_targets.csv`](../results/training_random_case/evaluation/test_per_reaction_metrics_active_targets.csv)
- [`test_per_case_metrics.csv`](../results/training_random_case/evaluation/test_per_case_metrics.csv)
- [`test_per_case_metrics_active_targets.csv`](../results/training_random_case/evaluation/test_per_case_metrics_active_targets.csv)
- [`test_relative_error_per_reaction.csv`](../results/training_random_case/evaluation/test_relative_error_per_reaction.csv)
- [`test_relative_error_per_case.csv`](../results/training_random_case/evaluation/test_relative_error_per_case.csv)
- [`test_smape_per_reaction.csv`](../results/training_random_case/evaluation/test_smape_per_reaction.csv)
- [`test_smape_per_case.csv`](../results/training_random_case/evaluation/test_smape_per_case.csv)
- [`worst_reactions_log_rmse.png`](../results/training_random_case/figures/worst_reactions_log_rmse.png)
- [`per_case_log_rmse_distribution.png`](../results/training_random_case/figures/per_case_log_rmse_distribution.png)
- [`relative_error_by_magnitude.png`](../results/training_random_case/figures/relative_error_by_magnitude.png)
- [`smape_by_magnitude.png`](../results/training_random_case/figures/smape_by_magnitude.png)

## Verification

The experiment also saved explicit verification checks for parser dimensions, inverse-transform accuracy, and PCA monotonicity.

- `parsed_case_count`: status `pass`, observed `83317.0`, expected `83317`
- `parsed_input_columns`: status `pass`, observed `49.0`, expected `49`
- `parsed_input_table_rows_match_target_rows`: status `pass`, observed `83317.0`, expected `83317`
- `parsed_input_table_column_count`: status `pass`, observed `60.0`, expected `60`
- `parsed_target_columns`: status `pass`, observed `204.0`, expected `204`
- `parsed_target_table_column_count`: status `pass`, observed `214.0`, expected `214`
- `density_group_count`: status `pass`, observed `2873.0`, expected `2873`
- `e_over_n_count`: status `pass`, observed `29.0`, expected `29`
- `inverse_transform_roundtrip_trainval`: status `pass`, observed `8.470329472543003e-22`, expected `< 1e-10`
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.145220203713898`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0952623923672974`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0780179946942839`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0430629967808295`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0131250049453332`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0072360135966806`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0030162210416687`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1454739597003885`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0963769222027026`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0790675162601852`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0440649423574354`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0135359103112694`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0075225445890008`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0033133782353796`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1444699189749402`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0953324489411933`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0778236576070693`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0427530701892605`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0131832978213585`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0074229746670811`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0032422972395796`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1459107309678547`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0961419675603258`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0787067391742912`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0434670608503847`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0131620947246093`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0073949868134491`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0033343736090506`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1441841861456916`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0938936092987807`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0769429623231373`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0420027243149958`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0129614900277529`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0074431363229837`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0033135224211984`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.1451636428031815`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.0944876486196268`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.0770496199717874`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.0423708907307648`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.0129295080598092`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.007467172518813`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.0033356733406948`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `2549592.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_random_case/verification_checks.csv)
- [`pca_scree.png`](../results/training_random_case/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_random_case/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_random_case/figures/residual_hist_log_space.png)

