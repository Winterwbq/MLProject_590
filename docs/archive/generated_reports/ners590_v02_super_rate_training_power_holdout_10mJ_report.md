# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/ners590_v03_super_rate/training_power_holdout_10mJ`
- Target family: `super_rate` (SUPER RATE)
- Selected model key: `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `all_inputs_plus_log_en_log_power`
- Selected latent dimension: `nan`
- Reconstructed output width: `204` total targets, `94` modeled directly, `110` constant-zero targets restored as zeros

Primary supporting outputs:

- [`selected_model.csv`](../results/training_power_holdout_10mJ/tuning/selected_model.csv)
- [`model_leaderboard_summary.csv`](../results/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv)
- [`search_stage_summary.csv`](../results/training_power_holdout_10mJ/tuning/search_stage_summary.csv)
- [`test_overall_metrics.csv`](../results/training_power_holdout_10mJ/evaluation/test_overall_metrics.csv)
- [`test_overall_metrics_active_targets.csv`](../results/training_power_holdout_10mJ/evaluation/test_overall_metrics_active_targets.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_power_holdout_10mJ/evaluation/test_relative_error_overall_summary.csv)
- [`test_smape_overall_summary.csv`](../results/training_power_holdout_10mJ/evaluation/test_smape_overall_summary.csv)
- [`oracle_test_overall_by_k.csv`](../results/training_power_holdout_10mJ/pca/oracle_test_overall_by_k.csv)
- [`target_metadata.csv`](../results/training_power_holdout_10mJ/data_snapshots/target_metadata.csv)
- [`verification_checks.csv`](../results/training_power_holdout_10mJ/verification_checks.csv)

## Main Findings

The winning configuration was `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.006036` and mean validation log-R2 `0.999992`.

On the locked test split, the final model achieved reconstructed-full-output log-space RMSE `0.003621`, log-space MAE `0.000282`, and log-space R2 `0.999941`. In original-value space, the model achieved RMSE `6.217041e-11` and MAE `4.691361e-12`.
The retained nontrivial target subset achieved log-space RMSE `0.005335`, log-space MAE `0.000612`, and log-space R2 `0.999873`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.005466`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_power_holdout_10mJ/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `6.705441e-05`, the 95th percentile was `4.342742e-03`, and `0.997390` of positive predictions were within 10% relative error. These summaries are based only on entries where the true target value is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `0.000000e+00`, the 95th percentile was `1.236085e-02`, and `0.958486` of predictions were within 10% SMAPE.

Supporting outputs:

- [`test_overall_metrics.csv`](../results/training_power_holdout_10mJ/evaluation/test_overall_metrics.csv)
- [`test_overall_metrics_active_targets.csv`](../results/training_power_holdout_10mJ/evaluation/test_overall_metrics_active_targets.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_power_holdout_10mJ/evaluation/test_relative_error_overall_summary.csv)
- [`test_relative_error_by_magnitude_bin.csv`](../results/training_power_holdout_10mJ/evaluation/test_relative_error_by_magnitude_bin.csv)
- [`test_smape_overall_summary.csv`](../results/training_power_holdout_10mJ/evaluation/test_smape_overall_summary.csv)
- [`test_smape_by_magnitude_bin.csv`](../results/training_power_holdout_10mJ/evaluation/test_smape_by_magnitude_bin.csv)
- [`worst_10_reactions.csv`](../results/training_power_holdout_10mJ/evaluation/worst_10_reactions.csv)
- [`worst_10_cases.csv`](../results/training_power_holdout_10mJ/evaluation/worst_10_cases.csv)
- [`model_leaderboard.png`](../results/training_power_holdout_10mJ/figures/model_leaderboard.png)
- [`oracle_reconstruction_error_by_k.png`](../results/training_power_holdout_10mJ/figures/oracle_reconstruction_error_by_k.png)
- [`relative_error_abs_histogram.png`](../results/training_power_holdout_10mJ/figures/relative_error_abs_histogram.png)
- [`smape_histogram.png`](../results/training_power_holdout_10mJ/figures/smape_histogram.png)

## Top Validation Configurations

- `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.006036`, log-R2 `0.999992`, factor-5 accuracy `0.999974`
- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.008577`, log-R2 `0.999970`, factor-5 accuracy `0.999967`
- `two_stage__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.008675`, log-R2 `0.999968`, factor-5 accuracy `0.999965`
- `two_stage__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.009305`, log-R2 `0.999981`, factor-5 accuracy `0.999969`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.015599`, log-R2 `0.996960`, factor-5 accuracy `0.999962`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_power_holdout_10mJ/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `H2 > H2(V=2)`: log-RMSE `0.032435`, log-MAE `0.006860`, factor-5 accuracy `1.000000`
- `O2-VIB 3`: log-RMSE `0.026915`, log-MAE `0.007214`, factor-5 accuracy `1.000000`
- `H2 > H2(V=1)`: log-RMSE `0.010364`, log-MAE `0.002799`, factor-5 accuracy `1.000000`
- `O2-VIB 2`: log-RMSE `0.008733`, log-MAE `0.002565`, factor-5 accuracy `1.000000`
- `O-1D`: log-RMSE `0.008395`, log-MAE `0.001943`, factor-5 accuracy `1.000000`
- `O2-VIB 1`: log-RMSE `0.008055`, log-MAE `0.002304`, factor-5 accuracy `1.000000`
- `O2-VIB 4`: log-RMSE `0.008018`, log-MAE `0.002469`, factor-5 accuracy `1.000000`
- `H2-ROTATIONAL (1-3)`: log-RMSE `0.007742`, log-MAE `0.002076`, factor-5 accuracy `1.000000`
- `O2-ROTATIONAL`: log-RMSE `0.007243`, log-MAE `0.001989`, factor-5 accuracy `1.000000`
- `AR4P > AR4D,6S,RYD`: log-RMSE `0.006914`, log-MAE `0.001587`, factor-5 accuracy `1.000000`

Worst test cases by log-RMSE:

- case `76912` (group `2653`, local case `4`): log-RMSE `0.041751`, factor-5 accuracy `1.000000`
- case `83292` (group `2873`, local case `4`): log-RMSE `0.040933`, factor-5 accuracy `1.000000`
- case `83205` (group `2870`, local case `4`): log-RMSE `0.040924`, factor-5 accuracy `1.000000`
- case `83263` (group `2872`, local case `4`): log-RMSE `0.040568`, factor-5 accuracy `1.000000`
- case `83234` (group `2871`, local case `4`): log-RMSE `0.040300`, factor-5 accuracy `1.000000`
- case `83176` (group `2869`, local case `4`): log-RMSE `0.038925`, factor-5 accuracy `1.000000`
- case `83147` (group `2868`, local case `4`): log-RMSE `0.036548`, factor-5 accuracy `1.000000`
- case `83204` (group `2870`, local case `3`): log-RMSE `0.035068`, factor-5 accuracy `1.000000`
- case `83118` (group `2867`, local case `4`): log-RMSE `0.034782`, factor-5 accuracy `1.000000`
- case `83262` (group `2872`, local case `3`): log-RMSE `0.034624`, factor-5 accuracy `1.000000`

Worst reactions by median absolute relative error:

- `O2-VIB 1`: median abs relative error `3.561477e-04`, 95th percentile `2.682113e-02`, within 10% `0.990638`
- `O2-VIB 2`: median abs relative error `3.295954e-04`, 95th percentile `3.289804e-02`, within 10% `0.990482`
- `O2-ROTATIONAL`: median abs relative error `3.262931e-04`, 95th percentile `2.167747e-02`, within 10% `0.992511`
- `E + H2O >O(3p5P)[777.4nm]+H2+E`: median abs relative error `1.665744e-04`, 95th percentile `1.593142e-03`, within 10% `1.000000`
- `E + H2OV>O(3p5P)[777.4nm]+H2+E`: median abs relative error `1.657777e-04`, 95th percentile `1.490519e-03`, within 10% `1.000000`
- `OH-E1PI`: median abs relative error `1.638235e-04`, 95th percentile `2.153785e-03`, within 10% `1.000000`
- `E + H2OV > H(2)[121.6nm] +OH+E`: median abs relative error `1.602849e-04`, 95th percentile `1.199655e-03`, within 10% `1.000000`
- `O2-VIB 4`: median abs relative error `1.590417e-04`, 95th percentile `3.209715e-02`, within 10% `0.991418`
- `E + H2O > H(2) [121.6nm] +OH+E`: median abs relative error `1.527895e-04`, 95th percentile `1.153366e-03`, within 10% `1.000000`
- `O2-ELECTRONIC 10EV`: median abs relative error `1.489638e-04`, 95th percentile `1.577253e-03`, within 10% `1.000000`

Worst test cases by median absolute relative error:

- case `83117` (group `2867`, local case `3`): median abs relative error `1.133196e-01`, within 10% `0.400000`
- case `83030` (group `2864`, local case `3`): median abs relative error `1.119794e-01`, within 10% `0.400000`
- case `83205` (group `2870`, local case `4`): median abs relative error `1.117398e-01`, within 10% `0.366667`
- case `83088` (group `2866`, local case `3`): median abs relative error `1.111339e-01`, within 10% `0.433333`
- case `83118` (group `2867`, local case `4`): median abs relative error `1.107638e-01`, within 10% `0.366667`
- case `83176` (group `2869`, local case `4`): median abs relative error `1.099433e-01`, within 10% `0.366667`
- case `83146` (group `2868`, local case `3`): median abs relative error `1.098264e-01`, within 10% `0.400000`
- case `83234` (group `2871`, local case `4`): median abs relative error `1.089463e-01`, within 10% `0.400000`
- case `83147` (group `2868`, local case `4`): median abs relative error `1.081875e-01`, within 10% `0.366667`
- case `83292` (group `2873`, local case `4`): median abs relative error `1.078146e-01`, within 10% `0.433333`

Worst reactions by median SMAPE:

- `O2-VIB 1`: median SMAPE `3.562111e-04`, 95th percentile `2.718570e-02`, within 10% `0.990170`
- `O2-VIB 2`: median SMAPE `3.296497e-04`, 95th percentile `3.344823e-02`, within 10% `0.988922`
- `O2-ROTATIONAL`: median SMAPE `3.263463e-04`, 95th percentile `2.191500e-02`, within 10% `0.991730`
- `E + H2OV > H(2)[121.6nm] +OH+E`: median SMAPE `2.970336e-04`, 95th percentile `2.000000e+00`, within 10% `0.772507`
- `E + H2O >O(3p5P)[777.4nm]+H2+E`: median SMAPE `2.409099e-04`, 95th percentile `2.000000e+00`, within 10% `0.794820`
- `O2-ELECTRONIC 10EV`: median SMAPE `2.297009e-04`, 95th percentile `2.000000e+00`, within 10% `0.793103`
- `E + H2O > O + 2H + E`: median SMAPE `2.102358e-04`, 95th percentile `2.000000e+00`, within 10% `0.793103`
- `E + H2OV > OH + H + E`: median SMAPE `2.084156e-04`, 95th percentile `2.000000e+00`, within 10% `0.793103`
- `E + H2OV > H(4)[486.1nm] +OH+E`: median SMAPE `2.054707e-04`, 95th percentile `2.000000e+00`, within 10% `0.758621`
- `02-130NM LINE EXCITE`: median SMAPE `1.888766e-04`, 95th percentile `2.000000e+00`, within 10% `0.793103`

Worst test cases by median SMAPE:

- case `76909` (group `2653`, local case `1`): median SMAPE `0.000000e+00`, within 10% `0.779412`
- case `76910` (group `2653`, local case `2`): median SMAPE `0.000000e+00`, within 10% `0.808824`
- case `76911` (group `2653`, local case `3`): median SMAPE `0.000000e+00`, within 10% `0.799020`
- case `76912` (group `2653`, local case `4`): median SMAPE `0.000000e+00`, within 10% `0.803922`
- case `76913` (group `2653`, local case `5`): median SMAPE `0.000000e+00`, within 10% `0.818627`
- case `76914` (group `2653`, local case `6`): median SMAPE `0.000000e+00`, within 10% `0.867647`
- case `76915` (group `2653`, local case `7`): median SMAPE `0.000000e+00`, within 10% `0.970588`
- case `76916` (group `2653`, local case `8`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `76917` (group `2653`, local case `9`): median SMAPE `0.000000e+00`, within 10% `1.000000`
- case `76918` (group `2653`, local case `10`): median SMAPE `0.000000e+00`, within 10% `1.000000`

Supporting outputs:

- [`test_per_reaction_metrics.csv`](../results/training_power_holdout_10mJ/evaluation/test_per_reaction_metrics.csv)
- [`test_per_reaction_metrics_active_targets.csv`](../results/training_power_holdout_10mJ/evaluation/test_per_reaction_metrics_active_targets.csv)
- [`test_per_case_metrics.csv`](../results/training_power_holdout_10mJ/evaluation/test_per_case_metrics.csv)
- [`test_per_case_metrics_active_targets.csv`](../results/training_power_holdout_10mJ/evaluation/test_per_case_metrics_active_targets.csv)
- [`test_relative_error_per_reaction.csv`](../results/training_power_holdout_10mJ/evaluation/test_relative_error_per_reaction.csv)
- [`test_relative_error_per_case.csv`](../results/training_power_holdout_10mJ/evaluation/test_relative_error_per_case.csv)
- [`test_smape_per_reaction.csv`](../results/training_power_holdout_10mJ/evaluation/test_smape_per_reaction.csv)
- [`test_smape_per_case.csv`](../results/training_power_holdout_10mJ/evaluation/test_smape_per_case.csv)
- [`worst_reactions_log_rmse.png`](../results/training_power_holdout_10mJ/figures/worst_reactions_log_rmse.png)
- [`per_case_log_rmse_distribution.png`](../results/training_power_holdout_10mJ/figures/per_case_log_rmse_distribution.png)
- [`relative_error_by_magnitude.png`](../results/training_power_holdout_10mJ/figures/relative_error_by_magnitude.png)
- [`smape_by_magnitude.png`](../results/training_power_holdout_10mJ/figures/smape_by_magnitude.png)

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
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1463300610252359`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0951187999655733`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.07816608095794`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0427857014671647`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0131563658392434`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0073216437777442`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0030481000961471`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1441186994787479`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0945598641201386`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0773828797288784`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0428026978682467`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.012991425913648`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0071173170301202`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0030038910072323`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.145201035059728`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0945666352888713`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0774350053379713`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0423444498328759`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0129797265221268`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0072806096417634`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0029677691794768`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1445431538069483`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0948518629191116`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0776016765359235`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0426241145233715`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0130508174517771`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0073082070026023`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0030580467515338`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.1446076964662924`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.0949641412766781`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.0777457618181712`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.0428705537738784`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.0130254985799913`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.007099958940456`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0030092069944818`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.1486902161552038`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.0970190959372813`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.0792751607603635`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.0445138259138676`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.0148069897322297`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0101074065647075`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.0054657938754009`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `1307436.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_power_holdout_10mJ/verification_checks.csv)
- [`pca_scree.png`](../results/training_power_holdout_10mJ/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_power_holdout_10mJ/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_power_holdout_10mJ/figures/residual_hist_log_space.png)

