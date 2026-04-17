# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `/Users/bingqingwang/Desktop/UMich/590_Machine learning/project/results/ners590_v03/training_power_holdout_10mJ`
- Target family: `rate_const` (RATE CONST)
- Selected model key: `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `composition_pca_plus_log_en_log_power`
- Selected latent dimension: `nan`
- Reconstructed output width: `204` total targets, `204` modeled directly, `0` constant-zero targets restored as zeros

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

The winning configuration was `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.069993` and mean validation log-R2 `0.999836`.

On the locked test split, the final model achieved reconstructed-full-output log-space RMSE `0.048740`, log-space MAE `0.008788`, and log-space R2 `0.999603`. In original-value space, the model achieved RMSE `1.338927e-05` and MAE `1.158805e-06`.
The retained nontrivial target subset achieved log-space RMSE `0.048740`, log-space MAE `0.008788`, and log-space R2 `0.999603`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.055988`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_power_holdout_10mJ/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `5.447369e-04`, the 95th percentile was `8.963542e-02`, and `0.952536` of positive predictions were within 10% relative error. These summaries are based only on entries where the true target value is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `4.148019e-04`, the 95th percentile was `4.301195e-01`, and `0.921710` of predictions were within 10% SMAPE.

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

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.069993`, log-R2 `0.999836`, factor-5 accuracy `0.997672`
- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.082113`, log-R2 `0.975166`, factor-5 accuracy `0.997373`
- `latent__random_forest__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.134231`, log-R2 `0.993353`, factor-5 accuracy `0.995050`
- `latent__mlp__composition_pca_plus_log_en_log_power__k_8__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.138702`, log-R2 `0.993309`, factor-5 accuracy `0.995420`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.144171`, log-R2 `0.993745`, factor-5 accuracy `0.993423`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_power_holdout_10mJ/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_power_holdout_10mJ/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `E + H2OV > O^ + 2H + 2E`: log-RMSE `0.200566`, log-MAE `0.070919`, factor-5 accuracy `0.984486`
- `AR+ IONIZATION`: log-RMSE `0.184949`, log-MAE `0.065659`, factor-5 accuracy `0.990498`
- `E + H2O > O^ + 2H + 2E`: log-RMSE `0.174336`, log-MAE `0.057591`, factor-5 accuracy `0.990498`
- `E + H2OV > H2^ + O + E`: log-RMSE `0.166994`, log-MAE `0.064848`, factor-5 accuracy `0.991629`
- `E + H2O > H2^ + O + E`: log-RMSE `0.146221`, log-MAE `0.051849`, factor-5 accuracy `0.995237`
- `E + H2O > H^ + OH + 2E`: log-RMSE `0.138750`, log-MAE `0.042654`, factor-5 accuracy `0.999177`
- `E + H2OV > H^ + OH + 2E`: log-RMSE `0.134603`, log-MAE `0.040534`, factor-5 accuracy `0.999177`
- `E + H2O > H(4) [486.1nm] +OH+E`: log-RMSE `0.122053`, log-MAE `0.035161`, factor-5 accuracy `0.999177`
- `E + H2OV > H(4)[486.1nm] +OH+E`: log-RMSE `0.116615`, log-MAE `0.033060`, factor-5 accuracy `0.999177`
- `E + H2O > H(3) [656.3nm] +OH+E`: log-RMSE `0.116185`, log-MAE `0.033000`, factor-5 accuracy `0.999177`

Worst test cases by log-RMSE:

- case `76916` (group `2653`, local case `8`): log-RMSE `1.258286`, factor-5 accuracy `0.870466`
- case `76915` (group `2653`, local case `7`): log-RMSE `0.987311`, factor-5 accuracy `0.828571`
- case `76917` (group `2653`, local case `9`): log-RMSE `0.846964`, factor-5 accuracy `0.922680`
- case `76918` (group `2653`, local case `10`): log-RMSE `0.786909`, factor-5 accuracy `0.918782`
- case `76919` (group `2653`, local case `11`): log-RMSE `0.715520`, factor-5 accuracy `0.949495`
- case `76920` (group `2653`, local case `12`): log-RMSE `0.436478`, factor-5 accuracy `0.974747`
- case `76911` (group `2653`, local case `3`): log-RMSE `0.301853`, factor-5 accuracy `0.848485`
- case `76912` (group `2653`, local case `4`): log-RMSE `0.284714`, factor-5 accuracy `0.888889`
- case `76910` (group `2653`, local case `2`): log-RMSE `0.248731`, factor-5 accuracy `0.877551`
- case `76921` (group `2653`, local case `13`): log-RMSE `0.211063`, factor-5 accuracy `0.974747`

Worst reactions by median absolute relative error:

- `O2- -MOMENTUM TRANSFER`: median abs relative error `1.986740e-02`, 95th percentile `6.737142e-02`, within 10% `0.985333`
- `H3+ -ELASTIC`: median abs relative error `1.985768e-02`, 95th percentile `6.736965e-02`, within 10% `0.985333`
- `OH- -MOM`: median abs relative error `1.985764e-02`, 95th percentile `6.736910e-02`, within 10% `0.985333`
- `AR+ -ELASTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `AR2+ -MOMTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `H+ -ELASTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `H2+ -ELASTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `H2O+ -MOMTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `O2+ -ELASTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`
- `O+ -ELASTIC`: median abs relative error `1.985583e-02`, 95th percentile `6.736860e-02`, within 10% `0.985333`

Worst test cases by median absolute relative error:

- case `82739` (group `2854`, local case `2`): median abs relative error `8.637292e-02`, within 10% `0.581633`
- case `82740` (group `2854`, local case `3`): median abs relative error `8.556324e-02`, within 10% `0.585859`
- case `83029` (group `2864`, local case `2`): median abs relative error `8.384529e-02`, within 10% `0.551020`
- case `83030` (group `2864`, local case `3`): median abs relative error `8.062035e-02`, within 10% `0.545455`
- case `83031` (group `2864`, local case `4`): median abs relative error `7.998494e-02`, within 10% `0.545455`
- case `83058` (group `2865`, local case `2`): median abs relative error `7.636673e-02`, within 10% `0.551020`
- case `82738` (group `2854`, local case `1`): median abs relative error `7.310070e-02`, within 10% `0.639535`
- case `83118` (group `2867`, local case `4`): median abs relative error `7.163496e-02`, within 10% `0.606061`
- case `83001` (group `2863`, local case `3`): median abs relative error `7.153039e-02`, within 10% `0.606061`
- case `83060` (group `2865`, local case `4`): median abs relative error `7.116228e-02`, within 10% `0.565657`

Worst reactions by median SMAPE:

- `AR3S > AR^^`: median SMAPE `2.000000e+00`, 95th percentile `2.000000e+00`, within 10% `0.448276`
- `O2- -MOMENTUM TRANSFER`: median SMAPE `1.967199e-02`, 95th percentile `6.540011e-02`, within 10% `0.986425`
- `H3+ -ELASTIC`: median SMAPE `1.966246e-02`, 95th percentile `6.540007e-02`, within 10% `0.986425`
- `OH- -MOM`: median SMAPE `1.966242e-02`, 95th percentile `6.540005e-02`, within 10% `0.986425`
- `AR+ -ELASTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`
- `AR2+ -MOMTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`
- `H+ -ELASTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`
- `H2+ -ELASTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`
- `H2O+ -MOMTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`
- `O2+ -ELASTIC`: median SMAPE `1.966064e-02`, 95th percentile `6.540003e-02`, within 10% `0.986425`

Worst test cases by median SMAPE:

- case `82739` (group `2854`, local case `2`): median SMAPE `6.346772e-02`, within 10% `0.637255`
- case `83029` (group `2864`, local case `2`): median SMAPE `6.326093e-02`, within 10% `0.612745`
- case `83030` (group `2864`, local case `3`): median SMAPE `6.062255e-02`, within 10% `0.607843`
- case `83031` (group `2864`, local case `4`): median SMAPE `5.962698e-02`, within 10% `0.612745`
- case `82740` (group `2854`, local case `3`): median SMAPE `5.900272e-02`, within 10% `0.642157`
- case `83058` (group `2865`, local case `2`): median SMAPE `5.625316e-02`, within 10% `0.612745`
- case `83293` (group `2873`, local case `5`): median SMAPE `5.322206e-02`, within 10% `0.671569`
- case `83147` (group `2868`, local case `4`): median SMAPE `5.256964e-02`, within 10% `0.642157`
- case `83118` (group `2867`, local case `4`): median SMAPE `5.229386e-02`, within 10% `0.622549`
- case `83087` (group `2866`, local case `2`): median SMAPE `5.182470e-02`, within 10% `0.627451`

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
- `inverse_transform_roundtrip_trainval`: status `pass`, observed `3.469446951953614e-18`, expected `< 1e-10`
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6754802103270051`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4375142049955722`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3042159642175359`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2072282666690523`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1294930184330761`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0879476418099095`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0588678554712072`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6739424165905289`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4303981328791839`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.303215896594606`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.205971505389203`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.130097119468205`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0871739834534558`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0584717917678454`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.675147406775165`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4353966174007903`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3044441250108318`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2057301102257533`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1292663357963121`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0870698234633162`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.058468783104069`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6723507525387529`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.428502542538812`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3027135000462184`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2052455159486077`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1284380670282549`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0870746045574947`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0589763244004922`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6726159993599441`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4293014713695761`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3038711375004164`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2048354404556488`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1281110124946684`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.086279237137196`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0582663927628544`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.6974055716938801`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.4563791019389188`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.3115582695397125`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.2090366031380971`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.1311380966629304`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0868693789558996`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.0559880020537193`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `1307436.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_power_holdout_10mJ/verification_checks.csv)
- [`pca_scree.png`](../results/training_power_holdout_10mJ/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_power_holdout_10mJ/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_power_holdout_10mJ/figures/residual_hist_log_space.png)

