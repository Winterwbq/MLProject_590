# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/ners590_v02/training_random_case`
- Selected model key: `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `composition_pca_plus_log_en_log_power`
- Selected latent dimension: `nan`

Primary supporting outputs:

- [`selected_model.csv`](../results/training_random_case/tuning/selected_model.csv)
- [`model_leaderboard_summary.csv`](../results/training_random_case/tuning/model_leaderboard_summary.csv)
- [`search_stage_summary.csv`](../results/training_random_case/tuning/search_stage_summary.csv)
- [`test_overall_metrics.csv`](../results/training_random_case/evaluation/test_overall_metrics.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- [`test_smape_overall_summary.csv`](../results/training_random_case/evaluation/test_smape_overall_summary.csv)
- [`oracle_test_overall_by_k.csv`](../results/training_random_case/pca/oracle_test_overall_by_k.csv)
- [`verification_checks.csv`](../results/training_random_case/verification_checks.csv)

## Main Findings

The winning configuration was `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.048725` and mean validation log-R2 `0.999920`.

On the locked test split, the final model achieved overall log-space RMSE `0.051212`, log-space MAE `0.006550`, and log-space R2 `0.999923`. In original rate-constant space, the model achieved RMSE `2.773644e-06` and MAE `2.152017e-07`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.052094`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_random_case/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `2.006515e-04`, the 95th percentile was `2.293077e-02`, and `0.967915` of positive predictions were within 10% relative error. These summaries are based only on entries where the true rate constant is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `1.566903e-04`, the 95th percentile was `3.468701e-01`, and `0.934257` of predictions were within 10% SMAPE.

Supporting outputs:

- [`test_overall_metrics.csv`](../results/training_random_case/evaluation/test_overall_metrics.csv)
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

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.048725`, log-R2 `0.999920`, factor-5 accuracy `0.999065`
- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.058656`, log-R2 `0.975321`, factor-5 accuracy `0.999252`
- `latent__random_forest__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.121843`, log-R2 `0.994819`, factor-5 accuracy `0.996712`
- `latent__mlp__composition_pca_plus_log_en_log_power__k_8__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.124401`, log-R2 `0.994712`, factor-5 accuracy `0.997000`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.126057`, log-R2 `0.995019`, factor-5 accuracy `0.995806`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_random_case/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_random_case/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `E + H2OV > O^ + 2H + 2E`: log-RMSE `0.237548`, log-MAE `0.071026`, factor-5 accuracy `0.964120`
- `AR+ IONIZATION`: log-RMSE `0.228004`, log-MAE `0.068792`, factor-5 accuracy `0.961573`
- `E + H2O > O^ + 2H + 2E`: log-RMSE `0.210558`, log-MAE `0.057999`, factor-5 accuracy `0.970953`
- `E + H2OV > H2^ + O + E`: log-RMSE `0.176095`, log-MAE `0.060789`, factor-5 accuracy `0.961876`
- `E + H2O > H2^ + O + E`: log-RMSE `0.161789`, log-MAE `0.050351`, factor-5 accuracy `0.969600`
- `E + H2O > H^ + OH + 2E`: log-RMSE `0.157853`, log-MAE `0.044110`, factor-5 accuracy `0.990904`
- `E + H2OV > H^ + OH + 2E`: log-RMSE `0.148924`, log-MAE `0.041745`, factor-5 accuracy `0.992558`
- `E + H2O > H(4) [486.1nm] +OH+E`: log-RMSE `0.132084`, log-MAE `0.036686`, factor-5 accuracy `0.993936`
- `E + H2OV > H(4)[486.1nm] +OH+E`: log-RMSE `0.122131`, log-MAE `0.034345`, factor-5 accuracy `0.996141`
- `E + H2O > H(3) [656.3nm] +OH+E`: log-RMSE `0.117826`, log-MAE `0.033765`, factor-5 accuracy `0.996968`

Worst test cases by log-RMSE:

- case `7` (group `1`, local case `7`): log-RMSE `0.796820`, factor-5 accuracy `0.857143`
- case `25646` (group `885`, local case `10`): log-RMSE `0.773440`, factor-5 accuracy `0.918782`
- case `19237` (group `664`, local case `10`): log-RMSE `0.702976`, factor-5 accuracy `0.918782`
- case `9` (group `1`, local case `9`): log-RMSE `0.616697`, factor-5 accuracy `0.927835`
- case `25643` (group `885`, local case `7`): log-RMSE `0.511692`, factor-5 accuracy `0.885714`
- case `6419` (group `222`, local case `10`): log-RMSE `0.495345`, factor-5 accuracy `0.934010`
- case `6420` (group `222`, local case `11`): log-RMSE `0.441166`, factor-5 accuracy `0.959596`
- case `19238` (group `664`, local case `11`): log-RMSE `0.438276`, factor-5 accuracy `0.959596`
- case `906` (group `32`, local case `7`): log-RMSE `0.312694`, factor-5 accuracy `0.880000`
- case `31` (group `2`, local case `2`): log-RMSE `0.303857`, factor-5 accuracy `0.846939`

Worst reactions by median absolute relative error:

- `E + H2O > H2^ + O + E`: median abs relative error `7.941553e-03`, 95th percentile `8.807683e-01`, within 10% `0.780800`
- `E + H2OV > H2^ + O + E`: median abs relative error `7.871646e-03`, 95th percentile `1.757592e+00`, within 10% `0.739486`
- `AR+ IONIZATION`: median abs relative error `7.511201e-03`, 95th percentile `1.229740e+00`, within 10% `0.744629`
- `E + H2O > O^ + 2H + 2E`: median abs relative error `6.721466e-03`, 95th percentile `9.204899e-01`, within 10% `0.757035`
- `E + H2OV > O^ + 2H + 2E`: median abs relative error `6.622615e-03`, 95th percentile `1.003992e+00`, within 10% `0.733796`
- `AR3S > AR^^`: median abs relative error `6.312157e-03`, 95th percentile `3.022655e-02`, within 10% `1.000000`
- `E + H2OV > H^ + OH + 2E`: median abs relative error `5.017047e-03`, 95th percentile `6.701515e-01`, within 10% `0.786935`
- `E + H2O > H^ + OH + 2E`: median abs relative error `5.004061e-03`, 95th percentile `6.857776e-01`, within 10% `0.781698`
- `O + O^ + E + E`: median abs relative error `4.707860e-03`, 95th percentile `4.642745e-01`, within 10% `0.837652`
- `E + H2O >O(3p5P)[777.4nm]+H2+E`: median abs relative error `4.483697e-03`, 95th percentile `5.637789e-01`, within 10% `0.828831`

Worst test cases by median absolute relative error:

- case `25873` (group `893`, local case `5`): median abs relative error `2.000658e-02`, within 10% `1.000000`
- case `25090` (group `866`, local case `5`): median abs relative error `1.810763e-02`, within 10% `0.868687`
- case `25903` (group `894`, local case `6`): median abs relative error `1.792195e-02`, within 10% `1.000000`
- case `24478` (group `845`, local case `2`): median abs relative error `1.785538e-02`, within 10% `0.826531`
- case `25874` (group `893`, local case `6`): median abs relative error `1.752257e-02`, within 10% `1.000000`
- case `30919` (group `1067`, local case `5`): median abs relative error `1.653850e-02`, within 10% `0.858586`
- case `25058` (group `865`, local case `2`): median abs relative error `1.641622e-02`, within 10% `0.908163`
- case `31467` (group `1086`, local case `2`): median abs relative error `1.558527e-02`, within 10% `0.877551`
- case `25957` (group `896`, local case `2`): median abs relative error `1.530653e-02`, within 10% `0.765306`
- case `15781` (group `545`, local case `5`): median abs relative error `1.442038e-02`, within 10% `1.000000`

Worst reactions by median SMAPE:

- `AR3S > AR^^`: median SMAPE `2.000000e+00`, 95th percentile `2.000000e+00`, within 10% `0.434783`
- `E + H2O > H^ + OH + 2E`: median SMAPE `4.374115e-03`, 95th percentile `6.040491e-01`, within 10% `0.834408`
- `E + H2OV > H^ + OH + 2E`: median SMAPE `4.150562e-03`, 95th percentile `5.852218e-01`, within 10% `0.838777`
- `OH- -MOM`: median SMAPE `3.854266e-03`, 95th percentile `1.465255e-02`, within 10% `0.996463`
- `AR+ -ELASTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`
- `AR2+ -MOMTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`
- `H+ -ELASTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`
- `H2+ -ELASTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`
- `H2O+ -MOMTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`
- `O2+ -ELASTIC`: median SMAPE `3.854265e-03`, 95th percentile `1.465254e-02`, within 10% `0.996463`

Worst test cases by median SMAPE:

- case `25873` (group `893`, local case `5`): median SMAPE `1.669464e-02`, within 10% `0.828431`
- case `25090` (group `866`, local case `5`): median SMAPE `1.571085e-02`, within 10% `0.759804`
- case `25903` (group `894`, local case `6`): median SMAPE `1.518663e-02`, within 10% `0.877451`
- case `25874` (group `893`, local case `6`): median SMAPE `1.419196e-02`, within 10% `0.877451`
- case `30919` (group `1067`, local case `5`): median SMAPE `1.386927e-02`, within 10% `0.759804`
- case `25058` (group `865`, local case `2`): median SMAPE `1.335230e-02`, within 10% `0.784314`
- case `24478` (group `845`, local case `2`): median SMAPE `1.288686e-02`, within 10% `0.745098`
- case `15781` (group `545`, local case `5`): median SMAPE `1.230114e-02`, within 10% `0.813725`
- case `27062` (group `934`, local case `5`): median SMAPE `1.154905e-02`, within 10% `0.828431`
- case `31467` (group `1086`, local case `2`): median SMAPE `1.129105e-02`, within 10% `0.769608`

Supporting outputs:

- [`test_per_reaction_metrics.csv`](../results/training_random_case/evaluation/test_per_reaction_metrics.csv)
- [`test_per_case_metrics.csv`](../results/training_random_case/evaluation/test_per_case_metrics.csv)
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

- `parsed_case_count`: status `pass`, observed `32045.0`, expected `32045`
- `parsed_input_columns`: status `pass`, observed `49.0`, expected `49`
- `parsed_input_table_rows_match_target_rows`: status `pass`, observed `32045.0`, expected `32045`
- `parsed_input_table_column_count`: status `pass`, observed `60.0`, expected `60`
- `parsed_target_columns`: status `pass`, observed `204.0`, expected `204`
- `parsed_target_table_column_count`: status `pass`, observed `214.0`, expected `214`
- `density_group_count`: status `pass`, observed `1105.0`, expected `1105`
- `e_over_n_count`: status `pass`, observed `29.0`, expected `29`
- `inverse_transform_roundtrip_trainval`: status `pass`, observed `3.469446951953614e-18`, expected `< 1e-10`
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6661571944906162`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4283736711679209`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2993211354128356`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1949814194212042`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.117999201333042`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0770376346453737`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0508845947630076`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.659987695632981`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.429112745099145`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2977771127661354`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1955802271130433`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.119777397705693`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0780007747515774`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0529556596723012`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6605334726423633`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4253519330125135`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2964478473011234`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1906940801066722`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1168558334382199`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0763554771361371`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0512452978301915`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6608474370128291`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4173407647448708`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2956356043073984`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1929883131420378`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1184195754498452`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0769256658230563`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0512789635524886`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6762256265641676`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4274041781441789`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2959713026574996`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1931415239043005`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1177631337581306`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0780173415187879`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0520126078072411`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.6711659743689368`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.4234535415405835`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.3014434628231989`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.1981063574759484`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.1204913732414069`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0795939527121956`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.0520943915273699`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `980628.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_random_case/verification_checks.csv)
- [`pca_scree.png`](../results/training_random_case/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_random_case/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_random_case/figures/residual_hist_log_space.png)

