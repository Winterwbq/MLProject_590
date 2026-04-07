# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/ners590_v02/training_power_holdout_5mJ`
- Selected model key: `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `composition_pca_plus_log_en_log_power`
- Selected latent dimension: `nan`

Primary supporting outputs:

- [`selected_model.csv`](../results/training_power_holdout_5mJ/tuning/selected_model.csv)
- [`model_leaderboard_summary.csv`](../results/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- [`search_stage_summary.csv`](../results/training_power_holdout_5mJ/tuning/search_stage_summary.csv)
- [`test_overall_metrics.csv`](../results/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv)
- [`test_smape_overall_summary.csv`](../results/training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv)
- [`oracle_test_overall_by_k.csv`](../results/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv)
- [`verification_checks.csv`](../results/training_power_holdout_5mJ/verification_checks.csv)

## Main Findings

The winning configuration was `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.048902` and mean validation log-R2 `0.999917`.

On the locked test split, the final model achieved overall log-space RMSE `0.047780`, log-space MAE `0.008852`, and log-space R2 `0.999722`. In original rate-constant space, the model achieved RMSE `1.288207e-05` and MAE `1.025436e-06`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.051636`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `5.360493e-04`, the 95th percentile was `8.028459e-02`, and `0.952621` of positive predictions were within 10% relative error. These summaries are based only on entries where the true rate constant is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `3.884313e-04`, the 95th percentile was `3.463052e-01`, and `0.927681` of predictions were within 10% SMAPE.

Supporting outputs:

- [`test_overall_metrics.csv`](../results/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv)
- [`test_relative_error_overall_summary.csv`](../results/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv)
- [`test_relative_error_by_magnitude_bin.csv`](../results/training_power_holdout_5mJ/evaluation/test_relative_error_by_magnitude_bin.csv)
- [`test_smape_overall_summary.csv`](../results/training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv)
- [`test_smape_by_magnitude_bin.csv`](../results/training_power_holdout_5mJ/evaluation/test_smape_by_magnitude_bin.csv)
- [`worst_10_reactions.csv`](../results/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv)
- [`worst_10_cases.csv`](../results/training_power_holdout_5mJ/evaluation/worst_10_cases.csv)
- [`model_leaderboard.png`](../results/training_power_holdout_5mJ/figures/model_leaderboard.png)
- [`oracle_reconstruction_error_by_k.png`](../results/training_power_holdout_5mJ/figures/oracle_reconstruction_error_by_k.png)
- [`relative_error_abs_histogram.png`](../results/training_power_holdout_5mJ/figures/relative_error_abs_histogram.png)
- [`smape_histogram.png`](../results/training_power_holdout_5mJ/figures/smape_histogram.png)

## Top Validation Configurations

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.048902`, log-R2 `0.999917`, factor-5 accuracy `0.999002`
- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.055362`, log-R2 `0.975328`, factor-5 accuracy `0.999303`
- `latent__random_forest__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.121544`, log-R2 `0.994806`, factor-5 accuracy `0.996755`
- `latent__mlp__composition_pca_plus_log_en_log_power__k_8__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.123854`, log-R2 `0.994720`, factor-5 accuracy `0.996984`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.127107`, log-R2 `0.995013`, factor-5 accuracy `0.995599`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_power_holdout_5mJ/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `E + H2OV > O^ + 2H + 2E`: log-RMSE `0.197593`, log-MAE `0.068568`, factor-5 accuracy `0.967895`
- `AR+ IONIZATION`: log-RMSE `0.185442`, log-MAE `0.063727`, factor-5 accuracy `0.969457`
- `E + H2OV > H2^ + O + E`: log-RMSE `0.181162`, log-MAE `0.066159`, factor-5 accuracy `0.974434`
- `E + H2O > O^ + 2H + 2E`: log-RMSE `0.165360`, log-MAE `0.054592`, factor-5 accuracy `0.977828`
- `E + H2O > H2^ + O + E`: log-RMSE `0.157914`, log-MAE `0.052223`, factor-5 accuracy `0.983329`
- `E + H2O > H^ + OH + 2E`: log-RMSE `0.122011`, log-MAE `0.043356`, factor-5 accuracy `0.995064`
- `E + H2OV > H^ + OH + 2E`: log-RMSE `0.117431`, log-MAE `0.041157`, factor-5 accuracy `0.995269`
- `E + H2O > H(4) [486.1nm] +OH+E`: log-RMSE `0.107146`, log-MAE `0.035999`, factor-5 accuracy `0.995886`
- `E + H2OV > H(4)[486.1nm] +OH+E`: log-RMSE `0.103013`, log-MAE `0.034074`, factor-5 accuracy `0.996092`
- `E + H2O > H(3) [656.3nm] +OH+E`: log-RMSE `0.102665`, log-MAE `0.034042`, factor-5 accuracy `0.995886`

Worst test cases by log-RMSE:

- case `25645` (group `885`, local case `9`): log-RMSE `0.398103`, factor-5 accuracy `0.932990`
- case `25644` (group `885`, local case `8`): log-RMSE `0.378203`, factor-5 accuracy `0.917098`
- case `25647` (group `885`, local case `11`): log-RMSE `0.367246`, factor-5 accuracy `0.964646`
- case `25646` (group `885`, local case `10`): log-RMSE `0.359950`, factor-5 accuracy `0.944162`
- case `31736` (group `1095`, local case `10`): log-RMSE `0.332161`, factor-5 accuracy `0.913706`
- case `31734` (group `1095`, local case `8`): log-RMSE `0.323402`, factor-5 accuracy `0.865285`
- case `31733` (group `1095`, local case `7`): log-RMSE `0.321680`, factor-5 accuracy `0.862857`
- case `31156` (group `1075`, local case `10`): log-RMSE `0.310048`, factor-5 accuracy `0.918782`
- case `31737` (group `1095`, local case `11`): log-RMSE `0.306226`, factor-5 accuracy `0.929293`
- case `31154` (group `1075`, local case `8`): log-RMSE `0.301910`, factor-5 accuracy `0.875648`

Worst reactions by median absolute relative error:

- `H3+ -ELASTIC`: median abs relative error `1.521412e-02`, 95th percentile `3.727634e-02`, within 10% `0.995163`
- `AR+ -ELASTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `AR2+ -MOMTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `H+ -ELASTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `H2O+ -MOMTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `O2+ -ELASTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `O+ -ELASTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `O- ELASTIC`: median abs relative error `1.520853e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `H2+ -ELASTIC`: median abs relative error `1.520560e-02`, 95th percentile `3.727633e-02`, within 10% `0.995163`
- `O2- -MOMENTUM TRANSFER`: median abs relative error `1.520470e-02`, 95th percentile `3.727634e-02`, within 10% `0.995163`

Worst test cases by median absolute relative error:

- case `31728` (group `1095`, local case `2`): median abs relative error `6.105734e-02`, within 10% `0.591837`
- case `31729` (group `1095`, local case `3`): median abs relative error `6.013025e-02`, within 10% `0.585859`
- case `31583` (group `1090`, local case `2`): median abs relative error `5.916854e-02`, within 10% `0.591837`
- case `31612` (group `1091`, local case `2`): median abs relative error `5.767799e-02`, within 10% `0.591837`
- case `31584` (group `1090`, local case `3`): median abs relative error `5.642398e-02`, within 10% `0.585859`
- case `31613` (group `1091`, local case `3`): median abs relative error `5.488298e-02`, within 10% `0.585859`
- case `31582` (group `1090`, local case `1`): median abs relative error `5.407747e-02`, within 10% `0.662791`
- case `31148` (group `1075`, local case `2`): median abs relative error `5.337777e-02`, within 10% `0.714286`
- case `31641` (group `1092`, local case `2`): median abs relative error `5.308809e-02`, within 10% `0.591837`
- case `31032` (group `1071`, local case `2`): median abs relative error `5.259887e-02`, within 10% `0.591837`

Worst reactions by median SMAPE:

- `AR3S > AR^^`: median SMAPE `2.000000e+00`, 95th percentile `2.000000e+00`, within 10% `0.448276`
- `H3+ -ELASTIC`: median SMAPE `1.509955e-02`, 95th percentile `3.659429e-02`, within 10% `0.996255`
- `H2+ -ELASTIC`: median SMAPE `1.509434e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `AR+ -ELASTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `AR2+ -MOMTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `H+ -ELASTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `H2O+ -MOMTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `O2+ -ELASTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `O+ -ELASTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`
- `O- ELASTIC`: median SMAPE `1.509433e-02`, 95th percentile `3.659428e-02`, within 10% `0.996255`

Worst test cases by median SMAPE:

- case `31729` (group `1095`, local case `3`): median SMAPE `4.065384e-02`, within 10% `0.651961`
- case `31728` (group `1095`, local case `2`): median SMAPE `3.899051e-02`, within 10% `0.651961`
- case `31583` (group `1090`, local case `2`): median SMAPE `3.804859e-02`, within 10% `0.656863`
- case `31584` (group `1090`, local case `3`): median SMAPE `3.794231e-02`, within 10% `0.656863`
- case `31613` (group `1091`, local case `3`): median SMAPE `3.685937e-02`, within 10% `0.656863`
- case `31612` (group `1091`, local case `2`): median SMAPE `3.667184e-02`, within 10% `0.661765`
- case `31149` (group `1075`, local case `3`): median SMAPE `3.503010e-02`, within 10% `0.715686`
- case `31641` (group `1092`, local case `2`): median SMAPE `3.388116e-02`, within 10% `0.720588`
- case `31148` (group `1075`, local case `2`): median SMAPE `3.386789e-02`, within 10% `0.720588`
- case `31699` (group `1094`, local case `2`): median SMAPE `3.344113e-02`, within 10% `0.720588`

Supporting outputs:

- [`test_per_reaction_metrics.csv`](../results/training_power_holdout_5mJ/evaluation/test_per_reaction_metrics.csv)
- [`test_per_case_metrics.csv`](../results/training_power_holdout_5mJ/evaluation/test_per_case_metrics.csv)
- [`test_relative_error_per_reaction.csv`](../results/training_power_holdout_5mJ/evaluation/test_relative_error_per_reaction.csv)
- [`test_relative_error_per_case.csv`](../results/training_power_holdout_5mJ/evaluation/test_relative_error_per_case.csv)
- [`test_smape_per_reaction.csv`](../results/training_power_holdout_5mJ/evaluation/test_smape_per_reaction.csv)
- [`test_smape_per_case.csv`](../results/training_power_holdout_5mJ/evaluation/test_smape_per_case.csv)
- [`worst_reactions_log_rmse.png`](../results/training_power_holdout_5mJ/figures/worst_reactions_log_rmse.png)
- [`per_case_log_rmse_distribution.png`](../results/training_power_holdout_5mJ/figures/per_case_log_rmse_distribution.png)
- [`relative_error_by_magnitude.png`](../results/training_power_holdout_5mJ/figures/relative_error_by_magnitude.png)
- [`smape_by_magnitude.png`](../results/training_power_holdout_5mJ/figures/smape_by_magnitude.png)

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
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6637667667622489`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4352744247147829`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2929465771169011`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1926637407303409`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1176391381356208`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0779098949889263`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0516401118062742`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6668773251838682`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.424576828652291`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3000985583318899`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1986133447943899`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1225729917893461`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0795384972339411`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0536817130078623`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6728342028641805`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4332011057579904`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2955057563422308`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1944763231831998`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.117613222612672`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0774996478090163`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0510382765564445`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6663253441419771`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4336512205443847`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2980719074324075`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1955838475847109`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1167784906988928`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.077815004254386`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0526387584834173`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6655420555342806`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4239503918339984`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2955302629227717`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1959235417455336`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1191737478470472`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0793374935761476`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0530181877734044`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.6776351125737791`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.4366227073162013`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.2977032333081518`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.1929604095309606`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.1169468366034504`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0764577181255589`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.0516356237024159`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `1307436.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_power_holdout_5mJ/verification_checks.csv)
- [`pca_scree.png`](../results/training_power_holdout_5mJ/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_power_holdout_5mJ/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_power_holdout_5mJ/figures/residual_hist_log_space.png)

