# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/ners590_v03/training_random_case`
- Target family: `rate_const` (RATE CONST)
- Selected model key: `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`
- Selected model family: `extra_trees`
- Selected feature set: `composition_pca_plus_log_en_log_power`
- Selected latent dimension: `nan`
- Reconstructed output width: `204` total targets, `204` modeled directly, `0` constant-zero targets restored as zeros

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

The winning configuration was `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.069550` and mean validation log-R2 `0.999828`.

On the locked test split, the final model achieved reconstructed-full-output log-space RMSE `0.068483`, log-space MAE `0.009375`, and log-space R2 `0.999839`. In original-value space, the model achieved RMSE `5.025485e-06` and MAE `3.783399e-07`.
The retained nontrivial target subset achieved log-space RMSE `0.068483`, log-space MAE `0.009375`, and log-space R2 `0.999839`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.057895`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/training_random_case/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `1.925288e-04`, the 95th percentile was `4.272989e-02`, and `0.960872` of positive predictions were within 10% relative error. These summaries are based only on entries where the true target value is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `1.473644e-04`, the 95th percentile was `3.918116e-01`, and `0.932523` of predictions were within 10% SMAPE.

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

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`: mean validation log-RMSE `0.069550`, log-R2 `0.999828`, factor-5 accuracy `0.997798`
- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.085413`, log-R2 `0.975117`, factor-5 accuracy `0.997290`
- `latent__random_forest__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.134085`, log-R2 `0.993259`, factor-5 accuracy `0.995125`
- `latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1`: mean validation log-RMSE `0.143601`, log-R2 `0.993620`, factor-5 accuracy `0.993622`
- `latent__mlp__composition_pca_plus_log_en_log_power__k_8__w_256__layers_3__drop_0.0__wd_1e-05`: mean validation log-RMSE `0.143604`, log-R2 `0.993251`, factor-5 accuracy `0.995297`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/training_random_case/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/training_random_case/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `E + H2OV > O^ + 2H + 2E`: log-RMSE `0.279043`, log-MAE `0.094769`, factor-5 accuracy `0.945231`
- `AR+ IONIZATION`: log-RMSE `0.261781`, log-MAE `0.086976`, factor-5 accuracy `0.950818`
- `E + H2O > O^ + 2H + 2E`: log-RMSE `0.229500`, log-MAE `0.073248`, factor-5 accuracy `0.959865`
- `E + H2OV > H2^ + O + E`: log-RMSE `0.222915`, log-MAE `0.079132`, factor-5 accuracy `0.948034`
- `E + H2O > H^ + OH + 2E`: log-RMSE `0.210701`, log-MAE `0.060713`, factor-5 accuracy `0.978553`
- `E + H2O > H2^ + O + E`: log-RMSE `0.205499`, log-MAE `0.066423`, factor-5 accuracy `0.961871`
- `E + H2OV > H^ + OH + 2E`: log-RMSE `0.205207`, log-MAE `0.057795`, factor-5 accuracy `0.980243`
- `E + H2O > H(4) [486.1nm] +OH+E`: log-RMSE `0.187264`, log-MAE `0.050986`, factor-5 accuracy `0.985843`
- `E + H2OV > H(4)[486.1nm] +OH+E`: log-RMSE `0.178695`, log-MAE `0.047938`, factor-5 accuracy `0.987005`
- `E + H2O > H(3) [656.3nm] +OH+E`: log-RMSE `0.177130`, log-MAE `0.047301`, factor-5 accuracy `0.986688`

Worst test cases by log-RMSE:

- case `12826` (group `443`, local case `8`): log-RMSE `1.426898`, factor-5 accuracy `0.880829`
- case `19235` (group `664`, local case `8`): log-RMSE `1.417474`, factor-5 accuracy `0.880829`
- case `51280` (group `1769`, local case `8`): log-RMSE `1.168750`, factor-5 accuracy `0.875648`
- case `64097` (group `2211`, local case `7`): log-RMSE `1.096137`, factor-5 accuracy `0.828571`
- case `51281` (group `1769`, local case `9`): log-RMSE `1.054989`, factor-5 accuracy `0.917526`
- case `64099` (group `2211`, local case `9`): log-RMSE `1.046065`, factor-5 accuracy `0.912371`
- case `32054` (group `1106`, local case `9`): log-RMSE `0.984291`, factor-5 accuracy `0.922680`
- case `38461` (group `1327`, local case `7`): log-RMSE `0.948366`, factor-5 accuracy `0.840000`
- case `70509` (group `2432`, local case `10`): log-RMSE `0.706769`, factor-5 accuracy `0.918782`
- case `6418` (group `222`, local case `9`): log-RMSE `0.668388`, factor-5 accuracy `0.927835`

Worst reactions by median absolute relative error:

- `AR+ -ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `AR2+ -MOMTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `H+ -ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `H2O+ -MOMTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `O2+ -ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `O+ -ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `O- ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `H3+ -ELASTIC`: median abs relative error `6.798603e-03`, 95th percentile `3.019090e-02`, within 10% `0.995919`
- `H2+ -ELASTIC`: median abs relative error `6.796862e-03`, 95th percentile `3.018645e-02`, within 10% `0.995919`
- `OH- -MOM`: median abs relative error `6.795386e-03`, 95th percentile `3.018929e-02`, within 10% `0.995919`

Worst test cases by median absolute relative error:

- case `82740` (group `2854`, local case `3`): median abs relative error `6.288083e-02`, within 10% `0.707071`
- case `63512` (group `2191`, local case `2`): median abs relative error `4.483134e-02`, within 10% `0.775510`
- case `72271` (group `2493`, local case `3`): median abs relative error `4.461475e-02`, within 10% `0.767677`
- case `83058` (group `2865`, local case `2`): median abs relative error `4.333646e-02`, within 10% `0.632653`
- case `57103` (group `1970`, local case `2`): median abs relative error `3.956231e-02`, within 10% `0.724490`
- case `76331` (group `2633`, local case `3`): median abs relative error `3.878120e-02`, within 10% `0.747475`
- case `83059` (group `2865`, local case `3`): median abs relative error `3.589415e-02`, within 10% `0.757576`
- case `69341` (group `2392`, local case `2`): median abs relative error `3.532054e-02`, within 10% `0.816327`
- case `56523` (group `1950`, local case `2`): median abs relative error `3.518017e-02`, within 10% `0.795918`
- case `74590` (group `2573`, local case `2`): median abs relative error `3.407406e-02`, within 10% `0.816327`

Worst reactions by median SMAPE:

- `OH- -MOM`: median SMAPE `6.805975e-03`, 95th percentile `3.001295e-02`, within 10% `0.996079`
- `AR+ -ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `AR2+ -MOMTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `H+ -ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `H2+ -ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `H2O+ -MOMTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `O2+ -ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `O+ -ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `O- ELASTIC`: median SMAPE `6.804495e-03`, 95th percentile `3.001300e-02`, within 10% `0.996079`
- `H3+ -ELASTIC`: median SMAPE `6.802821e-03`, 95th percentile `3.001292e-02`, within 10% `0.996079`

Worst test cases by median SMAPE:

- case `82740` (group `2854`, local case `3`): median SMAPE `4.445733e-02`, within 10% `0.700980`
- case `83058` (group `2865`, local case `2`): median SMAPE `3.090947e-02`, within 10% `0.725490`
- case `72271` (group `2493`, local case `3`): median SMAPE `2.997702e-02`, within 10% `0.730392`
- case `63512` (group `2191`, local case `2`): median SMAPE `2.836803e-02`, within 10% `0.730392`
- case `57103` (group `1970`, local case `2`): median SMAPE `2.746637e-02`, within 10% `0.705882`
- case `76331` (group `2633`, local case `3`): median SMAPE `2.699588e-02`, within 10% `0.725490`
- case `83059` (group `2865`, local case `3`): median SMAPE `2.665544e-02`, within 10% `0.720588`
- case `56523` (group `1950`, local case `2`): median SMAPE `2.484607e-02`, within 10% `0.730392`
- case `64327` (group `2219`, local case `5`): median SMAPE `2.423922e-02`, within 10% `0.823529`
- case `74590` (group `2573`, local case `2`): median SMAPE `2.421917e-02`, within 10% `0.750000`

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
- `inverse_transform_roundtrip_trainval`: status `pass`, observed `2.6020852139652106e-18`, expected `< 1e-10`
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6706615345592091`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4311282940076381`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3036679706402175`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.203489716971332`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1274180211110203`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0858605689589769`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0584373638389742`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6774436947892476`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4320826153453644`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3075782735990892`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2074215907922149`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.12981377401792`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0877623917171451`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0596891933105806`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6748824287000761`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4286175905296058`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3050659912421674`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2049467208117986`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1279681663404647`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0868041715922966`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0584525859228239`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.678344120659284`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4350994406578658`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3059956346583343`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2081003887538705`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1308493854669708`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0879889297646076`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0589139632489203`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6677787177338743`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4309457038390394`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.302110038205775`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2030726059525791`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1290068730934579`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0868350940613787`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0576113632262169`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.677927990589436`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.434854380498984`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.302472226753837`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.2040552567599445`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.1280308231030574`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0866964579174356`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.057894581211276`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `2549592.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/training_random_case/verification_checks.csv)
- [`pca_scree.png`](../results/training_random_case/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/training_random_case/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/training_random_case/figures/residual_hist_log_space.png)

