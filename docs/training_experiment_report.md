# Training Experiment Report

## Scope

This report summarizes the end-to-end training run from raw parsing through final testing.

- Results root: `results/full_training_pipeline`
- Selected model key: `direct__random_forest__composition_pca_plus_log_en__n_600__d_12__leaf_1`
- Selected model family: `random_forest`
- Selected feature set: `composition_pca_plus_log_en`
- Selected latent dimension: `nan`

Primary supporting outputs:

- [`selected_model.csv`](../results/full_training_pipeline/tuning/selected_model.csv)
- [`model_leaderboard_summary.csv`](../results/full_training_pipeline/tuning/model_leaderboard_summary.csv)
- [`search_stage_summary.csv`](../results/full_training_pipeline/tuning/search_stage_summary.csv)
- [`test_overall_metrics.csv`](../results/full_training_pipeline/evaluation/test_overall_metrics.csv)
- [`test_relative_error_overall_summary.csv`](../results/full_training_pipeline/evaluation/test_relative_error_overall_summary.csv)
- [`test_smape_overall_summary.csv`](../results/full_training_pipeline/evaluation/test_smape_overall_summary.csv)
- [`oracle_test_overall_by_k.csv`](../results/full_training_pipeline/pca/oracle_test_overall_by_k.csv)
- [`verification_checks.csv`](../results/full_training_pipeline/verification_checks.csv)

## Main Findings

The winning configuration was `direct__random_forest__composition_pca_plus_log_en__n_600__d_12__leaf_1` with mean validation reconstructed log-RMSE `0.167314` and mean validation log-R2 `0.997379`.

On the locked test split, the final model achieved overall log-space RMSE `0.203350`, log-space MAE `0.014383`, and log-space R2 `0.996972`. In original rate-constant space, the model achieved RMSE `2.802643e-05` and MAE `1.333844e-06`.

The best oracle PCA reconstruction on the locked test split was at `k=12` with reconstruction log-RMSE `0.055975`. This value is the compression-only ceiling from [`oracle_test_overall_by_k.csv`](../results/full_training_pipeline/pca/oracle_test_overall_by_k.csv).

For raw percent-style error on positive-ground-truth outputs, the median absolute relative error was `3.883201e-05`, the 95th percentile was `3.008085e-02`, and `0.972867` of positive predictions were within 10% relative error. These summaries are based only on entries where the true rate constant is greater than zero.

As a safer bounded alternative, the overall median `SMAPE` was `2.815656e-05`, the 95th percentile was `2.000000e+00`, and `0.926524` of predictions were within 10% SMAPE.

Supporting outputs:

- [`test_overall_metrics.csv`](../results/full_training_pipeline/evaluation/test_overall_metrics.csv)
- [`test_relative_error_overall_summary.csv`](../results/full_training_pipeline/evaluation/test_relative_error_overall_summary.csv)
- [`test_relative_error_by_magnitude_bin.csv`](../results/full_training_pipeline/evaluation/test_relative_error_by_magnitude_bin.csv)
- [`test_smape_overall_summary.csv`](../results/full_training_pipeline/evaluation/test_smape_overall_summary.csv)
- [`test_smape_by_magnitude_bin.csv`](../results/full_training_pipeline/evaluation/test_smape_by_magnitude_bin.csv)
- [`worst_10_reactions.csv`](../results/full_training_pipeline/evaluation/worst_10_reactions.csv)
- [`worst_10_cases.csv`](../results/full_training_pipeline/evaluation/worst_10_cases.csv)
- [`model_leaderboard.png`](../results/full_training_pipeline/figures/model_leaderboard.png)
- [`oracle_reconstruction_error_by_k.png`](../results/full_training_pipeline/figures/oracle_reconstruction_error_by_k.png)
- [`relative_error_abs_histogram.png`](../results/full_training_pipeline/figures/relative_error_abs_histogram.png)
- [`smape_histogram.png`](../results/full_training_pipeline/figures/smape_histogram.png)

## Top Validation Configurations

- `direct__random_forest__composition_pca_plus_log_en__n_600__d_12__leaf_1`: mean validation log-RMSE `0.167314`, log-R2 `0.997379`, factor-5 accuracy `0.997297`
- `direct__random_forest__composition_pca_plus_log_en__n_300__d_20__leaf_1`: mean validation log-RMSE `0.167338`, log-R2 `0.997420`, factor-5 accuracy `0.997350`
- `direct__random_forest__composition_pca_plus_log_en__n_300__d_none__leaf_1`: mean validation log-RMSE `0.167338`, log-R2 `0.997420`, factor-5 accuracy `0.997350`
- `direct__random_forest__composition_pca_plus_log_en__n_600__d_20__leaf_1`: mean validation log-RMSE `0.167340`, log-R2 `0.997379`, factor-5 accuracy `0.997297`
- `direct__random_forest__composition_pca_plus_log_en__n_600__d_none__leaf_1`: mean validation log-RMSE `0.167340`, log-R2 `0.997379`, factor-5 accuracy `0.997297`

Supporting outputs:

- [`model_leaderboard_summary.csv`](../results/full_training_pipeline/tuning/model_leaderboard_summary.csv)
- [`model_trials_foldwise.csv`](../results/full_training_pipeline/tuning/model_trials_foldwise.csv)

## Hardest Reactions and Cases

Worst reactions by test log-RMSE:

- `E + H2O > O^ + 2H + 2E`: log-RMSE `0.814428`, log-MAE `0.105499`, factor-5 accuracy `0.953846`
- `E + H2OV > O^ + 2H + 2E`: log-RMSE `0.806343`, log-MAE `0.104847`, factor-5 accuracy `0.954545`
- `AR+ IONIZATION`: log-RMSE `0.804537`, log-MAE `0.104643`, factor-5 accuracy `0.953846`
- `H*  > H**`: log-RMSE `0.601647`, log-MAE `0.075526`, factor-5 accuracy `0.988506`
- `AR4P > AR4D,6S,RYD`: log-RMSE `0.599353`, log-MAE `0.074979`, factor-5 accuracy `0.988506`
- `O-1D`: log-RMSE `0.581064`, log-MAE `0.072239`, factor-5 accuracy `0.988506`
- `H2 > H2(V=2)`: log-RMSE `0.550360`, log-MAE `0.064057`, factor-5 accuracy `0.988095`
- `O2-B-SINGLET SIGMA`: log-RMSE `0.549141`, log-MAE `0.068603`, factor-5 accuracy `0.988506`
- `H- -DETACHMENT`: log-RMSE `0.544621`, log-MAE `0.067941`, factor-5 accuracy `0.988506`
- `AR4SM>AR4P,3D,5S,5P`: log-RMSE `0.480804`, log-MAE `0.058945`, factor-5 accuracy `0.988506`

Worst test cases by log-RMSE:

- case `3` (group `1`, local case `3`): log-RMSE `1.330477`, factor-5 accuracy `0.787879`
- case `11` (group `1`, local case `11`): log-RMSE `1.059733`, factor-5 accuracy `0.929293`
- case `7` (group `1`, local case `7`): log-RMSE `0.918070`, factor-5 accuracy `0.800000`
- case `358` (group `13`, local case `10`): log-RMSE `0.134459`, factor-5 accuracy `0.984772`
- case `329` (group `12`, local case `10`): log-RMSE `0.103597`, factor-5 accuracy `0.984772`
- case `31` (group `2`, local case `2`): log-RMSE `0.099691`, factor-5 accuracy `1.000000`
- case `30` (group `2`, local case `1`): log-RMSE `0.077278`, factor-5 accuracy `1.000000`
- case `297` (group `11`, local case `7`): log-RMSE `0.075513`, factor-5 accuracy `1.000000`
- case `322` (group `12`, local case `3`): log-RMSE `0.072399`, factor-5 accuracy `1.000000`
- case `292` (group `11`, local case `2`): log-RMSE `0.060201`, factor-5 accuracy `1.000000`

Worst reactions by median absolute relative error:

- `O2- -MOMENTUM TRANSFER`: median abs relative error `2.868175e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `H3+ -ELASTIC`: median abs relative error `2.868133e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `OH- -MOM`: median abs relative error `2.868098e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `AR+ -ELASTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `AR2+ -MOMTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `H+ -ELASTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `H2+ -ELASTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `H2O+ -MOMTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `O2+ -ELASTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`
- `O+ -ELASTIC`: median abs relative error `2.868056e-03`, 95th percentile `4.696731e-02`, within 10% `0.956522`

Worst test cases by median absolute relative error:

- case `3` (group `1`, local case `3`): median abs relative error `5.633384e-02`, within 10% `0.585859`
- case `31` (group `2`, local case `2`): median abs relative error `3.480240e-02`, within 10% `0.724490`
- case `30` (group `2`, local case `1`): median abs relative error `3.053936e-02`, within 10% `0.825581`
- case `322` (group `12`, local case `3`): median abs relative error `2.436004e-02`, within 10% `0.767677`
- case `266` (group `10`, local case `5`): median abs relative error `1.385981e-02`, within 10% `0.989899`
- case `7` (group `1`, local case `7`): median abs relative error `1.364897e-02`, within 10% `0.668571`
- case `11` (group `1`, local case `11`): median abs relative error `1.149727e-02`, within 10% `0.742424`
- case `64` (group `3`, local case `6`): median abs relative error `6.241137e-03`, within 10% `1.000000`
- case `383` (group `14`, local case `6`): median abs relative error `5.599827e-03`, within 10% `0.959016`
- case `378` (group `14`, local case `1`): median abs relative error `5.345005e-03`, within 10% `0.906977`

Worst reactions by median SMAPE:

- `O2- -MOMENTUM TRANSFER`: median SMAPE `2.864062e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `H3+ -ELASTIC`: median SMAPE `2.864020e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `OH- -MOM`: median SMAPE `2.863985e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `AR+ -ELASTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `AR2+ -MOMTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `H+ -ELASTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `H2+ -ELASTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `H2O+ -MOMTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `O2+ -ELASTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`
- `O+ -ELASTIC`: median SMAPE `2.863943e-03`, 95th percentile `4.810468e-02`, within 10% `0.956522`

Worst test cases by median SMAPE:

- case `3` (group `1`, local case `3`): median SMAPE `4.481072e-02`, within 10% `0.583333`
- case `31` (group `2`, local case `2`): median SMAPE `2.646068e-02`, within 10% `0.651961`
- case `30` (group `2`, local case `1`): median SMAPE `2.558635e-02`, within 10% `0.686275`
- case `322` (group `12`, local case `3`): median SMAPE `1.886523e-02`, within 10% `0.671569`
- case `266` (group `10`, local case `5`): median SMAPE `1.293929e-02`, within 10% `0.779412`
- case `11` (group `1`, local case `11`): median SMAPE `1.031703e-02`, within 10% `0.745098`
- case `7` (group `1`, local case `7`): median SMAPE `9.905257e-03`, within 10% `0.696078`
- case `64` (group `3`, local case `6`): median SMAPE `5.079645e-03`, within 10% `0.848039`
- case `378` (group `14`, local case `1`): median SMAPE `5.050995e-03`, within 10% `0.720588`
- case `383` (group `14`, local case `6`): median SMAPE `4.900179e-03`, within 10% `0.823529`

Supporting outputs:

- [`test_per_reaction_metrics.csv`](../results/full_training_pipeline/evaluation/test_per_reaction_metrics.csv)
- [`test_per_case_metrics.csv`](../results/full_training_pipeline/evaluation/test_per_case_metrics.csv)
- [`test_relative_error_per_reaction.csv`](../results/full_training_pipeline/evaluation/test_relative_error_per_reaction.csv)
- [`test_relative_error_per_case.csv`](../results/full_training_pipeline/evaluation/test_relative_error_per_case.csv)
- [`test_smape_per_reaction.csv`](../results/full_training_pipeline/evaluation/test_smape_per_reaction.csv)
- [`test_smape_per_case.csv`](../results/full_training_pipeline/evaluation/test_smape_per_case.csv)
- [`worst_reactions_log_rmse.png`](../results/full_training_pipeline/figures/worst_reactions_log_rmse.png)
- [`per_case_log_rmse_distribution.png`](../results/full_training_pipeline/figures/per_case_log_rmse_distribution.png)
- [`relative_error_by_magnitude.png`](../results/full_training_pipeline/figures/relative_error_by_magnitude.png)
- [`smape_by_magnitude.png`](../results/full_training_pipeline/figures/smape_by_magnitude.png)

## Verification

The experiment also saved explicit verification checks for parser dimensions, inverse-transform accuracy, and PCA monotonicity.

- `parsed_case_count`: status `pass`, observed `609.0`, expected `609`
- `parsed_input_columns`: status `pass`, observed `53.0`, expected `53`
- `parsed_target_columns`: status `pass`, observed `207.0`, expected `207`
- `density_group_count`: status `pass`, observed `21.0`, expected `21`
- `e_over_n_count`: status `pass`, observed `29.0`, expected `29`
- `inverse_transform_roundtrip_trainval`: status `pass`, observed `3.469446951953614e-18`, expected `< 1e-10`
- `fold_1_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6289407158625845`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_3`: status `pass`, observed `0.4011179134854004`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3078143650837162`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_6`: status `pass`, observed `0.2055606881670513`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1287925907808451`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0749303675257121`, expected `<= previous k overall_log_rmse`
- `fold_1_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0419624155030962`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_2`: status `pass`, observed `0.6200788915707727`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_3`: status `pass`, observed `0.3783919782090899`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2740263426550687`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1807530544785631`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1182712782143865`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0626427077270715`, expected `<= previous k overall_log_rmse`
- `fold_2_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0490292972634857`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_2`: status `pass`, observed `0.5399042624141538`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_3`: status `pass`, observed `0.3624500978881839`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_4`: status `pass`, observed `0.3070259749332108`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1723283578122411`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1134364193006542`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0805802066834061`, expected `<= previous k overall_log_rmse`
- `fold_3_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0430284391005744`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_2`: status `pass`, observed `0.5680272675502644`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_3`: status `pass`, observed `0.3557362631524649`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_4`: status `pass`, observed `0.2830683332490222`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_6`: status `pass`, observed `0.193494749205`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1306888526366067`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0814111981735861`, expected `<= previous k overall_log_rmse`
- `fold_4_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0470305795344449`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_2`: status `pass`, observed `0.582711296208223`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_3`: status `pass`, observed `0.3733690895437332`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_4`: status `pass`, observed `0.310227442442532`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_6`: status `pass`, observed `0.1886955042938085`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_8`: status `pass`, observed `0.1255495623394339`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_10`: status `pass`, observed `0.0842789797009697`, expected `<= previous k overall_log_rmse`
- `fold_5_validation_oracle_monotonic_k_12`: status `pass`, observed `0.0448281477167382`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_2`: status `pass`, observed `0.623896023224691`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_3`: status `pass`, observed `0.3860840245583963`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_4`: status `pass`, observed `0.3110043692356759`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_6`: status `pass`, observed `0.2106657125823474`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_8`: status `pass`, observed `0.1420442657749531`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_10`: status `pass`, observed `0.0866807901186446`, expected `<= previous k overall_log_rmse`
- `locked_test_oracle_monotonic_k_12`: status `pass`, observed `0.055974639507005`, expected `<= previous k overall_log_rmse`
- `results_written`: status `pass`, observed `18768.0`, expected `non-empty evaluation outputs`

Supporting outputs:

- [`verification_checks.csv`](../results/full_training_pipeline/verification_checks.csv)
- [`pca_scree.png`](../results/full_training_pipeline/figures/pca_scree.png)
- [`parity_plot_log_space.png`](../results/full_training_pipeline/figures/parity_plot_log_space.png)
- [`residual_hist_log_space.png`](../results/full_training_pipeline/figures/residual_hist_log_space.png)

