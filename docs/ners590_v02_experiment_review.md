# NERS590 v02 Experiment Review

This report is a consolidated review of the two v02 training experiment sets. It is written to answer three questions clearly:

1. what were the two experiment sets
2. how were the models trained in each set
3. what did we learn from the results

## 1. Experiment Overview

The current v02 work used one merged dataset built from all five power-specific files, not five separate per-power training jobs. After parsing, the active dataset contained `32,045` cases total, formed by `5` power levels x `221` density groups per power x `29` local `E/N` positions. Power was carried into the model as an explicit input feature through `power_mj`.

On top of that one merged dataset, we ran two experiment sets:

- `training_random_case`
- `training_power_holdout_5mJ`

These are two different evaluation strategies applied to the same merged dataset and the same curated candidate-model catalog. So the difference between the two experiments is not “different model families” or “different raw files.” The difference is the generalization task we asked the model to solve.

Citations:
- [`results/ners590_v02/parsed/parser_summary.csv`](../results/ners590_v02/parsed/parser_summary.csv)
- [`results/ners590_v02/analysis/dataset_summary.csv`](../results/ners590_v02/analysis/dataset_summary.csv)
- [`results/ners590_v02/training_random_case/tuning/split_metadata.csv`](../results/ners590_v02/training_random_case/tuning/split_metadata.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv)

## 2. Purpose of the Two Experiment Sets

The first experiment set, `training_random_case`, answers the question:

“If we mix all powers together and randomly hold out cases, how well can the model interpolate within the overall observed data manifold?”

This is the easier and more standard supervised-learning benchmark. It mainly tests whether the model can learn the mapping once training and test data come from the same overall distribution.

The second experiment set, `training_power_holdout_5mJ`, answers the question:

“If the model never sees the `5mJ` file during testing, can it still generalize from lower powers to a new power level?”

This is a more scientific transfer-style benchmark. It is harder and more meaningful for this dataset because power is now a real control variable. This second experiment tells us whether the model is only interpolating within seen powers or whether it can also carry the learned structure to an unseen power condition.

Together, these two experiments separate two kinds of success:

- interpolation success inside the merged observed dataset
- cross-power generalization success to a held-out power level

Citations:
- [`docs/ners590_v02_experiment_plan.md`](./ners590_v02_experiment_plan.md)
- [`results/ners590_v02/training_random_case/tuning/split_metadata.csv`](../results/ners590_v02/training_random_case/tuning/split_metadata.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv)
- [`docs/ners590_v02_analysis_report.md`](./ners590_v02_analysis_report.md)

## 3. Common Training Pipeline Used in Both Sets

Both experiment sets used the same end-to-end training pipeline after parsing:

- parse the five `.out` files into one merged tabular dataset
- use the `49` input mole fractions plus `E/N` plus `power_mj` as the scientific inputs
- transform `E/N` and power on log scale
- keep the prediction target as the full `204`-dimensional `RATE CONST` vector
- train in transformed target space using `log10(rate_const + epsilon_j)` with per-reaction `epsilon` estimated from the training data only
- evaluate final predictions after inverse-transforming back to the full `204` reconstructed outputs

The feature sets tested in the curated v02 training sweep were:

- `full_nonconstant_plus_log_en_log_power`
- `composition_pca_plus_log_en_log_power`
- `all_inputs_plus_log_en_log_power`

The model families tested were:

- direct ridge
- direct extra trees
- direct FFN
- latent ridge
- latent random forest
- latent extra trees
- latent FFN

For the latent models, the target side was compressed with PCA and the latent dimension used in the curated sweep was `k=8`. These latent models were evaluated on the reconstructed full `204` outputs, not only in latent space. That is important because the final scientific task always requires the full rate vector.

Citations:
- [`results/ners590_v02/training_random_case/tuning/model_config_catalog.csv`](../results/ners590_v02/training_random_case/tuning/model_config_catalog.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/model_config_catalog.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/model_config_catalog.csv)
- [`results/ners590_v02/training_random_case/data_snapshots/feature_set_final_metadata.csv`](../results/ners590_v02/training_random_case/data_snapshots/feature_set_final_metadata.csv)
- [`results/ners590_v02/training_random_case/data_snapshots/target_epsilons_trainval.csv`](../results/ners590_v02/training_random_case/data_snapshots/target_epsilons_trainval.csv)
- [`results/ners590_v02/training_random_case/pca/target_pca_explained_variance.csv`](../results/ners590_v02/training_random_case/pca/target_pca_explained_variance.csv)

## 4. How the Random-Case Experiment Was Trained

In `training_random_case`, all `32,045` merged cases were pooled together. A locked random test split was created, and model selection was then performed with five resampled train/validation folds inside the remaining train/validation pool.

So this experiment did not isolate powers. Cases from `1mJ` through `5mJ` were all available to both training and testing, just in different random rows. That makes this experiment an interpolation benchmark over the merged dataset manifold.

The selected model for this experiment was:

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`

Its validation performance was:

- mean validation log-RMSE: `0.048725`
- mean validation log-R2: `0.999920`

The strongest alternative was the direct FFN:

- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`
- mean validation log-RMSE: `0.058656`

So the FFN was genuinely competitive on v02, but the direct extra-trees model still won by a clear margin on the main selection metric.

Citations:
- [`results/ners590_v02/training_random_case/tuning/selected_model.csv`](../results/ners590_v02/training_random_case/tuning/selected_model.csv)
- [`results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv)
- [`results/ners590_v02/training_random_case/figures/model_leaderboard.png`](../results/ners590_v02/training_random_case/figures/model_leaderboard.png)

## 5. What the Random-Case Experiment Achieved

On the locked random test split, the selected extra-trees model achieved:

- overall log-RMSE: `0.051212`
- overall log-MAE: `0.006550`
- overall log-R2: `0.999923`
- original-space RMSE: `2.773644e-06`
- original-space MAE: `2.152017e-07`

Its robust percent-style accuracy was also strong:

- median absolute relative error: `2.006515e-04`
- 95th percentile absolute relative error: `2.293077e-02`
- within 10% relative error: `0.967915`
- median SMAPE: `1.566903e-04`
- within 10% SMAPE: `0.934257`

These numbers indicate very strong interpolation performance. The model is not only good in aggregate log space, but also very precise in relative terms for most positive outputs.

The one caution is that the raw mean signed relative error is huge. That is not a model-failure signal here. It is the same denominator problem we discussed earlier: a small number of extremely tiny true rates make the raw mean relative error unstable. The more reliable summaries are the median and percentile relative-error statistics, plus SMAPE.

Citations:
- [`results/ners590_v02/training_random_case/evaluation/test_overall_metrics.csv`](../results/ners590_v02/training_random_case/evaluation/test_overall_metrics.csv)
- [`results/ners590_v02/training_random_case/evaluation/test_relative_error_overall_summary.csv`](../results/ners590_v02/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- [`results/ners590_v02/training_random_case/evaluation/test_smape_overall_summary.csv`](../results/ners590_v02/training_random_case/evaluation/test_smape_overall_summary.csv)
- [`results/ners590_v02/training_random_case/figures/relative_error_abs_histogram.png`](../results/ners590_v02/training_random_case/figures/relative_error_abs_histogram.png)
- [`results/ners590_v02/training_random_case/figures/smape_histogram.png`](../results/ners590_v02/training_random_case/figures/smape_histogram.png)

## 6. Where the Random-Case Experiment Was Still Hard

Even in the strong random-case run, the hardest reaction channels were not random. They concentrated in a physically coherent group of water / vibrational / ionization channels such as:

- `E + H2OV > O^ + 2H + 2E`
- `AR+ IONIZATION`
- `E + H2O > O^ + 2H + 2E`
- `E + H2OV > H2^ + O + E`
- `E + H2O > H^ + OH + 2E`

The hardest cases were also structured rather than arbitrary. They clustered around low-group / early-sweep cases such as:

- group `1`, local case `7`
- group `1`, local case `9`
- group `2`, local case `2`

So the difficult part of the mapping is still associated with certain low-`E/N` or regime-transition-like cases, even after moving to a much larger dataset.

Citations:
- [`results/ners590_v02/training_random_case/evaluation/worst_10_reactions.csv`](../results/ners590_v02/training_random_case/evaluation/worst_10_reactions.csv)
- [`results/ners590_v02/training_random_case/evaluation/worst_10_cases.csv`](../results/ners590_v02/training_random_case/evaluation/worst_10_cases.csv)
- [`results/ners590_v02/training_random_case/evaluation/test_per_reaction_metrics.csv`](../results/ners590_v02/training_random_case/evaluation/test_per_reaction_metrics.csv)
- [`results/ners590_v02/training_random_case/evaluation/test_per_case_metrics.csv`](../results/ners590_v02/training_random_case/evaluation/test_per_case_metrics.csv)
- [`results/ners590_v02/training_random_case/figures/worst_reactions_log_rmse.png`](../results/ners590_v02/training_random_case/figures/worst_reactions_log_rmse.png)
- [`results/ners590_v02/training_random_case/figures/per_case_log_rmse_distribution.png`](../results/ners590_v02/training_random_case/figures/per_case_log_rmse_distribution.png)

## 7. How the Power-Holdout Experiment Was Trained

In `training_power_holdout_5mJ`, we kept the same merged dataset and the same candidate-model catalog, but changed the split strategy:

- train/validation data came from `1mJ`, `2mJ`, `3mJ`, and `4mJ`
- the locked test set was the full held-out `5mJ` dataset

So this experiment tests cross-power generalization. The model must infer how the response surface evolves with power and then apply that understanding to an unseen power level.

Importantly, this is still a single unified power-aware model. It is not a bank of separate per-power models.

The selected model was again:

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`

The best direct FFN was again the runner-up:

- `direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05`

That repeatability matters. It suggests the model ranking is stable, not just a quirk of the random split.

Citations:
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/figures/model_leaderboard.png`](../results/ners590_v02/training_power_holdout_5mJ/figures/model_leaderboard.png)

## 8. What the Power-Holdout Experiment Achieved

On the fully held-out `5mJ` test set, the selected model achieved:

- overall log-RMSE: `0.047780`
- overall log-MAE: `0.008852`
- overall log-R2: `0.999722`
- original-space RMSE: `1.288207e-05`
- original-space MAE: `1.025436e-06`

Its robust percent-style metrics were:

- median absolute relative error: `5.360493e-04`
- 95th percentile absolute relative error: `8.028459e-02`
- within 10% relative error: `0.952621`
- median SMAPE: `3.884313e-04`
- within 10% SMAPE: `0.927681`

These results are extremely encouraging. In fact, the held-out-`5mJ` log-RMSE is slightly lower than the random-case test log-RMSE, though its absolute-relative and SMAPE summaries are a bit worse. The most reasonable interpretation is:

- the held-out `5mJ` dataset is not outside the learned manifold in a catastrophic way
- the model extrapolates across power quite well
- but the fine-grained percent-style calibration does degrade somewhat compared with pure random-case interpolation

Citations:
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/figures/relative_error_abs_histogram.png`](../results/ners590_v02/training_power_holdout_5mJ/figures/relative_error_abs_histogram.png)
- [`results/ners590_v02/training_power_holdout_5mJ/figures/smape_histogram.png`](../results/ners590_v02/training_power_holdout_5mJ/figures/smape_histogram.png)

## 9. Where the Power-Holdout Experiment Was Still Hard

The hardest reaction channels in the power-holdout experiment were nearly the same family as in the random-case experiment:

- `E + H2OV > O^ + 2H + 2E`
- `AR+ IONIZATION`
- `E + H2OV > H2^ + O + E`
- `E + H2O > O^ + 2H + 2E`
- `E + H2O > H2^ + O + E`

So the model difficulty is driven by a stable subset of reactions, not by arbitrary split noise.

The hardest held-out cases were concentrated in the `5mJ` file around:

- density-group-in-file `1`, local cases `8-11`
- density-group-in-file `211`, local cases `7-11`

This again points to structured difficulty in specific regions of the response surface, especially early-to-mid local-case regimes.

Citations:
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_cases.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_cases.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/test_per_reaction_metrics.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_per_reaction_metrics.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/evaluation/test_per_case_metrics.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_per_case_metrics.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/figures/worst_reactions_log_rmse.png`](../results/ners590_v02/training_power_holdout_5mJ/figures/worst_reactions_log_rmse.png)
- [`results/ners590_v02/training_power_holdout_5mJ/figures/per_case_log_rmse_distribution.png`](../results/ners590_v02/training_power_holdout_5mJ/figures/per_case_log_rmse_distribution.png)

## 10. Comparison Between the Two Experiment Sets

The two experiments together support five main conclusions.

First, the v02 dataset is large and regular enough that one unified power-aware model works very well. We do not currently need separate models for each power just to get strong baseline performance.

Second, the best current model family is still tree-based, but the direct FFN is now a serious competitor. That is a major change from the old tiny dataset era.

Third, direct prediction was better than the tested PCA-latent target variants in both split settings. So target compression is not currently buying us better end-to-end accuracy, even though the latent models are not terrible.

Fourth, the same model won both experiments. That gives us a stable baseline:

- direct extra trees
- composition PCA features
- `log10(E/N)`
- `log10(power_mj)`

Fifth, power generalization is not the main bottleneck right now. The harder issue appears to be a specific subset of reactions and a specific subset of low- to mid-local-case regimes.

Citations:
- [`results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- [`results/ners590_v02/training_random_case/tuning/selected_model.csv`](../results/ners590_v02/training_random_case/tuning/selected_model.csv)
- [`results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv)

## 11. Final Interpretation

At this stage, the project has a coherent and defensible training story:

- we merged the five raw files into one power-aware dataset
- we trained one unified model family on the merged data
- we evaluated it in two complementary ways
- we showed both interpolation quality and cross-power transfer quality

The current headline result is not merely that the model fits random held-out rows well. It is that a unified, power-aware model also generalizes strongly to a full unseen power level, while keeping strong accuracy on the final `204`-dimensional reconstructed output vector.

That makes the current v02 pipeline a strong publishable baseline.

Citations:
- [`docs/ners590_v02_dataset_overview.md`](./ners590_v02_dataset_overview.md)
- [`docs/ners590_v02_analysis_report.md`](./ners590_v02_analysis_report.md)
- [`docs/ners590_v02_training_report.md`](./ners590_v02_training_report.md)
