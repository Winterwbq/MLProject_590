# NERS590 v02 Training Report

This report summarizes the first baseline training runs on the merged v02 dataset:

- random-case generalization: [`results/ners590_v02/training_random_case`](../results/ners590_v02/training_random_case)
- `5mJ` power-held-out generalization: [`results/ners590_v02/training_power_holdout_5mJ`](../results/ners590_v02/training_power_holdout_5mJ)

## Main Result

In both split settings, the winning model was the same:

- `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1`

Supporting files:

- random-case selected model: [`training_random_case/tuning/selected_model.csv`](../results/ners590_v02/training_random_case/tuning/selected_model.csv)
- power-holdout selected model: [`training_power_holdout_5mJ/tuning/selected_model.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv)

So the best current baseline for v02 is a direct extra-trees regressor on:

- PCA-compressed composition features
- `log10(E/N)`
- `log10(power_mj)`

## 1. Random-Case Generalization

From [`training_random_case/evaluation/test_overall_metrics.csv`](../results/ners590_v02/training_random_case/evaluation/test_overall_metrics.csv):

- overall log-RMSE: `0.051212`
- overall log-MAE: `0.006550`
- overall log-R2: `0.999923`
- original-space RMSE: `3e-06`
- original-space MAE: `2.152017e-07`
- factor-of-2 accuracy: `0.993914`
- factor-of-5 accuracy: `0.999000`
- factor-of-10 accuracy: `0.999615`

From [`training_random_case/evaluation/test_relative_error_overall_summary.csv`](../results/ners590_v02/training_random_case/evaluation/test_relative_error_overall_summary.csv):

- median absolute relative error: `0.000201`
- 95th percentile absolute relative error: `0.022931`
- within 10% relative error: `0.967915`

From [`training_random_case/evaluation/test_smape_overall_summary.csv`](../results/ners590_v02/training_random_case/evaluation/test_smape_overall_summary.csv):

- median SMAPE: `0.000157`
- 95th percentile SMAPE: `0.346870`
- within 10% SMAPE: `0.934257`

This is a very strong random-case result. On the merged dataset, the model is much more accurate than the earlier small-data experiments on the old dataset.

## 2. Power-Held-Out Generalization on 5mJ

From [`training_power_holdout_5mJ/evaluation/test_overall_metrics.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv):

- overall log-RMSE: `0.047780`
- overall log-MAE: `0.008852`
- overall log-R2: `0.999722`
- original-space RMSE: `1.3e-05`
- original-space MAE: `1e-06`
- factor-of-2 accuracy: `0.994531`
- factor-of-5 accuracy: `0.999195`
- factor-of-10 accuracy: `0.999826`

From [`training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv):

- median absolute relative error: `0.000536`
- 95th percentile absolute relative error: `0.080285`
- within 10% relative error: `0.952621`

From [`training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_smape_overall_summary.csv):

- median SMAPE: `0.000388`
- 95th percentile SMAPE: `0.346305`
- within 10% SMAPE: `0.927681`

This is the most encouraging result of the first v02 training pass: even when the full `5mJ` power level is held out from the locked test set, the model still generalizes very well.

## 3. Model Ranking

The top validation leaderboard was similar in both split settings.

Random-case leaderboard:

- [`training_random_case/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv)

Power-holdout leaderboard:

- [`training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv`](../results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)

The main ranking pattern was:

1. direct extra trees on `composition_pca_plus_log_en_log_power`
2. direct FFN on `all_inputs_plus_log_en_log_power`
3. latent random forest on `composition_pca_plus_log_en_log_power`
4. latent FFN on `composition_pca_plus_log_en_log_power`
5. latent extra trees on `composition_pca_plus_log_en_log_power`
6. ridge variants far behind

This tells us:

- the larger v02 dataset makes FFNs much more competitive than before
- but the best current baseline is still a tree model
- direct prediction remains stronger than the tested latent PCA variants for this first-pass setup

## 4. Hardest Reactions and Cases

Random-case hardest reactions, from [`training_random_case/evaluation/worst_10_reactions.csv`](../results/ners590_v02/training_random_case/evaluation/worst_10_reactions.csv), are dominated by water-vibrational / ionization channels such as:

- `E + H2OV > O^ + 2H + 2E`
- `AR+ IONIZATION`
- `E + H2O > O^ + 2H + 2E`
- `E + H2OV > H2^ + O + E`
- `E + H2O > H^ + OH + 2E`

Power-holdout hardest reactions, from [`training_power_holdout_5mJ/evaluation/worst_10_reactions.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv), are very similar.

That alignment is important: it suggests the difficult part of the problem is not random noise in one split, but a consistent subset of physically sensitive reaction channels.

The hardest cases in both splits are concentrated in early local-case indices such as `7-11` for certain density groups, according to:

- [`training_random_case/evaluation/worst_10_cases.csv`](../results/ners590_v02/training_random_case/evaluation/worst_10_cases.csv)
- [`training_power_holdout_5mJ/evaluation/worst_10_cases.csv`](../results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_cases.csv)

So the difficult regime still appears to be the lower-`E/N` edge of the sweep, even in the larger dataset.

## 5. Interpretation

The most important training takeaway is that the v02 dataset is large enough that the model family picture changed:

- on the old tiny dataset, random forest dominated and FFNs lagged clearly
- on v02, the FFN is now a serious baseline
- but a power-aware tree model on compressed composition features still wins

That makes sense given the data analysis:

- composition lives on a structured manifold
- `E/N` and power are dominant control variables
- many outputs are strongly correlated
- some reaction channels remain locally nonlinear and regime-sensitive

The direct extra-trees model appears to benefit from both:

- strong local partitioning on the structured tabular inputs
- reduced noise / redundancy after composition PCA compression

## Bottom Line

The first full v02 training pass was successful. A direct extra-trees model using composition PCA plus `log10(E/N)` plus `log10(power_mj)` is the best current baseline, and it performs strongly both on random-case interpolation and on a full `5mJ` power-held-out test.
