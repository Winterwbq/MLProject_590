# NERS590 v02 Joint RATE CONST and SUPER RATE Experiment Report

## Scope

This report reviews the current v2 modeling results for both target families:

- `RATE CONST`
- `SUPER RATE (CC/S)`

It combines the four production experiment sets we have run so far:

1. `RATE CONST` with random-case splitting
2. `RATE CONST` with `5mJ` power holdout
3. `SUPER RATE` with random-case splitting
4. `SUPER RATE` with `5mJ` power holdout

The main goal is to compare the targets jointly and separately, explain which models worked best, and interpret why the winning model choices differ between the two outputs. Citations: [experiment_summary.csv](../../results/ners590_v02_joint_review/experiment_summary.csv), [ners590_v02_experiment_review.md](/Users/bingqingwang/Desktop/UMich/590_Machine%20learning/project/docs/active_v02/ners590_v02_experiment_review.md), [ners590_v02_super_rate_training_review.md](/Users/bingqingwang/Desktop/UMich/590_Machine%20learning/project/docs/active_v02/ners590_v02_super_rate_training_review.md).

## Experiment Matrix

The study currently covers one unified power-aware model per target family and split regime, not separate models per power file.

| Target | Split | Purpose | Winning model | Winning feature set |
| --- | --- | --- | --- | --- |
| RATE CONST | Random Case | interpolation on mixed-power cases | `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` | `composition_pca_plus_log_en_log_power` |
| RATE CONST | Power Holdout 5mJ | generalization to unseen power | `direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1` | `composition_pca_plus_log_en_log_power` |
| SUPER RATE | Random Case | interpolation on mixed-power cases | `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1` | `all_inputs_plus_log_en_log_power` |
| SUPER RATE | Power Holdout 5mJ | generalization to unseen power | `direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1` | `all_inputs_plus_log_en_log_power` |

The most important structural difference is on the target side. `RATE CONST` was modeled as a full 204-output regression problem. `SUPER RATE` was first cleaned by dropping `110` constant-zero channels and training only on the remaining `94` nontrivial outputs, then reconstructing the final 204-output vector by reinserting zeros. Citations: [experiment_summary.csv](../../results/ners590_v02_joint_review/experiment_summary.csv), [target_metadata.csv](../../results/ners590_v02_super_rate/training_random_case/data_snapshots/target_metadata.csv), [target_cleaning_map.csv](../../results/ners590_v02_super_rate/training_random_case/data_snapshots/target_cleaning_map.csv).

## Overall Outcome

The selected-model performance is summarized below.

| Target | Split | Test log RMSE | Test log R2 | Median abs relative error | P95 abs relative error | Within 10% relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RATE CONST | Random Case | 0.051212 | 0.999923 | 2.01e-04 | 2.29e-02 | 0.967915 |
| RATE CONST | Power Holdout 5mJ | 0.047780 | 0.999722 | 5.36e-04 | 8.03e-02 | 0.952621 |
| SUPER RATE | Random Case | 0.001243 | 0.999999 | 4.90e-05 | 8.40e-04 | 0.999802 |
| SUPER RATE | Power Holdout 5mJ | 0.003482 | 0.999973 | 3.29e-05 | 3.86e-03 | 0.997990 |

Two observations stand out immediately.

First, both targets are highly learnable on the new v2 dataset. All four experiments achieve near-perfect log-space `R2`, and even the harder power-holdout settings remain extremely accurate.

Second, `SUPER RATE` is easier to predict than `RATE CONST` after target cleaning, despite being sparser in the raw dataset. That is because the cleaned `SUPER RATE` problem removes the trivial always-zero channels and leaves a compact subset of structured nonzero outputs that the model can fit very well. By contrast, `RATE CONST` remains a denser and more heterogeneous 204-output regression problem. Citations: [experiment_summary.csv](../../results/ners590_v02_joint_review/experiment_summary.csv), ![Selected model test metrics](../../results/ners590_v02_joint_review/figures/selected_model_test_metrics.png), ![Selected model relative error](../../results/ners590_v02_joint_review/figures/selected_model_relative_error.png).

## Joint Comparison of Model Families

The leaderboard pattern is highly consistent across tasks:

- direct tree models are best
- MLPs are the strongest non-tree baseline
- latent PCA models are weaker than the best direct models
- Ridge is clearly the weakest family

The best-per-family summary is:

| Target | Split | Best ExtraTrees | Best MLP | Best latent model | Best Ridge |
| --- | --- | --- | --- | --- | --- |
| RATE CONST | Random Case | direct ExtraTrees on composition PCA | direct MLP on all inputs | latent RandomForest on composition PCA | latent Ridge on composition PCA |
| RATE CONST | Power Holdout 5mJ | direct ExtraTrees on composition PCA | direct MLP on all inputs | latent RandomForest on composition PCA | latent Ridge on composition PCA |
| SUPER RATE | Random Case | direct ExtraTrees on all inputs | direct MLP on all inputs | latent ExtraTrees on composition PCA | direct Ridge on full nonconstant inputs |
| SUPER RATE | Power Holdout 5mJ | direct ExtraTrees on all inputs | direct MLP on all inputs | latent ExtraTrees on composition PCA | direct Ridge on full nonconstant inputs |

This is a strong sign that we are in a classic nonlinear scientific-tabular regime rather than a representation-learning-limited regime. The data is structured, moderately large, and physics-driven; tree ensembles are capitalizing on nonlinear threshold-like interactions without paying the optimization cost of neural models or the compression bias of latent PCA pipelines. Citations: [best_model_per_family.csv](../../results/ners590_v02_joint_review/best_model_per_family.csv), [leaderboard_top7_combined.csv](../../results/ners590_v02_joint_review/leaderboard_top7_combined.csv), ![Leaderboard top-5 grid](../../results/ners590_v02_joint_review/figures/leaderboard_top5_grid.png).

## Why ExtraTrees Won

ExtraTrees is the best family for both targets, but it wins for slightly different reasons in each task.

For `RATE CONST`, the target is dense, highly correlated, and still rich enough that simple linear structure is not sufficient. ExtraTrees works well because it captures nonlinear interactions among composition, `E/N`, and power without requiring heavy scaling assumptions, and it remains stable on tabular data with mixed redundancy. The fact that the winning feature set is `composition_pca_plus_log_en_log_power` suggests the RATE CONST mapping mostly depends on a lower-dimensional composition manifold plus plasma-state controls. In other words, the PCA compression is helping the tree focus on the dominant chemistry variation instead of memorizing redundant species coordinates.

For `SUPER RATE`, the winning ExtraTrees model switches to `all_inputs_plus_log_en_log_power`. That tells us the sparse cleaned target still depends on finer species-level detail that composition PCA partially washes out. Here the tree benefits from direct access to all individual inputs because some reactions appear to hinge on sharper local composition cues rather than only the dominant low-dimensional composition trend.

So the same model family wins, but the preferred input representation changes with the target geometry. Citations: [selected_model.csv](../../results/ners590_v02/training_random_case/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02_super_rate/training_random_case/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/tuning/selected_model.csv), [leaderboard_top7_combined.csv](../../results/ners590_v02_joint_review/leaderboard_top7_combined.csv).

## Why the MLPs Were Good but Not Best

The MLP baselines were consistently second-tier rather than bad. For `RATE CONST`, the direct MLP is actually close to the winning ExtraTrees model, especially in the validation rankings. This suggests the RATE CONST mapping is smooth enough and the v2 dataset is large enough that a feed-forward network can learn a good approximation.

But the MLP still loses for three practical reasons.

First, the problem is still tabular rather than representation-heavy. ExtraTrees gets strong nonlinear partitioning almost for free, while the MLP has to learn those interactions through optimization.

Second, both targets have heavy-tailed transformed outputs and case-regime structure. Trees can separate these regimes quickly, whereas a single MLP must fit them with one global smooth function.

Third, the output dimensionality remains large, especially for `RATE CONST`, and the current MLP is still a fairly small baseline rather than a highly specialized architecture.

For `SUPER RATE`, the MLP falls further behind because sparsity and target imbalance make the regression surface harder to learn smoothly. After cleaning, the task is still sparse-aware in spirit, and trees remain better suited to that structure. Citations: [leaderboard_top7_combined.csv](../../results/ners590_v02_joint_review/leaderboard_top7_combined.csv), [model_leaderboard_summary.csv](../../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv), [model_leaderboard_summary.csv](../../results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv), [model_leaderboard_summary.csv](../../results/ners590_v02_super_rate/training_random_case/tuning/model_leaderboard_summary.csv), [model_leaderboard_summary.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv).

## Why Latent PCA Models Underperformed

The latent-output experiments are informative because they separate two questions:

1. are the outputs low-rank?
2. even if they are, is predicting latent PCA coordinates the best supervised strategy?

For `RATE CONST`, the oracle PCA reconstruction is surprisingly strong. With `k=12`, the oracle locked-test log RMSE is about `0.0521` for the random split and `0.0516` for the power-holdout split, which is already very close to the winning direct model. That means RATE CONST is indeed close to low-rank at the output level.

However, the actual latent supervised models still underperform the direct winner. That says the bottleneck is not only output redundancy, but the difficulty of mapping inputs to the exact latent coordinates without losing reaction-specific residual structure.

For `SUPER RATE`, the story is even clearer. The best oracle PCA reconstruction at `k=12` is `0.002341` for random-case and `0.002955` for power-holdout, both noticeably worse than the winning direct model on the full reconstructed outputs and also not clearly better than the active-target direct fit. So for SUPER RATE, fixed low-rank compression is simply discarding useful target detail.

This is why the latent models are scientifically interesting but practically not the best current choice. Citations: [oracle_test_overall_by_k.csv](../../results/ners590_v02/training_random_case/pca/oracle_test_overall_by_k.csv), [oracle_test_overall_by_k.csv](../../results/ners590_v02/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv), [oracle_test_overall_by_k.csv](../../results/ners590_v02_super_rate/training_random_case/pca/oracle_test_overall_by_k.csv), [oracle_test_overall_by_k.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv), ![Selected vs oracle](../../results/ners590_v02_joint_review/figures/selected_vs_oracle.png).

## Why Ridge Was Bad

Ridge is the clearest failure mode across all experiments. Its validation log RMSE is dramatically worse than every nonlinear baseline, and its `R2` falls to roughly `0.79` to `0.83`, far below the rest of the leaderboard.

This is exactly what we would expect from the dataset structure. Both targets depend on nonlinear interactions among composition, `E/N`, and power. The outputs also change across distinct local-case regimes. A linear model can absorb some global trend after log-transforming the target, but it cannot capture the regime transitions or multiplicative interactions cleanly enough.

So Ridge still has value as a sanity-check baseline, but it is not a realistic production model for this project. Citations: [best_model_per_family.csv](../../results/ners590_v02_joint_review/best_model_per_family.csv), [leaderboard_top7_combined.csv](../../results/ners590_v02_joint_review/leaderboard_top7_combined.csv).

## RATE CONST: Separate Analysis

`RATE CONST` remains the harder and more heterogeneous target family. Its best test log RMSE is around `0.05`, roughly one to two orders of magnitude larger than the cleaned SUPER RATE task. Relative-error metrics are also worse, especially in the `5mJ` holdout run, where the median absolute relative error rises from `2.01e-04` to `5.36e-04` and the 95th percentile rises from about `2.29e-02` to `8.03e-02`.

The hardest RATE CONST reactions cluster around ionization and dissociation channels such as `AR+ IONIZATION`, `E + H2O > O^ + 2H + 2E`, `E + H2OV > O^ + 2H + 2E`, and related water-fragmentation reactions. These are exactly the sorts of channels where small state changes can produce large relative variation, so they are plausible weak points rather than random model failures.

At the case level, RATE CONST difficulty is concentrated in the middle of the local-case sweep, especially around local cases `7-12`, where mean per-case log RMSE peaks. That suggests RATE CONST is hardest in the transition region where many channels become active and compete, not at the very lowest or very highest local-case positions. Citations: [test_overall_metrics.csv](../../results/ners590_v02/training_random_case/evaluation/test_overall_metrics.csv), [test_overall_metrics.csv](../../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv), [test_relative_error_overall_summary.csv](../../results/ners590_v02/training_random_case/evaluation/test_relative_error_overall_summary.csv), [test_relative_error_overall_summary.csv](../../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv), [worst_10_reactions.csv](../../results/ners590_v02/training_random_case/evaluation/worst_10_reactions.csv), [worst_10_reactions.csv](../../results/ners590_v02/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv), [local_case_error_profiles.csv](../../results/ners590_v02_joint_review/local_case_error_profiles.csv), ![Local case error profiles](../../results/ners590_v02_joint_review/figures/local_case_error_profiles.png).

## SUPER RATE: Separate Analysis

`SUPER RATE` is the opposite kind of problem. It starts from a much sparser raw target space, but after cleaning it becomes an extremely well-behaved supervised task. The direct ExtraTrees model reaches log RMSE `0.001243` on the random split and `0.003482` on the power-holdout split, with nearly perfect factor-5 accuracy and extremely small relative errors on positive entries.

The power-holdout degradation is real but modest. This is consistent with the earlier SUPER RATE analysis: power is not the dominant global driver, but it still produces detectable reaction-level shifts. The model generalizes to unseen `5mJ` very well overall, but the hardest reactions are again the oxygen vibrational and related small-magnitude channels, including `O2-VIB 1-4`, `H2 > H2(V=1/2)`, and `O-1D`.

At the case level, SUPER RATE errors concentrate much earlier in the sweep than RATE CONST errors. The hardest local cases are `3-4`, then `1-2`, with errors dropping sharply after `7`. That matches the underlying target structure: SUPER RATE sparsity and activation pattern change most rapidly in the early part of the local-case progression. Citations: [test_overall_metrics_full_reconstructed.csv](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_overall_metrics_full_reconstructed.csv), [test_overall_metrics_full_reconstructed.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/evaluation/test_overall_metrics_full_reconstructed.csv), [test_relative_error_overall_summary.csv](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_relative_error_overall_summary.csv), [test_relative_error_overall_summary.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/evaluation/test_relative_error_overall_summary.csv), [worst_10_reactions.csv](../../results/ners590_v02_super_rate/training_random_case/evaluation/worst_10_reactions.csv), [worst_10_reactions.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/evaluation/worst_10_reactions.csv), [local_case_error_profiles.csv](../../results/ners590_v02_joint_review/local_case_error_profiles.csv), ![Local case error profiles](../../results/ners590_v02_joint_review/figures/local_case_error_profiles.png).

## What the Results Mean for Joint Modeling

The current results support a nuanced conclusion.

On one hand, the two targets are similar enough that a shared-trunk model is scientifically plausible:

- the same unified multi-power training strategy works for both
- the same model family wins for both
- both generalize well to unseen `5mJ`

On the other hand, the preferred input representation differs:

- `RATE CONST` prefers composition PCA features
- `SUPER RATE` prefers all individual inputs

That means a naive “single shared output layer for everything” design is probably not ideal. If we build a joint model next, the best architecture is likely:

- a shared input encoder for global plasma-state structure
- one output head for `RATE CONST`
- one output head for `SUPER RATE`
- and possibly a slightly richer SUPER RATE branch because its sparse cleaned outputs seem to need more species-level detail

So the current production baseline should remain separate-task ExtraTrees models, while the next research step should be a shared-trunk, separate-head multi-task model evaluated directly against these baselines. Citations: [experiment_summary.csv](../../results/ners590_v02_joint_review/experiment_summary.csv), [best_model_per_family.csv](../../results/ners590_v02_joint_review/best_model_per_family.csv), [ners590_v02_super_rate_analysis_report.md](/Users/bingqingwang/Desktop/UMich/590_Machine%20learning/project/docs/active_v02/ners590_v02_super_rate_analysis_report.md), [ners590_v02_analysis_report.md](/Users/bingqingwang/Desktop/UMich/590_Machine%20learning/project/docs/active_v02/ners590_v02_analysis_report.md).

## Practical Recommendation

The current best operational setup is:

| Target | Recommended current baseline |
| --- | --- |
| RATE CONST | direct ExtraTrees on `composition_pca_plus_log_en_log_power` |
| SUPER RATE | direct ExtraTrees on `all_inputs_plus_log_en_log_power`, with constant-zero target cleaning |

That should remain the reference point for all future neural or multitask work. In other words, any new shared-head or joint-head model should be judged against these exact baselines, not only against weaker linear or latent-output models. Citations: [selected_model.csv](../../results/ners590_v02/training_random_case/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02/training_power_holdout_5mJ/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02_super_rate/training_random_case/tuning/selected_model.csv), [selected_model.csv](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/tuning/selected_model.csv).
