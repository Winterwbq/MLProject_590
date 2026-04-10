# NERS590 v02 Temporary Experiment Paper

This note is a temporary experiment paper for the `NERS590_data_v02` dataset while the larger v03 rerun is still in progress. It consolidates the v02 dataset analysis, cleaning and preprocessing protocol, model definitions, training procedure, and result interpretation into one self-contained document.

## Abstract

We study two supervised learning problems derived from the NERS590 v02 plasma-kinetics dataset: prediction of 204 `RATE CONST` outputs and prediction of 204 `SUPER RATE (CC/S)` outputs from 49 input gas mole fractions, `E/N`, and power. The v02 dataset contains 32,045 cases formed by a fixed 29-point `E/N` sweep repeated for 221 density groups at each of 5 power levels. We evaluate direct multi-output models and latent-output PCA models under both random-case generalization and power-holdout generalization. Across all four experiment settings, direct ExtraTrees models are selected as the best models. For `RATE CONST`, the best test log-RMSE is `0.0512` on the random split and `0.0478` on the 5 mJ holdout split. For `SUPER RATE`, the best test log-RMSE is `0.00124` on the random split and `0.00348` on the 5 mJ holdout split. Detailed magnitude-binned analysis shows that relative error is much larger for tiny target magnitudes because the denominator in percentage error becomes extremely small, even when absolute error remains tiny.

Evidence:
- Dataset scale: [`dataset_summary.csv`](../../results/ners590_v02/analysis/dataset_summary.csv)
- Final joint summary: [`experiment_summary.csv`](../../results/ners590_v02_joint_review/experiment_summary.csv)

## 1. Dataset Description and Basic Analysis

The v02 dataset is a merged multi-file dataset with 5 powers (`1mJ` to `5mJ`). Each file contains 6,409 cases, giving 32,045 total cases. Structurally, each power level contains 221 density groups, and each density group contains the same 29-point `E/N` sweep. The input side contains 49 gas-species mole fractions plus operating-condition metadata; the output side contains 204 reactions for `RATE CONST`, and the long-format raw tables also contain `SUPER RATE`.

The `E/N` grid is identical across powers, spanning `1e-20` to `1e-14` in the stored units, which is useful because it lets us compare power effects at matched composition and matched `E/N`. Across powers, some transport and plasma-state summaries drift only mildly, such as average electron energy (`5.8696 eV` at 1 mJ to `5.8657 eV` at 5 mJ), while others change strongly, such as updated electron density (`9.62e9` to `6.88e10`). This means that power is not redundant even though the `E/N` sweep itself is fixed.

On the output side, `SUPER RATE` is much sparser than `RATE CONST`: only `39.35%` of the 6,537,180 super-rate entries are positive, while the remainder are exactly zero. In addition, the super-rate analysis shows that many reactions have strong positive log-space correlation with the corresponding rate constant when both are active. Among the 94 reactions with enough positive pairs to compute a stable log-correlation, the median correlation is about `0.942`, and about `59.6%` of those valid reactions are above `0.9`. This supports the physical intuition that both targets are influenced by the same underlying electron-energy-distribution physics, even though they remain distinct supervised targets in the current pipeline.

Evidence:
- Overall dataset structure: [`dataset_summary.csv`](../../results/ners590_v02/analysis/dataset_summary.csv)
- Per-power structure: [`power_summary.csv`](../../results/ners590_v02/analysis/power_summary.csv)
- Per-power case features: [`case_features_by_power.csv`](../../results/ners590_v02/analysis/case_features_by_power.csv)
- RATE CONST sensitivity to power: [`reaction_power_sensitivity_summary.csv`](../../results/ners590_v02/analysis/reaction_power_sensitivity_summary.csv)
- SUPER RATE sparsity: [`super_rate_overall_summary.csv`](../../results/ners590_v02/super_rate_analysis/super_rate_overall_summary.csv)
- SUPER RATE by power: [`super_rate_by_power_summary.csv`](../../results/ners590_v02/super_rate_analysis/super_rate_by_power_summary.csv)
- RATE CONST/SUPER RATE relationship: [`super_rate_rate_const_relationship_summary.csv`](../../results/ners590_v02/super_rate_analysis/super_rate_rate_const_relationship_summary.csv)
- Supporting figures: [`figure_manifest.csv`](../../results/ners590_v02/figures/figure_manifest.csv), [`super_rate_figure_manifest.csv`](../../results/ners590_v02/super_rate_figures/super_rate_figure_manifest.csv)

## 2. Data Cleaning and Preprocessing

### 2.1 Parsing and Cleaning

The raw `.out` files are parsed into synchronized CSV tables for case-level features, long-format input fractions, long-format rate tables, wide training inputs, and wide training targets. During parsing we preserve case identity metadata such as `source_file`, `power_label`, `density_group_id`, and `local_case_id`, but those identifiers are used only for bookkeeping and reporting, not as predictive features.

The parser merges files only after checking that input-species order, reaction order, and power-deposition labels remain consistent across files. This avoids silent column misalignment when multiple power files are combined.

### 2.2 Feature Construction

Two main feature families were evaluated:

| Feature set | Construction | Final dimensionality in v02 |
| --- | --- | --- |
| `full_nonconstant_plus_log_en_log_power` | training-scope non-constant mole fractions + `log10(E/N)` + `log10(power)` | 39 |
| `composition_pca_plus_log_en_log_power` | standardize training-scope non-constant mole fractions, PCA to 99% variance, then append `log10(E/N)` and `log10(power)` | 13 |

In v02, training-scope constant-feature removal leaves 37 varying composition channels. For the PCA-based composition representation, these 37 channels are standardized with `StandardScaler`, then reduced to 11 principal components, which retain about `99.02%` of the variance. Appending `log10(E/N)` and `log10(power)` gives a 13-dimensional compressed input representation.

The use of `log10(E/N)` is mandatory in this project because the `E/N` grid spans six orders of magnitude. Operating in raw linear scale would force the models to fit strongly nonlinear behavior induced mainly by the scale itself.

### 2.3 Target Transformation

For each reaction \(j\), the raw target \(r_j\) is transformed as

$$
y_j = \log_{10}(r_j + \varepsilon_j),
\qquad
\varepsilon_j = \max\left(\frac{\min(r_j \mid r_j > 0)}{10}, 10^{-30}\right)
$$

where $\varepsilon_j$ is estimated only from the training portion of the split. This has three purposes:

1. It stabilizes the heavy-tailed target distribution.
2. It allows exact handling of zeros without dropping zero rows.
3. It makes evaluation in log space more physically meaningful for multiplicative errors.

At inference time, the inverse map is

$$
\hat r_j = \max(10^{\hat y_j} - \varepsilon_j, 0).
$$

For `RATE CONST`, all 204 targets remain active in v02. For `SUPER RATE`, 110 targets are constant-zero over the train/validation pool and are therefore dropped for training, leaving 94 active targets. Final evaluation is still reported on the full reconstructed 204-dimensional output space.

### 2.4 Split Protocol and Leakage Control

For random-case experiments, we use a locked 15% test split with seed `42`, then 5 shuffled train/validation resamples inside the remaining 85% with seeds `7, 21, 42, 84, 126`. For the power-holdout experiment, the entire `5mJ` file is used as the locked test set, and the same 5 resampled train/validation folds are created inside the remaining powers.

All feature filtering, scaling, PCA fitting, and epsilon estimation are fit only on the training side of each fold. This is the main leakage-control rule of the pipeline.

Evidence:
- Parsing implementation: [`data.py`](../../src/global_kin_ml/data.py)
- Preprocessing implementation: [`preprocessing.py`](../../src/global_kin_ml/preprocessing.py)
- Feature metadata: [`feature_set_final_metadata.csv`](../../results/ners590_v02/training_random_case/data_snapshots/feature_set_final_metadata.csv)
- RATE CONST epsilons: [`target_epsilons_trainval.csv`](../../results/ners590_v02/training_random_case/data_snapshots/target_epsilons_trainval.csv)
- SUPER RATE epsilons: [`super_rate_epsilons_trainval.csv`](../../results/ners590_v02_super_rate/training_random_case/data_snapshots/super_rate_epsilons_trainval.csv)
- SUPER RATE target pruning: [`target_metadata.csv`](../../results/ners590_v02_super_rate/training_random_case/data_snapshots/target_metadata.csv)
- Split protocol: [`split_metadata.csv`](../../results/ners590_v02/training_random_case/tuning/split_metadata.csv), [`split_metadata.csv`](../../results/ners590_v02/training_power_holdout_5mJ/tuning/split_metadata.csv)

## 3. Models and Training Objective

### 3.1 Direct Models

We benchmarked direct multi-output predictors for the transformed targets \(Y\):

#### Ridge regression

$$
\min_W \|Y - XW\|_F^2 + \alpha \|W\|_F^2
$$

Ridge provides a linear baseline with explicit shrinkage. It is fast and interpretable, but it can underfit strongly nonlinear plasma-chemistry response surfaces.

#### Random Forest / ExtraTrees

For tree ensembles, the predictor is the average of \(T\) trees:

$$
\hat y(x) = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

Random Forest uses bootstrap sampling and random feature subsampling; ExtraTrees injects more randomness in split generation. These models are a strong fit for this problem because they handle nonlinear interactions, mixed smooth and threshold-like behavior, and multi-output regression without strong feature scaling assumptions.

#### MLP

The MLP baseline applies stacked affine layers and nonlinearities:

$$
h^{(1)} = \sigma(W_1 x + b_1), \qquad
h^{(\ell)} = \sigma(W_\ell h^{(\ell-1)} + b_\ell), \qquad
\hat y = W_o h^{(L)} + b_o
$$

with ReLU nonlinearity, optional dropout, Adam optimization, early stopping, and internal `StandardScaler` normalization on both inputs and transformed targets.

### 3.2 Latent PCA Models

We also benchmarked PCA-latent models. For transformed targets \(Y\), PCA gives

\[
Y \approx ZP^\top + \mu
\]

where \(Z \in \mathbb{R}^{n \times k}\) is a low-dimensional latent code, \(P\) is the loading matrix, and \(k \in \{2,3,4,6,8,10,12\}\). A regression model is then trained to map \(X \to Z\), and predictions are decoded back to the full 204 outputs before evaluation.

This design tests whether the chemistry can be compressed into a low-dimensional latent manifold without losing too much predictive fidelity.

### 3.3 Two-Stage Sparse Tree Model for SUPER RATE

Because `SUPER RATE` is sparse, we also tested a two-stage ExtraTrees variant:

1. Regress the transformed target magnitude.
2. Classify whether each target is active or at its zero-floor state.
3. Replace predicted inactive outputs with the learned zero-log baseline.

This was designed to separate presence/absence from magnitude prediction.

### 3.4 Selection Rule

Configurations are selected by mean validation reconstructed log-space RMSE over the 5 folds. Tie-breakers are higher mean per-reaction log-\(R^2\), lower median per-reaction log-MAE, and higher factor-of-5 accuracy.

Evidence:
- Model implementations: [`models.py`](../../src/global_kin_ml/models.py)
- Training/evaluation pipeline: [`pipeline.py`](../../src/global_kin_ml/pipeline.py)
- RATE CONST leaderboard: [`model_leaderboard_summary.csv`](../../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv)
- SUPER RATE leaderboard: [`model_leaderboard_summary.csv`](../../results/ners590_v02_super_rate/training_random_case/tuning/model_leaderboard_summary.csv)
- Combined comparison: [`leaderboard_top7_combined.csv`](../../results/ners590_v02_joint_review/leaderboard_top7_combined.csv)
- Supporting figure: [`leaderboard_top5_grid.png`](../../results/ners590_v02_joint_review/figures/leaderboard_top5_grid.png)

## 4. Main Experimental Results

### 4.1 Selected Models

The same qualitative conclusion appears across all four experiment settings: a direct ExtraTrees model is selected every time.

| Task | Split | Selected model | Test log-RMSE | Test log-MAE | Test log-\(R^2\) | Factor-of-5 accuracy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RATE CONST | Random case | Direct ExtraTrees on composition PCA + `log10(E/N)` + `log10(power)` | 0.051212 | 0.006550 | 0.999923 | 0.999000 |
| RATE CONST | Holdout 5 mJ | Direct ExtraTrees on composition PCA + `log10(E/N)` + `log10(power)` | 0.047780 | 0.008852 | 0.999722 | 0.999195 |
| SUPER RATE | Random case | Direct ExtraTrees on all inputs + `log10(E/N)` + `log10(power)` | 0.001243 | 0.000057 | 0.999999 | 1.000000 |
| SUPER RATE | Holdout 5 mJ | Direct ExtraTrees on all inputs + `log10(E/N)` + `log10(power)` | 0.003482 | 0.000245 | 0.999973 | 1.000000 |

This outcome is physically and statistically sensible. The mapping from mixture composition, `E/N`, and power to reaction rates is nonlinear, but the dataset is still tabular and structured. Tree ensembles can exploit nonlinear feature interactions without needing the much larger sample budgets that neural networks often need for stable multi-output fitting.

### 4.2 Why ExtraTrees Won

For `RATE CONST`, the best direct ExtraTrees model clearly outperforms the direct MLP (`0.0487` vs `0.0587` validation log-RMSE on the random split) and all latent models (`~0.122` to `0.126` for the best latent models). For `SUPER RATE`, direct ExtraTrees on the full input set also beats the PCA-compressed input variant, the sparse two-stage tree, the latent PCA model, and the MLP. The margins are especially large for latent compression, which suggests that forcing the output into a 12-dimensional latent representation is too restrictive when the direct tree model can already fit the full output efficiently.

### 4.3 Oracle PCA Ceiling

The oracle PCA analysis gives a useful ceiling for latent compression. For `RATE CONST`, even an oracle 12-dimensional PCA reconstruction of the true targets gives test log-RMSE around `0.052`, which is already at the same level as the best direct model. That means there is very little room for a learned latent regressor to beat the direct predictor once the outputs are compressed that aggressively.

For `SUPER RATE`, the oracle 12-dimensional reconstruction error is `0.00234` on the random split and `0.00296` on the power holdout split. The selected direct ExtraTrees model does even better than those oracle-latent ceilings because it is not constrained by the 12-dimensional PCA bottleneck.

Evidence:
- Final summary table: [`experiment_summary.csv`](../../results/ners590_v02_joint_review/experiment_summary.csv)
- RATE CONST overall metrics: [`test_overall_metrics.csv`](../../results/ners590_v02/training_random_case/evaluation/test_overall_metrics.csv), [`test_overall_metrics.csv`](../../results/ners590_v02/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv)
- SUPER RATE overall metrics: [`test_overall_metrics.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_overall_metrics.csv), [`test_overall_metrics.csv`](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/evaluation/test_overall_metrics.csv)
- RATE CONST leaderboards: [`model_leaderboard_summary.csv`](../../results/ners590_v02/training_random_case/tuning/model_leaderboard_summary.csv), [`model_leaderboard_summary.csv`](../../results/ners590_v02/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- SUPER RATE leaderboards: [`model_leaderboard_summary.csv`](../../results/ners590_v02_super_rate/training_random_case/tuning/model_leaderboard_summary.csv), [`model_leaderboard_summary.csv`](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/tuning/model_leaderboard_summary.csv)
- Oracle PCA results: [`oracle_test_overall_by_k.csv`](../../results/ners590_v02/training_random_case/pca/oracle_test_overall_by_k.csv), [`oracle_test_overall_by_k.csv`](../../results/ners590_v02/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv), [`oracle_test_overall_by_k.csv`](../../results/ners590_v02_super_rate/training_random_case/pca/oracle_test_overall_by_k.csv), [`oracle_test_overall_by_k.csv`](../../results/ners590_v02_super_rate/training_power_holdout_5mJ/pca/oracle_test_overall_by_k.csv)
- Supporting figures: [`selected_model_test_metrics.png`](../../results/ners590_v02_joint_review/figures/selected_model_test_metrics.png), [`selected_vs_oracle.png`](../../results/ners590_v02_joint_review/figures/selected_vs_oracle.png)

## 5. What the Metrics Mean

The most important metric in this project is log-space RMSE on reconstructed outputs. A low log-RMSE means the model preserves multiplicative accuracy over many orders of magnitude. For example, a log error of about `0.05` is already very small on a 204-output plasma-chemistry problem with zeros and extreme tails.

`Factor-of-2`, `factor-of-5`, and `factor-of-10` accuracies are also helpful because they translate directly into multiplicative tolerances in the original physical space. On v02, the selected `RATE CONST` models are within a factor of 5 for about `99.9%` of positive test entries, and the selected `SUPER RATE` models are effectively at `100%` factor-of-5 accuracy on positive targets.

Raw relative error should be interpreted carefully. It is defined as

\[
\frac{\hat r - r}{r}
\]

for `r > 0`. This is useful, but it becomes unstable when `r` is extremely small. For that reason we also compute SMAPE:

\[
\text{SMAPE} = \frac{2|\hat r-r|}{|\hat r| + |r|}
\]

which is bounded in `[0, 2]` and behaves much more safely near zero.

Evidence:
- RATE CONST overall relative error: [`test_relative_error_overall_summary.csv`](../../results/ners590_v02/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- RATE CONST overall SMAPE: [`test_smape_overall_summary.csv`](../../results/ners590_v02/training_random_case/evaluation/test_smape_overall_summary.csv)
- SUPER RATE overall relative error: [`test_relative_error_overall_summary.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_relative_error_overall_summary.csv)
- SUPER RATE overall SMAPE: [`test_smape_overall_summary.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_smape_overall_summary.csv)

## 6. Detailed Error Analysis: Why Small-Magnitude Targets Look Worse

The magnitude-binned plots and CSV tables show a very consistent pattern: relative error is much larger for tiny true magnitudes than for moderate or large magnitudes.

### 6.1 RATE CONST

For `RATE CONST` on the random split:

| True range | Median abs relative error | P95 abs relative error |
| --- | ---: | ---: |
| `[1e-20, 1e-18)` | `2.22e-4` | `7.55e-1` |
| `[1e-18, 1e-16)` | `1.09e-1` | `6.42e-1` |
| `[1e-14, 1e-12)` | `3.46e-3` | `8.13e-2` |
| `[1e-10, 1e-8)` | `1.17e-4` | `2.99e-3` |

So the worst relative-error behavior is concentrated in the smallest bins, even though the same model is excellent overall. This does not mean the model is failing badly in absolute terms. It means that dividing by a tiny ground-truth value amplifies even a numerically tiny discrepancy into a large percentage.

That is why the overall mean signed relative error becomes absurdly large (`1.08e11`), while the median absolute relative error is only `2.01e-4`. A small number of tiny-denominator rows dominate the arithmetic mean.

### 6.2 SUPER RATE

For `SUPER RATE` the same pattern appears, but the model is even more accurate:

| True range | Median abs relative error | P95 abs relative error |
| --- | ---: | ---: |
| `[1e-18, 1e-16)` | `8.05e-2` | `3.99e-1` |
| `[1e-16, 1e-14)` | `1.96e-2` | `6.12e-2` |
| `[1e-14, 1e-12)` | `1.56e-3` | `3.36e-2` |
| `[1e-12, 1e-10)` | `1.67e-4` | `3.01e-3` |
| `[1e-10, 1e-8)` | `4.70e-5` | `6.46e-4` |

Again, the error curve improves monotonically as target magnitude grows.

### 6.3 Why This Happens

There are four main reasons:

1. Relative error divides by the true value, so tiny targets produce unstable percentages.
2. Near-zero reactions are closer to the epsilon floor used in the log transform, so a fixed log-space deviation corresponds to a larger percentage in original space.
3. Zero-heavy targets create a mixed regime of exact zeros, tiny positives, and physically negligible reactions; this makes percentage metrics noisier than log-space metrics.
4. Some of the hardest reactions are weak channels whose absolute scale is so small that even physically negligible absolute error looks large in percent units.

This is exactly why the paper should not rely on raw relative error alone. The safer reading is: use log-RMSE as the primary metric, use factor-accuracy as an interpretable multiplicative tolerance, and use SMAPE when a bounded near-zero-safe percentage view is needed.

Evidence:
- RATE CONST relative error by magnitude: [`test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v02/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv)
- RATE CONST SMAPE by magnitude: [`test_smape_by_magnitude_bin.csv`](../../results/ners590_v02/training_random_case/evaluation/test_smape_by_magnitude_bin.csv)
- SUPER RATE relative error by magnitude: [`test_relative_error_by_magnitude_bin.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_relative_error_by_magnitude_bin.csv)
- SUPER RATE SMAPE by magnitude: [`test_smape_by_magnitude_bin.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/test_smape_by_magnitude_bin.csv)
- Supporting plots: [`relative_error_by_magnitude.png`](../../results/ners590_v02/training_random_case/figures/relative_error_by_magnitude.png), [`relative_error_by_magnitude.png`](../../results/ners590_v02_super_rate/training_random_case/figures/relative_error_by_magnitude.png), [`smape_by_magnitude.png`](../../results/ners590_v02/training_random_case/figures/smape_by_magnitude.png), [`smape_by_magnitude.png`](../../results/ners590_v02_super_rate/training_random_case/figures/smape_by_magnitude.png)

## 7. Hardest Reactions and Hardest Cases

For `RATE CONST`, the hardest reactions are concentrated in water-vibrational and ionization-related channels such as `E + H2OV > O^ + 2H + 2E`, `AR+ IONIZATION`, and `E + H2O > O^ + 2H + 2E`. Even there, reaction-wise log-\(R^2\) remains extremely high (`~0.999`), so “hardest” here means relatively harder than the rest of an already very accurate model.

The worst `RATE CONST` cases are concentrated around low local-case indices such as 7, 9, 10, and 11, suggesting that some parts of the `E/N` sweep are more challenging than others. In contrast, the worst `SUPER RATE` cases still have small case-level log-RMSE values (`0.041` at worst in the random split), which shows that the super-rate model is robust even in its most difficult examples.

For `SUPER RATE`, the hardest reactions are mostly vibrational channels such as `H2 > H2(V=2)` and `O2-VIB 3`, but their log-\(R^2\) is still above `0.9999` in the random split.

Evidence:
- Combined worst reactions: [`worst_reactions_combined.csv`](../../results/ners590_v02_joint_review/worst_reactions_combined.csv)
- RATE CONST worst cases: [`worst_10_cases.csv`](../../results/ners590_v02/training_random_case/evaluation/worst_10_cases.csv)
- SUPER RATE worst cases: [`worst_10_cases.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/worst_10_cases.csv)
- SUPER RATE worst reactions: [`worst_10_reactions.csv`](../../results/ners590_v02_super_rate/training_random_case/evaluation/worst_10_reactions.csv)
- Supporting figures: [`worst_reactions_log_rmse.png`](../../results/ners590_v02/training_random_case/figures/worst_reactions_log_rmse.png), [`worst_reactions_log_rmse.png`](../../results/ners590_v02_super_rate/training_random_case/figures/worst_reactions_log_rmse.png), [`local_case_error_profiles.png`](../../results/ners590_v02_joint_review/figures/local_case_error_profiles.png)

## 8. Conclusions

The v02 experiments support four main conclusions.

1. Direct ExtraTrees is the best baseline family for both `RATE CONST` and `SUPER RATE` on this dataset.
2. Log-space target transformation with train-only epsilon estimation is essential because the targets are heavy-tailed and zero-inflated.
3. PCA compression is useful as an analysis tool, but it is not the best predictive strategy here; the direct models are better than latent PCA regressors, and for `SUPER RATE` the direct model even beats the 12-dimensional oracle PCA compression ceiling.
4. Large percentage errors at tiny magnitudes should not be read as catastrophic model failure. They are mostly a denominator effect. Log-RMSE, factor-accuracy, and SMAPE give a much more faithful view of end-to-end predictive quality.

Until the v03 rerun finishes, this v02 paper is the current best consolidated summary of the full pipeline and its behavior.

Evidence:
- Final combined review: [`experiment_summary.csv`](../../results/ners590_v02_joint_review/experiment_summary.csv)
- Combined visual summary: [`selected_model_test_metrics.png`](../../results/ners590_v02_joint_review/figures/selected_model_test_metrics.png), [`selected_model_relative_error.png`](../../results/ners590_v02_joint_review/figures/selected_model_relative_error.png), [`leaderboard_top5_grid.png`](../../results/ners590_v02_joint_review/figures/leaderboard_top5_grid.png)
