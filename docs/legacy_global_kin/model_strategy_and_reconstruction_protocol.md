# Model Strategy and PCA Reconstruction Protocol

## Why We Are Testing Both Direct and Latent Models

The current dataset is small but highly structured:

- `609` total cases
- `49` maximum input species plus `E/N`
- `204` rate-constant outputs

The advanced dataset analysis showed three properties that directly shape the training strategy:

1. `log10(E/N)` is the dominant predictor.
2. the output space is highly correlated and strongly low-rank.
3. random-case generalization is easy enough for nonlinear models, but output precision still needs to be checked at the full 204-output level.

Primary supporting analysis outputs:

- [`feature_set_comparison.csv`](../../outputs/advanced_analysis/feature_set_comparison.csv)
- [`baseline_random_split_metrics.csv`](../../outputs/advanced_analysis/baseline_random_split_metrics.csv)
- [`split_strategy_metrics.csv`](../../outputs/advanced_analysis/split_strategy_metrics.csv)
- [`low_rank_summary.csv`](../../outputs/advanced_analysis/low_rank_summary.csv)

Because of that, the training pipeline evaluates two model families:

- direct multi-output regressors that predict all 204 transformed targets directly
- PCA-latent regressors that predict a small number of latent target coefficients and then reconstruct the 204 outputs

The direct models are important because they avoid any compression loss. The latent models are important because the targets appear to live on a much smaller manifold, which may improve robustness and reduce overfitting.

## Why `log10(E/N)` Is Mandatory

Earlier analysis already showed that raw `E/N` is a poor representation for this problem compared with `log10(E/N)`.

Supporting output:

- [`feature_set_comparison.csv`](../../outputs/advanced_analysis/feature_set_comparison.csv)

That file showed that the same baseline model performs dramatically worse with raw `E/N` than with `log10(E/N)`. The end-to-end training pipeline therefore treats `log10(E/N)` as the standard representation for every feature set.

## Why PCA Reconstruction Precision Must Be Measured Explicitly

Explained variance alone is not enough to justify PCA compression.

Even if a small number of components explains most total target variance, we still need the final 204 reconstructed outputs to be accurate enough for downstream plasma experiments. A few low-variance reactions could still matter scientifically even if they barely affect total explained variance.

That is why the training pipeline evaluates two separate error sources:

1. **oracle PCA reconstruction error**
   Fit PCA on training targets only, encode the true evaluation targets, decode them, and compare the reconstructed 204 outputs against ground truth. This isolates compression loss.

2. **model-driven reconstruction error**
   Train a regressor to predict the PCA coefficients from inputs, decode the predicted coefficients, and compare the reconstructed 204 outputs against ground truth. This measures compression loss plus regression loss.

## Reconstruction Precision Criteria

Every PCA dimension `k in {2, 3, 4, 6, 8, 10, 12}` is evaluated with:

- overall log-space RMSE, MAE, and R²
- overall original-space RMSE and MAE
- per-reaction log-space RMSE, MAE, and R²
- per-case log-space RMSE and MAE
- factor-of-2, factor-of-5, and factor-of-10 accuracy in original space

The pipeline treats log-space precision as the primary selector because the rate constants are heavy-tailed and include many small values. Original-space metrics are still saved because they are easier to interpret physically.

## Model Selection Rule

The current project prioritizes **random-case generalization**. The data are split as:

- locked random test split: `15%`, seed `42`
- training/validation pool: remaining `85%`
- model selection folds inside the pool: `5` shuffled train/validation resamples with seeds `7, 21, 42, 84, 126`

The winning model is selected by mean validation **reconstructed log-space RMSE** over the full 204 outputs. Tie-breakers are:

- mean per-reaction log-space R²
- median per-reaction log-space MAE
- factor-of-5 accuracy in original space

## Expected Experiment Outputs

The training run writes:

- parsed data snapshots under `results/full_training_pipeline/data/`
- cleaned feature and target snapshots under `results/full_training_pipeline/data_snapshots/`
- model tuning outputs under `results/full_training_pipeline/tuning/`
- oracle PCA reconstruction outputs under `results/full_training_pipeline/pca/`
- final test metrics and prediction tables under `results/full_training_pipeline/evaluation/`
- figures under `results/full_training_pipeline/figures/`

The final experiment report in [`training_experiment_report.md`](./training_experiment_report.md) cites the exact result CSV and plot files that support each conclusion.
