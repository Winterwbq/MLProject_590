# NERS590 v03 Dataset Analysis Report

This report summarizes the expanded `NERS590_data_V03` dataset used for the new RATE CONST, SUPER RATE, and multitask experiments.

## Dataset Scope

The v03 dataset contains `83,317` cases across `13` power files, spanning `0.1mJ`, `0.2mJ`, `0.5mJ`, and integer powers from `1mJ` through `10mJ`. Each power file contributes `6,409` cases, corresponding to `221` density groups and a `29`-point local `E/N` sweep. The modeling input space remains `49` input gas mole-fraction channels plus electric-field density information, and the main target families each preserve the `204` reaction-channel layout.

This layout is important because v03 keeps the same structured per-power grid as v02 while adding more power regimes. That makes it appropriate to train unified power-aware models with `log10(power_mJ)` as an input, and also to keep a strict `10mJ` power-holdout benchmark for testing generalization to an unseen power file.

Citations: [`dataset_summary.csv`](../../results/ners590_v03_analysis/analysis/dataset_summary.csv), [`power_summary.csv`](../../results/ners590_v03_analysis/analysis/power_summary.csv), [`parser_file_summary.csv`](../../results/ners590_v03_analysis/parsed/parser_file_summary.csv), [`case_count_by_power.png`](../../results/ners590_v03_analysis/figures/case_count_by_power.png), [`e_over_n_grid_by_power.png`](../../results/ners590_v03_analysis/figures/e_over_n_grid_by_power.png).

## Input And Power Structure

The dominant input species remain physically interpretable. For example, `AR3S` is near `0.98`, `H2O` is near `0.02`, while reactive/intermediate species such as `E`, `H2O^`, `H`, `H2OV`, `H2`, and `OH` occupy smaller but nontrivial ranges. Several reactive species are nonzero in almost all cases but still have zeros in a small subset, so train-only constant-feature detection remains necessary.

The power grid is clean and balanced: every power has the same number of cases, density groups, and `E/N` values. Therefore, power should not be encoded as a file ID; it should remain a numeric physics input. We continue using `log10(power_mJ)` because the power levels cover a wide range from `0.1` to `10mJ`, and log scaling makes that range less dominated by the largest powers.

Citations: [`input_species_summary.csv`](../../results/ners590_v03_analysis/analysis/input_species_summary.csv), [`input_species_by_power_summary.csv`](../../results/ners590_v03_analysis/analysis/input_species_by_power_summary.csv), [`power_grid_consistency.csv`](../../results/ners590_v03_analysis/analysis/power_grid_consistency.csv), [`input_species_mean_heatmap_by_power.png`](../../results/ners590_v03_analysis/figures/input_species_mean_heatmap_by_power.png).

## RATE CONST Distribution

The RATE CONST target matrix has `16,996,668` scalar entries. It is heavy-tailed and partially zero-valued: `14,758,601` entries are positive, while `2,238,067` entries are zero. The median is only about `5.869e-10`, while the maximum reaches about `6.326e-03`, so raw-space loss would be poorly balanced across reaction channels and magnitudes.

Across powers, the overall RATE CONST mean decreases gradually from about `5.0e-05` at `0.1mJ` to lower values at larger powers, while the grid structure stays fixed. The reactions most sensitive to power include several O2 vibrational and ionization-related channels, which supports keeping a power-holdout split in addition to the random-case split.

Citations: [`rate_overall_summary.csv`](../../results/ners590_v03_analysis/analysis/rate_overall_summary.csv), [`rate_by_power_summary.csv`](../../results/ners590_v03_analysis/analysis/rate_by_power_summary.csv), [`reaction_power_sensitivity_summary.csv`](../../results/ners590_v03_analysis/analysis/reaction_power_sensitivity_summary.csv), [`median_rate_by_power.png`](../../results/ners590_v03_analysis/figures/median_rate_by_power.png), [`top_power_sensitive_reactions.png`](../../results/ners590_v03_analysis/figures/top_power_sensitive_reactions.png).

## SUPER RATE Distribution

SUPER RATE is much sparser than RATE CONST. It has the same `16,996,668` scalar entry count, but only `6,688,344` entries are positive, giving a positive fraction of about `0.3935`. This is why SUPER RATE training keeps the full 204-channel target definition for final reconstruction but drops constant-zero channels from the learned target matrix. In v03, the active SUPER RATE training target count remains `94`, while `110` constant-zero channels are restored as zeros during final evaluation.

The positive fraction is very stable across powers, which suggests that basic SUPER RATE sparsity is not a file-specific artifact. The local-case plots show the stronger structure: activity changes sharply across the `29` local `E/N` positions, so `log10(E/N)` remains mandatory.

Citations: [`super_rate_overall_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_overall_summary.csv), [`super_rate_by_power_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_by_power_summary.csv), [`super_rate_reaction_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_reaction_summary.csv), [`super_rate_positive_fraction_by_power.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_positive_fraction_by_power.png), [`super_rate_nonzero_count_by_local_case.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_nonzero_count_by_local_case.png).

## RATE CONST And SUPER RATE Relationship

The relationship analysis confirms the physics intuition: many SUPER RATE channels are strongly related to the corresponding RATE CONST channel because both are computed from the same underlying energy distribution, even though they use different coefficients. The strongest channels have nearly perfect positive log-correlation, and many active channels have matching support patterns.

However, this does not mean a joint neural model will necessarily outperform a strong separate-task model. The relationship summary says the two target families share signal; the training results must still decide whether the specific model class can exploit that shared signal better than the separate ExtraTrees baselines.

Citations: [`super_rate_rate_const_relationship_summary.csv`](../../results/ners590_v03_analysis/super_rate_analysis/super_rate_rate_const_relationship_summary.csv), [`super_rate_rate_const_correlation_top.png`](../../results/ners590_v03_analysis/super_rate_figures/super_rate_rate_const_correlation_top.png).

## Preprocessing Implications

The v03 analysis supports the same preprocessing choices as v02, with the added power input:

- Use mole fractions plus `log10(E/N)` and `log10(power_mJ)` as numeric features.
- Detect nonconstant composition features using only the training pool for each split.
- Transform targets as `log10(target + epsilon_j)`, with per-channel epsilon estimated from the training pool.
- Drop constant-zero SUPER RATE channels during training, but reconstruct the final 204-output vector before evaluation.
- Evaluate both random-case interpolation and 10mJ power-holdout generalization.

Citations: [`feature_set_final_metadata.csv`](../../results/ners590_v03_rate_const/training_random_case/data_snapshots/feature_set_final_metadata.csv), [`target_metadata.csv`](../../results/ners590_v03_super_rate/training_random_case/data_snapshots/target_metadata.csv), [`multitask_target_summary.csv`](../../results/ners590_v03_multitask/training_random_case/data_snapshots/multitask_target_summary.csv), [`split_assignments.csv`](../../results/ners590_v03_multitask/training_power_holdout_10mJ/data_snapshots/split_assignments.csv).

