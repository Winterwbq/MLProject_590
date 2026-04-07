# NERS590 v02 Analysis Report

This report summarizes the first full cleaning and analysis pass on the merged multi-file dataset. All conclusions below are based on the actual outputs in [`results/ners590_v02/analysis`](../results/ners590_v02/analysis) and [`results/ners590_v02/figures`](../results/ners590_v02/figures).

## Main Sources

- dataset summary: [`dataset_summary.csv`](../results/ners590_v02/analysis/dataset_summary.csv)
- power summary: [`power_summary.csv`](../results/ners590_v02/analysis/power_summary.csv)
- case-feature summary by power: [`case_features_by_power.csv`](../results/ners590_v02/analysis/case_features_by_power.csv)
- rate summary by power: [`rate_by_power_summary.csv`](../results/ners590_v02/analysis/rate_by_power_summary.csv)
- composition alignment across powers: [`composition_alignment_across_powers.csv`](../results/ners590_v02/analysis/composition_alignment_across_powers.csv)
- input species by power: [`input_species_by_power_summary.csv`](../results/ners590_v02/analysis/input_species_by_power_summary.csv)
- reaction sensitivity to power: [`reaction_power_sensitivity_summary.csv`](../results/ners590_v02/analysis/reaction_power_sensitivity_summary.csv)
- rate-case summary: [`rate_case_summary.csv`](../results/ners590_v02/analysis/rate_case_summary.csv)
- figures: [`results/ners590_v02/figures`](../results/ners590_v02/figures)

## 1. Structure of the New Dataset

The merged dataset contains `32,045` cases organized as `5` power levels x `221` density groups per power x `29` local `E/N` positions. This is recorded in [`dataset_summary.csv`](../results/ners590_v02/analysis/dataset_summary.csv) and [`power_summary.csv`](../results/ners590_v02/analysis/power_summary.csv).

Each power level contributes exactly:

- `6,409` cases
- `221` density groups
- `29` local cases
- the same `E/N` range from `1e-20` to `1e-14`

This consistency is also visible in [`e_over_n_by_power.csv`](../results/ners590_v02/analysis/e_over_n_by_power.csv) and the figure [`e_over_n_grid_by_power.png`](../results/ners590_v02/figures/e_over_n_grid_by_power.png).

## 2. Power Is a Real New Control Variable, Not Just a Duplicate Label

Power changes the dataset in a systematic but not catastrophic way.

From [`composition_alignment_across_powers.csv`](../results/ners590_v02/analysis/composition_alignment_across_powers.csv):

- density-group index `1` is exactly unchanged across powers
- most other matched density groups do change across powers
- for many groups, `34-35` species show some change across power
- the changes are small in absolute mole-fraction scale, but they are not zero

This means the same `density_group_in_file_id` across powers appears to represent the same nominal trajectory index, but not an identical composition vector. So power should be treated as an explicit model input, not as something that can be ignored after merging.

## 3. The Dominant Composition Shifts Across Power Are Physically Coherent

The strongest cross-power mean shifts, from [`input_species_by_power_summary.csv`](../results/ners590_v02/analysis/input_species_by_power_summary.csv), are:

- `H2O`: mean drops from about `1.9943e-02` at `1mJ` to `1.9705e-02` at `5mJ`
- `AR3S`: mean drops slightly from `9.8000e-01` to `9.799149e-01`
- `H2`, `H`, `OH`, `H2OV`, and `H2O2` all increase with power

So higher power appears to move the mixture away from the base `AR3S + H2O` state and toward more dissociation / excited / secondary-product content. The figure [`input_species_mean_heatmap_by_power.png`](../results/ners590_v02/figures/input_species_mean_heatmap_by_power.png) helps visualize that shift.

## 4. The E/N Grid Is Stable Across Power

The `29`-point `E/N` sweep is reused identically across all five power levels according to [`power_summary.csv`](../results/ners590_v02/analysis/power_summary.csv), [`e_over_n_by_power.csv`](../results/ners590_v02/analysis/e_over_n_by_power.csv), and [`power_grid_consistency.csv`](../results/ners590_v02/analysis/power_grid_consistency.csv).

This is very useful for learning because it means power effects can be compared at matched local-case positions rather than being confounded by a different `E/N` grid.

## 5. Case-Level Plasma Features Change with Power in an Uneven Way

From [`case_features_by_power.csv`](../results/ners590_v02/analysis/case_features_by_power.csv):

- mean electron energy changes only slightly: about `5.8696 eV` at `1mJ` to `5.8657 eV` at `5mJ`
- mean updated electron density increases strongly: about `9.62e9` to `6.88e10`
- mean drift velocity stays almost flat around `1.17e7`
- mean mobility decreases gradually from about `2425` to `2269`

So power has a strong effect on density-related quantities, but a much weaker effect on average electron energy. The figure [`electron_energy_vs_power.png`](../results/ners590_v02/figures/electron_energy_vs_power.png) makes the weak energy trend visible.

## 6. Target Sparsity Is Stable, but Target Magnitudes Drift with Power

The average number of nonzero rate constants per case remains constant at about `177.14` for all power levels according to [`rate_case_summary.csv`](../results/ners590_v02/analysis/rate_case_summary.csv).

But the magnitudes shift:

- mean max rate constant per case decreases from about `7.47e-04` at `1mJ` to `6.46e-04` at `5mJ`
- mean sum of rate constants per case decreases from about `8.968e-03` at `1mJ` to `7.763e-03` at `5mJ`
- overall rate mean in [`rate_by_power_summary.csv`](../results/ners590_v02/analysis/rate_by_power_summary.csv) also decreases steadily with power

This means power does not simply turn many reactions on or off. Instead, it shifts the magnitude profile of an already active multi-reaction system.

The figure [`median_rate_by_power.png`](../results/ners590_v02/figures/median_rate_by_power.png) shows the same directional effect at the distribution level.

## 7. Power Sensitivity Is Reaction-Specific and Concentrated in Certain Channels

The most power-sensitive reactions in [`reaction_power_sensitivity_summary.csv`](../results/ners590_v02/analysis/reaction_power_sensitivity_summary.csv) include:

- `E + H2OV > H2^ + O + E`
- `E + H2OV > O^ + 2H + 2E`
- `O2-VIB 4`
- `O2-VIB 3`
- `E + H2O > H^ + OH + 2E`
- `E + H2OV > H^ + OH + 2E`
- `H2 > H2(V=1)`
- `AR+ IONIZATION`

Their mean log-space standard deviation across powers is around `0.048` to `0.056`, which is noticeable but still moderate. So power sensitivity is real, but it is not so large that the mapping becomes discontinuous or chaotic.

This is visualized in [`top_power_sensitive_reactions.png`](../results/ners590_v02/figures/top_power_sensitive_reactions.png).

## 8. Overall Modeling Implication from the Analysis

The v02 dataset looks much more favorable for machine learning than the original `609`-case dataset:

- it is much larger (`32,045` cases)
- it preserves a clean repeated-grid structure
- `E/N` remains a stable ordered axis
- power is an additional continuous control variable
- matched compositions across powers are similar enough to support transfer learning, but different enough that power should not be dropped

The most important modeling implication is:

- use power explicitly as a feature
- keep `E/N` and power on log scales
- expect some reactions to be more power-sensitive than others
- expect generalization across unseen power to be harder than random-case interpolation, but still learnable

## Bottom Line

The merged v02 dataset is internally consistent, power-aware, and suitable for the next-stage ML pipeline. Power changes both input compositions and target magnitudes in systematic ways, while preserving the same underlying `29`-point `E/N` sweep. That makes the new dataset a stronger scientific learning problem than the old one, not just a larger copy of it.
