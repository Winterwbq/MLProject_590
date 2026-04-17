# Advanced Dataset Analysis Report

## Scope

This report summarizes the second-stage diagnostics run over the 609 parsed cases from `global_kin_boltz.out`.

The analysis covered the following 12 method families:

1. correlation and redundancy analysis
2. group-aware variation decomposition
3. target transform diagnostics
4. reaction activity analysis
5. monotonicity and trend analysis over `E/N`
6. sensitivity analysis
7. clustering
8. low-rank structure analysis
9. baseline predictability analysis
10. split strategy analysis
11. outlier and anomaly analysis
12. feature engineering analysis

The generated outputs are stored in:

- `outputs/advanced_analysis/`
- `outputs/advanced_figures/`

The file `outputs/advanced_analysis/advanced_output_manifest.csv` documents the purpose of every generated CSV and figure.

Primary source index:

- [advanced_output_manifest.csv](../../outputs/advanced_analysis/advanced_output_manifest.csv)

## Executive Summary

The advanced diagnostics strengthen the same overall conclusion from the first-pass analysis:

- the dataset is highly structured
- `E/N` is by far the dominant driver of most outputs
- composition differences matter, but much less than `E/N` for most reactions
- the target space is strongly low-rank
- the dataset appears to contain a sharp regime transition early in the 29-point `E/N` sweep

Three results stand out the most:

- The 204-target space is extremely low-dimensional. PCA on log-transformed rate constants shows that the first component explains about `87.7%` of the target variance, and the first 3 components explain about `97.1%`.
- Most reactions are governed primarily by `E/N`, not by density-group composition. For 199 out of 204 reactions, the largest variance component is the local-case axis, which corresponds directly to the `E/N` sweep.
- The 609 cases naturally split into two major behavioral regimes. Case clustering separates local cases `1` to `6` from local cases `7` to `29`, independent of density group.

These results suggest that the dataset is not an arbitrary 609-row tabular regression problem. It is much closer to a low-dimensional scientific response surface indexed by:

- one dominant control variable: `E/N`
- a smaller composition manifold across the 21 density groups

Main supporting outputs:

- [low_rank_summary.csv](../../outputs/advanced_analysis/low_rank_summary.csv)
- [reaction_variance_decomposition.csv](../../outputs/advanced_analysis/reaction_variance_decomposition.csv)
- [case_cluster_assignments.csv](../../outputs/advanced_analysis/case_cluster_assignments.csv)
- [case_clustering_target_pca.png](../../outputs/advanced_figures/case_clustering_target_pca.png)

## 1. Correlation and Redundancy Analysis

### Inputs

The input space contains substantial redundancy.

- Constant input features: 14
- Non-constant input features: 35
- Near-perfect correlated composition-feature pairs with absolute correlation at least `0.95`: 54

The strongest input redundancies include:

- `H2O2` with `O2`
- `AR4SPM` with `AR4SPR`
- `AR4SR` with `AR4SPM`
- `AR4SM` with `AR4SPR`
- `OH-` with `O-`
- `OH`, `H2OV`, and several minor oxygen-related species

This means many input species move together across the 21 density groups rather than independently.

Supporting outputs:

- [input_feature_metadata.csv](../../outputs/advanced_analysis/input_feature_metadata.csv)
- [input_correlation_top_pairs.csv](../../outputs/advanced_analysis/input_correlation_top_pairs.csv)
- [input_correlation_heatmap_top_features.png](../../outputs/advanced_figures/input_correlation_heatmap_top_features.png)

### Targets

The target space is even more redundant than the input space.

- Among the 30 most variable targets, 202 reaction pairs have absolute correlation at least `0.95`

Examples of nearly identical reaction-behavior pairs include:

- `O-3s 3S0` with `E + H2O > H + OH(A) + E`
- `OH-A3SIGMA` with `E + H2OV > H + OH(A) + E`
- `H2 > H + H` with `OH-A3SIGMA`
- `OH-A3PI` with `O2-ELECTRONIC 6EV`

The target correlations indicate that many outputs are not evolving independently. Instead, large groups of reactions share almost the same response shape across the dataset.

Supporting outputs:

- [reaction_correlation_top_pairs.csv](../../outputs/advanced_analysis/reaction_correlation_top_pairs.csv)
- [target_correlation_heatmap_top_reactions.png](../../outputs/advanced_figures/target_correlation_heatmap_top_reactions.png)

### Input-to-Target Correlations

The strongest single predictor by far is `log10(E/N)`.

Its mean absolute Spearman correlation with the 204 log-targets is:

- `0.8509`

Every non-`E/N` composition feature is far weaker. The next-largest mean absolute sensitivity is only about:

- `0.0110`

This is a decisive result. It says:

- `E/N` dominates the response behavior
- composition does matter
- but composition effects are small relative to the sweep direction

Supporting outputs:

- [feature_target_spearman_top_pairs.csv](../../outputs/advanced_analysis/feature_target_spearman_top_pairs.csv)
- [feature_sensitivity_summary.csv](../../outputs/advanced_analysis/feature_sensitivity_summary.csv)

## 2. Group-Aware Variation Decomposition

Variance decomposition was run using the full `21 x 29` structure, separating each variable into:

- density-group share
- local-case share
- interaction share

### Case-Level Physical Summaries

The dominant source of variation for all major case-level outputs is the local-case axis, which corresponds to `E/N`.

Examples:

- `average_electron_energy_ev`: local-case share is effectively `1.000000`
- `equivalent_electron_temperature_ev`: local-case share is effectively `1.000000`
- `total_power_loss_ev_cm3_per_s`: local-case share is effectively `1.000000`
- `updated_electron_density_per_cc`: local-case share `0.909`
- `ionization_coefficient_per_cm`: local-case share `0.800`

Even for the more composition-sensitive summaries, `E/N` remains the largest source of variance.

Supporting outputs:

- [case_feature_variance_decomposition.csv](../../outputs/advanced_analysis/case_feature_variance_decomposition.csv)

### Reaction Targets

For the 204 log-transformed rate constants:

- 199 reactions are dominated by local-case share
- 5 reactions are dominated by group share

Summary statistics for the reaction variance shares:

- mean local-case share: `0.971`
- median local-case share: `0.99991`
- mean group share: `0.000557`
- median group share: `3.31e-06`
- mean interaction share: `0.00391`

This means the overwhelming majority of targets are effectively controlled by `E/N`.

The few reactions with the largest composition contribution are mostly:

- `O2-VIB 2`
- `H2 > H2(V=1)`
- `E + H2O > H2OV(001) + E`
- `E + H2O > H2OV(100) + E`
- `O2-VIB 3`

Even these are still mostly local-case-driven. Their composition share is just relatively larger than the others.

Supporting outputs:

- [reaction_variance_decomposition.csv](../../outputs/advanced_analysis/reaction_variance_decomposition.csv)
- [reaction_variance_decomposition_top_local_case.png](../../outputs/advanced_figures/reaction_variance_decomposition_top_local_case.png)

## 3. Target Transform Diagnostics

Three transforms were compared:

- raw rate constants
- `log10(rate + eps)`
- `asinh(rate / eps)`

At face value, raw targets show lower average absolute skewness than the transformed targets. However, that result is partly caused by zero inflation and by reactions that are constant or almost constant over large parts of the dataset.

The more important practical interpretation is:

- raw targets keep the enormous dynamic range intact
- log and asinh transforms compress the tail and better expose relative differences among active reactions

The reaction-level transform summaries show that several transformed targets remain highly skewed, especially for intermittent reactions such as:

- `OH-V1`
- `AR4SPR>AR4P,3D,5S,5P`
- `AR4SPM>AR4P,3D,5S,5P`
- `AR4D > AR^`

So no transform fully removes the regime structure, but transformed targets are still more appropriate for modeling because the raw scale spans many orders of magnitude.

Supporting outputs:

- [target_transform_global_summary.csv](../../outputs/advanced_analysis/target_transform_global_summary.csv)
- [target_transform_reaction_summary.csv](../../outputs/advanced_analysis/target_transform_reaction_summary.csv)
- [target_transform_skewness_comparison.png](../../outputs/advanced_figures/target_transform_skewness_comparison.png)

## 4. Reaction Activity Analysis

Reaction activity is highly structured and almost entirely synchronized across density groups.

### Always-Off and Always-On Reactions

- 5 reactions are always zero
- 86 reactions are always nonzero
- 113 reactions are conditionally active

The always-zero reactions are exactly the same five "Not Used" channels found earlier.

### Activation Thresholds

Many conditionally active reactions switch on at the same local-case index in every density group.

Examples:

- `AR3S > AR^^` first activates at local case 17 in every group
- `E + H2O > H2^ + O + E` first activates at local case 11 in every group
- `AR+ IONIZATION` first activates at local case 10 in every group
- `AR3S > AR-RYD` first activates at local case 8 in every group

This is a strong sign that many activation thresholds are effectively determined by `E/N` alone, not by composition.

### Active Reaction Count by Local Case

The mean number of active reactions is identical across all density groups at each local case:

- case 1: 86
- case 2: 98
- case 6: 122
- case 7: 175
- case 8: 193
- case 17 onward: 199

This is one of the clearest findings in the whole analysis. It means the dataset has a deterministic activation staircase over the 29-point sweep.

Supporting outputs:

- [reaction_activity_summary.csv](../../outputs/advanced_analysis/reaction_activity_summary.csv)
- [reaction_activation_by_local_case.csv](../../outputs/advanced_analysis/reaction_activation_by_local_case.csv)
- [reaction_activity_summary.png](../../outputs/advanced_figures/reaction_activity_summary.png)
- [reaction_activation_by_local_case.png](../../outputs/advanced_figures/reaction_activation_by_local_case.png)

## 5. Monotonicity and Trend Analysis Over `E/N`

### Case-Level Summaries

Several physical outputs are perfectly monotonic with `E/N` across every density group:

- `average_electron_energy_ev`
- `equivalent_electron_temperature_ev`
- `total_power_loss_ev_cm3_per_s`

Each of these has:

- mean Spearman rho = `1.0`
- significant monotonicity in all 21 groups

Other variables remain strongly monotonic, though less perfectly:

- `updated_electron_density_per_cc`: mean rho `0.756`
- `ionization_coefficient_per_cm`: mean rho `0.700`
- `drift_velocity_cm_per_s`: mean rho `0.694`

Supporting outputs:

- [case_feature_monotonicity.csv](../../outputs/advanced_analysis/case_feature_monotonicity.csv)

### Reaction Targets

Many reactions are also perfectly monotonic across all 21 density groups.

Strong positive monotonic examples:

- `AR4SM-ELASTIC`
- `AR4SR > AR3S`
- `AR4D-ELASTIC`
- `H* -ELASTIC`
- `O-MOMENTUM`

Strong negative monotonic examples:

- `H3+ > H2 + H`
- `AR2+ -RECOMBINATION`
- `H2O+ -RECOMBINATION`
- `O2+ -RECOMBINATION`
- `H- -ELASTIC`

Weak-monotonic or nearly non-monotonic reactions include:

- `OH-V1`
- `AR4SM > AR4SR`
- `H2-ELASTIC`
- `H2* -ELASTIC`
- `O2-VIB 2`

This means a large fraction of the target space follows smooth one-directional trends in `E/N`, while a smaller subset contains more shape complexity.

Supporting outputs:

- [reaction_monotonicity_summary.csv](../../outputs/advanced_analysis/reaction_monotonicity_summary.csv)
- [reaction_monotonicity_histogram.png](../../outputs/advanced_figures/reaction_monotonicity_histogram.png)

## 6. Sensitivity Analysis

Sensitivity analysis confirmed that composition features have only weak global associations with the targets compared with `E/N`.

Key result:

- `log10(E/N)` mean absolute target correlation: `0.8509`
- best non-`E/N` feature mean absolute target correlation: about `0.011`

The most sensitive composition-side features are small argon-water plasma byproducts such as:

- `AR2*`
- `OH*`
- `E`
- `AR4SM`
- `AR4SR`
- `AR4SPM`

But their effects are still tiny compared with the `E/N` term.

Several non-constant composition features show essentially zero measurable global sensitivity, including:

- `H2^`
- `H-`
- `O^`
- `O2-`
- `H^`
- `HO2`
- `O2^`
- `OH^`
- `AR4D`
- `H3^`
- `AR2^`
- `H*`
- `H2*`

This does not necessarily mean those species are physically irrelevant. It means their marginal variation across the 21 compositions is too small or too collinear with other species to stand out in a simple global correlation analysis.

Supporting outputs:

- [feature_sensitivity_summary.csv](../../outputs/advanced_analysis/feature_sensitivity_summary.csv)
- [feature_target_spearman_top_pairs.csv](../../outputs/advanced_analysis/feature_target_spearman_top_pairs.csv)

## 7. Clustering

### Density-Group Clustering

Clustering the 21 composition groups gave the best silhouette score at:

- `k = 4`

Cluster sizes:

- cluster 0: 3 groups
- cluster 1: 8 groups
- cluster 2: 8 groups
- cluster 3: 2 groups

These clusters are mainly separated by the increasing concentrations of:

- `H`
- `OH`
- `H2OV`
- `H2O2`
- `H2`

So the composition progression is not random; it forms a small number of well-defined mixture regimes.

Supporting outputs:

- [density_group_cluster_scores.csv](../../outputs/advanced_analysis/density_group_cluster_scores.csv)
- [density_group_cluster_assignments.csv](../../outputs/advanced_analysis/density_group_cluster_assignments.csv)
- [density_group_cluster_profiles.csv](../../outputs/advanced_analysis/density_group_cluster_profiles.csv)
- [density_group_clustering_pca.png](../../outputs/advanced_figures/density_group_clustering_pca.png)

### Reaction Clustering

Reaction clustering produced a surprisingly simple result:

- cluster 0: 199 reactions
- cluster 1: 5 reactions

The second cluster is exactly the always-zero reactions.

This means the remaining 199 reactions are so strongly aligned in their broad response structure that unsupervised clustering does not find further stable coarse clusters beyond the trivial separation of always-zero channels.

Supporting outputs:

- [reaction_cluster_scores.csv](../../outputs/advanced_analysis/reaction_cluster_scores.csv)
- [reaction_cluster_assignments.csv](../../outputs/advanced_analysis/reaction_cluster_assignments.csv)
- [reaction_cluster_profiles.csv](../../outputs/advanced_analysis/reaction_cluster_profiles.csv)

### Case Clustering

Case clustering gave the best silhouette score at:

- `k = 2`

The clusters separate perfectly by local-case index:

- cluster 1 contains local cases 1 through 6
- cluster 0 contains local cases 7 through 29

This is an important result. It indicates a natural low-`E/N` versus higher-`E/N` regime break near the transition from case 6 to case 7.

Supporting outputs:

- [case_cluster_scores.csv](../../outputs/advanced_analysis/case_cluster_scores.csv)
- [case_cluster_assignments.csv](../../outputs/advanced_analysis/case_cluster_assignments.csv)
- [case_clustering_target_pca.png](../../outputs/advanced_figures/case_clustering_target_pca.png)

## 8. Low-Rank Structure

### Input Composition Manifold

PCA on the 21 unique density-group compositions shows:

- PC1 explains `47.2%`
- PCs 1-2 explain `71.3%`
- PCs 1-3 explain `92.1%`
- PCs 1-4 explain `95.7%`

So the composition space is effectively low-dimensional. Three to four latent composition axes capture almost all of the structured variation.

Supporting outputs:

- [composition_pca_explained_variance.csv](../../outputs/advanced_analysis/composition_pca_explained_variance.csv)

### Output Target Manifold

PCA on the log-transformed target matrix shows:

- PC1 explains `87.7%`
- PCs 1-2 explain `94.0%`
- PCs 1-3 explain `97.1%`
- PCs 1-6 explain `99.35%`

The effective rank estimates are:

- inputs: about `3.79`
- targets: about `1.73`

This is one of the strongest modeling clues in the project. It suggests the 204 rate constants live on an extremely low-dimensional manifold once the main control direction is captured.

Supporting outputs:

- [target_pca_explained_variance.csv](../../outputs/advanced_analysis/target_pca_explained_variance.csv)
- [low_rank_summary.csv](../../outputs/advanced_analysis/low_rank_summary.csv)
- [target_pca_scree.png](../../outputs/advanced_figures/target_pca_scree.png)

## 9. Baseline Predictability Analysis

Baseline models were evaluated on a random 80/20 case split using the `full_nonconstant_plus_log_en` feature set.

Results:

- Random forest: overall `R2 = 0.996`
- PLS: overall `R2 = 0.815`
- Ridge: overall `R2 = 0.813`
- Linear regression: overall `R2 = 0.807`

Interpretation:

- The problem is very learnable under random splitting.
- Even simple linear latent models already explain more than 80% of the variance in log-target space.
- The random forest captures the structured manifold extremely well on interpolation-type splits.

This is encouraging, but random splits are the easiest setting for this dataset.

Supporting outputs:

- [baseline_random_split_metrics.csv](../../outputs/advanced_analysis/baseline_random_split_metrics.csv)
- [baseline_model_comparison.png](../../outputs/advanced_figures/baseline_model_comparison.png)

## 10. Split Strategy Analysis

Three split strategies were compared:

- random case split
- density-group holdout
- local-case holdout

### Random Split

Best model:

- random forest with overall `R2 = 0.996`

### Density-Group Holdout

Held-out groups:

- `[1, 2, 9, 16, 18]`

Best model:

- random forest with overall `R2 = 0.988`

Linear latent models still perform well here:

- Ridge: `0.790`
- PLS: `0.790`

This suggests the 21 density groups are similar enough that the model can generalize across unseen compositions fairly well.

### Local-Case Holdout

Held-out local cases:

- `[9, 10, 13, 17, 23, 28]`

Results:

- random forest: overall `R2 = 0.579`
- Ridge: overall `R2 = -1713`
- PLS: overall `R2 = -1717`

This is the hardest split by far.

Interpretation:

- holding out entire `E/N` positions is much harder than holding out density groups
- the mapping over `E/N` is not well approximated by a single linear relationship
- local-case generalization is the key modeling challenge

This also explains why linear models issued numerical warnings in this analysis section. They are effectively failing on a strongly structured held-out `E/N` interpolation problem.

Supporting outputs:

- [split_strategy_metadata.csv](../../outputs/advanced_analysis/split_strategy_metadata.csv)
- [split_strategy_metrics.csv](../../outputs/advanced_analysis/split_strategy_metrics.csv)
- [split_strategy_comparison.png](../../outputs/advanced_figures/split_strategy_comparison.png)

## 11. Outlier and Anomaly Analysis

The strongest anomalies are concentrated at two kinds of cases:

- the highest `E/N` cases, especially local case 29
- the early transition region around local cases 4 to 6

Top anomalies include:

- case `(density group 1, local case 29)`
- case `(density group 1, local case 5)`
- case `(density group 21, local case 29)`
- case `(density group 12, local case 29)`

This pattern makes sense physically and structurally:

- very high `E/N` cases sit at the extreme end of the target manifold
- early transition cases sit near the regime shift where the number of active reactions grows rapidly

At the reaction level, the largest outlier propensities occur for:

- `H2 > H2(V=2)`
- `AR4P > AR4D,6S,RYD`
- `O2-SINGLET DELTA`
- `AR4D > AR^`
- `OH-V1`

These are likely the reactions with the sharpest localized excursions relative to their own typical behavior.

Supporting outputs:

- [case_anomaly_scores.csv](../../outputs/advanced_analysis/case_anomaly_scores.csv)
- [top_case_anomalies.csv](../../outputs/advanced_analysis/top_case_anomalies.csv)
- [reaction_outlier_propensity.csv](../../outputs/advanced_analysis/reaction_outlier_propensity.csv)
- [case_anomaly_scatter.png](../../outputs/advanced_figures/case_anomaly_scatter.png)

## 12. Feature Engineering Analysis

Four feature sets were compared using a Ridge baseline on a random split.

Results:

- `composition_pca5_plus_log_en`: `R2 = 0.816`
- `major_species_plus_log_en`: `R2 = 0.816`
- `full_nonconstant_plus_log_en`: `R2 = 0.813`
- `full_nonconstant_plus_raw_en`: `R2 = 0.178`

Two findings are especially important:

- using `log10(E/N)` is dramatically better than using raw `E/N`
- compact feature sets perform as well as or slightly better than the full non-constant feature set

This means the dataset does not need a high-dimensional raw feature representation to be predictable. A compact engineered representation may actually be better.

Supporting outputs:

- [feature_set_comparison.csv](../../outputs/advanced_analysis/feature_set_comparison.csv)
- [feature_set_comparison.png](../../outputs/advanced_figures/feature_set_comparison.png)

## Overall Interpretation

Putting all 12 analyses together, the dataset appears to have the following structure:

### What drives the data

- `E/N` is the main control variable for almost everything.
- Density-group composition matters, but usually only as a secondary modifier.
- Many reaction activation thresholds are effectively fixed by local-case index alone.

### Geometry of the problem

- The input compositions lie on a low-dimensional manifold.
- The 204-dimensional target space is even lower dimensional than the inputs.
- Most target variation is explained by one dominant response direction plus a few smaller correction modes.

### Regime behavior

- The dataset contains a strong low-`E/N` regime covering local cases 1 to 6.
- A different regime starts around local case 7.
- This regime transition appears in case clustering, reaction activation counts, anomaly ranking, and model generalization behavior.

### Modeling implications

- Random row-wise splitting is overly optimistic for this dataset.
- Density-group holdout is useful, but not the hardest test.
- Local-case holdout is the most revealing split for generalization across the `E/N` axis.
- `log10(E/N)` should be preferred over raw `E/N`.
- Compact input representations such as composition PCA or major-species subsets are reasonable and may be preferable.
- The target side should likely be modeled in a transformed space, and possibly with a low-rank or latent-target strategy.

## Recommended Next ML Steps

Based on the full analysis, the strongest next steps would be:

1. Build train/validation/test protocols that explicitly include local-case holdout and density-group holdout.
2. Use `log10(E/N)` rather than raw `E/N`.
3. Remove always-zero species and always-zero reactions.
4. Start with compact feature sets such as:
   - composition PCA plus `log10(E/N)`
   - major species plus `log10(E/N)`
5. Consider multi-output models that can exploit the strong low-rank structure of the targets.
6. Consider a two-stage target strategy for harder reactions:
   - reaction active/inactive classification
   - conditional magnitude regression

## Bottom Line

This is a highly learnable dataset, but it is learnable because it is strongly structured.

The most useful way to think about it is not:

- 609 independent samples with 49 inputs and 204 outputs

but rather:

- 21 related compositions
- evaluated over a deterministic 29-point `E/N` sweep
- producing targets that live on a very low-dimensional manifold with a clear early regime transition

That structure is an advantage for modeling, as long as the validation strategy respects it.
