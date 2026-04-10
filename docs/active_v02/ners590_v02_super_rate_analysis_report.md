# NERS590 v02 SUPER RATE Analysis Report

## Scope

This report summarizes a dedicated exploratory analysis of the `SUPER RATE (CC/S)` outputs in the active multi-file dataset under [NERS590_data_v02](../../NERS590_data_v02). The goal is to understand the distribution, sparsity, power dependence, reaction-level structure, and relationship to the already-studied `RATE CONST` target so we can decide how to model `SUPER RATE` more effectively.

The analysis was run on the parsed merged v2 dataset, using the long-form reaction table at [rate_constants_long.csv](../../results/ners590_v02/parsed/rate_constants_long.csv), and exported to CSV-first outputs under [super_rate_analysis](../../results/ners590_v02/super_rate_analysis) with companion plots under [super_rate_figures](../../results/ners590_v02/super_rate_figures). Supporting sources: [super_rate_analysis_manifest.csv](../../results/ners590_v02/super_rate_analysis/super_rate_analysis_manifest.csv), [super_rate_figure_manifest.csv](../../results/ners590_v02/super_rate_figures/super_rate_figure_manifest.csv), [rate_constants_long.csv](../../results/ners590_v02/parsed/rate_constants_long.csv).

## Dataset Shape for SUPER RATE

The `SUPER RATE` analysis uses the same merged v2 case grid as the rest of the project: `32,045` total cases from `5` power files, each file contributing `6,409` cases arranged as `221` density groups x `29` local cases. Because `SUPER RATE` is reported for each of the `204` reactions, the full target table contains `6,537,180` scalar entries (`32,045 x 204`).

This is important for modeling because the target is large enough to support richer diagnostics than the earlier legacy dataset, and it preserves the same structured hierarchy: power, density-group composition, and local-case / `E/N` sweep. Supporting sources: [dataset_summary.csv](../../results/ners590_v02/analysis/dataset_summary.csv), [parser_summary_copy.csv](../../results/ners590_v02/super_rate_analysis/parser_summary_copy.csv), [super_rate_overall_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_overall_summary.csv).

## Overall Distribution

`SUPER RATE` is substantially sparser than `RATE CONST`. Across all `6,537,180` target entries, `2,572,440` are positive and `3,964,740` are exactly zero, so only about `39.35%` of the matrix is active. The overall median is exactly `0`, and even the 75th percentile is only about `7.509e-10`, which means more than half of all reaction outputs are zero in a typical case and most nonzero values are still quite small.

At the same time, the positive tail is meaningful rather than negligible. The positive-only median is about `2.191e-09`, the positive-only 95th percentile is about `1.945e-07`, and the maximum observed value reaches `1.103e-06`. So the target combines heavy sparsity with a physically relevant nonzero band.

This immediately suggests that direct regression on raw values will be harder than for `RATE CONST`. A future training pipeline for `SUPER RATE` should probably use log-space on positive values, explicitly track zero vs nonzero behavior, and consider whether always-zero channels should be removed before training. Supporting sources: [super_rate_overall_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_overall_summary.csv).

## Power Dependence at the Global Level

At the global distribution level, power has only a weak effect on `SUPER RATE`. The fraction of positive entries is identical across all five powers at `0.393509`, and the mean, upper quantiles, and positive-only median change only slightly from `1mJ` to `5mJ`. The same pattern appears at the case-summary level: the average number of nonzero `SUPER RATE` channels per case is `80.275862` for every power, while the average case sum and mean case maximum drift only mildly with increasing power.

This means power should still be included as an input feature, but it does not look like the dominant source of variation for `SUPER RATE` when we aggregate across the entire dataset. In other words, power matters, but much less strongly than one might guess just from the file organization.

For prediction, that points toward a unified power-aware model being reasonable. We likely do not need separate per-power models just to capture the basic global `SUPER RATE` distribution. Supporting sources: [super_rate_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_by_power_summary.csv), [super_rate_case_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_case_by_power_summary.csv), ![Positive fraction by power](../../results/ners590_v02/super_rate_figures/super_rate_positive_fraction_by_power.png), ![Positive median by power](../../results/ners590_v02/super_rate_figures/super_rate_positive_median_by_power.png), ![Case sum vs power](../../results/ners590_v02/super_rate_figures/super_rate_case_sum_vs_power.png).

## Local-Case / E-over-N Structure

The strongest dataset-level driver of `SUPER RATE` is the 29-point local-case sweep, which corresponds to the structured within-group plasma-state progression used throughout the dataset. The mean number of nonzero `SUPER RATE` outputs per case rises sharply from `19` in local case `1` to `39` by local case `6`, then jumps to `83` in local case `7` and stabilizes around `94` active reactions from local cases `8` through `29`.

The total `SUPER RATE` mass per case also increases along the sweep. The mean case sum grows from about `5.33e-07` in local case `1` to about `4e-06` in local cases `28-29`, while the mean case maximum rises from about `2.45e-07` to `1.103e-06`. In contrast, the per-case median remains `0.0` for every local case, confirming that sparsity remains central even when activity increases.

This is a strong modeling clue: the target is not only sparse, it is regime-structured. A model that captures smooth dependence on the local-case / `E/N` coordinate should do much better than one that treats cases as an unstructured cloud. Supporting sources: [super_rate_local_case_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_local_case_summary.csv), ![Nonzero count by local case](../../results/ners590_v02/super_rate_figures/super_rate_nonzero_count_by_local_case.png), ![Sum by local case](../../results/ners590_v02/super_rate_figures/super_rate_sum_by_local_case.png).

## Reaction-Level Structure

`SUPER RATE` is highly uneven across the 204 reactions. A relatively small set of channels carries most of the mass. The highest-mean reactions include `H2O2-MOM`, several `AR4S* -> AR4P,3D,5S,5P` channels, `AR4SM > AR4SR`, `AR4P > AR4D,6S,RYD`, and ion channels like `H2+ > H+ + H` and `H3+ > H+ + H`.

At the same time, `SUPER RATE` contains a very large always-zero subset: `110` of the `204` reactions have positive fraction exactly `0.0` across the entire dataset. These include many elastic and transfer channels in the `AR3S` and `AR4S*` families. Among the reactions that are not always zero, several channels only activate in about `75.86%` of cases, including multiple excited-water dissociation and emission channels, `H2 > H*(ALPHA)`, `AR3S > AR-RYD`, and `H3+ > H^ + H + H`.

From a prediction standpoint, this matters a lot. A model trained on all 204 outputs without any target screening will spend a large fraction of its capacity learning trivial always-zero channels. It would be sensible to test a cleaned target set that drops the always-zero reactions, and possibly to treat partially active reactions differently from always-active ones. Supporting sources: [super_rate_reaction_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_reaction_summary.csv), ![Top reactions by mean SUPER RATE](../../results/ners590_v02/super_rate_figures/top_super_rate_reactions_by_mean.png).

## Power Sensitivity by Reaction

Although power is weak at the aggregate level, some reactions are noticeably more power-sensitive than others. The most power-sensitive channels by mean log-standard-deviation across matched density/local-case pairs are `O2-VIB 3`, `H2 > H2(V=2)`, `O2-VIB 2`, `O2-VIB 4`, `H2 > H2(V=1)`, `O2-VIB 1`, and a few rotational or oxygen excited-state channels.

Even here, the magnitudes are modest relative to the full target dynamic range, which reinforces the earlier conclusion that power is a secondary modifier rather than the primary driver of `SUPER RATE`. Still, these reaction-specific shifts are exactly the kind of structure that a good multi-output model should learn, especially if power remains an explicit continuous feature in the input set.

This section suggests that any future model comparison should include a power-holdout benchmark, but we should not expect power generalization to be the hardest part of the problem. Supporting sources: [super_rate_power_sensitivity_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_power_sensitivity_summary.csv), [super_rate_reaction_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_reaction_by_power_summary.csv), ![Power-sensitive reactions](../../results/ners590_v02/super_rate_figures/top_super_rate_power_sensitive_reactions.png).

## Relationship to RATE CONST

`SUPER RATE` is related to `RATE CONST`, but it is not simply a rescaled duplicate. For many reactions, the support overlap is perfect and the positive-only log correlation is extremely high. Examples include `H2O2-MOM`, `H2+ > H+ + H`, `O2-VIB 1(res)`, `O2-VIB 2(res)`, `H2-ROTATIONAL (0-2)`, and `H2-ROTATIONAL (1-3)`, where the log correlation is essentially `1`.

However, some reactions show weak or even negative positive-only log correlation with `RATE CONST`. The strongest negative examples are `E + H2O > H2OV(010) + E`, `E + H2O > H2OV(100) + E`, `O2-VIB 2`, `O2-ELECTRONIC 6EV`, `OH-A3PI`, and `E + H2O > H2OV(001) + E`. This means `SUPER RATE` and `RATE CONST` share activation structure for many channels, but their magnitudes can evolve differently.

This is one of the most useful findings for future ML design. It argues against assuming that a model for `RATE CONST` can be reused directly for `SUPER RATE` without retraining. At the same time, it suggests that multi-task learning or shared representations across the two targets may still help, because many channels do share strong support and trend information. Supporting sources: [super_rate_rate_const_relationship_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_rate_const_relationship_summary.csv), ![Top SUPER RATE vs RATE CONST correlations](../../results/ners590_v02/super_rate_figures/super_rate_rate_const_correlation_top.png).

## What This Means for Modeling

The `SUPER RATE` target looks more challenging than `RATE CONST` in one key way: it is much more sparse. But the structure is also favorable in several ways. First, all values are nonnegative. Second, more than half of the reactions are trivial zeros or near-trivial sparse channels, which can be screened or handled explicitly. Third, the strongest systematic variation seems to follow the local-case / `E/N` sweep rather than chaotic cross-case noise.

Based on the current analysis, the most promising prediction strategy would likely include:

- dropping always-zero `SUPER RATE` channels from the training target set and re-expanding them as zeros after prediction
- using log-space on positive targets, with explicit handling of zeros
- keeping `power_mj` as an input, but not treating it as the dominant variable
- evaluating whether a two-stage pipeline works better: activity prediction first, positive magnitude regression second
- considering joint or multi-task learning with `RATE CONST`, because many reactions have nearly identical support and strongly aligned log trends

If we want the first baseline to stay close to the existing codebase, the simplest next step is probably: reuse the current v2 training pipeline, swap the target from `RATE CONST` to `SUPER RATE`, remove always-zero channels, and evaluate both a direct multi-output regressor and a cleaned positive-only / sparse-aware variant. Supporting sources: [super_rate_overall_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_overall_summary.csv), [super_rate_reaction_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_reaction_summary.csv), [super_rate_local_case_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_local_case_summary.csv), [super_rate_rate_const_relationship_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_rate_const_relationship_summary.csv).

## Output Inventory

The main analysis tables and plots produced for this SUPER RATE study are:

- [super_rate_overall_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_overall_summary.csv)
- [super_rate_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_by_power_summary.csv)
- [super_rate_case_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_case_summary.csv)
- [super_rate_case_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_case_by_power_summary.csv)
- [super_rate_local_case_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_local_case_summary.csv)
- [super_rate_reaction_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_reaction_summary.csv)
- [super_rate_reaction_by_power_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_reaction_by_power_summary.csv)
- [super_rate_power_sensitivity_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_power_sensitivity_summary.csv)
- [super_rate_rate_const_relationship_summary.csv](../../results/ners590_v02/super_rate_analysis/super_rate_rate_const_relationship_summary.csv)
- [super_rate_positive_fraction_by_power.png](../../results/ners590_v02/super_rate_figures/super_rate_positive_fraction_by_power.png)
- [super_rate_positive_median_by_power.png](../../results/ners590_v02/super_rate_figures/super_rate_positive_median_by_power.png)
- [super_rate_nonzero_count_by_local_case.png](../../results/ners590_v02/super_rate_figures/super_rate_nonzero_count_by_local_case.png)
- [super_rate_sum_by_local_case.png](../../results/ners590_v02/super_rate_figures/super_rate_sum_by_local_case.png)
- [top_super_rate_reactions_by_mean.png](../../results/ners590_v02/super_rate_figures/top_super_rate_reactions_by_mean.png)
- [top_super_rate_power_sensitive_reactions.png](../../results/ners590_v02/super_rate_figures/top_super_rate_power_sensitive_reactions.png)
- [super_rate_rate_const_correlation_top.png](../../results/ners590_v02/super_rate_figures/super_rate_rate_const_correlation_top.png)
- [super_rate_case_sum_vs_power.png](../../results/ners590_v02/super_rate_figures/super_rate_case_sum_vs_power.png)

