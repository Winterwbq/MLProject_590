# Dataset Analysis Findings for `global_kin_boltz.out`

## Scope

This report summarizes the current analysis results for the 609 parsed cases generated from `global_kin_boltz.out`.

The findings are based on the CSV outputs in:

- `outputs/parsed/`
- `outputs/analysis/`
- `outputs/figures/`

## Dataset Shape

The dataset structure is internally consistent and matches the intended interpretation.

- Total cases: 609
- Density groups: 21
- `E/N` values per density group: 29
- Input species per case: 49
- Rate constants per case: 204

This means the dataset forms a complete `21 x 29` grid:

- 21 different number-density groups
- 29 `E/N` conditions for each group

There are no duplicate parsed feature rows and no duplicate parsed target rows.

## Structure of Inputs

### Density-Group Behavior

Each density group contains exactly 29 cases and spans the same 29-value `E/N` sweep.

Within each density group:

- the input mole fractions are constant across all 29 `E/N` values
- the number of varying species within the group is 0

This is important for modeling because it means:

- the 21 density groups define 21 distinct compositions
- the `E/N` sweep is the only changing input inside each group

### Input Composition Summary

Across all 609 cases, the input composition is dominated by two species:

- `AR3S`, mean about `0.979981`
- `H2O`, mean about `0.019939`

The only species that are nonzero in every case are:

- `E`
- `AR3S`
- `H2O`
- `H2O^`

There are 14 species that are always zero across the entire dataset:

- `HV_096`
- `HV_106`
- `HV_126`
- `ONE`
- `TE`
- `TGAS`
- `EB`
- `POS`
- `POSITION`
- `SPEED`
- `EDEP`
- `PDEP`
- `M`
- `MIS`

These always-zero species can be removed from the model input without losing information.

### Two Input Regimes

The dataset has a noticeable split in input complexity:

- Density group 1 has only 4 nonzero input species
- Density groups 2 through 21 each have 35 nonzero input species

So the full dataset is not made of equally complex mixtures. One group is a very sparse baseline composition, while the other 20 groups are richer mixtures.

### Composition Trends Across Density Groups

Across the 21 density groups, the composition changes gradually rather than randomly.

The strongest trends are:

- `H2O` decreases from `0.02000` to `0.01987`
- `AR3S` decreases from `0.9800` to `0.9799`
- `H` increases from `0` to `6.449e-05`
- `OH` increases from `0` to `5.084e-05`
- `H2OV` increases from `0` to `4.602e-05`
- `H2O2` increases from `0` to `1.244e-05`
- `H2` increases from `0` to `7.574e-06`

This suggests that the 21 groups are better interpreted as a structured composition progression than as unrelated random mixtures.

### Input Normalization Quality

The mole-fraction sums are extremely close to 1 in every density group:

- minimum sum: `0.9999486`
- maximum sum: `1.0000513`

These small deviations are consistent with source rounding and do not suggest parsing problems.

## Structure of `E/N`

The `E/N` sweep is perfectly regular.

- Minimum `E/N`: `1.000e-20 V-cm2`
- Maximum `E/N`: `1.000e-14 V-cm2`
- Unique `E/N` values: 29

Each `E/N` value appears exactly 21 times, once for each density group.

The 29 values are:

- `1.000e-20`
- `1.000e-19`
- `3.000e-19`
- `1.000e-18`
- `3.000e-18`
- `1.000e-17`
- `3.000e-17`
- `6.000e-17`
- `1.000e-16`
- `1.500e-16`
- `2.000e-16`
- `3.000e-16`
- `4.000e-16`
- `5.000e-16`
- `6.000e-16`
- `8.000e-16`
- `1.000e-15`
- `1.250e-15`
- `1.500e-15`
- `1.750e-15`
- `2.000e-15`
- `2.330e-15`
- `2.660e-15`
- `3.000e-15`
- `3.500e-15`
- `4.000e-15`
- `6.000e-15`
- `8.000e-15`
- `1.000e-14`

## Rate-Constant Target Behavior

### Overall Distribution

The target matrix contains:

- 124,236 total rate-constant entries
- 16,359 zeros
- 107,877 positive values

Important summary statistics:

- min: `0`
- median: `5.8705e-10`
- mean: `4.0793e-05`
- 95th percentile: `1.184e-05`
- 99th percentile: `2.232e-03`
- max: `6.326e-03`

This is a strongly right-skewed distribution with a long positive tail. For model training, this strongly suggests that raw rate constants will be difficult to learn directly without a target transform such as a log-based scaling.

### Sparsity Pattern

The 204 reactions split into three classes:

- 86 reactions are always nonzero
- 113 reactions are conditionally active
- 5 reactions are always zero

The 5 always-zero reactions are:

- `OH^-X2SIGMA--Not Used`
- `OH^-A2PI--Not Used`
- `OH^-B2SIGMA--Not Used`
- `OH-DISSOC, C^+O--Not Used`
- `OH-DISSOC,C+O^--Not used`

These five reactions can be removed from the supervised target set if desired.

### Dominant Reactions

The largest mean rate constants are dominated by elastic or momentum-transfer channels for ion species. The top reactions by mean rate constant are:

- `H- -ELASTIC`
- `O2- -MOMENTUM TRANSFER`
- `H3+ -ELASTIC`
- `OH- -MOM`
- `H2+ -ELASTIC`
- `AR+ -ELASTIC`
- `O- ELASTIC`
- `O+ -ELASTIC`
- `AR2+ -MOMTIC`
- `O2+ -ELASTIC`

Many of these share nearly identical summary statistics, which suggests that the target space contains some highly similar channels.

### Which Reaction Dominates Each Case

The maximum rate constant in each case is almost always one of two reactions:

- `AR+ -ELASTIC` is the dominant reaction in 390 cases
- `H- -ELASTIC` is the dominant reaction in 219 cases

So although the target dimension is 204, the case-level maximum behavior is strongly concentrated in just two channels.

## How the Targets Change with `E/N`

### Nonzero Reaction Count per Case

The number of active rate constants rises dramatically as `E/N` increases.

Mean nonzero count by local case:

- case 1: 86
- case 2: 98
- case 3: 99
- case 6: 122
- case 7: 175
- case 8: 193
- cases 17 to 29: 199

This shows a clear activation pattern:

- low `E/N` cases are sparse
- high `E/N` cases activate nearly all reactions

### Magnitude of Dominant Rates

At the same time, the largest rate constant in a case drops sharply as `E/N` increases.

Examples of mean maximum rate constant by local case:

- case 1: `4.242e-03`
- case 5: `2.397e-03`
- case 10: `2.4e-05`
- case 20: `1.5e-05`
- case 29: `8e-06`

The total sum of all 204 rate constants in a case also decreases strongly over the sweep:

- case 1 mean sum: `5.0905e-02`
- case 29 mean sum: `1.04e-04`

This means the target space changes regime over `E/N`:

- at low `E/N`, fewer reactions are active but the dominant channels are much larger
- at high `E/N`, many more reactions are active but the largest rates are smaller

## Plasma Summary Variables

The case-level physical summaries show strong dependence on `E/N`.

Across all 609 cases:

- average electron energy ranges from `0.06389 eV` to `16.33 eV`
- equivalent electron temperature ranges from `0.04259 eV` to `10.89 eV`
- updated electron density ranges from `9.646e+07 /cc` to `3.304e+11 /cc`
- drift velocity ranges from `-3.887e+06` to `8.711e+07 cm/s`
- total power loss ranges from `-2.916e-14` to `5.058e-07 eV-cm3/s`

Correlations with `E/N` are especially strong for:

- drift velocity: `0.983`
- total power loss: `0.963`
- average electron energy: `0.882`
- equivalent electron temperature: `0.882`
- updated electron density: `0.855`

This confirms that `E/N` is a major control variable in the dataset.

## Sign Changes and Regime Transitions

Some case-level outputs change sign in specific parts of the sweep:

- 140 cases have negative drift velocity
- 140 cases have negative mobility
- 140 cases have negative ionization coefficient
- 36 cases have negative total power loss
- 176 cases have negative `JE/COLLISION POWER`

These negative values are concentrated in particular local-case ranges, not scattered randomly. That makes them look like real solver-output behavior or physics-regime transitions rather than parsing artifacts.

Notable patterns:

- negative total power loss appears only in local cases 1 and 2
- negative drift velocity, mobility, and ionization coefficient cluster mainly in local cases 8 through 19

These cases should be kept in mind during preprocessing because they may represent behavior that is harder for a model to fit smoothly.

## Cross-Group Similarity at High `E/N`

One interesting pattern is that several case-level plasma summaries become almost identical across density groups at high `E/N`.

For example, average electron energy becomes effectively constant across groups at the printed precision for many of the highest local-case indices.

This suggests:

- composition differences matter strongly in some parts of the sweep
- but at high enough `E/N`, certain outputs become dominated by the field condition rather than by small composition differences

## Modeling Implications

The current dataset has several important implications for model design.

### Inputs

- The raw 49-species input space is over-complete because 14 species are always zero.
- One density group is a sparse baseline, while the other 20 groups are richer mixtures.
- Since composition is constant within each 29-case sweep, the dataset has strong repeated structure that should be respected in train/validation/test splitting.

### Targets

- The target distribution is heavy-tailed and partly sparse.
- Many reactions are always on, many are conditionally on, and a few are always off.
- A log transform or carefully designed scaling strategy will likely help substantially.

### Splitting Strategy

Because the same composition repeats across 29 `E/N` values, random row-wise splitting could leak composition structure across train and test sets.

A stronger evaluation design would likely split by density group, or at least test both:

- interpolation across `E/N` within known compositions
- generalization to unseen density groups

## Bottom Line

The current 609-case dataset is well-structured and model-usable.

Its most important properties are:

- a complete `21 x 29` design
- constant composition within each density group
- strongly structured, non-random composition changes across groups
- a highly skewed and partially sparse 204-dimensional target space
- strong `E/N` dependence in both physical summaries and rate-constant activation patterns

Overall, this is a good supervised-learning dataset, but it should be treated as a structured scientific dataset rather than as a generic i.i.d. table.
