# Dataset Cleaning Notes for `global_kin_boltz.out`

## Main Findings Relevant to Cleaning

The raw dataset is readable by eye, but several formatting details will matter when we parse it for model training.

## Structural Regularity

- The file is highly regular: 609 case blocks
- Every case block is exactly 344 lines
- Each case includes the same major sections

This is good news for preprocessing because it means a deterministic parser should work well.

## Repeated Experimental Sweep

- The printed case number only runs from 1 to 29
- That 29-case sequence repeats across the file
- The `E/N` values appear to define the repeated sweep

This means the printed case number is not a unique row identifier for the whole dataset. During cleaning, we should create our own unique identifier such as:

- `global_case_id` for all 609 blocks
- `sweep_case_id` for the local 1..29 case number within each sweep
- `sweep_id` for which repeated sweep the case belongs to

## Two Different Species Tables

Each case contains two different species-based sections:

1. `INPUT GAS MOLE FRACTIONS`
2. `FRACTIONAL POWER DEPOSITION BY SPECIES`

Both sections list 49 species, but they represent different quantities and should not be merged into one table without a column that identifies their meaning.

## Reaction Table Characteristics

The reaction-rate section contains 204 rows per case with:

- rate constant
- super rate
- fractional power
- reaction label

This section is naturally suited to a long-format table with columns like:

- `global_case_id`
- `reaction_id`
- `reaction_label`
- `rate_const`
- `super_rate_cc_per_s`
- `frac_power`

## Formatting Issues and Parsing Risks

Several details will require care:

- Species labels contain special characters such as `^`, `*`, `-`, and parentheses
- Reaction labels also contain punctuation, spaces, wavelength annotations, and uneven spacing
- Many values are exactly zero
- Some `FRAC POWER` values are negative
- There are visible typos in labels such as `CONVERGANCE` and `DEPOSIITON`

These issues are not fatal, but they mean we should parse by section structure and numeric position rather than by fragile string assumptions.

## Suspicious Numeric Formatting

One value in the summary section appears as:

- `1.317-316`

This likely represents a malformed scientific notation value, probably intended to look like something such as `1.317E-316`. We should treat entries like this as data-quality flags during parsing.

## Composition Variation Across Sweeps

Although the `E/N` sweep repeats, the gas composition is not perfectly identical from one repeated sweep to the next. For example, the `H2O` fraction changes slightly across repeated `CASE=1` blocks.

This suggests the full dataset spans multiple compositions, not just one composition evaluated at many electric-field settings.

## Recommended Cleaning Outputs

For model preparation, the cleaned dataset should probably be exported into separate files or tables:

1. Case-level metadata table
2. Input species composition table
3. Reaction-rate long table
4. Fractional power deposition table

Possible case-level columns:

- `global_case_id`
- `sweep_id`
- `sweep_case_id`
- `temperature_gas_k`
- `electron_fractional_ionization`
- `initial_electron_temperature_ev`
- `e_over_n_v_cm2`
- `average_electron_energy_ev`
- `equivalent_electron_temperature_ev`
- `updated_electron_density_per_cc`
- `drift_velocity_cm_per_s`
- `mobility_cm2_per_v_s`
- `diffusion_coefficient_cm2_per_s`
- `ionization_coefficient_per_cm`
- `momentum_transfer_collision_frequency_per_s`
- `total_power_loss_ev_cm3_per_s`

## Suggested Validation Checks

When we build the parser, we should validate at least the following:

- every block has 344 lines
- every block contains 49 input species rows
- every block contains 204 reaction rows
- every block contains 49 power-deposition species rows
- every block has one `E/N` value
- parsed numeric fields convert cleanly or are flagged

## Bottom Line

The dataset is in a strong position for cleaning because the layout is consistent. The main work is not discovering the structure, but converting the repeated text blocks into tidy, uniquely indexed, machine-readable tables while preserving reaction labels and species information accurately.
