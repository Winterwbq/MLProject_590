# Dataset Overview: `global_kin_boltz.out`

## Source File

- File: `global_kin_boltz.out`
- Location: project root
- Size: about 12.8 MB
- Total lines: 209,496

## High-Level Structure

The dataset is stored as a large fixed-format text file produced as a sequence of repeated case blocks.

- Total case blocks: 609
- Lines per case block: 344
- Each case block contains the same section layout
- The visible case number runs from 1 to 29 and then resets

Based on the repeated 1..29 pattern, the file appears to represent a 29-point sweep over electric field over number density (`E/N`) that is repeated for multiple gas compositions. This is an inference from the structure of the file.

## Per-Case Layout

Each case block contains the following sections:

1. Case header
2. Input options and scalar settings
3. Input gas mole fractions
4. Final values
5. Transport coefficients
6. Reaction rate table
7. Total power loss
8. Fractional power deposition by species

## Important Fields Observed

### Input Scalars

The early part of each case includes scalar values such as:

- Gas/ion temperature
- Electron fractional ionization
- Initial electron temperature
- Maximum iterations for EDF
- Electric field/number density (`E/N`)

### Species Input Table

Each case includes an `INPUT GAS MOLE FRACTIONS` section with 49 listed species. Examples include:

- `E`
- `AR3S`
- `H2O`
- `H2O^`
- `O2`
- `O`

### Final and Transport Summaries

Each case also contains summary outputs such as:

- Average electron energy
- Equivalent electron temperature
- Updated electron density
- Drift velocity
- Mobility
- Diffusion coefficient
- Ionization coefficient
- Momentum transfer collision frequency

### Reaction Rate Table

Each case contains a reaction table with 204 rows and the following columns:

- `RATE CONST`
- `SUPER RATE (CC/S)`
- `FRAC POWER`
- Reaction label

### Species Power Deposition Table

At the end of each case, the file contains:

- Total power loss by electrons
- Fractional power deposition by species

This last species table again contains 49 species entries, but it is a different table from the earlier input mole fraction list.

## Observed `E/N` Sweep

The first 29 cases show a monotonic `E/N` sweep, starting at:

- `1.000E-20 V-cm2`

and continuing through values such as:

- `1.000E-19`
- `3.000E-19`
- `1.000E-18`
- `1.000E-17`
- `1.000E-16`
- `1.000E-15`
- `1.000E-14 V-cm2`

This same 29-value sweep appears to repeat across the full file.

## Initial Interpretation for ML Preparation

The raw file is structured enough to parse reliably, but it is not yet model-ready. A useful cleaned representation will likely separate the data into multiple tidy tables:

- One case-level table for scalar metadata
- One species input composition table
- One long-format reaction-rate table
- One species power-deposition table

That structure should preserve the original information while making downstream feature engineering and training much easier.
