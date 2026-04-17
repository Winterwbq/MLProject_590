# NERS590 V03 Documentation Index

This is the active reading path for the V03 experiment. Files are grouped by topic and numbered in the suggested order.

## 0. Dataset and Preprocessing

- [dataset/0_v03_dataset_analysis_report.md](dataset/0_v03_dataset_analysis_report.md)
  - V03 raw data structure, power levels, E/N grid, RATE CONST and SUPER RATE distribution analysis.
- [dataset/1_data_cleaning_scaling_and_preprocessing.md](dataset/1_data_cleaning_scaling_and_preprocessing.md)
  - Data cleaning, train-only preprocessing, input scaling, output log transforms, and why `epsilon` is used.

## 1. Model Definitions

- [models/0_model_name_notations.md](models/0_model_name_notations.md)
  - Short labels used in figures and reports, including separate-task, direct, latent, and joint-model notation.

## 2. Experiment Execution

- [experiments/0_code_structure_and_run_guide.md](experiments/0_code_structure_and_run_guide.md)
  - How `scripts/` and `src/` connect, canonical commands, and result folder layout.
- [experiments/1_v03_multitask_training_report.md](experiments/1_v03_multitask_training_report.md)
  - V03 multitask training design, branch evaluation, and joint-vs-separate comparison.

## 3. Results and Conclusions

- [results/0_v03_paper_draft.md](results/0_v03_paper_draft.md)
  - Paper-style draft covering methods, preprocessing, models, results, analysis, and conclusions.

## 4. Ideas and Historical Lessons

- [ideas/0_v02_experiment_plan_inspiration.md](ideas/0_v02_experiment_plan_inspiration.md)
  - Early power-aware experiment plan that shaped the V03 design.
- [ideas/1_super_rate_prediction_plan_inspiration.md](ideas/1_super_rate_prediction_plan_inspiration.md)
  - SUPER RATE-specific analysis and modeling plan.
- [ideas/2_joint_rate_super_rate_lessons_from_v02.md](ideas/2_joint_rate_super_rate_lessons_from_v02.md)
  - Lessons from separate vs joint RATE CONST and SUPER RATE experiments.
- [ideas/3_super_rate_training_lessons_from_v02.md](ideas/3_super_rate_training_lessons_from_v02.md)
  - Sparse SUPER RATE target-cleaning and modeling lessons.
- [ideas/4_v02_temporary_experiment_paper_reference.md](ideas/4_v02_temporary_experiment_paper_reference.md)
  - Full V02 temporary paper retained as a methodological reference.
- [ideas/5_legacy_model_strategy_and_reconstruction_protocol.md](ideas/5_legacy_model_strategy_and_reconstruction_protocol.md)
  - Original model-choice and PCA reconstruction-precision protocol.
- [ideas/6_ffn_vs_tree_model_lessons.md](ideas/6_ffn_vs_tree_model_lessons.md)
  - Why FFN baselines struggled relative to tree ensembles in earlier experiments.

## Archive

Older full documentation remains in [../archive](../archive) for provenance, but it is not the primary V03 reading path.
