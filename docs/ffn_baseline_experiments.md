# FFN Baseline Experiments

This report summarizes the two dedicated feed-forward neural-network baselines added after the main full-training pipeline.

Scenarios:

- `direct_all_inputs_end_to_end`: use all 49 input species plus `log10(E/N)` and predict the 204 transformed targets directly.
- `rf_replacement_composition_pca`: replace the winning random-forest model with a direct FFN while keeping the PCA-compressed composition feature path and the rest of the preprocessing pipeline unchanged.

Shared protocol:

- same locked random test split as the main pipeline
- same 5 validation resamples inside train/validation
- same per-reaction epsilon target transform and inverse transform
- model selection by validation reconstructed log-RMSE

## Final Comparison

- `direct_all_inputs_end_to_end`: test log-RMSE `0.246912`, log-R2 `0.972000`, median abs relative error `3.235723e-02`, median SMAPE `3.532242e-02`
- `rf_replacement_composition_pca`: test log-RMSE `0.263759`, log-R2 `0.971658`, median abs relative error `2.367100e-02`, median SMAPE `2.568115e-02`

Primary outputs:

- [`ffn_baseline_comparison_summary.csv`](../results/ffn_baselines/ffn_baseline_comparison_summary.csv)
- [`direct_all_inputs_end_to_end/test_overall_metrics.csv`](../results/ffn_baselines/direct_all_inputs_end_to_end/evaluation/test_overall_metrics.csv)
- [`rf_replacement_composition_pca/test_overall_metrics.csv`](../results/ffn_baselines/rf_replacement_composition_pca/evaluation/test_overall_metrics.csv)
- [`direct_all_inputs_end_to_end/test_relative_error_overall_summary.csv`](../results/ffn_baselines/direct_all_inputs_end_to_end/evaluation/test_relative_error_overall_summary.csv)
- [`rf_replacement_composition_pca/test_relative_error_overall_summary.csv`](../results/ffn_baselines/rf_replacement_composition_pca/evaluation/test_relative_error_overall_summary.csv)

