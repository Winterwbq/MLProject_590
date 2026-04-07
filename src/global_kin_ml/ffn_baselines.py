from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import load_parsed_dataset, parse_raw_dataset, validate_parsed_shapes
from .evaluation import (
    build_prediction_frame,
    compute_overall_metrics,
    compute_per_case_metrics,
    compute_per_reaction_metrics,
    compute_relative_error_outputs,
    compute_smape_outputs,
    plot_case_error_distribution,
    plot_log_residual_histogram,
    plot_model_leaderboard,
    plot_parity,
    plot_relative_error_by_magnitude,
    plot_relative_error_histogram,
    plot_smape_by_magnitude,
    plot_smape_histogram,
    save_csv,
)
from .models import ModelConfig, build_model
from .preprocessing import (
    TARGET_PREFIX,
    TargetTransformer,
    build_feature_transformer,
    build_split_assignment_frame,
    create_random_case_splits,
    get_case_metadata_columns,
)


def _mlp_configs(feature_set: str, scenario_name: str) -> list[ModelConfig]:
    configs: list[ModelConfig] = []
    for hidden_width in (128, 256):
        for hidden_layers in (2, 3):
            for dropout in (0.0, 0.1):
                for weight_decay in (0.0, 1e-5, 1e-4):
                    configs.append(
                        ModelConfig(
                            model_key=(
                                f"{scenario_name}__mlp__{feature_set}__w_{hidden_width}"
                                f"__layers_{hidden_layers}__drop_{dropout}__wd_{weight_decay}"
                            ),
                            model_family="mlp",
                            feature_set=feature_set,
                            latent_k=None,
                            hyperparameters={
                                "hidden_width": hidden_width,
                                "hidden_layers": hidden_layers,
                                "dropout": dropout,
                                "weight_decay": weight_decay,
                            },
                        )
                    )
    return configs


def _aggregate_trials(trial_frame: pd.DataFrame) -> pd.DataFrame:
    successful = trial_frame[trial_frame["status"] == "success"].copy()
    metric_columns = [
        "validation_overall_log_rmse",
        "validation_overall_log_mae",
        "validation_overall_log_r2",
        "validation_overall_original_rmse",
        "validation_overall_original_mae",
        "validation_factor2_accuracy_positive_only",
        "validation_factor5_accuracy_positive_only",
        "validation_factor10_accuracy_positive_only",
        "validation_mean_reaction_log_r2",
        "validation_median_reaction_log_mae",
        "runtime_seconds",
    ]
    summary = (
        successful.groupby(
            ["model_key", "model_family", "feature_set", "latent_k"], dropna=False
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        .replace("validation_", "")
        .replace("runtime_seconds_mean", "mean_runtime_seconds")
        .replace("runtime_seconds_std", "std_runtime_seconds")
        for column in summary.columns.to_flat_index()
    ]
    summary = summary.rename(
        columns={
            "overall_log_rmse_mean": "mean_validation_log_rmse",
            "overall_log_rmse_std": "std_validation_log_rmse",
            "overall_log_mae_mean": "mean_validation_log_mae",
            "overall_log_mae_std": "std_validation_log_mae",
            "overall_log_r2_mean": "mean_validation_log_r2",
            "overall_log_r2_std": "std_validation_log_r2",
            "overall_original_rmse_mean": "mean_validation_original_rmse",
            "overall_original_rmse_std": "std_validation_original_rmse",
            "overall_original_mae_mean": "mean_validation_original_mae",
            "overall_original_mae_std": "std_validation_original_mae",
            "factor2_accuracy_positive_only_mean": "mean_validation_factor2_accuracy",
            "factor2_accuracy_positive_only_std": "std_validation_factor2_accuracy",
            "factor5_accuracy_positive_only_mean": "mean_validation_factor5_accuracy",
            "factor5_accuracy_positive_only_std": "std_validation_factor5_accuracy",
            "factor10_accuracy_positive_only_mean": "mean_validation_factor10_accuracy",
            "factor10_accuracy_positive_only_std": "std_validation_factor10_accuracy",
            "mean_reaction_log_r2_mean": "mean_validation_reaction_log_r2",
            "mean_reaction_log_r2_std": "std_validation_reaction_log_r2",
            "median_reaction_log_mae_mean": "mean_validation_reaction_median_log_mae",
            "median_reaction_log_mae_std": "std_validation_reaction_median_log_mae",
        }
    )
    first_rows = (
        successful.sort_values("model_key")
        .groupby("model_key", dropna=False)
        .first()
        .reset_index()[["model_key", "hyperparameters_json"]]
    )
    summary = summary.merge(first_rows, on="model_key", how="left")
    summary["leaderboard_rank"] = np.arange(1, len(summary) + 1)
    summary = summary.sort_values(
        [
            "mean_validation_log_rmse",
            "mean_validation_reaction_log_r2",
            "mean_validation_reaction_median_log_mae",
            "mean_validation_factor5_accuracy",
        ],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    summary["leaderboard_rank"] = np.arange(1, len(summary) + 1)
    return summary


def _write_report(results_dir: Path, comparison_frame: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# FFN Baseline Experiments",
        "",
        "This report summarizes the two dedicated feed-forward neural-network baselines added after the main full-training pipeline.",
        "",
        "Scenarios:",
        "",
        "- `direct_all_inputs_end_to_end`: use all 49 input species plus `log10(E/N)` and predict the 204 transformed targets directly.",
        "- `rf_replacement_composition_pca`: replace the winning random-forest model with a direct FFN while keeping the PCA-compressed composition feature path and the rest of the preprocessing pipeline unchanged.",
        "",
        "Shared protocol:",
        "",
        "- same locked random test split as the main pipeline",
        "- same 5 validation resamples inside train/validation",
        "- same per-reaction epsilon target transform and inverse transform",
        "- model selection by validation reconstructed log-RMSE",
        "",
        "## Final Comparison",
        "",
    ]
    for _, row in comparison_frame.iterrows():
        lines.append(
            f"- `{row['scenario_name']}`: test log-RMSE `{row['overall_log_rmse']:.6f}`, "
            f"log-R2 `{row['overall_log_r2']:.6f}`, median abs relative error `{row['median_absolute_relative_error']:.6e}`, "
            f"median SMAPE `{row['median_smape']:.6e}`"
        )

    lines.extend(
        [
            "",
            "Primary outputs:",
            "",
            f"- [`ffn_baseline_comparison_summary.csv`](../results/{results_dir.name}/ffn_baseline_comparison_summary.csv)",
            f"- [`direct_all_inputs_end_to_end/test_overall_metrics.csv`](../results/{results_dir.name}/direct_all_inputs_end_to_end/evaluation/test_overall_metrics.csv)",
            f"- [`rf_replacement_composition_pca/test_overall_metrics.csv`](../results/{results_dir.name}/rf_replacement_composition_pca/evaluation/test_overall_metrics.csv)",
            f"- [`direct_all_inputs_end_to_end/test_relative_error_overall_summary.csv`](../results/{results_dir.name}/direct_all_inputs_end_to_end/evaluation/test_relative_error_overall_summary.csv)",
            f"- [`rf_replacement_composition_pca/test_relative_error_overall_summary.csv`](../results/{results_dir.name}/rf_replacement_composition_pca/evaluation/test_relative_error_overall_summary.csv)",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffn_baseline_experiments(
    raw_dataset_path: Path,
    results_dir: Path,
    report_output_path: Path,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = results_dir / "data" / "parsed"
    parse_raw_dataset(raw_dataset_path, parsed_dir)
    dataset = load_parsed_dataset(parsed_dir)
    inputs = dataset.training_inputs.copy()
    targets = dataset.training_targets.copy()
    reaction_map = dataset.reaction_map.copy()
    target_columns = [column for column in targets.columns if column.startswith(TARGET_PREFIX)]
    metadata_columns = get_case_metadata_columns(inputs)

    verification_frame = pd.DataFrame(validate_parsed_shapes(dataset))
    save_csv(verification_frame, results_dir / "verification_checks.csv")

    splits = create_random_case_splits(total_cases=len(inputs))
    split_assignment = build_split_assignment_frame(inputs, splits)
    save_csv(split_assignment, results_dir / "split_assignments.csv")

    scenarios = [
        {
            "scenario_name": "direct_all_inputs_end_to_end",
            "feature_set": "all_inputs_plus_log_en",
            "description": "Direct FFN on all 49 inputs plus log10(E/N).",
        },
        {
            "scenario_name": "rf_replacement_composition_pca",
            "feature_set": "composition_pca_plus_log_en",
            "description": "Direct FFN on the winning RF feature path with composition PCA plus log10(E/N).",
        },
    ]

    comparison_rows = []

    for scenario in scenarios:
        scenario_root = results_dir / scenario["scenario_name"]
        tuning_dir = scenario_root / "tuning"
        evaluation_dir = scenario_root / "evaluation"
        figures_dir = scenario_root / "figures"
        data_dir = scenario_root / "data_snapshots"
        for path in (tuning_dir, evaluation_dir, figures_dir, data_dir):
            path.mkdir(parents=True, exist_ok=True)

        configs = _mlp_configs(scenario["feature_set"], scenario["scenario_name"])
        save_csv(
            pd.DataFrame(
                [
                    {
                        **config.as_dict(),
                        "hyperparameters_json": str(config.hyperparameters),
                    }
                    for config in configs
                ]
            ),
            tuning_dir / "config_catalog.csv",
        )

        trainval_inputs = inputs.loc[splits.trainval_indices]
        feature_transformer = build_feature_transformer(scenario["feature_set"]).fit(trainval_inputs)
        save_csv(
            pd.concat(
                [
                    inputs[metadata_columns].reset_index(drop=True),
                    split_assignment[["locked_split"]].reset_index(drop=True),
                    feature_transformer.transform(inputs).reset_index(drop=True),
                ],
                axis=1,
            ),
            data_dir / f"{scenario['feature_set']}_all_cases.csv",
        )

        trial_rows = []
        for fold in splits.validation_folds:
            train_inputs = inputs.loc[fold.train_indices]
            val_inputs = inputs.loc[fold.val_indices]
            train_targets = targets.loc[fold.train_indices]
            val_targets = targets.loc[fold.val_indices]

            fold_feature_transformer = build_feature_transformer(scenario["feature_set"]).fit(train_inputs)
            x_train = fold_feature_transformer.transform(train_inputs).to_numpy(dtype=float)
            x_val = fold_feature_transformer.transform(val_inputs).to_numpy(dtype=float)

            target_transformer = TargetTransformer(target_columns).fit(train_targets)
            y_train_log = target_transformer.transform(train_targets).to_numpy(dtype=float)
            y_val_log = target_transformer.transform(val_targets).to_numpy(dtype=float)
            y_val_original = val_targets[target_columns].to_numpy(dtype=float)

            for config in configs:
                row = {
                    "scenario_name": scenario["scenario_name"],
                    "fold_id": fold.fold_id,
                    "fold_seed": fold.seed,
                    "status": "success",
                    "error_message": "",
                    **config.as_dict(),
                    "hyperparameters_json": str(config.hyperparameters),
                }
                start_time = time.perf_counter()
                try:
                    model = build_model(config)
                    callback = lambda predicted_log, truth=y_val_log: float(
                        math.sqrt(np.mean((predicted_log - truth) ** 2))
                    )
                    model.fit(
                        x_train=x_train,
                        y_train=y_train_log,
                        x_val=x_val,
                        y_val=y_val_log,
                        validation_score_callback=callback,
                    )
                    y_val_log_pred = model.predict(x_val)
                    y_val_original_pred = target_transformer.inverse_transform_array(y_val_log_pred)
                    metrics = compute_overall_metrics(
                        y_true_log=y_val_log,
                        y_pred_log=y_val_log_pred,
                        y_true_original=y_val_original,
                        y_pred_original=y_val_original_pred,
                    )
                    per_reaction = compute_per_reaction_metrics(
                        y_true_log=y_val_log,
                        y_pred_log=y_val_log_pred,
                        y_true_original=y_val_original,
                        y_pred_original=y_val_original_pred,
                        reaction_map=reaction_map,
                    )
                    row.update({f"validation_{key}": value for key, value in metrics.items()})
                    row["validation_mean_reaction_log_r2"] = float(
                        per_reaction["log_r2"].dropna().mean()
                    )
                    row["validation_median_reaction_log_mae"] = float(
                        per_reaction["log_mae"].median()
                    )
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "failed"
                    row["error_message"] = f"{type(exc).__name__}: {exc}"
                row["runtime_seconds"] = time.perf_counter() - start_time
                trial_rows.append(row)

        trial_frame = pd.DataFrame(trial_rows)
        save_csv(trial_frame, tuning_dir / "model_trials_foldwise.csv")
        leaderboard = _aggregate_trials(trial_frame)
        save_csv(leaderboard, tuning_dir / "model_leaderboard_summary.csv")
        selected = leaderboard.iloc[0]
        save_csv(pd.DataFrame([selected]), tuning_dir / "selected_model.csv")
        plot_model_leaderboard(leaderboard, figures_dir / "model_leaderboard.png", top_n=12)

        selected_config = next(config for config in configs if config.model_key == selected["model_key"])
        final_feature_transformer = build_feature_transformer(scenario["feature_set"]).fit(
            inputs.loc[splits.trainval_indices]
        )
        x_trainval = final_feature_transformer.transform(inputs.loc[splits.trainval_indices]).to_numpy(dtype=float)
        x_test = final_feature_transformer.transform(inputs.loc[splits.test_indices]).to_numpy(dtype=float)

        final_target_transformer = TargetTransformer(target_columns).fit(targets.loc[splits.trainval_indices])
        y_trainval_log = final_target_transformer.transform(targets.loc[splits.trainval_indices]).to_numpy(dtype=float)
        y_test_log = final_target_transformer.transform(targets.loc[splits.test_indices]).to_numpy(dtype=float)
        y_test_original = targets.loc[splits.test_indices, target_columns].to_numpy(dtype=float)

        final_model = build_model(selected_config)
        final_model.fit(x_train=x_trainval, y_train=y_trainval_log)
        y_test_log_pred = final_model.predict(x_test)
        y_test_original_pred = final_target_transformer.inverse_transform_array(y_test_log_pred)

        overall_metrics = compute_overall_metrics(
            y_true_log=y_test_log,
            y_pred_log=y_test_log_pred,
            y_true_original=y_test_original,
            y_pred_original=y_test_original_pred,
        )
        per_reaction_metrics = compute_per_reaction_metrics(
            y_true_log=y_test_log,
            y_pred_log=y_test_log_pred,
            y_true_original=y_test_original,
            y_pred_original=y_test_original_pred,
            reaction_map=reaction_map,
        )
        per_case_metrics = compute_per_case_metrics(
            y_true_log=y_test_log,
            y_pred_log=y_test_log_pred,
            y_true_original=y_test_original,
            y_pred_original=y_test_original_pred,
            case_ids=targets.loc[splits.test_indices, metadata_columns],
        )
        prediction_frame = build_prediction_frame(
            case_ids=targets.loc[splits.test_indices, metadata_columns],
            reaction_map=reaction_map,
            y_true_original=y_test_original,
            y_pred_original=y_test_original_pred,
            y_true_log=y_test_log,
            y_pred_log=y_test_log_pred,
        )
        relative_overall, relative_per_reaction, relative_per_case, relative_by_magnitude = (
            compute_relative_error_outputs(prediction_frame)
        )
        smape_overall, smape_per_reaction, smape_per_case, smape_by_magnitude = compute_smape_outputs(
            prediction_frame
        )

        save_csv(pd.DataFrame([overall_metrics]), evaluation_dir / "test_overall_metrics.csv")
        save_csv(per_reaction_metrics, evaluation_dir / "test_per_reaction_metrics.csv")
        save_csv(per_case_metrics, evaluation_dir / "test_per_case_metrics.csv")
        save_csv(prediction_frame, evaluation_dir / "test_predictions_long.csv")
        save_csv(relative_overall, evaluation_dir / "test_relative_error_overall_summary.csv")
        save_csv(relative_per_reaction, evaluation_dir / "test_relative_error_per_reaction.csv")
        save_csv(relative_per_case, evaluation_dir / "test_relative_error_per_case.csv")
        save_csv(relative_by_magnitude, evaluation_dir / "test_relative_error_by_magnitude_bin.csv")
        save_csv(smape_overall, evaluation_dir / "test_smape_overall_summary.csv")
        save_csv(smape_per_reaction, evaluation_dir / "test_smape_per_reaction.csv")
        save_csv(smape_per_case, evaluation_dir / "test_smape_per_case.csv")
        save_csv(smape_by_magnitude, evaluation_dir / "test_smape_by_magnitude_bin.csv")
        save_csv(per_reaction_metrics.nlargest(10, "log_rmse"), evaluation_dir / "worst_10_reactions.csv")
        save_csv(per_case_metrics.nlargest(10, "log_rmse"), evaluation_dir / "worst_10_cases.csv")

        plot_parity(
            y_true=y_test_log,
            y_pred=y_test_log_pred,
            path=figures_dir / "parity_plot_log_space.png",
            title=f"{scenario['scenario_name']} Predicted vs True Log10 Rates",
            x_label="True log10(rate + epsilon)",
            y_label="Predicted log10(rate + epsilon)",
        )
        plot_log_residual_histogram(prediction_frame, figures_dir / "residual_hist_log_space.png")
        plot_case_error_distribution(per_case_metrics, figures_dir / "per_case_log_rmse_distribution.png")
        plot_relative_error_histogram(
            relative_overall,
            prediction_frame.assign(
                relative_error_defined=prediction_frame["true_rate_const"] > 0,
                relative_error_abs=np.where(
                    prediction_frame["true_rate_const"] > 0,
                    (
                        (prediction_frame["predicted_rate_const"] - prediction_frame["true_rate_const"])
                        / prediction_frame["true_rate_const"]
                    ).abs(),
                    np.nan,
                ),
            ),
            figures_dir / "relative_error_abs_histogram.png",
        )
        plot_relative_error_by_magnitude(relative_by_magnitude, figures_dir / "relative_error_by_magnitude.png")
        plot_smape_histogram(
            smape_overall,
            prediction_frame.assign(
                smape=2.0
                * prediction_frame["absolute_error"]
                / (
                    prediction_frame["predicted_rate_const"].abs()
                    + prediction_frame["true_rate_const"].abs()
                ).replace(0.0, np.nan)
            ).fillna({"smape": 0.0}),
            figures_dir / "smape_histogram.png",
        )
        plot_smape_by_magnitude(smape_by_magnitude, figures_dir / "smape_by_magnitude.png")

        comparison_rows.append(
            {
                "scenario_name": scenario["scenario_name"],
                "description": scenario["description"],
                "feature_set": scenario["feature_set"],
                "selected_model_key": selected["model_key"],
                "selected_hyperparameters": selected["hyperparameters_json"],
                **overall_metrics,
                "median_absolute_relative_error": float(
                    relative_overall.loc[0, "median_absolute_relative_error"]
                ),
                "p95_absolute_relative_error": float(
                    relative_overall.loc[0, "p95_absolute_relative_error"]
                ),
                "within_10pct_relative_error": float(relative_overall.loc[0, "within_10pct"]),
                "median_smape": float(smape_overall.loc[0, "median_smape"]),
                "p95_smape": float(smape_overall.loc[0, "p95_smape"]),
                "within_10pct_smape": float(smape_overall.loc[0, "within_10pct_smape"]),
            }
        )

    comparison_frame = pd.DataFrame(comparison_rows).sort_values("overall_log_rmse").reset_index(drop=True)
    save_csv(comparison_frame, results_dir / "ffn_baseline_comparison_summary.csv")
    _write_report(results_dir, comparison_frame, report_output_path)
    return {
        "results_dir": results_dir,
        "comparison_summary": results_dir / "ffn_baseline_comparison_summary.csv",
        "report": report_output_path,
    }
