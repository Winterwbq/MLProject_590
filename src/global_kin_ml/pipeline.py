from __future__ import annotations

import math
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_parsed_dataset, parse_raw_dataset, validate_parsed_shapes
from .evaluation import (
    build_prediction_frame,
    build_relative_error_frame,
    build_smape_frame,
    compute_overall_metrics,
    compute_per_case_metrics,
    compute_per_reaction_metrics,
    compute_relative_error_outputs,
    compute_smape_outputs,
    oracle_reconstruction,
    pca_explained_variance_frame,
    plot_case_error_distribution,
    plot_log_residual_histogram,
    plot_model_leaderboard,
    plot_oracle_error,
    plot_parity,
    plot_pca_scree,
    plot_relative_error_by_magnitude,
    plot_relative_error_histogram,
    plot_smape_by_magnitude,
    plot_smape_histogram,
    plot_worst_reactions,
    save_csv,
)
from .models import ModelConfig, build_model, build_model_configs
from .preprocessing import (
    TargetSpec,
    TargetTransformer,
    build_feature_transformer,
    build_split_assignment_frame,
    create_group_holdout_splits,
    create_random_case_splits,
    filter_constant_target_columns,
    get_case_metadata_columns,
    get_target_spec,
)
from .reporting import export_experiment_report

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


RESULT_SUBDIRS = {
    "data": "data",
    "data_snapshots": "data_snapshots",
    "tuning": "tuning",
    "pca": "pca",
    "evaluation": "evaluation",
    "figures": "figures",
}


def _prefixed_metric_row(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _aggregate_trial_summary(trial_frame: pd.DataFrame) -> pd.DataFrame:
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
    fold_counts = (
        successful.groupby(["model_key"], dropna=False)["fold_id"]
        .nunique()
        .rename("evaluated_fold_count")
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        .replace("validation_", "")
        .replace("runtime_seconds_mean", "mean_runtime_seconds")
        .replace("runtime_seconds_std", "std_runtime_seconds")
        for column in summary.columns.to_flat_index()
    ]
    rename_map = {
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
    summary = summary.rename(columns=rename_map)
    config_columns = [
        "model_key",
        "model_family",
        "feature_set",
        "latent_k",
    ]
    first_rows = (
        successful.sort_values("model_key")
        .groupby("model_key", dropna=False)
        .first()
        .reset_index()[["model_key", "hyperparameters_json"]]
    )
    summary = summary.merge(first_rows, on="model_key", how="left")
    summary = summary.merge(fold_counts, on="model_key", how="left")
    summary = summary.sort_values(
        by=[
            "evaluated_fold_count",
            "mean_validation_log_rmse",
            "mean_validation_reaction_log_r2",
            "mean_validation_reaction_median_log_mae",
            "mean_validation_factor5_accuracy",
        ],
        ascending=[False, True, False, True, False],
    ).reset_index(drop=True)
    summary["leaderboard_rank"] = np.arange(1, len(summary) + 1)
    ordered_columns = (
        ["leaderboard_rank"]
        + config_columns
        + [column for column in summary.columns if column not in (["leaderboard_rank"] + config_columns)]
    )
    return summary[ordered_columns]


def _build_config_frame(configs: list[ModelConfig]) -> pd.DataFrame:
    rows = []
    for config in configs:
        row = config.as_dict()
        row["hyperparameters_json"] = str(config.hyperparameters)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_best_model(summary: pd.DataFrame) -> pd.Series:
    eligible = summary[summary["evaluated_fold_count"] == 5].copy()
    if eligible.empty:
        eligible = summary.copy()
    return eligible.iloc[0]


def _select_stage2_survivors(stage1_frame: pd.DataFrame, configs: list[ModelConfig]) -> list[ModelConfig]:
    successful = stage1_frame[stage1_frame["status"] == "success"].copy()
    if successful.empty:
        return configs

    successful = successful.sort_values("validation_overall_log_rmse").reset_index(drop=True)
    top_n = max(64, int(math.ceil(0.20 * len(successful))))
    survivor_keys = set(successful.head(top_n)["model_key"].tolist())

    mandatory = (
        successful.assign(
            latent_group=pd.to_numeric(successful["latent_k"], errors="coerce").fillna(-1).astype(int)
        )
        .sort_values("validation_overall_log_rmse")
        .groupby(["model_family", "feature_set", "latent_group"], dropna=False)
        .head(1)
    )
    survivor_keys.update(mandatory["model_key"].tolist())
    return [config for config in configs if config.model_key in survivor_keys]


def _save_feature_snapshots(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    split_assignment: pd.DataFrame,
    trainval_indices: np.ndarray,
    target_columns: list[str],
    data_snapshot_dir: Path,
    target_stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trainval_inputs = inputs.loc[trainval_indices]
    metadata_columns = get_case_metadata_columns(inputs)
    feature_metadata_rows = []

    for feature_set in (
        "full_nonconstant_plus_log_en",
        "composition_pca_plus_log_en",
        "full_nonconstant_plus_log_en_log_power",
        "composition_pca_plus_log_en_log_power",
    ):
        if feature_set.endswith("_log_power") and "power_mj" not in inputs.columns:
            continue
        transformer = build_feature_transformer(feature_set).fit(trainval_inputs)
        transformed = transformer.transform(inputs)
        snapshot = pd.concat(
            [
                inputs[metadata_columns].reset_index(drop=True),
                split_assignment[["locked_split"]].reset_index(drop=True),
                transformed.reset_index(drop=True),
            ],
            axis=1,
        )
        save_csv(snapshot, data_snapshot_dir / f"{feature_set}_all_cases.csv")
        metadata_row = transformer.metadata()
        metadata_row["fit_scope"] = "trainval"
        feature_metadata_rows.append(metadata_row)

    target_transformer = TargetTransformer(target_columns).fit(targets.loc[trainval_indices])
    transformed_targets = pd.concat(
        [
            targets[metadata_columns].reset_index(drop=True),
            split_assignment[["locked_split"]].reset_index(drop=True),
            target_transformer.transform(targets).reset_index(drop=True),
        ],
        axis=1,
    )
    save_csv(transformed_targets, data_snapshot_dir / f"log_transformed_{target_stem}_all_cases.csv")
    epsilon_frame = target_transformer.epsilon_frame()
    save_csv(epsilon_frame, data_snapshot_dir / f"{target_stem}_epsilons_trainval.csv")
    return pd.DataFrame(feature_metadata_rows), epsilon_frame


def _build_target_frame(
    dataset,
    inputs: pd.DataFrame,
    target_spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = get_case_metadata_columns(inputs)
    reaction_map = dataset.reaction_map[
        ["reaction_id", "reaction_label", target_spec.reaction_map_column]
    ].copy()
    ordered_target_columns = reaction_map[target_spec.reaction_map_column].tolist()

    if target_spec.name == "rate_const":
        target_frame = inputs[metadata_columns].merge(
            dataset.training_targets[metadata_columns + ordered_target_columns],
            on=metadata_columns,
            how="left",
        )
        return target_frame, reaction_map

    long_frame = dataset.rate_constants_long[
        metadata_columns + ["reaction_id", target_spec.long_value_column]
    ].copy()
    wide = (
        long_frame.pivot_table(
            index=metadata_columns,
            columns="reaction_id",
            values=target_spec.long_value_column,
            aggfunc="first",
        )
        .reset_index()
    )
    rename_map = dict(zip(reaction_map["reaction_id"], reaction_map[target_spec.reaction_map_column]))
    wide = wide.rename(columns=rename_map)
    for column in ordered_target_columns:
        if column not in wide.columns:
            wide[column] = 0.0
    target_frame = inputs[metadata_columns].merge(
        wide[metadata_columns + ordered_target_columns],
        on=metadata_columns,
        how="left",
    )
    return target_frame, reaction_map


def _expand_target_matrix(
    active_matrix: np.ndarray,
    kept_target_columns: list[str],
    full_target_columns: list[str],
) -> np.ndarray:
    expanded = np.zeros((active_matrix.shape[0], len(full_target_columns)), dtype=float)
    column_to_index = {column: index for index, column in enumerate(full_target_columns)}
    for active_index, column in enumerate(kept_target_columns):
        expanded[:, column_to_index[column]] = active_matrix[:, active_index]
    return expanded


def _oracle_frames_for_eval(
    y_train_log: np.ndarray,
    y_eval_log: np.ndarray,
    y_eval_original: np.ndarray,
    case_ids: pd.DataFrame,
    reaction_map: pd.DataFrame,
    target_transformer: TargetTransformer,
    prefix: str,
    target_column_field: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    per_reaction_rows = []
    per_case_rows = []
    monotonic_rows = []

    previous_rmse = None
    for latent_k in (2, 3, 4, 6, 8, 10, 12):
        reconstructed_log, _ = oracle_reconstruction(y_train_log, y_eval_log, latent_k)
        reconstructed_original = target_transformer.inverse_transform_array(reconstructed_log)
        metrics = compute_overall_metrics(
            y_true_log=y_eval_log,
            y_pred_log=reconstructed_log,
            y_true_original=y_eval_original,
            y_pred_original=reconstructed_original,
        )
        overall_row = {"latent_k": latent_k, **metrics}
        overall_rows.append(overall_row)

        per_reaction = compute_per_reaction_metrics(
            y_true_log=y_eval_log,
            y_pred_log=reconstructed_log,
            y_true_original=y_eval_original,
            y_pred_original=reconstructed_original,
            reaction_map=reaction_map,
            target_column_field=target_column_field,
        )
        per_reaction.insert(0, "latent_k", latent_k)
        per_reaction_rows.append(per_reaction)

        per_case = compute_per_case_metrics(
            y_true_log=y_eval_log,
            y_pred_log=reconstructed_log,
            y_true_original=y_eval_original,
            y_pred_original=reconstructed_original,
            case_ids=case_ids,
        )
        per_case.insert(0, "latent_k", latent_k)
        per_case_rows.append(per_case)

        monotonic_rows.append(
            {
                "check_name": f"{prefix}_oracle_monotonic_k_{latent_k}",
                "status": "pass"
                if previous_rmse is None or metrics["overall_log_rmse"] <= previous_rmse + 1e-9
                else "fail",
                "expected": "<= previous k overall_log_rmse",
                "observed": metrics["overall_log_rmse"],
                "details": f"Oracle reconstruction RMSE at k={latent_k}",
            }
        )
        previous_rmse = metrics["overall_log_rmse"]

    return (
        pd.DataFrame(overall_rows),
        pd.concat(per_reaction_rows, ignore_index=True),
        pd.concat(per_case_rows, ignore_index=True),
        pd.DataFrame(monotonic_rows),
    )


def run_training_experiment(
    raw_dataset_path: Path,
    results_dir: Path,
    report_output_path: Path | None = None,
    max_configs: int | None = None,
    feature_sets: list[str] | None = None,
    split_strategy: str = "random_case",
    holdout_power_labels: list[str] | None = None,
    configs: list[ModelConfig] | None = None,
    target_name: str = "rate_const",
    drop_constant_targets: bool = False,
) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = {name: results_dir / subdir for name, subdir in RESULT_SUBDIRS.items()}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    verification_rows: list[dict[str, object]] = []

    parsed_dir = output_dirs["data"] / "parsed"
    parse_raw_dataset(raw_dataset_path, parsed_dir)
    dataset = load_parsed_dataset(parsed_dir)
    verification_rows.extend(validate_parsed_shapes(dataset))
    manifest_rows.append(
        {
            "category": "data",
            "path": str(parsed_dir),
            "description": "Parsed dataset tables regenerated from the raw .out file.",
        }
    )

    inputs = dataset.training_inputs.copy()
    target_spec = get_target_spec(target_name)
    full_targets, full_reaction_map = _build_target_frame(dataset, inputs, target_spec)
    metadata_columns = get_case_metadata_columns(inputs)

    if split_strategy == "random_case":
        splits = create_random_case_splits(total_cases=len(inputs))
    elif split_strategy == "power_holdout":
        if "power_label" not in inputs.columns:
            raise ValueError("power_holdout split requires power_label in training_inputs.")
        if not holdout_power_labels:
            raise ValueError("holdout_power_labels must be provided for power_holdout.")
        splits = create_group_holdout_splits(inputs["power_label"], holdout_power_labels)
    else:
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")
    split_assignment = build_split_assignment_frame(inputs, splits)
    save_csv(split_assignment, output_dirs["data_snapshots"] / "split_assignments.csv")
    manifest_rows.append(
        {
            "category": "data_snapshots",
            "path": str(output_dirs["data_snapshots"] / "split_assignments.csv"),
            "description": "Locked test split and fold-specific train/validation assignments.",
        }
    )

    split_metadata_rows = [
        {
            "split_name": "locked_test",
            "split_strategy": split_strategy,
            "seed": 42,
            "trainval_case_count": len(splits.trainval_indices),
            "test_case_count": len(splits.test_indices),
            "holdout_power_labels": "|".join(holdout_power_labels or []),
            "target_name": target_spec.name,
            "target_display_name": target_spec.display_name,
        }
    ]
    for fold in splits.validation_folds:
        split_metadata_rows.append(
            {
                "split_name": f"fold_{fold.fold_id}",
                "seed": fold.seed,
                "train_case_count": len(fold.train_indices),
                "validation_case_count": len(fold.val_indices),
            }
        )
    save_csv(pd.DataFrame(split_metadata_rows), output_dirs["tuning"] / "split_metadata.csv")

    full_target_columns = [column for column in full_targets.columns if column.startswith(target_spec.prefix)]
    kept_target_columns = list(full_target_columns)
    dropped_target_columns: list[str] = []
    if drop_constant_targets:
        kept_target_columns, dropped_target_columns = filter_constant_target_columns(
            full_targets.loc[splits.trainval_indices],
            full_target_columns,
        )
    targets = full_targets[metadata_columns + kept_target_columns].copy()
    reaction_map = full_reaction_map[
        full_reaction_map[target_spec.reaction_map_column].isin(kept_target_columns)
    ].copy()
    reaction_map["_target_order"] = reaction_map[target_spec.reaction_map_column].map(
        {column: index for index, column in enumerate(kept_target_columns)}
    )
    reaction_map = reaction_map.sort_values("_target_order").drop(columns="_target_order").reset_index(drop=True)
    target_columns = kept_target_columns

    target_cleaning_rows = []
    for _, reaction_row in full_reaction_map.iterrows():
        target_column = reaction_row[target_spec.reaction_map_column]
        target_cleaning_rows.append(
            {
                "reaction_id": reaction_row["reaction_id"],
                "reaction_label": reaction_row["reaction_label"],
                "target_column": target_column,
                "target_name": target_spec.name,
                "kept_for_training": target_column in kept_target_columns,
                "dropped_as_constant": target_column in dropped_target_columns,
            }
        )
    save_csv(pd.DataFrame(target_cleaning_rows), output_dirs["data_snapshots"] / "target_cleaning_map.csv")
    save_csv(
        pd.DataFrame(
            [
                {
                    "target_name": target_spec.name,
                    "target_display_name": target_spec.display_name,
                    "full_target_count": len(full_target_columns),
                    "kept_target_count": len(kept_target_columns),
                    "dropped_target_count": len(dropped_target_columns),
                    "drop_constant_targets": bool(drop_constant_targets),
                }
            ]
        ),
        output_dirs["data_snapshots"] / "target_metadata.csv",
    )

    feature_metadata, epsilon_frame = _save_feature_snapshots(
        inputs=inputs,
        targets=targets,
        split_assignment=split_assignment,
        trainval_indices=splits.trainval_indices,
        target_columns=target_columns,
        data_snapshot_dir=output_dirs["data_snapshots"],
        target_stem=target_spec.name,
    )
    save_csv(feature_metadata, output_dirs["data_snapshots"] / "feature_set_final_metadata.csv")

    target_transformer_trainval = TargetTransformer(target_columns).fit(targets.loc[splits.trainval_indices])
    inverse_check = target_transformer_trainval.inverse_transform_array(
        target_transformer_trainval.transform(targets.loc[splits.trainval_indices]).to_numpy(dtype=float)
    )
    original_trainval = targets.loc[splits.trainval_indices, target_columns].to_numpy(dtype=float)
    inverse_max_abs_error = float(np.max(np.abs(inverse_check - original_trainval)))
    verification_rows.append(
        {
            "check_name": "inverse_transform_roundtrip_trainval",
            "status": "pass" if inverse_max_abs_error < 1e-10 else "fail",
            "expected": "< 1e-10",
            "observed": inverse_max_abs_error,
            "details": "Maximum absolute error after transform + inverse_transform on trainval targets.",
        }
    )

    if configs is None:
        if feature_sets is None:
            feature_sets = ["full_nonconstant_plus_log_en", "composition_pca_plus_log_en"]
        configs = build_model_configs(feature_sets)
        if max_configs is not None:
            configs = configs[:max_configs]
    else:
        if max_configs is not None:
            configs = configs[:max_configs]
        if feature_sets is None:
            feature_sets = sorted({config.feature_set for config in configs})
    save_csv(_build_config_frame(configs), output_dirs["tuning"] / "model_config_catalog.csv")

    trial_rows = []
    oracle_validation_overall_rows = []
    oracle_validation_per_reaction_frames = []
    oracle_validation_per_case_frames = []

    active_configs = list(configs)
    total_possible_fits = len(configs) + len(configs) * max(0, len(splits.validation_folds) - 1)
    completed_configs = 0
    search_stage_rows = []

    for fold in splits.validation_folds:
        print(f"[training] fold {fold.fold_id}/5: preparing preprocessing artifacts")
        train_inputs = inputs.loc[fold.train_indices]
        val_inputs = inputs.loc[fold.val_indices]
        train_targets = targets.loc[fold.train_indices]
        val_targets = targets.loc[fold.val_indices]

        target_transformer = TargetTransformer(target_columns).fit(train_targets)
        y_train_log = target_transformer.transform(train_targets).to_numpy(dtype=float)
        y_val_log = target_transformer.transform(val_targets).to_numpy(dtype=float)
        y_val_original = val_targets[target_columns].to_numpy(dtype=float)

        oracle_overall, oracle_per_reaction, oracle_per_case, oracle_monotonic = _oracle_frames_for_eval(
            y_train_log=y_train_log,
            y_eval_log=y_val_log,
            y_eval_original=y_val_original,
            case_ids=val_targets[metadata_columns],
            reaction_map=reaction_map,
            target_transformer=target_transformer,
            prefix=f"fold_{fold.fold_id}_validation",
            target_column_field=target_spec.reaction_map_column,
        )
        oracle_overall.insert(0, "fold_id", fold.fold_id)
        oracle_per_reaction.insert(0, "fold_id", fold.fold_id)
        oracle_per_case.insert(0, "fold_id", fold.fold_id)
        oracle_validation_overall_rows.append(oracle_overall)
        oracle_validation_per_reaction_frames.append(oracle_per_reaction)
        oracle_validation_per_case_frames.append(oracle_per_case)
        verification_rows.extend(oracle_monotonic.to_dict(orient="records"))

        feature_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for feature_set in feature_sets:
            transformer = build_feature_transformer(feature_set).fit(train_inputs)
            x_train = transformer.transform(train_inputs).to_numpy(dtype=float)
            x_val = transformer.transform(val_inputs).to_numpy(dtype=float)
            feature_matrices[feature_set] = (x_train, x_val)

        for config in active_configs:
            completed_configs += 1
            if completed_configs == 1 or completed_configs % 50 == 0:
                print(
                    f"[training] fold {fold.fold_id}: evaluating config {completed_configs}/{total_possible_fits} "
                    f"({config.model_key})"
                )
            x_train, x_val = feature_matrices[config.feature_set]
            row = {
                "fold_id": fold.fold_id,
                "fold_seed": fold.seed,
                "search_stage": "stage1" if fold.fold_id == 1 else "stage2",
                "status": "success",
                "error_message": "",
                **config.as_dict(),
                "hyperparameters_json": str(config.hyperparameters),
            }
            start_time = time.perf_counter()
            try:
                model = build_model(config)
                if config.latent_k is not None:
                    validation_callback = lambda reconstructed_log, truth=y_val_log: float(
                        math.sqrt(np.mean((reconstructed_log - truth) ** 2))
                    )
                else:
                    validation_callback = lambda predicted_log, truth=y_val_log: float(
                        math.sqrt(np.mean((predicted_log - truth) ** 2))
                    )
                model.fit(
                    x_train=x_train,
                    y_train=y_train_log,
                    x_val=x_val,
                    y_val=y_val_log,
                    validation_score_callback=validation_callback,
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
                    target_column_field=target_spec.reaction_map_column,
                )
                row.update(_prefixed_metric_row("validation", metrics))
                row["validation_mean_reaction_log_r2"] = float(
                    per_reaction["log_r2"].dropna().mean()
                )
                row["validation_median_reaction_log_mae"] = float(
                    per_reaction["log_mae"].median()
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error_message"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=5)
            row["runtime_seconds"] = time.perf_counter() - start_time
            trial_rows.append(row)

        if fold.fold_id == 1 and max_configs is None:
            stage1_frame = pd.DataFrame(trial_rows)
            survivors = _select_stage2_survivors(stage1_frame[stage1_frame["fold_id"] == 1], configs)
            search_stage_rows.append(
                {
                    "stage_name": "stage1_full_catalog",
                    "evaluated_config_count": len(configs),
                    "survivor_config_count": len(survivors),
                    "survivor_fraction": len(survivors) / len(configs),
                    "selection_rule": "top_20pct_plus_best_per_family_feature_latent_group",
                }
            )
            active_configs = survivors
            total_possible_fits = len(configs) + len(active_configs) * max(
                0, len(splits.validation_folds) - 1
            )
            print(
                f"[training] adaptive search retained {len(active_configs)} survivors for folds 2-5"
            )

    trial_frame = pd.DataFrame(trial_rows)
    save_csv(trial_frame, output_dirs["tuning"] / "model_trials_foldwise.csv")
    manifest_rows.append(
        {
            "category": "tuning",
            "path": str(output_dirs["tuning"] / "model_trials_foldwise.csv"),
            "description": "Per-fold validation metrics for every attempted configuration.",
        }
    )
    summary = _aggregate_trial_summary(trial_frame)
    save_csv(summary, output_dirs["tuning"] / "model_leaderboard_summary.csv")
    selected = _select_best_model(summary)
    save_csv(pd.DataFrame([selected]), output_dirs["tuning"] / "selected_model.csv")
    if search_stage_rows:
        save_csv(pd.DataFrame(search_stage_rows), output_dirs["tuning"] / "search_stage_summary.csv")

    validation_oracle_overall = pd.concat(oracle_validation_overall_rows, ignore_index=True)
    validation_oracle_per_reaction = pd.concat(oracle_validation_per_reaction_frames, ignore_index=True)
    validation_oracle_per_case = pd.concat(oracle_validation_per_case_frames, ignore_index=True)
    save_csv(validation_oracle_overall, output_dirs["pca"] / "oracle_validation_overall_by_k.csv")
    save_csv(validation_oracle_per_reaction, output_dirs["pca"] / "oracle_validation_per_reaction_by_k.csv")
    save_csv(validation_oracle_per_case, output_dirs["pca"] / "oracle_validation_per_case_by_k.csv")

    trainval_inputs = inputs.loc[splits.trainval_indices]
    test_inputs = inputs.loc[splits.test_indices]
    trainval_targets = targets.loc[splits.trainval_indices]
    test_targets = targets.loc[splits.test_indices]

    final_target_transformer = TargetTransformer(target_columns).fit(trainval_targets)
    y_trainval_log = final_target_transformer.transform(trainval_targets).to_numpy(dtype=float)
    y_test_log = final_target_transformer.transform(test_targets).to_numpy(dtype=float)
    y_test_original = test_targets[target_columns].to_numpy(dtype=float)

    explained_variance = pca_explained_variance_frame(y_trainval_log)
    save_csv(explained_variance, output_dirs["pca"] / "target_pca_explained_variance.csv")
    oracle_test_overall, oracle_test_per_reaction, oracle_test_per_case, oracle_test_monotonic = _oracle_frames_for_eval(
        y_train_log=y_trainval_log,
        y_eval_log=y_test_log,
        y_eval_original=y_test_original,
        case_ids=test_targets[metadata_columns],
        reaction_map=reaction_map,
        target_transformer=final_target_transformer,
        prefix="locked_test",
        target_column_field=target_spec.reaction_map_column,
    )
    save_csv(oracle_test_overall, output_dirs["pca"] / "oracle_test_overall_by_k.csv")
    save_csv(oracle_test_per_reaction, output_dirs["pca"] / "oracle_test_per_reaction_by_k.csv")
    save_csv(oracle_test_per_case, output_dirs["pca"] / "oracle_test_per_case_by_k.csv")
    verification_rows.extend(oracle_test_monotonic.to_dict(orient="records"))

    selected_config = next(config for config in configs if config.model_key == selected["model_key"])
    final_feature_transformer = build_feature_transformer(selected_config.feature_set).fit(trainval_inputs)
    x_trainval = final_feature_transformer.transform(trainval_inputs).to_numpy(dtype=float)
    x_test = final_feature_transformer.transform(test_inputs).to_numpy(dtype=float)

    print(f"[training] final fit using selected model: {selected_config.model_key}")
    final_model = build_model(selected_config)
    final_model.fit(x_train=x_trainval, y_train=y_trainval_log)
    y_test_log_pred = final_model.predict(x_test)
    y_test_original_pred = final_target_transformer.inverse_transform_array(y_test_log_pred)

    active_overall_metrics = compute_overall_metrics(
        y_true_log=y_test_log,
        y_pred_log=y_test_log_pred,
        y_true_original=y_test_original,
        y_pred_original=y_test_original_pred,
    )
    active_per_reaction_metrics = compute_per_reaction_metrics(
        y_true_log=y_test_log,
        y_pred_log=y_test_log_pred,
        y_true_original=y_test_original,
        y_pred_original=y_test_original_pred,
        reaction_map=reaction_map,
        target_column_field=target_spec.reaction_map_column,
    )
    active_per_case_metrics = compute_per_case_metrics(
        y_true_log=y_test_log,
        y_pred_log=y_test_log_pred,
        y_true_original=y_test_original,
        y_pred_original=y_test_original_pred,
        case_ids=test_targets[metadata_columns],
    )
    active_prediction_frame = build_prediction_frame(
        case_ids=test_targets[metadata_columns],
        reaction_map=reaction_map,
        y_true_original=y_test_original,
        y_pred_original=y_test_original_pred,
        y_true_log=y_test_log,
        y_pred_log=y_test_log_pred,
        target_column_field=target_spec.reaction_map_column,
        target_value_label=target_spec.name,
    )

    full_target_transformer = TargetTransformer(full_target_columns).fit(
        full_targets.loc[splits.trainval_indices]
    )
    y_test_full_original = full_targets.loc[splits.test_indices, full_target_columns].to_numpy(dtype=float)
    y_test_full_original_pred = _expand_target_matrix(
        y_test_original_pred,
        kept_target_columns=kept_target_columns,
        full_target_columns=full_target_columns,
    )
    y_test_full_log = full_target_transformer.transform_array(y_test_full_original)
    y_test_full_log_pred = full_target_transformer.transform_array(y_test_full_original_pred)
    full_overall_metrics = compute_overall_metrics(
        y_true_log=y_test_full_log,
        y_pred_log=y_test_full_log_pred,
        y_true_original=y_test_full_original,
        y_pred_original=y_test_full_original_pred,
    )
    full_per_reaction_metrics = compute_per_reaction_metrics(
        y_true_log=y_test_full_log,
        y_pred_log=y_test_full_log_pred,
        y_true_original=y_test_full_original,
        y_pred_original=y_test_full_original_pred,
        reaction_map=full_reaction_map,
        target_column_field=target_spec.reaction_map_column,
    )
    full_per_case_metrics = compute_per_case_metrics(
        y_true_log=y_test_full_log,
        y_pred_log=y_test_full_log_pred,
        y_true_original=y_test_full_original,
        y_pred_original=y_test_full_original_pred,
        case_ids=test_targets[metadata_columns],
    )
    prediction_frame = build_prediction_frame(
        case_ids=test_targets[metadata_columns],
        reaction_map=full_reaction_map,
        y_true_original=y_test_full_original,
        y_pred_original=y_test_full_original_pred,
        y_true_log=y_test_full_log,
        y_pred_log=y_test_full_log_pred,
        target_column_field=target_spec.reaction_map_column,
        target_value_label=target_spec.name,
    )
    relative_error_overall, relative_error_per_reaction, relative_error_per_case, relative_error_by_magnitude = (
        compute_relative_error_outputs(prediction_frame, target_value_label=target_spec.name)
    )
    relative_error_frame = build_relative_error_frame(prediction_frame, target_value_label=target_spec.name)
    smape_overall, smape_per_reaction, smape_per_case, smape_by_magnitude = compute_smape_outputs(
        prediction_frame,
        target_value_label=target_spec.name,
    )
    smape_frame = build_smape_frame(prediction_frame, target_value_label=target_spec.name)

    save_csv(pd.DataFrame([active_overall_metrics]), output_dirs["evaluation"] / "test_overall_metrics_active_targets.csv")
    save_csv(active_per_reaction_metrics, output_dirs["evaluation"] / "test_per_reaction_metrics_active_targets.csv")
    save_csv(active_per_case_metrics, output_dirs["evaluation"] / "test_per_case_metrics_active_targets.csv")
    save_csv(active_prediction_frame, output_dirs["evaluation"] / "test_predictions_long_active_targets.csv")
    save_csv(pd.DataFrame([full_overall_metrics]), output_dirs["evaluation"] / "test_overall_metrics_full_reconstructed.csv")
    save_csv(full_per_reaction_metrics, output_dirs["evaluation"] / "test_per_reaction_metrics_full_reconstructed.csv")
    save_csv(full_per_case_metrics, output_dirs["evaluation"] / "test_per_case_metrics_full_reconstructed.csv")
    save_csv(prediction_frame, output_dirs["evaluation"] / "test_predictions_long_full_reconstructed.csv")
    save_csv(pd.DataFrame([full_overall_metrics]), output_dirs["evaluation"] / "test_overall_metrics.csv")
    save_csv(full_per_reaction_metrics, output_dirs["evaluation"] / "test_per_reaction_metrics.csv")
    save_csv(full_per_case_metrics, output_dirs["evaluation"] / "test_per_case_metrics.csv")
    save_csv(prediction_frame, output_dirs["evaluation"] / "test_predictions_long.csv")
    save_csv(relative_error_overall, output_dirs["evaluation"] / "test_relative_error_overall_summary.csv")
    save_csv(relative_error_per_reaction, output_dirs["evaluation"] / "test_relative_error_per_reaction.csv")
    save_csv(relative_error_per_case, output_dirs["evaluation"] / "test_relative_error_per_case.csv")
    save_csv(relative_error_by_magnitude, output_dirs["evaluation"] / "test_relative_error_by_magnitude_bin.csv")
    save_csv(smape_overall, output_dirs["evaluation"] / "test_smape_overall_summary.csv")
    save_csv(smape_per_reaction, output_dirs["evaluation"] / "test_smape_per_reaction.csv")
    save_csv(smape_per_case, output_dirs["evaluation"] / "test_smape_per_case.csv")
    save_csv(smape_by_magnitude, output_dirs["evaluation"] / "test_smape_by_magnitude_bin.csv")
    save_csv(
        full_per_reaction_metrics.nlargest(10, "log_rmse"),
        output_dirs["evaluation"] / "worst_10_reactions.csv",
    )
    save_csv(
        full_per_case_metrics.nlargest(10, "log_rmse"),
        output_dirs["evaluation"] / "worst_10_cases.csv",
    )
    save_csv(
        relative_error_per_reaction.nlargest(10, "median_absolute_relative_error"),
        output_dirs["evaluation"] / "worst_10_reactions_by_relative_error.csv",
    )
    save_csv(
        relative_error_per_case.nlargest(10, "median_absolute_relative_error"),
        output_dirs["evaluation"] / "worst_10_cases_by_relative_error.csv",
    )
    save_csv(
        smape_per_reaction.nlargest(10, "median_smape"),
        output_dirs["evaluation"] / "worst_10_reactions_by_smape.csv",
    )
    save_csv(
        smape_per_case.nlargest(10, "median_smape"),
        output_dirs["evaluation"] / "worst_10_cases_by_smape.csv",
    )

    verification_rows.append(
        {
            "check_name": "results_written",
            "status": "pass",
            "expected": "non-empty evaluation outputs",
            "observed": len(prediction_frame),
            "details": "Number of row-level prediction outputs saved for the locked test split.",
        }
    )
    verification_frame = pd.DataFrame(verification_rows)
    save_csv(verification_frame, results_dir / "verification_checks.csv")

    plot_pca_scree(explained_variance, output_dirs["figures"] / "pca_scree.png")
    plot_oracle_error(oracle_test_overall, output_dirs["figures"] / "oracle_reconstruction_error_by_k.png")
    plot_model_leaderboard(summary, output_dirs["figures"] / "model_leaderboard.png")
    plot_parity(
        y_true=y_test_full_log,
        y_pred=y_test_full_log_pred,
        path=output_dirs["figures"] / "parity_plot_log_space.png",
        title=f"Predicted vs True Log10 {target_spec.display_name}",
        x_label=f"True log10({target_spec.singular_label} + epsilon)",
        y_label=f"Predicted log10({target_spec.singular_label} + epsilon)",
    )
    plot_parity(
        y_true=np.log10(np.maximum(y_test_full_original, 1e-300)),
        y_pred=np.log10(np.maximum(y_test_full_original_pred, 1e-300)),
        path=output_dirs["figures"] / "parity_plot_original_space.png",
        title=f"Predicted vs True {target_spec.display_name} (Log10 View)",
        x_label=f"True log10({target_spec.name})",
        y_label=f"Predicted log10({target_spec.name})",
    )
    plot_log_residual_histogram(prediction_frame, output_dirs["figures"] / "residual_hist_log_space.png")
    plot_worst_reactions(full_per_reaction_metrics, output_dirs["figures"] / "worst_reactions_log_rmse.png")
    plot_case_error_distribution(full_per_case_metrics, output_dirs["figures"] / "per_case_log_rmse_distribution.png")
    plot_relative_error_histogram(
        relative_error_overall,
        relative_error_frame,
        output_dirs["figures"] / "relative_error_abs_histogram.png",
    )
    plot_relative_error_by_magnitude(
        relative_error_by_magnitude,
        output_dirs["figures"] / "relative_error_by_magnitude.png",
    )
    plot_smape_histogram(
        smape_overall,
        smape_frame,
        output_dirs["figures"] / "smape_histogram.png",
    )
    plot_smape_by_magnitude(
        smape_by_magnitude,
        output_dirs["figures"] / "smape_by_magnitude.png",
    )

    manifest_rows.extend(
        [
            {
                "category": "tuning",
                "path": str(output_dirs["tuning"] / "model_leaderboard_summary.csv"),
                "description": "Aggregated validation leaderboard used for model selection.",
            },
            {
                "category": "pca",
                "path": str(output_dirs["pca"] / "oracle_test_overall_by_k.csv"),
                "description": "Compression-only oracle PCA reconstruction metrics on the locked test split.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_overall_metrics.csv"),
                "description": "Final locked-test overall metrics for the selected model.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_per_reaction_metrics.csv"),
                "description": "Per-reaction test metrics for all 204 outputs.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_per_case_metrics.csv"),
                "description": "Per-case test metrics on the locked test split.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_relative_error_overall_summary.csv"),
                "description": "Overall relative-error summary for positive-ground-truth rate constants.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_relative_error_per_reaction.csv"),
                "description": "Per-reaction relative-error summary for positive-ground-truth rate constants.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_relative_error_per_case.csv"),
                "description": "Per-case relative-error summary for positive-ground-truth rate constants.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_smape_overall_summary.csv"),
                "description": "Overall SMAPE summary over all prediction rows.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_smape_per_reaction.csv"),
                "description": "Per-reaction SMAPE summary over all prediction rows.",
            },
            {
                "category": "evaluation",
                "path": str(output_dirs["evaluation"] / "test_smape_per_case.csv"),
                "description": "Per-case SMAPE summary over all prediction rows.",
            },
            {
                "category": "figures",
                "path": str(output_dirs["figures"] / "model_leaderboard.png"),
                "description": "Visualization of the top validation configurations.",
            },
            {
                "category": "figures",
                "path": str(output_dirs["figures"] / "oracle_reconstruction_error_by_k.png"),
                "description": "Oracle PCA reconstruction log-RMSE versus latent dimension.",
            },
            {
                "category": "figures",
                "path": str(output_dirs["figures"] / "relative_error_abs_histogram.png"),
                "description": "Histogram of absolute relative errors on positive-ground-truth outputs.",
            },
            {
                "category": "figures",
                "path": str(output_dirs["figures"] / "smape_histogram.png"),
                "description": "Histogram of SMAPE values over all prediction rows.",
            },
        ]
    )
    manifest_frame = pd.DataFrame(manifest_rows)
    save_csv(manifest_frame, results_dir / "output_manifest.csv")

    if report_output_path is not None:
        export_experiment_report(results_dir=results_dir, output_path=report_output_path)

    return {
        "results_dir": results_dir,
        "manifest": results_dir / "output_manifest.csv",
        "selected_model": output_dirs["tuning"] / "selected_model.csv",
        "overall_metrics": output_dirs["evaluation"] / "test_overall_metrics.csv",
        "report": report_output_path if report_output_path is not None else Path(),
    }
