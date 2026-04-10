from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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
    plot_case_error_distribution,
    plot_log_residual_histogram,
    plot_parity,
    plot_relative_error_by_magnitude,
    plot_relative_error_histogram,
    plot_smape_by_magnitude,
    plot_smape_histogram,
    plot_worst_reactions,
    save_csv,
    save_figure,
)
from .models import DirectMLPModel, ModelConfig, RANDOM_STATE
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


RESULT_SUBDIRS = {
    "data": "data",
    "data_snapshots": "data_snapshots",
    "tuning": "tuning",
    "evaluation": "evaluation",
    "figures": "figures",
}


@dataclass
class MultitaskTargetBundle:
    target_spec: TargetSpec
    full_targets: pd.DataFrame
    active_targets: pd.DataFrame
    full_reaction_map: pd.DataFrame
    active_reaction_map: pd.DataFrame
    full_target_columns: list[str]
    kept_target_columns: list[str]
    dropped_target_columns: list[str]


def _prefixed_metric_row(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _estimate_eta_seconds(completed: int, total: int, elapsed_seconds: float) -> float | None:
    if completed <= 0 or total <= 0 or elapsed_seconds <= 0:
        return None
    remaining = max(total - completed, 0)
    if remaining == 0:
        return 0.0
    return (elapsed_seconds / completed) * remaining


def _build_target_frame(dataset, inputs: pd.DataFrame, target_spec: TargetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _build_task_bundle(
    dataset,
    inputs: pd.DataFrame,
    target_name: str,
    trainval_indices: np.ndarray,
    drop_constant_targets: bool,
) -> MultitaskTargetBundle:
    target_spec = get_target_spec(target_name)
    full_targets, full_reaction_map = _build_target_frame(dataset, inputs, target_spec)
    metadata_columns = get_case_metadata_columns(inputs)
    full_target_columns = [column for column in full_targets.columns if column.startswith(target_spec.prefix)]
    kept_target_columns = list(full_target_columns)
    dropped_target_columns: list[str] = []
    if drop_constant_targets:
        kept_target_columns, dropped_target_columns = filter_constant_target_columns(
            full_targets.loc[trainval_indices],
            full_target_columns,
        )

    active_targets = full_targets[metadata_columns + kept_target_columns].copy()
    active_reaction_map = full_reaction_map[
        full_reaction_map[target_spec.reaction_map_column].isin(kept_target_columns)
    ].copy()
    active_reaction_map["_target_order"] = active_reaction_map[target_spec.reaction_map_column].map(
        {column: index for index, column in enumerate(kept_target_columns)}
    )
    active_reaction_map = (
        active_reaction_map.sort_values("_target_order")
        .drop(columns="_target_order")
        .reset_index(drop=True)
    )
    return MultitaskTargetBundle(
        target_spec=target_spec,
        full_targets=full_targets,
        active_targets=active_targets,
        full_reaction_map=full_reaction_map,
        active_reaction_map=active_reaction_map,
        full_target_columns=full_target_columns,
        kept_target_columns=kept_target_columns,
        dropped_target_columns=dropped_target_columns,
    )


class JointSingleHeadModelWrapper:
    def __init__(self, config: ModelConfig, rate_dim: int) -> None:
        params = config.hyperparameters
        self.rate_dim = rate_dim
        self.model = DirectMLPModel(
            hidden_width=int(params["hidden_width"]),
            hidden_layers=int(params["hidden_layers"]),
            dropout=float(params["dropout"]),
            weight_decay=float(params["weight_decay"]),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            batch_size=int(params.get("batch_size", 256)),
            max_epochs=int(params.get("max_epochs", 120)),
            patience=int(params.get("patience", 15)),
        )

    def fit(
        self,
        x_train: np.ndarray,
        y_rate_train: np.ndarray,
        y_super_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_rate_val: np.ndarray | None = None,
        y_super_val: np.ndarray | None = None,
        validation_score_callback=None,
    ) -> "JointSingleHeadModelWrapper":
        y_train = np.concatenate([y_rate_train, y_super_train], axis=1)
        y_val = None
        callback = None
        if y_rate_val is not None and y_super_val is not None:
            y_val = np.concatenate([y_rate_val, y_super_val], axis=1)
        if validation_score_callback is not None:
            def callback(concat_predictions: np.ndarray) -> float:
                rate_pred, super_pred = self.predict_split_array(concat_predictions)
                return float(validation_score_callback(rate_pred, super_pred))

        self.model.fit(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            validation_score_callback=callback,
        )
        return self

    def predict_split_array(self, concat_predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rate_pred = concat_predictions[:, : self.rate_dim]
        super_pred = concat_predictions[:, self.rate_dim :]
        return rate_pred, super_pred

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        concat_predictions = self.model.predict(x)
        return self.predict_split_array(np.asarray(concat_predictions, dtype=float))


class _DualHeadNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        hidden_layers: int,
        dropout: float,
        rate_dim: int,
        super_dim: int,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(last_dim, hidden_width))
            layers.append(torch.nn.ReLU())
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            last_dim = hidden_width
        self.backbone = torch.nn.Sequential(*layers)
        self.rate_head = torch.nn.Linear(last_dim, rate_dim)
        self.super_head = torch.nn.Linear(last_dim, super_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.rate_head(features), self.super_head(features)


class SharedBackboneTwoHeadModel:
    def __init__(self, config: ModelConfig, rate_dim: int, super_dim: int) -> None:
        params = config.hyperparameters
        self.hidden_width = int(params["hidden_width"])
        self.hidden_layers = int(params["hidden_layers"])
        self.dropout = float(params["dropout"])
        self.weight_decay = float(params["weight_decay"])
        self.learning_rate = float(params.get("learning_rate", 1e-3))
        self.batch_size = int(params.get("batch_size", 256))
        self.max_epochs = int(params.get("max_epochs", 120))
        self.patience = int(params.get("patience", 15))
        self.rate_loss_weight = float(params.get("rate_loss_weight", 0.5))
        self.super_loss_weight = float(params.get("super_loss_weight", 0.5))
        self.rate_dim = rate_dim
        self.super_dim = super_dim
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.rate_mean_: np.ndarray | None = None
        self.rate_scale_: np.ndarray | None = None
        self.super_mean_: np.ndarray | None = None
        self.super_scale_: np.ndarray | None = None
        self.network: _DualHeadNetwork | None = None
        self.device = torch.device("cpu")

    def _fit_standardizer(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return mean.astype(float), scale.astype(float)

    def _transform(self, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - mean) / scale

    def _inverse(self, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * scale + mean

    def fit(
        self,
        x_train: np.ndarray,
        y_rate_train: np.ndarray,
        y_super_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_rate_val: np.ndarray | None = None,
        y_super_val: np.ndarray | None = None,
        validation_score_callback=None,
    ) -> "SharedBackboneTwoHeadModel":
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        self.x_mean_, self.x_scale_ = self._fit_standardizer(x_train)
        self.rate_mean_, self.rate_scale_ = self._fit_standardizer(y_rate_train)
        self.super_mean_, self.super_scale_ = self._fit_standardizer(y_super_train)

        x_train_scaled = self._transform(x_train, self.x_mean_, self.x_scale_)
        y_rate_train_scaled = self._transform(y_rate_train, self.rate_mean_, self.rate_scale_)
        y_super_train_scaled = self._transform(y_super_train, self.super_mean_, self.super_scale_)

        x_val_scaled = None
        y_rate_val_scaled = None
        y_super_val_scaled = None
        if x_val is not None and y_rate_val is not None and y_super_val is not None:
            x_val_scaled = self._transform(x_val, self.x_mean_, self.x_scale_)
            y_rate_val_scaled = self._transform(y_rate_val, self.rate_mean_, self.rate_scale_)
            y_super_val_scaled = self._transform(y_super_val, self.super_mean_, self.super_scale_)

        self.network = _DualHeadNetwork(
            input_dim=x_train.shape[1],
            hidden_width=self.hidden_width,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            rate_dim=self.rate_dim,
            super_dim=self.super_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.MSELoss()

        train_x = torch.tensor(x_train_scaled, dtype=torch.float32, device=self.device)
        train_rate = torch.tensor(y_rate_train_scaled, dtype=torch.float32, device=self.device)
        train_super = torch.tensor(y_super_train_scaled, dtype=torch.float32, device=self.device)

        if x_val_scaled is not None and y_rate_val_scaled is not None and y_super_val_scaled is not None:
            val_x = torch.tensor(x_val_scaled, dtype=torch.float32, device=self.device)
            val_rate = torch.tensor(y_rate_val_scaled, dtype=torch.float32, device=self.device)
            val_super = torch.tensor(y_super_val_scaled, dtype=torch.float32, device=self.device)
        else:
            val_x = None
            val_rate = None
            val_super = None

        best_state = copy.deepcopy(self.network.state_dict())
        best_score = float("inf")
        patience_left = self.patience

        for _epoch in range(self.max_epochs):
            permutation = np.random.permutation(train_x.shape[0])
            self.network.train()
            for start in range(0, len(permutation), self.batch_size):
                batch_indices = permutation[start : start + self.batch_size]
                batch_x = train_x[batch_indices]
                batch_rate = train_rate[batch_indices]
                batch_super = train_super[batch_indices]
                optimizer.zero_grad()
                pred_rate, pred_super = self.network(batch_x)
                loss = (
                    self.rate_loss_weight * loss_fn(pred_rate, batch_rate)
                    + self.super_loss_weight * loss_fn(pred_super, batch_super)
                )
                loss.backward()
                optimizer.step()

            self.network.eval()
            with torch.no_grad():
                if val_x is not None and val_rate is not None and val_super is not None:
                    pred_rate, pred_super = self.network(val_x)
                    if validation_score_callback is not None:
                        unscaled_rate = self._inverse(
                            pred_rate.cpu().numpy(),
                            self.rate_mean_,
                            self.rate_scale_,
                        )
                        unscaled_super = self._inverse(
                            pred_super.cpu().numpy(),
                            self.super_mean_,
                            self.super_scale_,
                        )
                        score = float(validation_score_callback(unscaled_rate, unscaled_super))
                    else:
                        score = float(
                            self.rate_loss_weight * loss_fn(pred_rate, val_rate).item()
                            + self.super_loss_weight * loss_fn(pred_super, val_super).item()
                        )
                else:
                    pred_rate, pred_super = self.network(train_x)
                    score = float(
                        self.rate_loss_weight * loss_fn(pred_rate, train_rate).item()
                        + self.super_loss_weight * loss_fn(pred_super, train_super).item()
                    )

            if score < best_score - 1e-8:
                best_score = score
                best_state = copy.deepcopy(self.network.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.network.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.network is None
            or self.x_mean_ is None
            or self.x_scale_ is None
            or self.rate_mean_ is None
            or self.rate_scale_ is None
            or self.super_mean_ is None
            or self.super_scale_ is None
        ):
            raise RuntimeError("Model must be fit before prediction.")
        x_scaled = self._transform(x, self.x_mean_, self.x_scale_)
        tensor_x = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            pred_rate, pred_super = self.network(tensor_x)
        rate_unscaled = self._inverse(pred_rate.cpu().numpy(), self.rate_mean_, self.rate_scale_)
        super_unscaled = self._inverse(pred_super.cpu().numpy(), self.super_mean_, self.super_scale_)
        return rate_unscaled, super_unscaled


def build_multitask_model(
    config: ModelConfig,
    rate_dim: int,
    super_dim: int,
) -> JointSingleHeadModelWrapper | SharedBackboneTwoHeadModel:
    if config.model_family == "joint_single_head_mlp":
        return JointSingleHeadModelWrapper(config=config, rate_dim=rate_dim)
    if config.model_family == "joint_two_head_mlp":
        return SharedBackboneTwoHeadModel(config=config, rate_dim=rate_dim, super_dim=super_dim)
    raise ValueError(f"Unsupported multitask model family: {config.model_family}")


def _save_feature_snapshots(
    inputs: pd.DataFrame,
    split_assignment: pd.DataFrame,
    trainval_indices: np.ndarray,
    feature_sets: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    trainval_inputs = inputs.loc[trainval_indices]
    metadata_columns = get_case_metadata_columns(inputs)
    rows = []
    for feature_set in feature_sets:
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
        save_csv(snapshot, output_dir / f"{feature_set}_all_cases.csv")
        meta = transformer.metadata()
        meta["fit_scope"] = "trainval"
        rows.append(meta)
    return pd.DataFrame(rows)


def _save_target_snapshots(
    bundle: MultitaskTargetBundle,
    split_assignment: pd.DataFrame,
    trainval_indices: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    metadata_columns = get_case_metadata_columns(bundle.active_targets)
    transformer = TargetTransformer(bundle.kept_target_columns).fit(
        bundle.active_targets.loc[trainval_indices]
    )
    transformed = pd.concat(
        [
            bundle.active_targets[metadata_columns].reset_index(drop=True),
            split_assignment[["locked_split"]].reset_index(drop=True),
            transformer.transform(bundle.active_targets).reset_index(drop=True),
        ],
        axis=1,
    )
    save_csv(transformed, output_dir / f"log_transformed_{bundle.target_spec.name}_all_cases.csv")
    save_csv(transformer.epsilon_frame(), output_dir / f"{bundle.target_spec.name}_epsilons_trainval.csv")

    target_cleaning_rows = []
    for _, reaction_row in bundle.full_reaction_map.iterrows():
        target_column = reaction_row[bundle.target_spec.reaction_map_column]
        target_cleaning_rows.append(
            {
                "reaction_id": reaction_row["reaction_id"],
                "reaction_label": reaction_row["reaction_label"],
                "target_column": target_column,
                "target_name": bundle.target_spec.name,
                "kept_for_training": target_column in bundle.kept_target_columns,
                "dropped_as_constant": target_column in bundle.dropped_target_columns,
            }
        )
    save_csv(
        pd.DataFrame(target_cleaning_rows),
        output_dir / f"{bundle.target_spec.name}_target_cleaning_map.csv",
    )

    metadata_frame = pd.DataFrame(
        [
            {
                "target_name": bundle.target_spec.name,
                "target_display_name": bundle.target_spec.display_name,
                "full_target_count": len(bundle.full_target_columns),
                "kept_target_count": len(bundle.kept_target_columns),
                "dropped_target_count": len(bundle.dropped_target_columns),
                "drop_constant_targets": bool(bundle.dropped_target_columns),
            }
        ]
    )
    save_csv(metadata_frame, output_dir / f"{bundle.target_spec.name}_target_metadata.csv")
    return metadata_frame


def _aggregate_trial_summary(trial_frame: pd.DataFrame) -> pd.DataFrame:
    successful = trial_frame[trial_frame["status"] == "success"].copy()
    metric_columns = [
        "validation_joint_log_rmse",
        "validation_rate_overall_log_rmse",
        "validation_rate_overall_log_r2",
        "validation_rate_factor5_accuracy_positive_only",
        "validation_super_active_overall_log_rmse",
        "validation_super_active_overall_log_r2",
        "validation_super_active_factor5_accuracy_positive_only",
        "validation_super_full_overall_log_rmse",
        "validation_super_full_overall_log_r2",
        "runtime_seconds",
    ]
    summary = (
        successful.groupby(["model_key", "model_family", "feature_set"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in summary.columns.to_flat_index()
    ]
    rename_map = {
        "validation_joint_log_rmse_mean": "mean_validation_joint_log_rmse",
        "validation_joint_log_rmse_std": "std_validation_joint_log_rmse",
        "validation_rate_overall_log_rmse_mean": "mean_validation_rate_log_rmse",
        "validation_rate_overall_log_rmse_std": "std_validation_rate_log_rmse",
        "validation_rate_overall_log_r2_mean": "mean_validation_rate_log_r2",
        "validation_rate_overall_log_r2_std": "std_validation_rate_log_r2",
        "validation_rate_factor5_accuracy_positive_only_mean": "mean_validation_rate_factor5_accuracy",
        "validation_rate_factor5_accuracy_positive_only_std": "std_validation_rate_factor5_accuracy",
        "validation_super_active_overall_log_rmse_mean": "mean_validation_super_active_log_rmse",
        "validation_super_active_overall_log_rmse_std": "std_validation_super_active_log_rmse",
        "validation_super_active_overall_log_r2_mean": "mean_validation_super_active_log_r2",
        "validation_super_active_overall_log_r2_std": "std_validation_super_active_log_r2",
        "validation_super_active_factor5_accuracy_positive_only_mean": "mean_validation_super_active_factor5_accuracy",
        "validation_super_active_factor5_accuracy_positive_only_std": "std_validation_super_active_factor5_accuracy",
        "validation_super_full_overall_log_rmse_mean": "mean_validation_super_full_log_rmse",
        "validation_super_full_overall_log_rmse_std": "std_validation_super_full_log_rmse",
        "validation_super_full_overall_log_r2_mean": "mean_validation_super_full_log_r2",
        "validation_super_full_overall_log_r2_std": "std_validation_super_full_log_r2",
        "runtime_seconds_mean": "mean_runtime_seconds",
        "runtime_seconds_std": "std_runtime_seconds",
    }
    summary = summary.rename(columns=rename_map)
    fold_counts = (
        successful.groupby("model_key", dropna=False)["fold_id"]
        .nunique()
        .rename("evaluated_fold_count")
        .reset_index()
    )
    first_rows = (
        successful.sort_values("model_key")
        .groupby("model_key", dropna=False)
        .first()
        .reset_index()[["model_key", "hyperparameters_json"]]
    )
    summary = summary.merge(fold_counts, on="model_key", how="left")
    summary = summary.merge(first_rows, on="model_key", how="left")
    summary = summary.sort_values(
        by=[
            "evaluated_fold_count",
            "mean_validation_joint_log_rmse",
            "mean_validation_rate_log_r2",
            "mean_validation_super_active_log_r2",
        ],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)
    summary["leaderboard_rank"] = np.arange(1, len(summary) + 1)
    ordered = [
        "leaderboard_rank",
        "model_key",
        "model_family",
        "feature_set",
        "hyperparameters_json",
        "evaluated_fold_count",
    ] + [
        column
        for column in summary.columns
        if column
        not in {
            "leaderboard_rank",
            "model_key",
            "model_family",
            "feature_set",
            "hyperparameters_json",
            "evaluated_fold_count",
        }
    ]
    return summary[ordered]


def _select_best_model(summary: pd.DataFrame) -> pd.Series:
    eligible = summary[summary["evaluated_fold_count"] == 5].copy()
    if eligible.empty:
        eligible = summary.copy()
    return eligible.iloc[0]


def _plot_multitask_leaderboard(summary: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    top = summary.nsmallest(top_n, "mean_validation_joint_log_rmse").copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=top, x="mean_validation_joint_log_rmse", y="model_key", hue="model_family", dodge=False, ax=ax)
    ax.set_xlabel("Mean validation joint log RMSE")
    ax.set_ylabel("Model")
    ax.set_title(f"Top {top_n} Multitask Validation Configurations")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def _save_task_evaluation_outputs(
    bundle: MultitaskTargetBundle,
    case_ids: pd.DataFrame,
    y_active_true_log: np.ndarray,
    y_active_pred_log: np.ndarray,
    y_active_true_original: np.ndarray,
    y_active_pred_original: np.ndarray,
    y_full_true_log: np.ndarray,
    y_full_pred_log: np.ndarray,
    y_full_true_original: np.ndarray,
    y_full_pred_original: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    prefix = bundle.target_spec.name
    active_overall = compute_overall_metrics(
        y_true_log=y_active_true_log,
        y_pred_log=y_active_pred_log,
        y_true_original=y_active_true_original,
        y_pred_original=y_active_pred_original,
    )
    active_per_reaction = compute_per_reaction_metrics(
        y_true_log=y_active_true_log,
        y_pred_log=y_active_pred_log,
        y_true_original=y_active_true_original,
        y_pred_original=y_active_pred_original,
        reaction_map=bundle.active_reaction_map,
        target_column_field=bundle.target_spec.reaction_map_column,
    )
    active_per_case = compute_per_case_metrics(
        y_true_log=y_active_true_log,
        y_pred_log=y_active_pred_log,
        y_true_original=y_active_true_original,
        y_pred_original=y_active_pred_original,
        case_ids=case_ids,
    )
    active_prediction_frame = build_prediction_frame(
        case_ids=case_ids,
        reaction_map=bundle.active_reaction_map,
        y_true_original=y_active_true_original,
        y_pred_original=y_active_pred_original,
        y_true_log=y_active_true_log,
        y_pred_log=y_active_pred_log,
        target_column_field=bundle.target_spec.reaction_map_column,
        target_value_label=prefix,
    )

    full_overall = compute_overall_metrics(
        y_true_log=y_full_true_log,
        y_pred_log=y_full_pred_log,
        y_true_original=y_full_true_original,
        y_pred_original=y_full_pred_original,
    )
    full_per_reaction = compute_per_reaction_metrics(
        y_true_log=y_full_true_log,
        y_pred_log=y_full_pred_log,
        y_true_original=y_full_true_original,
        y_pred_original=y_full_pred_original,
        reaction_map=bundle.full_reaction_map,
        target_column_field=bundle.target_spec.reaction_map_column,
    )
    full_per_case = compute_per_case_metrics(
        y_true_log=y_full_true_log,
        y_pred_log=y_full_pred_log,
        y_true_original=y_full_true_original,
        y_pred_original=y_full_pred_original,
        case_ids=case_ids,
    )
    full_prediction_frame = build_prediction_frame(
        case_ids=case_ids,
        reaction_map=bundle.full_reaction_map,
        y_true_original=y_full_true_original,
        y_pred_original=y_full_pred_original,
        y_true_log=y_full_true_log,
        y_pred_log=y_full_pred_log,
        target_column_field=bundle.target_spec.reaction_map_column,
        target_value_label=prefix,
    )

    relative_overall, relative_per_reaction, relative_per_case, relative_by_magnitude = (
        compute_relative_error_outputs(full_prediction_frame, target_value_label=prefix)
    )
    relative_frame = build_relative_error_frame(full_prediction_frame, target_value_label=prefix)
    smape_overall, smape_per_reaction, smape_per_case, smape_by_magnitude = compute_smape_outputs(
        full_prediction_frame,
        target_value_label=prefix,
    )
    smape_frame = build_smape_frame(full_prediction_frame, target_value_label=prefix)

    save_csv(pd.DataFrame([active_overall]), output_dir / f"{prefix}_test_overall_metrics_active_targets.csv")
    save_csv(active_per_reaction, output_dir / f"{prefix}_test_per_reaction_metrics_active_targets.csv")
    save_csv(active_per_case, output_dir / f"{prefix}_test_per_case_metrics_active_targets.csv")
    save_csv(active_prediction_frame, output_dir / f"{prefix}_test_predictions_long_active_targets.csv")

    save_csv(pd.DataFrame([full_overall]), output_dir / f"{prefix}_test_overall_metrics_full_reconstructed.csv")
    save_csv(full_per_reaction, output_dir / f"{prefix}_test_per_reaction_metrics_full_reconstructed.csv")
    save_csv(full_per_case, output_dir / f"{prefix}_test_per_case_metrics_full_reconstructed.csv")
    save_csv(full_prediction_frame, output_dir / f"{prefix}_test_predictions_long_full_reconstructed.csv")

    save_csv(pd.DataFrame([full_overall]), output_dir / f"{prefix}_test_overall_metrics.csv")
    save_csv(full_per_reaction, output_dir / f"{prefix}_test_per_reaction_metrics.csv")
    save_csv(full_per_case, output_dir / f"{prefix}_test_per_case_metrics.csv")
    save_csv(full_prediction_frame, output_dir / f"{prefix}_test_predictions_long.csv")

    save_csv(relative_overall, output_dir / f"{prefix}_test_relative_error_overall_summary.csv")
    save_csv(relative_per_reaction, output_dir / f"{prefix}_test_relative_error_per_reaction.csv")
    save_csv(relative_per_case, output_dir / f"{prefix}_test_relative_error_per_case.csv")
    save_csv(relative_by_magnitude, output_dir / f"{prefix}_test_relative_error_by_magnitude_bin.csv")

    save_csv(smape_overall, output_dir / f"{prefix}_test_smape_overall_summary.csv")
    save_csv(smape_per_reaction, output_dir / f"{prefix}_test_smape_per_reaction.csv")
    save_csv(smape_per_case, output_dir / f"{prefix}_test_smape_per_case.csv")
    save_csv(smape_by_magnitude, output_dir / f"{prefix}_test_smape_by_magnitude_bin.csv")

    save_csv(full_per_reaction.nlargest(10, "log_rmse"), output_dir / f"{prefix}_worst_10_reactions.csv")
    save_csv(full_per_case.nlargest(10, "log_rmse"), output_dir / f"{prefix}_worst_10_cases.csv")
    save_csv(
        relative_per_reaction.nlargest(10, "median_absolute_relative_error"),
        output_dir / f"{prefix}_worst_10_reactions_by_relative_error.csv",
    )
    save_csv(
        relative_per_case.nlargest(10, "median_absolute_relative_error"),
        output_dir / f"{prefix}_worst_10_cases_by_relative_error.csv",
    )
    save_csv(
        smape_per_reaction.nlargest(10, "median_smape"),
        output_dir / f"{prefix}_worst_10_reactions_by_smape.csv",
    )
    save_csv(
        smape_per_case.nlargest(10, "median_smape"),
        output_dir / f"{prefix}_worst_10_cases_by_smape.csv",
    )

    plot_parity(
        y_true=y_full_true_log,
        y_pred=y_full_pred_log,
        path=output_dir.parent / "figures" / f"{prefix}_parity_plot_log_space.png",
        title=f"Predicted vs True Log10 {bundle.target_spec.display_name}",
        x_label=f"True log10({bundle.target_spec.singular_label} + epsilon)",
        y_label=f"Predicted log10({bundle.target_spec.singular_label} + epsilon)",
    )
    plot_parity(
        y_true=np.log10(np.maximum(y_full_true_original, 1e-300)),
        y_pred=np.log10(np.maximum(y_full_pred_original, 1e-300)),
        path=output_dir.parent / "figures" / f"{prefix}_parity_plot_original_space.png",
        title=f"Predicted vs True {bundle.target_spec.display_name} (Log10 View)",
        x_label=f"True log10({prefix})",
        y_label=f"Predicted log10({prefix})",
    )
    plot_log_residual_histogram(full_prediction_frame, output_dir.parent / "figures" / f"{prefix}_residual_hist_log_space.png")
    plot_worst_reactions(full_per_reaction, output_dir.parent / "figures" / f"{prefix}_worst_reactions_log_rmse.png")
    plot_case_error_distribution(full_per_case, output_dir.parent / "figures" / f"{prefix}_per_case_log_rmse_distribution.png")
    plot_relative_error_histogram(
        relative_overall,
        relative_frame,
        output_dir.parent / "figures" / f"{prefix}_relative_error_abs_histogram.png",
    )
    plot_relative_error_by_magnitude(
        relative_by_magnitude,
        output_dir.parent / "figures" / f"{prefix}_relative_error_by_magnitude.png",
    )
    plot_smape_histogram(
        smape_overall,
        smape_frame,
        output_dir.parent / "figures" / f"{prefix}_smape_histogram.png",
    )
    plot_smape_by_magnitude(
        smape_by_magnitude,
        output_dir.parent / "figures" / f"{prefix}_smape_by_magnitude.png",
    )
    return {
        "overall_log_rmse": full_overall["overall_log_rmse"],
        "overall_log_r2": full_overall["overall_log_r2"],
        "median_absolute_relative_error": float(relative_overall.iloc[0]["median_absolute_relative_error"]),
        "median_smape": float(smape_overall.iloc[0]["median_smape"]),
    }


def run_multitask_training_experiment(
    raw_dataset_path: Path,
    results_dir: Path,
    feature_sets: list[str],
    configs: list[ModelConfig],
    split_strategy: str = "random_case",
    holdout_power_labels: list[str] | None = None,
) -> dict[str, Path]:
    experiment_start = time.perf_counter()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = {name: results_dir / subdir for name, subdir in RESULT_SUBDIRS.items()}
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print(
        "[multitask] start | "
        f"split={split_strategy} | raw={raw_dataset_path} | results={results_dir}",
        flush=True,
    )

    parsed_dir = output_dirs["data"] / "parsed"
    parse_raw_dataset(raw_dataset_path, parsed_dir)
    dataset = load_parsed_dataset(parsed_dir)
    verification_rows = validate_parsed_shapes(dataset)

    inputs = dataset.training_inputs.copy()
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

    rate_bundle = _build_task_bundle(
        dataset=dataset,
        inputs=inputs,
        target_name="rate_const",
        trainval_indices=splits.trainval_indices,
        drop_constant_targets=False,
    )
    super_bundle = _build_task_bundle(
        dataset=dataset,
        inputs=inputs,
        target_name="super_rate",
        trainval_indices=splits.trainval_indices,
        drop_constant_targets=True,
    )
    metadata_columns = get_case_metadata_columns(inputs)

    feature_meta = _save_feature_snapshots(
        inputs=inputs,
        split_assignment=split_assignment,
        trainval_indices=splits.trainval_indices,
        feature_sets=feature_sets,
        output_dir=output_dirs["data_snapshots"],
    )
    save_csv(feature_meta, output_dirs["data_snapshots"] / "feature_set_final_metadata.csv")

    rate_meta = _save_target_snapshots(
        bundle=rate_bundle,
        split_assignment=split_assignment,
        trainval_indices=splits.trainval_indices,
        output_dir=output_dirs["data_snapshots"],
    )
    super_meta = _save_target_snapshots(
        bundle=super_bundle,
        split_assignment=split_assignment,
        trainval_indices=splits.trainval_indices,
        output_dir=output_dirs["data_snapshots"],
    )
    multitask_summary = pd.concat([rate_meta, super_meta], ignore_index=True)
    save_csv(multitask_summary, output_dirs["data_snapshots"] / "multitask_target_summary.csv")

    rate_roundtrip = TargetTransformer(rate_bundle.kept_target_columns).fit(
        rate_bundle.active_targets.loc[splits.trainval_indices]
    )
    super_roundtrip = TargetTransformer(super_bundle.kept_target_columns).fit(
        super_bundle.active_targets.loc[splits.trainval_indices]
    )
    rate_inverse = rate_roundtrip.inverse_transform_array(
        rate_roundtrip.transform(rate_bundle.active_targets.loc[splits.trainval_indices]).to_numpy(dtype=float)
    )
    super_inverse = super_roundtrip.inverse_transform_array(
        super_roundtrip.transform(super_bundle.active_targets.loc[splits.trainval_indices]).to_numpy(dtype=float)
    )
    verification_rows.extend(
        [
            {
                "check_name": "rate_const_inverse_transform_roundtrip_trainval",
                "status": "pass"
                if float(np.max(np.abs(
                    rate_inverse
                    - rate_bundle.active_targets.loc[splits.trainval_indices, rate_bundle.kept_target_columns].to_numpy(dtype=float)
                ))) < 1e-10
                else "fail",
                "expected": "< 1e-10",
                "observed": float(np.max(np.abs(
                    rate_inverse
                    - rate_bundle.active_targets.loc[splits.trainval_indices, rate_bundle.kept_target_columns].to_numpy(dtype=float)
                ))),
                "details": "RATE CONST transform + inverse_transform max absolute error on trainval.",
            },
            {
                "check_name": "super_rate_inverse_transform_roundtrip_trainval",
                "status": "pass"
                if float(np.max(np.abs(
                    super_inverse
                    - super_bundle.active_targets.loc[splits.trainval_indices, super_bundle.kept_target_columns].to_numpy(dtype=float)
                ))) < 1e-10
                else "fail",
                "expected": "< 1e-10",
                "observed": float(np.max(np.abs(
                    super_inverse
                    - super_bundle.active_targets.loc[splits.trainval_indices, super_bundle.kept_target_columns].to_numpy(dtype=float)
                ))),
                "details": "SUPER RATE transform + inverse_transform max absolute error on trainval.",
            },
        ]
    )

    trial_rows: list[dict[str, object]] = []
    total_fits = len(configs) * len(splits.validation_folds)
    completed_fits = 0
    search_start = time.perf_counter()
    print(
        "[multitask] tuning start | "
        f"configs={len(configs)} | folds={len(splits.validation_folds)} | expected_fits={total_fits}",
        flush=True,
    )

    for fold in splits.validation_folds:
        fold_start = time.perf_counter()
        train_inputs = inputs.loc[fold.train_indices]
        val_inputs = inputs.loc[fold.val_indices]

        feature_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for feature_set in feature_sets:
            transformer = build_feature_transformer(feature_set).fit(train_inputs)
            feature_matrices[feature_set] = (
                transformer.transform(train_inputs).to_numpy(dtype=float),
                transformer.transform(val_inputs).to_numpy(dtype=float),
            )

        rate_train = rate_bundle.active_targets.loc[fold.train_indices, rate_bundle.kept_target_columns]
        rate_val = rate_bundle.active_targets.loc[fold.val_indices, rate_bundle.kept_target_columns]
        rate_transformer = TargetTransformer(rate_bundle.kept_target_columns).fit(rate_train)
        y_rate_train_log = rate_transformer.transform(rate_train).to_numpy(dtype=float)
        y_rate_val_log = rate_transformer.transform(rate_val).to_numpy(dtype=float)
        y_rate_val_original = rate_val.to_numpy(dtype=float)

        super_train = super_bundle.active_targets.loc[fold.train_indices, super_bundle.kept_target_columns]
        super_val = super_bundle.active_targets.loc[fold.val_indices, super_bundle.kept_target_columns]
        super_transformer = TargetTransformer(super_bundle.kept_target_columns).fit(super_train)
        y_super_train_log = super_transformer.transform(super_train).to_numpy(dtype=float)
        y_super_val_log = super_transformer.transform(super_val).to_numpy(dtype=float)
        y_super_val_original = super_val.to_numpy(dtype=float)

        super_full_transformer = TargetTransformer(super_bundle.full_target_columns).fit(
            super_bundle.full_targets.loc[fold.train_indices, super_bundle.full_target_columns]
        )
        y_super_val_full_original = super_bundle.full_targets.loc[
            fold.val_indices, super_bundle.full_target_columns
        ].to_numpy(dtype=float)

        for config_index, config in enumerate(configs, start=1):
            completed_fits += 1
            row = {
                "fold_id": fold.fold_id,
                "fold_seed": fold.seed,
                "status": "success",
                "error_message": "",
                **config.as_dict(),
                "hyperparameters_json": str(config.hyperparameters),
            }
            fit_start = time.perf_counter()
            try:
                x_train, x_val = feature_matrices[config.feature_set]
                model = build_multitask_model(
                    config=config,
                    rate_dim=y_rate_train_log.shape[1],
                    super_dim=y_super_train_log.shape[1],
                )

                def validation_callback(rate_pred_log: np.ndarray, super_pred_log: np.ndarray) -> float:
                    rate_metrics = compute_overall_metrics(
                        y_true_log=y_rate_val_log,
                        y_pred_log=rate_pred_log,
                        y_true_original=y_rate_val_original,
                        y_pred_original=rate_transformer.inverse_transform_array(rate_pred_log),
                    )
                    super_metrics = compute_overall_metrics(
                        y_true_log=y_super_val_log,
                        y_pred_log=super_pred_log,
                        y_true_original=y_super_val_original,
                        y_pred_original=super_transformer.inverse_transform_array(super_pred_log),
                    )
                    return 0.5 * (
                        rate_metrics["overall_log_rmse"] + super_metrics["overall_log_rmse"]
                    )

                model.fit(
                    x_train=x_train,
                    y_rate_train=y_rate_train_log,
                    y_super_train=y_super_train_log,
                    x_val=x_val,
                    y_rate_val=y_rate_val_log,
                    y_super_val=y_super_val_log,
                    validation_score_callback=validation_callback,
                )
                y_rate_val_pred_log, y_super_val_pred_log = model.predict(x_val)
                y_rate_val_pred_original = rate_transformer.inverse_transform_array(y_rate_val_pred_log)
                y_super_val_pred_original = super_transformer.inverse_transform_array(y_super_val_pred_log)

                y_super_val_full_pred_original = _expand_target_matrix(
                    y_super_val_pred_original,
                    kept_target_columns=super_bundle.kept_target_columns,
                    full_target_columns=super_bundle.full_target_columns,
                )
                y_super_val_full_log = super_full_transformer.transform_array(y_super_val_full_original)
                y_super_val_full_pred_log = super_full_transformer.transform_array(y_super_val_full_pred_original)

                rate_metrics = compute_overall_metrics(
                    y_true_log=y_rate_val_log,
                    y_pred_log=y_rate_val_pred_log,
                    y_true_original=y_rate_val_original,
                    y_pred_original=y_rate_val_pred_original,
                )
                super_active_metrics = compute_overall_metrics(
                    y_true_log=y_super_val_log,
                    y_pred_log=y_super_val_pred_log,
                    y_true_original=y_super_val_original,
                    y_pred_original=y_super_val_pred_original,
                )
                super_full_metrics = compute_overall_metrics(
                    y_true_log=y_super_val_full_log,
                    y_pred_log=y_super_val_full_pred_log,
                    y_true_original=y_super_val_full_original,
                    y_pred_original=y_super_val_full_pred_original,
                )
                row["validation_joint_log_rmse"] = 0.5 * (
                    rate_metrics["overall_log_rmse"] + super_active_metrics["overall_log_rmse"]
                )
                row.update(_prefixed_metric_row("validation_rate", rate_metrics))
                row.update(_prefixed_metric_row("validation_super_active", super_active_metrics))
                row.update(_prefixed_metric_row("validation_super_full", super_full_metrics))
            except Exception as exc:  # noqa: BLE001
                row["status"] = "failed"
                row["error_message"] = f"{type(exc).__name__}: {exc}"
            row["runtime_seconds"] = time.perf_counter() - fit_start
            trial_rows.append(row)

            tuning_elapsed = time.perf_counter() - search_start
            eta_seconds = _estimate_eta_seconds(completed_fits, total_fits, tuning_elapsed)
            metric_text = (
                f"{row['validation_joint_log_rmse']:.6f}"
                if "validation_joint_log_rmse" in row and isinstance(row["validation_joint_log_rmse"], (int, float))
                else "n/a"
            )
            print(
                f"[multitask] fold {fold.fold_id}/{len(splits.validation_folds)} | "
                f"config {config_index}/{len(configs)} | global {completed_fits}/{total_fits} | "
                f"status={row['status']} | joint_val_log_rmse={metric_text} | "
                f"fit={_format_seconds(row['runtime_seconds'])} | "
                f"elapsed={_format_seconds(tuning_elapsed)} | eta={_format_seconds(eta_seconds)} | "
                f"model={config.model_key}",
                flush=True,
            )

        fold_elapsed = time.perf_counter() - fold_start
        print(
            f"[multitask] fold complete | fold={fold.fold_id}/{len(splits.validation_folds)} | "
            f"elapsed={_format_seconds(fold_elapsed)}",
            flush=True,
        )
        save_csv(
            pd.DataFrame(trial_rows),
            output_dirs["tuning"] / "model_trials_foldwise_partial.csv",
        )

    trial_frame = pd.DataFrame(trial_rows)
    save_csv(trial_frame, output_dirs["tuning"] / "model_trials_foldwise.csv")
    summary = _aggregate_trial_summary(trial_frame)
    save_csv(summary, output_dirs["tuning"] / "model_leaderboard_summary.csv")
    selected = _select_best_model(summary)
    save_csv(pd.DataFrame([selected]), output_dirs["tuning"] / "selected_model.csv")
    print(
        "[multitask] model selected | "
        f"model={selected['model_key']} | mean_joint_val_log_rmse={selected['mean_validation_joint_log_rmse']:.6f}",
        flush=True,
    )

    trainval_inputs = inputs.loc[splits.trainval_indices]
    test_inputs = inputs.loc[splits.test_indices]
    rate_trainval = rate_bundle.active_targets.loc[splits.trainval_indices, rate_bundle.kept_target_columns]
    rate_test = rate_bundle.active_targets.loc[splits.test_indices, rate_bundle.kept_target_columns]
    super_trainval = super_bundle.active_targets.loc[splits.trainval_indices, super_bundle.kept_target_columns]
    super_test = super_bundle.active_targets.loc[splits.test_indices, super_bundle.kept_target_columns]

    rate_transformer = TargetTransformer(rate_bundle.kept_target_columns).fit(rate_trainval)
    super_transformer = TargetTransformer(super_bundle.kept_target_columns).fit(super_trainval)
    super_full_transformer = TargetTransformer(super_bundle.full_target_columns).fit(
        super_bundle.full_targets.loc[splits.trainval_indices, super_bundle.full_target_columns]
    )

    y_rate_trainval_log = rate_transformer.transform(rate_trainval).to_numpy(dtype=float)
    y_rate_test_log = rate_transformer.transform(rate_test).to_numpy(dtype=float)
    y_rate_test_original = rate_test.to_numpy(dtype=float)

    y_super_trainval_log = super_transformer.transform(super_trainval).to_numpy(dtype=float)
    y_super_test_log = super_transformer.transform(super_test).to_numpy(dtype=float)
    y_super_test_original = super_test.to_numpy(dtype=float)

    selected_config = next(config for config in configs if config.model_key == selected["model_key"])
    feature_transformer = build_feature_transformer(selected_config.feature_set).fit(trainval_inputs)
    x_trainval = feature_transformer.transform(trainval_inputs).to_numpy(dtype=float)
    x_test = feature_transformer.transform(test_inputs).to_numpy(dtype=float)

    print(
        "[multitask] final fit start | "
        f"model={selected_config.model_key} | trainval_cases={x_trainval.shape[0]} | test_cases={x_test.shape[0]}",
        flush=True,
    )
    final_model = build_multitask_model(
        config=selected_config,
        rate_dim=y_rate_trainval_log.shape[1],
        super_dim=y_super_trainval_log.shape[1],
    )
    final_model.fit(
        x_train=x_trainval,
        y_rate_train=y_rate_trainval_log,
        y_super_train=y_super_trainval_log,
    )
    y_rate_test_pred_log, y_super_test_pred_log = final_model.predict(x_test)
    y_rate_test_pred_original = rate_transformer.inverse_transform_array(y_rate_test_pred_log)
    y_super_test_pred_original = super_transformer.inverse_transform_array(y_super_test_pred_log)

    y_super_test_full_original = super_bundle.full_targets.loc[
        splits.test_indices, super_bundle.full_target_columns
    ].to_numpy(dtype=float)
    y_super_test_full_pred_original = _expand_target_matrix(
        y_super_test_pred_original,
        kept_target_columns=super_bundle.kept_target_columns,
        full_target_columns=super_bundle.full_target_columns,
    )
    y_super_test_full_log = super_full_transformer.transform_array(y_super_test_full_original)
    y_super_test_full_pred_log = super_full_transformer.transform_array(y_super_test_full_pred_original)

    rate_summary = _save_task_evaluation_outputs(
        bundle=rate_bundle,
        case_ids=rate_bundle.active_targets.loc[splits.test_indices, metadata_columns],
        y_active_true_log=y_rate_test_log,
        y_active_pred_log=y_rate_test_pred_log,
        y_active_true_original=y_rate_test_original,
        y_active_pred_original=y_rate_test_pred_original,
        y_full_true_log=y_rate_test_log,
        y_full_pred_log=y_rate_test_pred_log,
        y_full_true_original=y_rate_test_original,
        y_full_pred_original=y_rate_test_pred_original,
        output_dir=output_dirs["evaluation"],
    )
    super_summary = _save_task_evaluation_outputs(
        bundle=super_bundle,
        case_ids=super_bundle.active_targets.loc[splits.test_indices, metadata_columns],
        y_active_true_log=y_super_test_log,
        y_active_pred_log=y_super_test_pred_log,
        y_active_true_original=y_super_test_original,
        y_active_pred_original=y_super_test_pred_original,
        y_full_true_log=y_super_test_full_log,
        y_full_pred_log=y_super_test_full_pred_log,
        y_full_true_original=y_super_test_full_original,
        y_full_pred_original=y_super_test_full_pred_original,
        output_dir=output_dirs["evaluation"],
    )

    overall_summary = pd.DataFrame(
        [
            {
                "selected_model_key": selected["model_key"],
                "selected_model_family": selected["model_family"],
                "selected_feature_set": selected["feature_set"],
                "mean_validation_joint_log_rmse": selected["mean_validation_joint_log_rmse"],
                "rate_const_test_overall_log_rmse": rate_summary["overall_log_rmse"],
                "rate_const_test_overall_log_r2": rate_summary["overall_log_r2"],
                "rate_const_median_absolute_relative_error": rate_summary["median_absolute_relative_error"],
                "rate_const_median_smape": rate_summary["median_smape"],
                "super_rate_test_overall_log_rmse": super_summary["overall_log_rmse"],
                "super_rate_test_overall_log_r2": super_summary["overall_log_r2"],
                "super_rate_median_absolute_relative_error": super_summary["median_absolute_relative_error"],
                "super_rate_median_smape": super_summary["median_smape"],
                "joint_test_log_rmse_mean": 0.5 * (
                    rate_summary["overall_log_rmse"] + super_summary["overall_log_rmse"]
                ),
            }
        ]
    )
    save_csv(overall_summary, output_dirs["evaluation"] / "multitask_test_summary.csv")

    _plot_multitask_leaderboard(summary, output_dirs["figures"] / "multitask_model_leaderboard.png")
    verification_rows.append(
        {
            "check_name": "multitask_results_written",
            "status": "pass",
            "expected": "non-empty task evaluation outputs",
            "observed": int(len(rate_bundle.active_targets.loc[splits.test_indices])),
            "details": "Number of test cases saved in multitask evaluation outputs.",
        }
    )
    save_csv(pd.DataFrame(verification_rows), results_dir / "verification_checks.csv")

    manifest_rows = [
        {
            "category": "tuning",
            "path": str(output_dirs["tuning"] / "selected_model.csv"),
            "description": "Selected multitask model by mean validation joint log RMSE.",
        },
        {
            "category": "evaluation",
            "path": str(output_dirs["evaluation"] / "multitask_test_summary.csv"),
            "description": "Combined test summary for the selected multitask model.",
        },
        {
            "category": "evaluation",
            "path": str(output_dirs["evaluation"] / "rate_const_test_overall_metrics.csv"),
            "description": "RATE CONST test metrics from the multitask model.",
        },
        {
            "category": "evaluation",
            "path": str(output_dirs["evaluation"] / "super_rate_test_overall_metrics.csv"),
            "description": "SUPER RATE test metrics from the multitask model.",
        },
        {
            "category": "figures",
            "path": str(output_dirs["figures"] / "multitask_model_leaderboard.png"),
            "description": "Validation leaderboard for multitask configurations.",
        },
    ]
    save_csv(pd.DataFrame(manifest_rows), results_dir / "output_manifest.csv")

    print(
        "[multitask] done | "
        f"results={results_dir} | total_elapsed={_format_seconds(time.perf_counter() - experiment_start)}",
        flush=True,
    )
    return {
        "results_dir": results_dir,
        "selected_model": output_dirs["tuning"] / "selected_model.csv",
        "summary": output_dirs["evaluation"] / "multitask_test_summary.csv",
    }


def run_multitask_family_finalist_evaluations(
    results_dir: Path,
    configs: list[ModelConfig],
    split_strategy: str,
    holdout_power_labels: list[str] | None = None,
) -> Path:
    """Refit and test the best saved validation config for each multitask model family."""
    branch_root = results_dir / "branch_evaluation"
    branch_root.mkdir(parents=True, exist_ok=True)
    parsed_dir = results_dir / "data" / "parsed"
    dataset = load_parsed_dataset(parsed_dir)
    inputs = dataset.training_inputs.copy()

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

    rate_bundle = _build_task_bundle(
        dataset=dataset,
        inputs=inputs,
        target_name="rate_const",
        trainval_indices=splits.trainval_indices,
        drop_constant_targets=False,
    )
    super_bundle = _build_task_bundle(
        dataset=dataset,
        inputs=inputs,
        target_name="super_rate",
        trainval_indices=splits.trainval_indices,
        drop_constant_targets=True,
    )
    metadata_columns = get_case_metadata_columns(inputs)
    config_by_key = {config.model_key: config for config in configs}

    leaderboard = pd.read_csv(results_dir / "tuning" / "model_leaderboard_summary.csv")
    family_winners = (
        leaderboard.sort_values("mean_validation_joint_log_rmse")
        .groupby("model_family", dropna=False)
        .head(1)
        .sort_values("model_family")
        .reset_index(drop=True)
    )

    trainval_inputs = inputs.loc[splits.trainval_indices]
    test_inputs = inputs.loc[splits.test_indices]
    rate_trainval = rate_bundle.active_targets.loc[splits.trainval_indices, rate_bundle.kept_target_columns]
    rate_test = rate_bundle.active_targets.loc[splits.test_indices, rate_bundle.kept_target_columns]
    super_trainval = super_bundle.active_targets.loc[splits.trainval_indices, super_bundle.kept_target_columns]
    super_test = super_bundle.active_targets.loc[splits.test_indices, super_bundle.kept_target_columns]

    rate_transformer = TargetTransformer(rate_bundle.kept_target_columns).fit(rate_trainval)
    super_transformer = TargetTransformer(super_bundle.kept_target_columns).fit(super_trainval)
    super_full_transformer = TargetTransformer(super_bundle.full_target_columns).fit(
        super_bundle.full_targets.loc[splits.trainval_indices, super_bundle.full_target_columns]
    )

    y_rate_trainval_log = rate_transformer.transform(rate_trainval).to_numpy(dtype=float)
    y_rate_test_log = rate_transformer.transform(rate_test).to_numpy(dtype=float)
    y_rate_test_original = rate_test.to_numpy(dtype=float)

    y_super_trainval_log = super_transformer.transform(super_trainval).to_numpy(dtype=float)
    y_super_test_log = super_transformer.transform(super_test).to_numpy(dtype=float)
    y_super_test_original = super_test.to_numpy(dtype=float)
    y_super_test_full_original = super_bundle.full_targets.loc[
        splits.test_indices, super_bundle.full_target_columns
    ].to_numpy(dtype=float)
    y_super_test_full_log = super_full_transformer.transform_array(y_super_test_full_original)

    branch_rows = []
    for _, winner in family_winners.iterrows():
        model_key = str(winner["model_key"])
        if model_key not in config_by_key:
            raise KeyError(f"Saved model key is not present in provided configs: {model_key}")
        config = config_by_key[model_key]
        branch_dir = branch_root / str(winner["model_family"])
        evaluation_dir = branch_dir / "evaluation"
        figures_dir = branch_dir / "figures"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_csv(pd.DataFrame([winner]), branch_dir / "selected_model.csv")

        print(
            "[multitask-finalist] final fit start | "
            f"family={winner['model_family']} | model={model_key} | results={branch_dir}",
            flush=True,
        )
        feature_transformer = build_feature_transformer(config.feature_set).fit(trainval_inputs)
        x_trainval = feature_transformer.transform(trainval_inputs).to_numpy(dtype=float)
        x_test = feature_transformer.transform(test_inputs).to_numpy(dtype=float)
        model = build_multitask_model(
            config=config,
            rate_dim=y_rate_trainval_log.shape[1],
            super_dim=y_super_trainval_log.shape[1],
        )
        fit_start = time.perf_counter()
        model.fit(
            x_train=x_trainval,
            y_rate_train=y_rate_trainval_log,
            y_super_train=y_super_trainval_log,
        )
        y_rate_test_pred_log, y_super_test_pred_log = model.predict(x_test)
        y_rate_test_pred_original = rate_transformer.inverse_transform_array(y_rate_test_pred_log)
        y_super_test_pred_original = super_transformer.inverse_transform_array(y_super_test_pred_log)
        y_super_test_full_pred_original = _expand_target_matrix(
            y_super_test_pred_original,
            kept_target_columns=super_bundle.kept_target_columns,
            full_target_columns=super_bundle.full_target_columns,
        )
        y_super_test_full_pred_log = super_full_transformer.transform_array(y_super_test_full_pred_original)

        rate_summary = _save_task_evaluation_outputs(
            bundle=rate_bundle,
            case_ids=rate_bundle.active_targets.loc[splits.test_indices, metadata_columns],
            y_active_true_log=y_rate_test_log,
            y_active_pred_log=y_rate_test_pred_log,
            y_active_true_original=y_rate_test_original,
            y_active_pred_original=y_rate_test_pred_original,
            y_full_true_log=y_rate_test_log,
            y_full_pred_log=y_rate_test_pred_log,
            y_full_true_original=y_rate_test_original,
            y_full_pred_original=y_rate_test_pred_original,
            output_dir=evaluation_dir,
        )
        super_summary = _save_task_evaluation_outputs(
            bundle=super_bundle,
            case_ids=super_bundle.active_targets.loc[splits.test_indices, metadata_columns],
            y_active_true_log=y_super_test_log,
            y_active_pred_log=y_super_test_pred_log,
            y_active_true_original=y_super_test_original,
            y_active_pred_original=y_super_test_pred_original,
            y_full_true_log=y_super_test_full_log,
            y_full_pred_log=y_super_test_full_pred_log,
            y_full_true_original=y_super_test_full_original,
            y_full_pred_original=y_super_test_full_pred_original,
            output_dir=evaluation_dir,
        )
        fit_elapsed = time.perf_counter() - fit_start
        branch_rows.append(
            {
                "model_family": winner["model_family"],
                "model_key": model_key,
                "feature_set": config.feature_set,
                "mean_validation_joint_log_rmse": winner["mean_validation_joint_log_rmse"],
                "rate_const_test_overall_log_rmse": rate_summary["overall_log_rmse"],
                "rate_const_test_overall_log_r2": rate_summary["overall_log_r2"],
                "super_rate_test_overall_log_rmse": super_summary["overall_log_rmse"],
                "super_rate_test_overall_log_r2": super_summary["overall_log_r2"],
                "joint_test_log_rmse_mean": 0.5
                * (rate_summary["overall_log_rmse"] + super_summary["overall_log_rmse"]),
                "runtime_seconds": fit_elapsed,
                "branch_results_dir": str(branch_dir),
            }
        )
        print(
            "[multitask-finalist] final fit complete | "
            f"family={winner['model_family']} | elapsed={_format_seconds(fit_elapsed)} | "
            f"rate_log_rmse={rate_summary['overall_log_rmse']:.6f} | "
            f"super_log_rmse={super_summary['overall_log_rmse']:.6f}",
            flush=True,
        )

    summary = pd.DataFrame(branch_rows).sort_values("joint_test_log_rmse_mean").reset_index(drop=True)
    save_csv(summary, branch_root / "branch_test_summary.csv")
    return branch_root / "branch_test_summary.csv"
