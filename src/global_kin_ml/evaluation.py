from __future__ import annotations

import math
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.var(y_true) < 1e-18:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _factor_accuracy(y_true: np.ndarray, y_pred: np.ndarray, factor: float) -> tuple[float, int]:
    mask = y_true > 0.0
    if not np.any(mask):
        return float("nan"), 0
    true_values = y_true[mask]
    pred_values = np.maximum(y_pred[mask], 1e-300)
    ratios = np.maximum(pred_values / true_values, true_values / pred_values)
    return float(np.mean(ratios <= factor)), int(mask.sum())


def compute_overall_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
) -> dict[str, float]:
    factor2, positive_count = _factor_accuracy(y_true_original.ravel(), y_pred_original.ravel(), 2.0)
    factor5, _ = _factor_accuracy(y_true_original.ravel(), y_pred_original.ravel(), 5.0)
    factor10, _ = _factor_accuracy(y_true_original.ravel(), y_pred_original.ravel(), 10.0)
    return {
        "overall_log_rmse": float(math.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "overall_log_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "overall_log_r2": float(r2_score(y_true_log, y_pred_log, multioutput="uniform_average")),
        "overall_original_rmse": float(math.sqrt(mean_squared_error(y_true_original, y_pred_original))),
        "overall_original_mae": float(mean_absolute_error(y_true_original, y_pred_original)),
        "factor2_accuracy_positive_only": factor2,
        "factor5_accuracy_positive_only": factor5,
        "factor10_accuracy_positive_only": factor10,
        "positive_factor_denominator": positive_count,
    }


def compute_per_reaction_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
    reaction_map: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for column_index, reaction in reaction_map.iterrows():
        true_log = y_true_log[:, column_index]
        pred_log = y_pred_log[:, column_index]
        true_original = y_true_original[:, column_index]
        pred_original = y_pred_original[:, column_index]
        factor2, positive_count = _factor_accuracy(true_original, pred_original, 2.0)
        factor5, _ = _factor_accuracy(true_original, pred_original, 5.0)
        factor10, _ = _factor_accuracy(true_original, pred_original, 10.0)
        rows.append(
            {
                "reaction_id": reaction["reaction_id"],
                "reaction_label": reaction["reaction_label"],
                "target_column": reaction["rate_const_column"],
                "log_rmse": float(math.sqrt(mean_squared_error(true_log, pred_log))),
                "log_mae": float(mean_absolute_error(true_log, pred_log)),
                "log_r2": _safe_r2(true_log, pred_log),
                "original_rmse": float(math.sqrt(mean_squared_error(true_original, pred_original))),
                "original_mae": float(mean_absolute_error(true_original, pred_original)),
                "factor2_accuracy_positive_only": factor2,
                "factor5_accuracy_positive_only": factor5,
                "factor10_accuracy_positive_only": factor10,
                "positive_factor_denominator": positive_count,
            }
        )
    return pd.DataFrame(rows)


def compute_per_case_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
    case_ids: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row_index, (_, case_row) in enumerate(case_ids.iterrows()):
        true_log = y_true_log[row_index, :]
        pred_log = y_pred_log[row_index, :]
        true_original = y_true_original[row_index, :]
        pred_original = y_pred_original[row_index, :]
        factor2, positive_count = _factor_accuracy(true_original, pred_original, 2.0)
        factor5, _ = _factor_accuracy(true_original, pred_original, 5.0)
        factor10, _ = _factor_accuracy(true_original, pred_original, 10.0)
        row = dict(case_row.to_dict())
        row.update(
            {
                "log_rmse": float(math.sqrt(mean_squared_error(true_log, pred_log))),
                "log_mae": float(mean_absolute_error(true_log, pred_log)),
                "original_rmse": float(math.sqrt(mean_squared_error(true_original, pred_original))),
                "original_mae": float(mean_absolute_error(true_original, pred_original)),
                "factor2_accuracy_positive_only": factor2,
                "factor5_accuracy_positive_only": factor5,
                "factor10_accuracy_positive_only": factor10,
                "positive_factor_denominator": positive_count,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_reconstruction(y_train_log: np.ndarray, y_eval_log: np.ndarray, latent_k: int) -> tuple[np.ndarray, PCA]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(n_components=latent_k, random_state=42, svd_solver="full")
        encoded = pca.fit_transform(y_train_log)
        _ = encoded
        reconstructed = pca.inverse_transform(pca.transform(y_eval_log))
    return reconstructed, pca


def pca_explained_variance_frame(y_train_log: np.ndarray) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA(random_state=42, svd_solver="full")
        pca.fit(y_train_log)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return pd.DataFrame(
        {
            "component_index": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )


def build_prediction_frame(
    case_ids: pd.DataFrame,
    reaction_map: pd.DataFrame,
    y_true_original: np.ndarray,
    y_pred_original: np.ndarray,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for case_position, (_, case_row) in enumerate(case_ids.iterrows()):
        for reaction_index, reaction in reaction_map.iterrows():
            row = dict(case_row.to_dict())
            row.update(
                {
                    "reaction_id": reaction["reaction_id"],
                    "reaction_label": reaction["reaction_label"],
                    "target_column": reaction["rate_const_column"],
                    "true_rate_const": y_true_original[case_position, reaction_index],
                    "predicted_rate_const": y_pred_original[case_position, reaction_index],
                    "absolute_error": abs(
                        y_pred_original[case_position, reaction_index]
                        - y_true_original[case_position, reaction_index]
                    ),
                    "true_log10_rate": y_true_log[case_position, reaction_index],
                    "predicted_log10_rate": y_pred_log[case_position, reaction_index],
                    "absolute_log_error": abs(
                        y_pred_log[case_position, reaction_index]
                        - y_true_log[case_position, reaction_index]
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_relative_error_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    relative_frame = prediction_frame.copy()
    relative_frame["relative_error_defined"] = relative_frame["true_rate_const"] > 0.0
    relative_frame["relative_error_signed"] = np.where(
        relative_frame["relative_error_defined"],
        (
            relative_frame["predicted_rate_const"] - relative_frame["true_rate_const"]
        )
        / relative_frame["true_rate_const"],
        np.nan,
    )
    relative_frame["relative_error_abs"] = relative_frame["relative_error_signed"].abs()
    return relative_frame


def build_smape_frame(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    smape_frame = prediction_frame.copy()
    denominator = smape_frame["true_rate_const"].abs() + smape_frame["predicted_rate_const"].abs()
    smape_frame["smape_defined"] = denominator > 0.0
    smape_frame["smape"] = np.where(
        smape_frame["smape_defined"],
        2.0 * smape_frame["absolute_error"] / denominator,
        0.0,
    )
    return smape_frame


def _relative_error_summary(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    if clean.empty:
        return {
            "mean_signed_relative_error": float("nan"),
            "median_signed_relative_error": float("nan"),
            "mean_absolute_relative_error": float("nan"),
            "median_absolute_relative_error": float("nan"),
            "p75_absolute_relative_error": float("nan"),
            "p90_absolute_relative_error": float("nan"),
            "p95_absolute_relative_error": float("nan"),
            "p99_absolute_relative_error": float("nan"),
            "within_1pct": float("nan"),
            "within_5pct": float("nan"),
            "within_10pct": float("nan"),
            "within_20pct": float("nan"),
            "within_50pct": float("nan"),
            "within_100pct": float("nan"),
            "max_absolute_relative_error": float("nan"),
        }
    absolute = clean.abs()
    return {
        "mean_signed_relative_error": float(clean.mean()),
        "median_signed_relative_error": float(clean.median()),
        "mean_absolute_relative_error": float(absolute.mean()),
        "median_absolute_relative_error": float(absolute.median()),
        "p75_absolute_relative_error": float(absolute.quantile(0.75)),
        "p90_absolute_relative_error": float(absolute.quantile(0.90)),
        "p95_absolute_relative_error": float(absolute.quantile(0.95)),
        "p99_absolute_relative_error": float(absolute.quantile(0.99)),
        "within_1pct": float((absolute <= 0.01).mean()),
        "within_5pct": float((absolute <= 0.05).mean()),
        "within_10pct": float((absolute <= 0.10).mean()),
        "within_20pct": float((absolute <= 0.20).mean()),
        "within_50pct": float((absolute <= 0.50).mean()),
        "within_100pct": float((absolute <= 1.00).mean()),
        "max_absolute_relative_error": float(absolute.max()),
    }


def _smape_summary(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    if clean.empty:
        return {
            "mean_smape": float("nan"),
            "median_smape": float("nan"),
            "p75_smape": float("nan"),
            "p90_smape": float("nan"),
            "p95_smape": float("nan"),
            "p99_smape": float("nan"),
            "within_1pct_smape": float("nan"),
            "within_5pct_smape": float("nan"),
            "within_10pct_smape": float("nan"),
            "within_20pct_smape": float("nan"),
            "within_50pct_smape": float("nan"),
            "within_100pct_smape": float("nan"),
            "max_smape": float("nan"),
        }
    return {
        "mean_smape": float(clean.mean()),
        "median_smape": float(clean.median()),
        "p75_smape": float(clean.quantile(0.75)),
        "p90_smape": float(clean.quantile(0.90)),
        "p95_smape": float(clean.quantile(0.95)),
        "p99_smape": float(clean.quantile(0.99)),
        "within_1pct_smape": float((clean <= 0.01).mean()),
        "within_5pct_smape": float((clean <= 0.05).mean()),
        "within_10pct_smape": float((clean <= 0.10).mean()),
        "within_20pct_smape": float((clean <= 0.20).mean()),
        "within_50pct_smape": float((clean <= 0.50).mean()),
        "within_100pct_smape": float((clean <= 1.00).mean()),
        "max_smape": float(clean.max()),
    }


def compute_relative_error_outputs(
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    relative_frame = build_relative_error_frame(prediction_frame)
    defined = relative_frame[relative_frame["relative_error_defined"]].copy()

    overall_row = {
        "total_prediction_count": int(len(relative_frame)),
        "positive_groundtruth_count": int(len(defined)),
        "zero_groundtruth_count": int((~relative_frame["relative_error_defined"]).sum()),
        "relative_error_definition": "(predicted_rate_const - true_rate_const) / true_rate_const for true_rate_const > 0",
        "zero_groundtruth_policy": "relative error undefined when true_rate_const == 0",
        **_relative_error_summary(defined["relative_error_signed"]),
    }
    overall = pd.DataFrame([overall_row])

    per_reaction = []
    for (reaction_id, reaction_label, target_column), group in relative_frame.groupby(
        ["reaction_id", "reaction_label", "target_column"], dropna=False
    ):
        defined_group = group[group["relative_error_defined"]]
        row = {
            "reaction_id": reaction_id,
            "reaction_label": reaction_label,
            "target_column": target_column,
            "total_prediction_count": int(len(group)),
            "positive_groundtruth_count": int(len(defined_group)),
            "zero_groundtruth_count": int((~group["relative_error_defined"]).sum()),
        }
        row.update(_relative_error_summary(defined_group["relative_error_signed"]))
        per_reaction.append(row)
    per_reaction_frame = pd.DataFrame(per_reaction)

    per_case = []
    for (global_case_id, density_group_id, local_case_id), group in relative_frame.groupby(
        ["global_case_id", "density_group_id", "local_case_id"], dropna=False
    ):
        defined_group = group[group["relative_error_defined"]]
        row = {
            "global_case_id": global_case_id,
            "density_group_id": density_group_id,
            "local_case_id": local_case_id,
            "total_prediction_count": int(len(group)),
            "positive_groundtruth_count": int(len(defined_group)),
            "zero_groundtruth_count": int((~group["relative_error_defined"]).sum()),
        }
        row.update(_relative_error_summary(defined_group["relative_error_signed"]))
        per_case.append(row)
    per_case_frame = pd.DataFrame(per_case)

    magnitude_bins = [
        (1e-20, 1e-18),
        (1e-18, 1e-16),
        (1e-16, 1e-14),
        (1e-14, 1e-12),
        (1e-12, 1e-10),
        (1e-10, 1e-8),
        (1e-8, 1e-6),
        (1e-6, 1e-4),
        (1e-4, 1e-2),
        (1e-2, 1e0),
    ]
    magnitude_rows = []
    for lower, upper in magnitude_bins:
        subset = defined[
            (defined["true_rate_const"] >= lower) & (defined["true_rate_const"] < upper)
        ]
        if subset.empty:
            continue
        row = {
            "true_rate_range": f"[{lower:.0e}, {upper:.0e})",
            "lower_bound": lower,
            "upper_bound_exclusive": upper,
            "positive_groundtruth_count": int(len(subset)),
        }
        row.update(_relative_error_summary(subset["relative_error_signed"]))
        magnitude_rows.append(row)
    magnitude_frame = pd.DataFrame(magnitude_rows)
    return overall, per_reaction_frame, per_case_frame, magnitude_frame


def compute_smape_outputs(
    prediction_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    smape_frame = build_smape_frame(prediction_frame)

    overall_row = {
        "total_prediction_count": int(len(smape_frame)),
        "smape_definition": "2 * abs(predicted_rate_const - true_rate_const) / (abs(predicted_rate_const) + abs(true_rate_const))",
        "smape_range": "[0, 2], where 0 is perfect and 2 is the maximum disagreement",
        **_smape_summary(smape_frame["smape"]),
    }
    overall = pd.DataFrame([overall_row])

    per_reaction = []
    for (reaction_id, reaction_label, target_column), group in smape_frame.groupby(
        ["reaction_id", "reaction_label", "target_column"], dropna=False
    ):
        row = {
            "reaction_id": reaction_id,
            "reaction_label": reaction_label,
            "target_column": target_column,
            "prediction_count": int(len(group)),
        }
        row.update(_smape_summary(group["smape"]))
        per_reaction.append(row)
    per_reaction_frame = pd.DataFrame(per_reaction)

    per_case = []
    for (global_case_id, density_group_id, local_case_id), group in smape_frame.groupby(
        ["global_case_id", "density_group_id", "local_case_id"], dropna=False
    ):
        row = {
            "global_case_id": global_case_id,
            "density_group_id": density_group_id,
            "local_case_id": local_case_id,
            "prediction_count": int(len(group)),
        }
        row.update(_smape_summary(group["smape"]))
        per_case.append(row)
    per_case_frame = pd.DataFrame(per_case)

    magnitude_bins = [
        (0.0, 1e-20),
        (1e-20, 1e-18),
        (1e-18, 1e-16),
        (1e-16, 1e-14),
        (1e-14, 1e-12),
        (1e-12, 1e-10),
        (1e-10, 1e-8),
        (1e-8, 1e-6),
        (1e-6, 1e-4),
        (1e-4, 1e-2),
    ]
    magnitude_rows = []
    for lower, upper in magnitude_bins:
        subset = smape_frame[
            (smape_frame["true_rate_const"] >= lower)
            & (smape_frame["true_rate_const"] < upper)
        ]
        if subset.empty:
            continue
        row = {
            "true_rate_range": f"[{lower:.0e}, {upper:.0e})",
            "lower_bound": lower,
            "upper_bound_exclusive": upper,
            "prediction_count": int(len(subset)),
        }
        row.update(_smape_summary(subset["smape"]))
        magnitude_rows.append(row)
    magnitude_frame = pd.DataFrame(magnitude_rows)
    return overall, per_reaction_frame, per_case_frame, magnitude_frame


def plot_pca_scree(explained_variance: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        explained_variance["component_index"],
        explained_variance["cumulative_explained_variance_ratio"],
        marker="o",
    )
    ax.set_xlabel("PCA Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Target PCA Scree")
    ax.grid(True, alpha=0.25)
    save_figure(fig, path)


def plot_oracle_error(oracle_overall: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(oracle_overall["latent_k"], oracle_overall["overall_log_rmse"], marker="o", label="Test log RMSE")
    ax.set_xlabel("Latent PCA Components")
    ax.set_ylabel("Oracle Reconstruction Log RMSE")
    ax.set_title("Oracle PCA Reconstruction Error")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, path)


def plot_model_leaderboard(leaderboard: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    top = leaderboard.nsmallest(top_n, "mean_validation_log_rmse").copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top, x="mean_validation_log_rmse", y="model_key", hue="model_family", dodge=False, ax=ax)
    ax.set_xlabel("Mean Validation Reconstructed Log RMSE")
    ax.set_ylabel("Model")
    ax.set_title(f"Top {top_n} Validation Configurations")
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_parity(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str, x_label: str, y_label: str) -> None:
    flat_true = y_true.ravel()
    flat_pred = y_pred.ravel()
    if flat_true.size > 6000:
        sample = np.linspace(0, flat_true.size - 1, 6000, dtype=int)
        flat_true = flat_true[sample]
        flat_pred = flat_pred[sample]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(flat_true, flat_pred, s=10, alpha=0.35)
    lower = float(min(flat_true.min(), flat_pred.min()))
    upper = float(max(flat_true.max(), flat_pred.max()))
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    save_figure(fig, path)


def plot_log_residual_histogram(prediction_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(prediction_frame["absolute_log_error"], bins=40, color="#2f6db2", alpha=0.85)
    ax.set_xlabel("Absolute Log10 Error")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Absolute Log-Space Errors")
    save_figure(fig, path)


def plot_worst_reactions(per_reaction_metrics: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    top = per_reaction_metrics.nlargest(top_n, "log_rmse").copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=top, x="log_rmse", y="reaction_label", color="#c05746", ax=ax)
    ax.set_xlabel("Log RMSE")
    ax.set_ylabel("Reaction")
    ax.set_title(f"Worst {top_n} Reactions by Log RMSE")
    save_figure(fig, path)


def plot_case_error_distribution(per_case_metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_case_metrics["log_rmse"], bins=20, color="#4b9c64", alpha=0.85)
    ax.set_xlabel("Per-Case Log RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Test-Case Log RMSE Distribution")
    save_figure(fig, path)


def plot_relative_error_histogram(relative_overall: pd.DataFrame, relative_frame: pd.DataFrame, path: Path) -> None:
    defined = relative_frame[relative_frame["relative_error_defined"]].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(defined["relative_error_abs"], bins=40, range=(0.0, 1.0), color="#a04e8a", alpha=0.85)
    ax.set_xlabel("Absolute Relative Error")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Relative Error Distribution (Clipped to 100%)")
    summary = relative_overall.iloc[0]
    ax.text(
        0.98,
        0.98,
        f"Median: {summary['median_absolute_relative_error']:.3e}\nP95: {summary['p95_absolute_relative_error']:.3e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    save_figure(fig, path)


def plot_relative_error_by_magnitude(magnitude_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        magnitude_frame["true_rate_range"],
        magnitude_frame["median_absolute_relative_error"],
        marker="o",
        label="Median abs relative error",
    )
    ax.plot(
        magnitude_frame["true_rate_range"],
        magnitude_frame["p95_absolute_relative_error"],
        marker="o",
        label="P95 abs relative error",
    )
    ax.set_xlabel("True Rate Magnitude Bin")
    ax.set_ylabel("Absolute Relative Error")
    ax.set_title("Relative Error by True Rate Magnitude")
    ax.tick_params(axis="x", rotation=40)
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_figure(fig, path)


def plot_smape_histogram(smape_overall: pd.DataFrame, smape_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(smape_frame["smape"], bins=40, range=(0.0, 1.0), color="#d17a22", alpha=0.85)
    ax.set_xlabel("SMAPE")
    ax.set_ylabel("Count")
    ax.set_title("SMAPE Distribution (Clipped to 100%)")
    summary = smape_overall.iloc[0]
    ax.text(
        0.98,
        0.98,
        f"Median: {summary['median_smape']:.3e}\nP95: {summary['p95_smape']:.3e}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    save_figure(fig, path)


def plot_smape_by_magnitude(magnitude_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        magnitude_frame["true_rate_range"],
        magnitude_frame["median_smape"],
        marker="o",
        label="Median SMAPE",
    )
    ax.plot(
        magnitude_frame["true_rate_range"],
        magnitude_frame["p95_smape"],
        marker="o",
        label="P95 SMAPE",
    )
    ax.set_xlabel("True Rate Magnitude Bin")
    ax.set_ylabel("SMAPE")
    ax.set_title("SMAPE by True Rate Magnitude")
    ax.tick_params(axis="x", rotation=40)
    ax.legend()
    ax.grid(True, alpha=0.25)
    save_figure(fig, path)
