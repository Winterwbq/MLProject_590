from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
OUTPUT_MANIFEST_COLUMNS = ["category", "method", "path", "description"]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_directory(path.parent)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def safe_skew(values: np.ndarray) -> float:
    if np.allclose(values, values[0]):
        return 0.0
    return float(skew(values, bias=False))


def safe_kurtosis(values: np.ndarray) -> float:
    if np.allclose(values, values[0]):
        return 0.0
    return float(kurtosis(values, fisher=True, bias=False))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_value = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0, 1.0
    return float(rho), float(p_value)


def top_abs_corr_pairs(df: pd.DataFrame, label_map: dict[str, str], top_n: int) -> pd.DataFrame:
    corr = df.corr()
    rows = []
    columns = list(corr.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a = columns[i]
            col_b = columns[j]
            value = corr.iloc[i, j]
            if np.isnan(value):
                continue
            rows.append(
                {
                    "feature_a": col_a,
                    "label_a": label_map.get(col_a, col_a),
                    "feature_b": col_b,
                    "label_b": label_map.get(col_b, col_b),
                    "correlation": float(value),
                    "abs_correlation": abs(float(value)),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values("abs_correlation", ascending=False).head(top_n)


def compute_variance_decomposition(matrix: np.ndarray) -> dict[str, float]:
    grand_mean = matrix.mean()
    group_means = matrix.mean(axis=1)
    local_means = matrix.mean(axis=0)
    ss_total = float(np.sum((matrix - grand_mean) ** 2))
    if ss_total == 0.0:
        return {
            "total_variance": 0.0,
            "group_share": 0.0,
            "local_case_share": 0.0,
            "interaction_share": 0.0,
        }
    ss_group = float(matrix.shape[1] * np.sum((group_means - grand_mean) ** 2))
    ss_local = float(matrix.shape[0] * np.sum((local_means - grand_mean) ** 2))
    ss_interaction = float(
        np.sum(
            (
                matrix
                - group_means[:, None]
                - local_means[None, :]
                + grand_mean
            )
            ** 2
        )
    )
    return {
        "total_variance": ss_total,
        "group_share": ss_group / ss_total,
        "local_case_share": ss_local / ss_total,
        "interaction_share": ss_interaction / ss_total,
    }


def choose_best_k(data: np.ndarray, k_values: list[int]) -> tuple[int, pd.DataFrame]:
    rows = []
    best_k = k_values[0]
    best_score = -1.0
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(data)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(data, labels))
        rows.append({"k": k, "silhouette_score": score})
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, pd.DataFrame(rows)


def effective_rank(explained_variance_ratio: np.ndarray) -> float:
    positive = explained_variance_ratio[explained_variance_ratio > 0]
    if len(positive) == 0:
        return 0.0
    entropy = -np.sum(positive * np.log(positive))
    return float(np.exp(entropy))


def components_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold) + 1)


def evaluate_model(model, x_train, x_test, y_train, y_test) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    overall_r2 = float(r2_score(y_test, predictions, multioutput="uniform_average"))
    flat_rmse = float(math.sqrt(mean_squared_error(y_test, predictions)))
    flat_mae = float(mean_absolute_error(y_test, predictions))

    per_target_r2 = []
    for index in range(y_test.shape[1]):
        y_true = y_test[:, index]
        if np.var(y_true) < 1e-18:
            continue
        per_target_r2.append(float(r2_score(y_true, predictions[:, index])))
    if per_target_r2:
        mean_target_r2 = float(np.mean(per_target_r2))
        median_target_r2 = float(np.median(per_target_r2))
        negative_target_fraction = float(np.mean(np.array(per_target_r2) < 0))
    else:
        mean_target_r2 = float("nan")
        median_target_r2 = float("nan")
        negative_target_fraction = float("nan")

    return {
        "overall_r2": overall_r2,
        "mean_target_r2": mean_target_r2,
        "median_target_r2": median_target_r2,
        "flat_rmse": flat_rmse,
        "flat_mae": flat_mae,
        "evaluated_target_count": len(per_target_r2),
        "negative_target_fraction": negative_target_fraction,
    }


def build_feature_sets(
    x_inputs: pd.DataFrame,
    nonconstant_input_cols: list[str],
    log_e_over_n: np.ndarray,
) -> dict[str, pd.DataFrame]:
    top_varying_cols = (
        x_inputs[nonconstant_input_cols]
        .var()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    composition_nonconstant = x_inputs[nonconstant_input_cols].copy()
    scaler = StandardScaler()
    composition_scaled = scaler.fit_transform(composition_nonconstant)
    pca_components = min(5, composition_scaled.shape[1], composition_scaled.shape[0])
    composition_pca = PCA(n_components=pca_components, random_state=RANDOM_STATE).fit_transform(
        composition_scaled
    )
    composition_pca_df = pd.DataFrame(
        composition_pca,
        columns=[f"composition_pc_{idx:02d}" for idx in range(1, pca_components + 1)],
        index=x_inputs.index,
    )

    minor_cols = [col for col in nonconstant_input_cols if col not in top_varying_cols]
    major_species_df = x_inputs[top_varying_cols].copy()
    major_species_df["minor_species_sum"] = (
        x_inputs[minor_cols].sum(axis=1) if minor_cols else 0.0
    )

    feature_sets = {
        "full_nonconstant_plus_raw_en": pd.concat(
            [x_inputs[["e_over_n_v_cm2"]], composition_nonconstant], axis=1
        ),
        "full_nonconstant_plus_log_en": pd.concat(
            [
                pd.DataFrame({"log10_e_over_n": log_e_over_n}, index=x_inputs.index),
                composition_nonconstant,
            ],
            axis=1,
        ),
        "major_species_plus_log_en": pd.concat(
            [
                pd.DataFrame({"log10_e_over_n": log_e_over_n}, index=x_inputs.index),
                major_species_df,
            ],
            axis=1,
        ),
        "composition_pca5_plus_log_en": pd.concat(
            [
                pd.DataFrame({"log10_e_over_n": log_e_over_n}, index=x_inputs.index),
                composition_pca_df,
            ],
            axis=1,
        ),
    }
    return feature_sets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run advanced diagnostics over the parsed global_kin dataset."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("outputs/parsed"),
        help="Directory containing parsed CSV outputs.",
    )
    parser.add_argument(
        "--basic-analysis-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Directory containing first-stage analysis CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/advanced_analysis"),
        help="Directory for advanced CSV outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/advanced_figures"),
        help="Directory for advanced figures.",
    )
    args = parser.parse_args()

    ensure_directory(args.output_dir)
    ensure_directory(args.figure_dir)

    sns.set_theme(style="whitegrid", context="talk")

    training_inputs = pd.read_csv(args.parsed_dir / "training_inputs.csv")
    training_targets = pd.read_csv(args.parsed_dir / "training_targets.csv")
    case_features = pd.read_csv(args.parsed_dir / "case_features.csv")
    reaction_map = pd.read_csv(args.parsed_dir / "reaction_map.csv")
    species_map = pd.read_csv(args.parsed_dir / "species_map.csv")
    input_species_summary = pd.read_csv(args.basic_analysis_dir / "input_species_summary.csv")
    rate_reaction_summary = pd.read_csv(args.basic_analysis_dir / "rate_reaction_summary.csv")

    manifest_rows: list[dict[str, str]] = []

    def register_output(category: str, method: str, path: Path, description: str) -> None:
        manifest_rows.append(
            {
                "category": category,
                "method": method,
                "path": str(path),
                "description": description,
            }
        )

    id_cols = ["global_case_id", "density_group_id", "local_case_id"]
    input_cols = [col for col in training_inputs.columns if col.startswith("input_")]
    target_cols = [col for col in training_targets.columns if col.startswith("rate_const_")]

    species_label_by_column = dict(
        zip(species_map["species_column"], species_map["species_label"])
    )
    reaction_label_by_column = dict(
        zip(reaction_map["rate_const_column"], reaction_map["reaction_label"])
    )
    reaction_id_by_column = dict(
        zip(reaction_map["rate_const_column"], reaction_map["reaction_id"])
    )

    nonconstant_input_cols = [
        col for col in input_cols if training_inputs[col].nunique() > 1
    ]
    constant_input_cols = [col for col in input_cols if col not in nonconstant_input_cols]

    min_positive_target = float(training_targets[target_cols].replace(0, np.nan).min().min())
    target_eps = min_positive_target / 10.0
    y_raw = training_targets[target_cols].to_numpy(dtype=float)
    y_log = np.log10(y_raw + target_eps)
    y_asinh = np.arcsinh(y_raw / target_eps)

    x_model = training_inputs[["e_over_n_v_cm2"] + nonconstant_input_cols].copy()
    log_e_over_n = np.log10(training_inputs["e_over_n_v_cm2"].to_numpy(dtype=float))
    x_model["log10_e_over_n"] = log_e_over_n
    x_model = x_model.drop(columns=["e_over_n_v_cm2"])

    group_compositions = (
        training_inputs.groupby("density_group_id")[nonconstant_input_cols].first().sort_index()
    )
    group_compositions_scaled = StandardScaler().fit_transform(group_compositions)

    local_case_values = np.sort(training_inputs["local_case_id"].unique())
    density_group_values = np.sort(training_inputs["density_group_id"].unique())
    log_e_unique = np.log10(
        training_inputs.groupby("local_case_id")["e_over_n_v_cm2"].first().sort_index().to_numpy()
    )

    # Method 1: correlation and redundancy analysis
    input_feature_metadata_rows = []
    for col in input_cols:
        series = training_inputs[col]
        input_feature_metadata_rows.append(
            {
                "feature": col,
                "species_label": species_label_by_column.get(col, col),
                "is_constant": bool(col in constant_input_cols),
                "nonzero_fraction": float((series != 0).mean()),
                "mean_value": float(series.mean()),
                "std_value": float(series.std(ddof=0)),
                "variance_value": float(series.var(ddof=0)),
                "n_unique": int(series.nunique()),
            }
        )
    input_feature_metadata = pd.DataFrame(input_feature_metadata_rows).sort_values(
        ["is_constant", "variance_value", "mean_value"],
        ascending=[True, False, False],
    )
    input_feature_metadata_path = args.output_dir / "input_feature_metadata.csv"
    save_csv(input_feature_metadata, input_feature_metadata_path)
    register_output(
        "csv",
        "correlation_redundancy",
        input_feature_metadata_path,
        "Feature-level metadata for all 49 input species, including constant-feature flags.",
    )

    input_corr_top_pairs = top_abs_corr_pairs(
        group_compositions,
        species_label_by_column,
        top_n=200,
    )
    input_corr_top_pairs_path = args.output_dir / "input_correlation_top_pairs.csv"
    save_csv(input_corr_top_pairs, input_corr_top_pairs_path)
    register_output(
        "csv",
        "correlation_redundancy",
        input_corr_top_pairs_path,
        "Strongest absolute pairwise correlations among non-constant composition features at the density-group level.",
    )

    reaction_variance = pd.Series(y_log.var(axis=0), index=target_cols).sort_values(ascending=False)
    top_reaction_corr_cols = reaction_variance.head(30).index.tolist()
    reaction_corr_top_pairs = top_abs_corr_pairs(
        pd.DataFrame(y_log, columns=target_cols)[top_reaction_corr_cols],
        reaction_label_by_column,
        top_n=300,
    )
    reaction_corr_top_pairs_path = args.output_dir / "reaction_correlation_top_pairs.csv"
    save_csv(reaction_corr_top_pairs, reaction_corr_top_pairs_path)
    register_output(
        "csv",
        "correlation_redundancy",
        reaction_corr_top_pairs_path,
        "Strongest pairwise correlations among the 30 most variable log-transformed rate constants.",
    )

    feature_target_rows = []
    for feature in ["log10_e_over_n"] + nonconstant_input_cols:
        x_values = x_model[feature].to_numpy()
        for target_idx, target_col in enumerate(target_cols):
            rho, p_value = safe_spearman(x_values, y_log[:, target_idx])
            feature_target_rows.append(
                {
                    "feature": feature,
                    "feature_label": species_label_by_column.get(feature, feature),
                    "target_column": target_col,
                    "reaction_id": int(reaction_id_by_column[target_col]),
                    "reaction_label": reaction_label_by_column[target_col],
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "p_value": p_value,
                }
            )
    feature_target_sensitivity = pd.DataFrame(feature_target_rows)
    feature_target_top_pairs = feature_target_sensitivity.sort_values(
        "abs_spearman_rho", ascending=False
    ).head(300)
    feature_target_top_pairs_path = args.output_dir / "feature_target_spearman_top_pairs.csv"
    save_csv(feature_target_top_pairs, feature_target_top_pairs_path)
    register_output(
        "csv",
        "correlation_redundancy",
        feature_target_top_pairs_path,
        "Strongest Spearman associations between model inputs and log-transformed rate constants.",
    )

    redundancy_summary = pd.DataFrame(
        [
            {
                "metric": "constant_input_feature_count",
                "value": len(constant_input_cols),
                "note": "Input species that never change across the 609 cases.",
            },
            {
                "metric": "nonconstant_input_feature_count",
                "value": len(nonconstant_input_cols),
                "note": "Input species with at least two distinct values across the 609 cases.",
            },
            {
                "metric": "near_perfect_input_pairs_abs_corr_ge_0_95",
                "value": int((input_corr_top_pairs["abs_correlation"] >= 0.95).sum()),
                "note": "Highly redundant composition-feature pairs among the top correlations.",
            },
            {
                "metric": "near_perfect_reaction_pairs_abs_corr_ge_0_95",
                "value": int((reaction_corr_top_pairs["abs_correlation"] >= 0.95).sum()),
                "note": "Highly redundant reaction pairs among the 30 most variable targets.",
            },
        ]
    )
    redundancy_summary_path = args.output_dir / "redundancy_summary.csv"
    save_csv(redundancy_summary, redundancy_summary_path)
    register_output(
        "csv",
        "correlation_redundancy",
        redundancy_summary_path,
        "High-level redundancy statistics for the input and target spaces.",
    )

    # Method 2: group-aware variation decomposition
    case_feature_var_rows = []
    selected_case_feature_cols = [
        "average_electron_energy_ev",
        "equivalent_electron_temperature_ev",
        "updated_electron_density_per_cc",
        "drift_velocity_cm_per_s",
        "mobility_cm2_per_v_s",
        "ionization_coefficient_per_cm",
        "total_power_loss_ev_cm3_per_s",
    ]
    for feature in selected_case_feature_cols:
        matrix = (
            case_features.pivot(
                index="density_group_id",
                columns="local_case_id",
                values=feature,
            )
            .sort_index()
            .to_numpy()
        )
        stats = compute_variance_decomposition(matrix)
        case_feature_var_rows.append(
            {
                "feature": feature,
                **stats,
                "dominant_source": max(
                    ("group_share", stats["group_share"]),
                    ("local_case_share", stats["local_case_share"]),
                    ("interaction_share", stats["interaction_share"]),
                    key=lambda item: item[1],
                )[0],
            }
        )
    case_feature_variance_decomposition = pd.DataFrame(case_feature_var_rows).sort_values(
        "local_case_share", ascending=False
    )
    case_feature_variance_path = args.output_dir / "case_feature_variance_decomposition.csv"
    save_csv(case_feature_variance_decomposition, case_feature_variance_path)
    register_output(
        "csv",
        "group_variation_decomposition",
        case_feature_variance_path,
        "Variance decomposition of selected case-level summary variables into density-group, E/N, and interaction shares.",
    )

    reaction_var_rows = []
    y_log_df = pd.DataFrame(y_log, columns=target_cols)
    for target_col in target_cols:
        matrix = (
            pd.concat([training_targets[id_cols], y_log_df[[target_col]]], axis=1)
            .pivot(index="density_group_id", columns="local_case_id", values=target_col)
            .sort_index()
            .to_numpy()
        )
        stats = compute_variance_decomposition(matrix)
        reaction_var_rows.append(
            {
                "target_column": target_col,
                "reaction_id": int(reaction_id_by_column[target_col]),
                "reaction_label": reaction_label_by_column[target_col],
                **stats,
                "dominant_source": max(
                    ("group_share", stats["group_share"]),
                    ("local_case_share", stats["local_case_share"]),
                    ("interaction_share", stats["interaction_share"]),
                    key=lambda item: item[1],
                )[0],
            }
        )
    reaction_variance_decomposition = pd.DataFrame(reaction_var_rows).sort_values(
        "local_case_share", ascending=False
    )
    reaction_variance_path = args.output_dir / "reaction_variance_decomposition.csv"
    save_csv(reaction_variance_decomposition, reaction_variance_path)
    register_output(
        "csv",
        "group_variation_decomposition",
        reaction_variance_path,
        "Variance decomposition of all 204 log-transformed rate constants into density-group, E/N, and interaction shares.",
    )

    # Method 3: target transform diagnostics
    transforms = {
        "raw_rate_const": y_raw,
        "log10_rate_const_plus_eps": y_log,
        "asinh_rate_const_scaled": y_asinh,
    }
    transform_global_rows = []
    transform_reaction_rows = []
    for transform_name, array in transforms.items():
        flat = array.reshape(-1)
        reaction_skews = []
        reaction_kurtoses = []
        for target_idx, target_col in enumerate(target_cols):
            values = array[:, target_idx]
            reaction_skews.append(safe_skew(values))
            reaction_kurtoses.append(safe_kurtosis(values))
            transform_reaction_rows.append(
                {
                    "transform": transform_name,
                    "target_column": target_col,
                    "reaction_id": int(reaction_id_by_column[target_col]),
                    "reaction_label": reaction_label_by_column[target_col],
                    "skewness": safe_skew(values),
                    "kurtosis": safe_kurtosis(values),
                    "std": float(np.std(values)),
                    "mean": float(np.mean(values)),
                }
            )
        transform_global_rows.append(
            {
                "transform": transform_name,
                "overall_min": float(np.min(flat)),
                "overall_median": float(np.median(flat)),
                "overall_mean": float(np.mean(flat)),
                "overall_max": float(np.max(flat)),
                "mean_reaction_skewness": float(np.mean(reaction_skews)),
                "median_reaction_skewness": float(np.median(reaction_skews)),
                "mean_abs_reaction_skewness": float(np.mean(np.abs(reaction_skews))),
                "mean_reaction_kurtosis": float(np.mean(reaction_kurtoses)),
                "median_reaction_kurtosis": float(np.median(reaction_kurtoses)),
                "note": "Lower absolute skewness usually indicates a friendlier target transform for regression.",
            }
        )
    target_transform_global = pd.DataFrame(transform_global_rows)
    target_transform_reaction = pd.DataFrame(transform_reaction_rows)
    target_transform_global_path = args.output_dir / "target_transform_global_summary.csv"
    target_transform_reaction_path = args.output_dir / "target_transform_reaction_summary.csv"
    save_csv(target_transform_global, target_transform_global_path)
    save_csv(target_transform_reaction, target_transform_reaction_path)
    register_output(
        "csv",
        "target_transform_diagnostics",
        target_transform_global_path,
        "Global comparison of raw, log10, and asinh transforms applied to the 204 rate constants.",
    )
    register_output(
        "csv",
        "target_transform_diagnostics",
        target_transform_reaction_path,
        "Reaction-level skewness and spread statistics for each candidate target transform.",
    )

    # Method 4: reaction activity analysis
    reaction_activity_rows = []
    reaction_activation_local_rows = []
    for target_col in target_cols:
        group_first_active = []
        group_last_active = []
        group_active_count = []
        active_group_count = 0
        for density_group_id in density_group_values:
            subset = training_targets[training_targets["density_group_id"] == density_group_id]
            active_local_cases = subset.loc[subset[target_col] > 0, "local_case_id"].tolist()
            if active_local_cases:
                active_group_count += 1
                group_first_active.append(min(active_local_cases))
                group_last_active.append(max(active_local_cases))
                group_active_count.append(len(active_local_cases))
        series = training_targets[target_col]
        reaction_activity_rows.append(
            {
                "target_column": target_col,
                "reaction_id": int(reaction_id_by_column[target_col]),
                "reaction_label": reaction_label_by_column[target_col],
                "nonzero_fraction": float((series > 0).mean()),
                "active_case_count": int((series > 0).sum()),
                "active_density_group_count": active_group_count,
                "is_always_zero": bool((series > 0).sum() == 0),
                "is_always_nonzero": bool((series > 0).sum() == len(series)),
                "min_first_active_local_case": min(group_first_active) if group_first_active else np.nan,
                "median_first_active_local_case": float(np.median(group_first_active))
                if group_first_active
                else np.nan,
                "max_first_active_local_case": max(group_first_active) if group_first_active else np.nan,
                "median_last_active_local_case": float(np.median(group_last_active))
                if group_last_active
                else np.nan,
                "mean_active_local_case_count": float(np.mean(group_active_count))
                if group_active_count
                else 0.0,
            }
        )

    for local_case_id in local_case_values:
        subset = training_targets[training_targets["local_case_id"] == local_case_id]
        active_counts = (subset[target_cols] > 0).sum(axis=1)
        reaction_activation_local_rows.append(
            {
                "local_case_id": int(local_case_id),
                "mean_active_reaction_count": float(active_counts.mean()),
                "min_active_reaction_count": int(active_counts.min()),
                "max_active_reaction_count": int(active_counts.max()),
            }
        )

    reaction_activity_summary = pd.DataFrame(reaction_activity_rows).sort_values(
        ["nonzero_fraction", "active_case_count", "reaction_id"],
        ascending=[False, False, True],
    )
    reaction_activity_path = args.output_dir / "reaction_activity_summary.csv"
    save_csv(reaction_activity_summary, reaction_activity_path)
    register_output(
        "csv",
        "reaction_activity",
        reaction_activity_path,
        "How often each reaction is active, and when along the 29-point E/N sweep it first turns on.",
    )

    reaction_activation_by_local_case = pd.DataFrame(reaction_activation_local_rows)
    reaction_activation_local_path = args.output_dir / "reaction_activation_by_local_case.csv"
    save_csv(reaction_activation_by_local_case, reaction_activation_local_path)
    register_output(
        "csv",
        "reaction_activity",
        reaction_activation_local_path,
        "Average number of active reactions for each local E/N case index.",
    )

    # Method 5: monotonicity and trend analysis
    case_feature_monotonicity_rows = []
    for feature in selected_case_feature_cols:
        rhos = []
        p_values = []
        for density_group_id in density_group_values:
            subset = case_features[case_features["density_group_id"] == density_group_id].sort_values(
                "local_case_id"
            )
            rho, p_value = safe_spearman(log_e_unique, subset[feature].to_numpy())
            rhos.append(rho)
            p_values.append(p_value)
        case_feature_monotonicity_rows.append(
            {
                "feature": feature,
                "mean_spearman_rho": float(np.mean(rhos)),
                "median_spearman_rho": float(np.median(rhos)),
                "mean_abs_spearman_rho": float(np.mean(np.abs(rhos))),
                "positive_group_count_rho_gt_0_7": int(np.sum(np.array(rhos) > 0.7)),
                "negative_group_count_rho_lt_neg_0_7": int(np.sum(np.array(rhos) < -0.7)),
                "significant_group_count_p_lt_0_05": int(np.sum(np.array(p_values) < 0.05)),
            }
        )
    case_feature_monotonicity = pd.DataFrame(case_feature_monotonicity_rows).sort_values(
        "mean_abs_spearman_rho", ascending=False
    )
    case_feature_monotonicity_path = args.output_dir / "case_feature_monotonicity.csv"
    save_csv(case_feature_monotonicity, case_feature_monotonicity_path)
    register_output(
        "csv",
        "monotonicity_trends",
        case_feature_monotonicity_path,
        "Density-group-wise monotonicity diagnostics for selected case-level summary variables across E/N.",
    )

    reaction_monotonicity_rows = []
    for target_col in target_cols:
        rhos = []
        p_values = []
        for density_group_id in density_group_values:
            mask = training_targets["density_group_id"] == density_group_id
            values = y_log_df.loc[mask, target_col].to_numpy()
            rho, p_value = safe_spearman(log_e_unique, values)
            rhos.append(rho)
            p_values.append(p_value)
        reaction_monotonicity_rows.append(
            {
                "target_column": target_col,
                "reaction_id": int(reaction_id_by_column[target_col]),
                "reaction_label": reaction_label_by_column[target_col],
                "mean_spearman_rho": float(np.mean(rhos)),
                "median_spearman_rho": float(np.median(rhos)),
                "mean_abs_spearman_rho": float(np.mean(np.abs(rhos))),
                "positive_group_count_rho_gt_0_7": int(np.sum(np.array(rhos) > 0.7)),
                "negative_group_count_rho_lt_neg_0_7": int(np.sum(np.array(rhos) < -0.7)),
                "significant_group_count_p_lt_0_05": int(np.sum(np.array(p_values) < 0.05)),
            }
        )
    reaction_monotonicity = pd.DataFrame(reaction_monotonicity_rows).sort_values(
        "mean_abs_spearman_rho", ascending=False
    )
    reaction_monotonicity_path = args.output_dir / "reaction_monotonicity_summary.csv"
    save_csv(reaction_monotonicity, reaction_monotonicity_path)
    register_output(
        "csv",
        "monotonicity_trends",
        reaction_monotonicity_path,
        "Reaction-level monotonicity diagnostics for log-transformed rate constants across E/N.",
    )

    # Method 6: sensitivity analysis
    feature_sensitivity_summary_rows = []
    for feature in ["log10_e_over_n"] + nonconstant_input_cols:
        subset = feature_target_sensitivity[feature_target_sensitivity["feature"] == feature]
        strongest = subset.loc[subset["abs_spearman_rho"].idxmax()]
        feature_sensitivity_summary_rows.append(
            {
                "feature": feature,
                "feature_label": species_label_by_column.get(feature, feature),
                "mean_abs_spearman_rho": float(subset["abs_spearman_rho"].mean()),
                "median_abs_spearman_rho": float(subset["abs_spearman_rho"].median()),
                "max_abs_spearman_rho": float(strongest["abs_spearman_rho"]),
                "strongest_reaction_id": int(strongest["reaction_id"]),
                "strongest_reaction_label": strongest["reaction_label"],
                "strongest_signed_rho": float(strongest["spearman_rho"]),
            }
        )
    feature_sensitivity_summary = pd.DataFrame(feature_sensitivity_summary_rows).sort_values(
        "mean_abs_spearman_rho", ascending=False
    )
    feature_sensitivity_summary_path = args.output_dir / "feature_sensitivity_summary.csv"
    save_csv(feature_sensitivity_summary, feature_sensitivity_summary_path)
    register_output(
        "csv",
        "sensitivity_analysis",
        feature_sensitivity_summary_path,
        "Per-feature summary of input-target Spearman sensitivity over all 204 targets.",
    )

    # Method 7: clustering
    density_group_best_k, density_group_cluster_scores = choose_best_k(
        group_compositions_scaled,
        k_values=list(range(2, min(6, len(group_compositions) - 1) + 1)),
    )
    density_group_cluster_scores_path = args.output_dir / "density_group_cluster_scores.csv"
    save_csv(density_group_cluster_scores, density_group_cluster_scores_path)
    register_output(
        "csv",
        "clustering",
        density_group_cluster_scores_path,
        "Silhouette-score grid for density-group clustering.",
    )

    density_group_kmeans = KMeans(
        n_clusters=density_group_best_k, random_state=RANDOM_STATE, n_init=20
    ).fit(group_compositions_scaled)
    density_group_assignments = pd.DataFrame(
        {
            "density_group_id": group_compositions.index,
            "cluster_id": density_group_kmeans.labels_,
        }
    )
    density_group_assignments_path = args.output_dir / "density_group_cluster_assignments.csv"
    save_csv(density_group_assignments, density_group_assignments_path)
    register_output(
        "csv",
        "clustering",
        density_group_assignments_path,
        "Cluster assignment for the 21 density-group compositions.",
    )

    density_group_profiles = (
        density_group_assignments.merge(
            group_compositions.reset_index(), on="density_group_id", how="left"
        )
        .groupby("cluster_id")[nonconstant_input_cols]
        .mean()
        .reset_index()
    )
    density_group_profiles_path = args.output_dir / "density_group_cluster_profiles.csv"
    save_csv(density_group_profiles, density_group_profiles_path)
    register_output(
        "csv",
        "clustering",
        density_group_profiles_path,
        "Mean composition profile of each density-group cluster.",
    )

    reaction_profile_matrix = StandardScaler().fit_transform(y_log.T)
    reaction_pca_for_cluster = PCA(
        n_components=min(10, reaction_profile_matrix.shape[0], reaction_profile_matrix.shape[1]),
        random_state=RANDOM_STATE,
    ).fit_transform(reaction_profile_matrix)
    reaction_best_k, reaction_cluster_scores = choose_best_k(
        reaction_pca_for_cluster,
        k_values=list(range(2, 9)),
    )
    reaction_cluster_scores_path = args.output_dir / "reaction_cluster_scores.csv"
    save_csv(reaction_cluster_scores, reaction_cluster_scores_path)
    register_output(
        "csv",
        "clustering",
        reaction_cluster_scores_path,
        "Silhouette-score grid for clustering the 204 reaction profiles.",
    )

    reaction_kmeans = KMeans(n_clusters=reaction_best_k, random_state=RANDOM_STATE, n_init=20).fit(
        reaction_pca_for_cluster
    )
    reaction_cluster_assignments = pd.DataFrame(
        {
            "reaction_id": reaction_map["reaction_id"],
            "reaction_label": reaction_map["reaction_label"],
            "cluster_id": reaction_kmeans.labels_,
        }
    ).merge(
        reaction_activity_summary[
            ["reaction_id", "nonzero_fraction", "active_density_group_count"]
        ],
        on="reaction_id",
        how="left",
    )
    reaction_cluster_assignments_path = args.output_dir / "reaction_cluster_assignments.csv"
    save_csv(reaction_cluster_assignments, reaction_cluster_assignments_path)
    register_output(
        "csv",
        "clustering",
        reaction_cluster_assignments_path,
        "Cluster assignment for the 204 reaction target profiles.",
    )

    reaction_cluster_profiles = (
        reaction_cluster_assignments.groupby("cluster_id")
        .agg(
            reaction_count=("reaction_id", "count"),
            mean_nonzero_fraction=("nonzero_fraction", "mean"),
            mean_active_density_group_count=("active_density_group_count", "mean"),
        )
        .reset_index()
    )
    reaction_cluster_profiles_path = args.output_dir / "reaction_cluster_profiles.csv"
    save_csv(reaction_cluster_profiles, reaction_cluster_profiles_path)
    register_output(
        "csv",
        "clustering",
        reaction_cluster_profiles_path,
        "Size and activity characteristics of the learned reaction clusters.",
    )

    case_target_scaled = StandardScaler().fit_transform(y_log)
    case_target_pca = PCA(
        n_components=min(10, case_target_scaled.shape[0], case_target_scaled.shape[1]),
        random_state=RANDOM_STATE,
    ).fit_transform(case_target_scaled)
    case_best_k, case_cluster_scores = choose_best_k(
        case_target_pca,
        k_values=list(range(2, 9)),
    )
    case_cluster_scores_path = args.output_dir / "case_cluster_scores.csv"
    save_csv(case_cluster_scores, case_cluster_scores_path)
    register_output(
        "csv",
        "clustering",
        case_cluster_scores_path,
        "Silhouette-score grid for clustering the 609 case target profiles.",
    )

    case_kmeans = KMeans(n_clusters=case_best_k, random_state=RANDOM_STATE, n_init=20).fit(
        case_target_pca
    )
    case_cluster_assignments = training_targets[id_cols].copy()
    case_cluster_assignments["cluster_id"] = case_kmeans.labels_
    case_cluster_assignments_path = args.output_dir / "case_cluster_assignments.csv"
    save_csv(case_cluster_assignments, case_cluster_assignments_path)
    register_output(
        "csv",
        "clustering",
        case_cluster_assignments_path,
        "Cluster assignment for the 609 cases using PCA-reduced target profiles.",
    )

    # Method 8: low-rank structure
    composition_pca_model = PCA(
        n_components=min(group_compositions_scaled.shape),
        random_state=RANDOM_STATE,
    ).fit(group_compositions_scaled)
    composition_pca_explained = pd.DataFrame(
        {
            "component_index": np.arange(1, len(composition_pca_model.explained_variance_ratio_) + 1),
            "explained_variance_ratio": composition_pca_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                composition_pca_model.explained_variance_ratio_
            ),
        }
    )
    composition_pca_explained_path = args.output_dir / "composition_pca_explained_variance.csv"
    save_csv(composition_pca_explained, composition_pca_explained_path)
    register_output(
        "csv",
        "low_rank_structure",
        composition_pca_explained_path,
        "Explained-variance curve for PCA applied to the 21 unique density-group compositions.",
    )

    target_pca_model = PCA(
        n_components=min(case_target_scaled.shape),
        random_state=RANDOM_STATE,
    ).fit(case_target_scaled)
    target_pca_explained = pd.DataFrame(
        {
            "component_index": np.arange(1, len(target_pca_model.explained_variance_ratio_) + 1),
            "explained_variance_ratio": target_pca_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                target_pca_model.explained_variance_ratio_
            ),
        }
    )
    target_pca_explained_path = args.output_dir / "target_pca_explained_variance.csv"
    save_csv(target_pca_explained, target_pca_explained_path)
    register_output(
        "csv",
        "low_rank_structure",
        target_pca_explained_path,
        "Explained-variance curve for PCA applied to standardized log-transformed rate constants.",
    )

    low_rank_summary = pd.DataFrame(
        [
            {
                "matrix_name": "composition_inputs",
                "effective_rank": effective_rank(composition_pca_model.explained_variance_ratio_),
                "components_for_80pct": components_for_threshold(
                    np.cumsum(composition_pca_model.explained_variance_ratio_), 0.80
                ),
                "components_for_90pct": components_for_threshold(
                    np.cumsum(composition_pca_model.explained_variance_ratio_), 0.90
                ),
                "components_for_95pct": components_for_threshold(
                    np.cumsum(composition_pca_model.explained_variance_ratio_), 0.95
                ),
                "components_for_99pct": components_for_threshold(
                    np.cumsum(composition_pca_model.explained_variance_ratio_), 0.99
                ),
            },
            {
                "matrix_name": "log_rate_constants",
                "effective_rank": effective_rank(target_pca_model.explained_variance_ratio_),
                "components_for_80pct": components_for_threshold(
                    np.cumsum(target_pca_model.explained_variance_ratio_), 0.80
                ),
                "components_for_90pct": components_for_threshold(
                    np.cumsum(target_pca_model.explained_variance_ratio_), 0.90
                ),
                "components_for_95pct": components_for_threshold(
                    np.cumsum(target_pca_model.explained_variance_ratio_), 0.95
                ),
                "components_for_99pct": components_for_threshold(
                    np.cumsum(target_pca_model.explained_variance_ratio_), 0.99
                ),
            },
        ]
    )
    low_rank_summary_path = args.output_dir / "low_rank_summary.csv"
    save_csv(low_rank_summary, low_rank_summary_path)
    register_output(
        "csv",
        "low_rank_structure",
        low_rank_summary_path,
        "Effective-rank and explained-variance thresholds for the composition and target spaces.",
    )

    # Method 9: baseline predictability analysis
    x_default = build_feature_sets(training_inputs, nonconstant_input_cols, log_e_over_n)[
        "full_nonconstant_plus_log_en"
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        x_default.to_numpy(),
        y_log,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    baseline_models = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "ridge_alpha_1": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "pls_n8": Pipeline(
            [("scaler", StandardScaler()), ("model", PLSRegression(n_components=8, scale=False))]
        ),
        "random_forest_150": RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    baseline_rows = []
    for model_name, model in baseline_models.items():
        metrics = evaluate_model(model, x_train, x_test, y_train, y_test)
        baseline_rows.append(
            {
                "model_name": model_name,
                "split_strategy": "random_case_split",
                "feature_set": "full_nonconstant_plus_log_en",
                **metrics,
            }
        )
    baseline_random_split_metrics = pd.DataFrame(baseline_rows).sort_values(
        "overall_r2", ascending=False
    )
    baseline_random_split_path = args.output_dir / "baseline_random_split_metrics.csv"
    save_csv(baseline_random_split_metrics, baseline_random_split_path)
    register_output(
        "csv",
        "baseline_predictability",
        baseline_random_split_path,
        "Baseline multi-output regression metrics on a random 80/20 case split using log-transformed targets.",
    )

    # Method 10: split strategy analysis
    split_rows = []
    split_metadata_rows = []
    split_definitions = {}

    random_train_idx, random_test_idx = train_test_split(
        np.arange(len(x_default)), test_size=0.2, random_state=RANDOM_STATE
    )
    split_definitions["random_case_split"] = (random_train_idx, random_test_idx)
    split_metadata_rows.append(
        {
            "split_strategy": "random_case_split",
            "test_descriptor": "20 percent random holdout of the 609 cases",
            "train_size": len(random_train_idx),
            "test_size": len(random_test_idx),
        }
    )

    group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    group_train_idx, group_test_idx = next(
        group_splitter.split(x_default, groups=training_inputs["density_group_id"])
    )
    held_out_groups = sorted(training_inputs.iloc[group_test_idx]["density_group_id"].unique().tolist())
    split_definitions["density_group_holdout"] = (group_train_idx, group_test_idx)
    split_metadata_rows.append(
        {
            "split_strategy": "density_group_holdout",
            "test_descriptor": f"held-out density groups: {held_out_groups}",
            "train_size": len(group_train_idx),
            "test_size": len(group_test_idx),
        }
    )

    local_splitter = GroupShuffleSplit(n_splits=1, test_size=6 / 29, random_state=RANDOM_STATE)
    local_train_idx, local_test_idx = next(
        local_splitter.split(x_default, groups=training_inputs["local_case_id"])
    )
    held_out_local_cases = sorted(
        training_inputs.iloc[local_test_idx]["local_case_id"].unique().tolist()
    )
    split_definitions["local_case_holdout"] = (local_train_idx, local_test_idx)
    split_metadata_rows.append(
        {
            "split_strategy": "local_case_holdout",
            "test_descriptor": f"held-out local_case_id values: {held_out_local_cases}",
            "train_size": len(local_train_idx),
            "test_size": len(local_test_idx),
        }
    )

    split_metadata = pd.DataFrame(split_metadata_rows)
    split_metadata_path = args.output_dir / "split_strategy_metadata.csv"
    save_csv(split_metadata, split_metadata_path)
    register_output(
        "csv",
        "split_strategy_analysis",
        split_metadata_path,
        "Exact train/test definitions used for the random, density-group, and local-case holdout splits.",
    )

    split_models = {
        "ridge_alpha_1": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        ),
        "pls_n8": Pipeline(
            [("scaler", StandardScaler()), ("model", PLSRegression(n_components=8, scale=False))]
        ),
        "random_forest_150": RandomForestRegressor(
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    x_default_values = x_default.to_numpy()
    for split_name, (train_idx, test_idx) in split_definitions.items():
        for model_name, model in split_models.items():
            metrics = evaluate_model(
                model,
                x_default_values[train_idx],
                x_default_values[test_idx],
                y_log[train_idx],
                y_log[test_idx],
            )
            split_rows.append(
                {
                    "split_strategy": split_name,
                    "model_name": model_name,
                    "feature_set": "full_nonconstant_plus_log_en",
                    **metrics,
                }
            )
    split_strategy_metrics = pd.DataFrame(split_rows).sort_values(
        ["split_strategy", "overall_r2"], ascending=[True, False]
    )
    split_strategy_metrics_path = args.output_dir / "split_strategy_metrics.csv"
    save_csv(split_strategy_metrics, split_strategy_metrics_path)
    register_output(
        "csv",
        "split_strategy_analysis",
        split_strategy_metrics_path,
        "Model performance comparison under random case split, density-group holdout, and local-case holdout.",
    )

    # Method 11: outlier and anomaly analysis
    x_anomaly_scaled = StandardScaler().fit_transform(x_default)
    y_anomaly_scaled = StandardScaler().fit_transform(y_log)
    case_feature_anomaly_scaled = StandardScaler().fit_transform(
        case_features[selected_case_feature_cols]
    )

    input_iforest = IsolationForest(random_state=RANDOM_STATE, contamination="auto").fit(
        x_anomaly_scaled
    )
    target_iforest = IsolationForest(random_state=RANDOM_STATE, contamination="auto").fit(
        y_anomaly_scaled
    )
    case_feature_iforest = IsolationForest(
        random_state=RANDOM_STATE, contamination="auto"
    ).fit(case_feature_anomaly_scaled)

    case_anomaly_scores = training_inputs[id_cols].copy()
    case_anomaly_scores["input_anomaly_score"] = -input_iforest.score_samples(x_anomaly_scaled)
    case_anomaly_scores["target_anomaly_score"] = -target_iforest.score_samples(y_anomaly_scaled)
    case_anomaly_scores["case_feature_anomaly_score"] = -case_feature_iforest.score_samples(
        case_feature_anomaly_scaled
    )
    standardized_case_features = (
        case_features[selected_case_feature_cols] - case_features[selected_case_feature_cols].mean()
    ) / case_features[selected_case_feature_cols].std(ddof=0)
    case_anomaly_scores["max_abs_case_feature_zscore"] = standardized_case_features.abs().max(axis=1)
    case_anomaly_scores["combined_anomaly_rank_score"] = (
        case_anomaly_scores["input_anomaly_score"].rank(pct=True)
        + case_anomaly_scores["target_anomaly_score"].rank(pct=True)
        + case_anomaly_scores["case_feature_anomaly_score"].rank(pct=True)
        + case_anomaly_scores["max_abs_case_feature_zscore"].rank(pct=True)
    )
    case_anomaly_scores = case_anomaly_scores.sort_values(
        "combined_anomaly_rank_score", ascending=False
    )
    case_anomaly_scores_path = args.output_dir / "case_anomaly_scores.csv"
    save_csv(case_anomaly_scores, case_anomaly_scores_path)
    register_output(
        "csv",
        "outlier_anomaly_analysis",
        case_anomaly_scores_path,
        "Per-case anomaly scores based on inputs, targets, and case-level physical summary variables.",
    )

    top_case_anomalies = case_anomaly_scores.head(30)
    top_case_anomalies_path = args.output_dir / "top_case_anomalies.csv"
    save_csv(top_case_anomalies, top_case_anomalies_path)
    register_output(
        "csv",
        "outlier_anomaly_analysis",
        top_case_anomalies_path,
        "Top 30 most unusual cases according to the combined anomaly ranking.",
    )

    reaction_outlier_propensity_rows = []
    for target_idx, target_col in enumerate(target_cols):
        values = y_log[:, target_idx]
        centered = values - values.mean()
        std = values.std(ddof=0)
        if std == 0:
            outlier_fraction = 0.0
            max_abs_z = 0.0
        else:
            z_scores = centered / std
            outlier_fraction = float(np.mean(np.abs(z_scores) > 3.0))
            max_abs_z = float(np.max(np.abs(z_scores)))
        reaction_outlier_propensity_rows.append(
            {
                "reaction_id": int(reaction_id_by_column[target_col]),
                "reaction_label": reaction_label_by_column[target_col],
                "outlier_fraction_abs_z_gt_3": outlier_fraction,
                "max_abs_zscore": max_abs_z,
            }
        )
    reaction_outlier_propensity = pd.DataFrame(reaction_outlier_propensity_rows).sort_values(
        "outlier_fraction_abs_z_gt_3", ascending=False
    )
    reaction_outlier_propensity_path = args.output_dir / "reaction_outlier_propensity.csv"
    save_csv(reaction_outlier_propensity, reaction_outlier_propensity_path)
    register_output(
        "csv",
        "outlier_anomaly_analysis",
        reaction_outlier_propensity_path,
        "Reaction-level outlier propensity measured on log-transformed targets using z-score thresholds.",
    )

    # Method 12: feature engineering analysis
    feature_sets = build_feature_sets(training_inputs, nonconstant_input_cols, log_e_over_n)
    feature_set_rows = []
    for feature_set_name, feature_df in feature_sets.items():
        feat_train, feat_test, y_train, y_test = train_test_split(
            feature_df.to_numpy(),
            y_log,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        metrics = evaluate_model(
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
            feat_train,
            feat_test,
            y_train,
            y_test,
        )
        feature_set_rows.append(
            {
                "feature_set": feature_set_name,
                "feature_count": feature_df.shape[1],
                **metrics,
            }
        )
    feature_set_comparison = pd.DataFrame(feature_set_rows).sort_values(
        "overall_r2", ascending=False
    )
    feature_set_comparison_path = args.output_dir / "feature_set_comparison.csv"
    save_csv(feature_set_comparison, feature_set_comparison_path)
    register_output(
        "csv",
        "feature_engineering_analysis",
        feature_set_comparison_path,
        "Comparison of candidate engineered input feature sets using a Ridge baseline on a random split.",
    )

    # Figures
    # 1 input correlation heatmap
    top_input_heatmap_cols = (
        group_compositions.var().sort_values(ascending=False).head(15).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        group_compositions[top_input_heatmap_cols].corr(),
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Top-Varying Composition Feature Correlations")
    input_corr_heatmap_path = args.figure_dir / "input_correlation_heatmap_top_features.png"
    save_figure(fig, input_corr_heatmap_path)
    register_output(
        "figure",
        "correlation_redundancy",
        input_corr_heatmap_path,
        "Correlation heatmap for the 15 most variable composition features.",
    )

    # 2 target correlation heatmap
    top_target_heatmap_cols = reaction_variance.head(20).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        pd.DataFrame(y_log, columns=target_cols)[top_target_heatmap_cols].corr(),
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Top-Variance Reaction Correlations (Log Targets)")
    target_corr_heatmap_path = args.figure_dir / "target_correlation_heatmap_top_reactions.png"
    save_figure(fig, target_corr_heatmap_path)
    register_output(
        "figure",
        "correlation_redundancy",
        target_corr_heatmap_path,
        "Correlation heatmap for the 20 most variable log-transformed reaction targets.",
    )

    # 3 variance decomposition stacked bar
    top_local_share = reaction_variance_decomposition.sort_values(
        "local_case_share", ascending=False
    ).head(15)
    fig, ax = plt.subplots(figsize=(15, 8))
    x_positions = np.arange(len(top_local_share))
    ax.bar(x_positions, top_local_share["group_share"], label="density_group")
    ax.bar(
        x_positions,
        top_local_share["local_case_share"],
        bottom=top_local_share["group_share"],
        label="local_case",
    )
    ax.bar(
        x_positions,
        top_local_share["interaction_share"],
        bottom=top_local_share["group_share"] + top_local_share["local_case_share"],
        label="interaction",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top_local_share["reaction_label"], rotation=70, ha="right")
    ax.set_ylabel("Variance Share")
    ax.set_title("Variance Decomposition of Reactions Most Driven by E/N")
    ax.legend()
    variance_decomposition_plot_path = args.figure_dir / "reaction_variance_decomposition_top_local_case.png"
    save_figure(fig, variance_decomposition_plot_path)
    register_output(
        "figure",
        "group_variation_decomposition",
        variance_decomposition_plot_path,
        "Stacked variance-share plot for reactions with the largest E/N-driven variance component.",
    )

    # 4 transform skewness comparison
    transform_plot_df = target_transform_reaction[
        ["transform", "skewness"]
    ].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=transform_plot_df, x="transform", y="skewness", ax=ax)
    ax.set_title("Reaction-Level Skewness Under Different Target Transforms")
    ax.tick_params(axis="x", rotation=20)
    transform_plot_path = args.figure_dir / "target_transform_skewness_comparison.png"
    save_figure(fig, transform_plot_path)
    register_output(
        "figure",
        "target_transform_diagnostics",
        transform_plot_path,
        "Boxplot comparing reaction-level skewness for raw, log10, and asinh target transforms.",
    )

    # 5 reaction activity summary plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(reaction_activity_summary["nonzero_fraction"], bins=20, ax=axes[0], color="#457b9d")
    axes[0].set_title("Reaction Nonzero Fraction Distribution")
    axes[0].set_xlabel("Fraction of Cases With Positive Rate Constant")
    sns.histplot(
        reaction_activity_summary["median_first_active_local_case"].dropna(),
        bins=15,
        ax=axes[1],
        color="#e76f51",
    )
    axes[1].set_title("When Reactions First Activate Across E/N")
    axes[1].set_xlabel("Median First Active Local Case")
    reaction_activity_plot_path = args.figure_dir / "reaction_activity_summary.png"
    save_figure(fig, reaction_activity_plot_path)
    register_output(
        "figure",
        "reaction_activity",
        reaction_activity_plot_path,
        "Histogram summary of how frequently reactions are active and when they first activate.",
    )

    # 6 active reaction count by local case
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=reaction_activation_by_local_case,
        x="local_case_id",
        y="mean_active_reaction_count",
        marker="o",
        ax=ax,
    )
    ax.set_title("Mean Active Reaction Count Across the 29 E/N Points")
    ax.set_xlabel("Local Case ID")
    ax.set_ylabel("Mean Active Reaction Count")
    activation_line_path = args.figure_dir / "reaction_activation_by_local_case.png"
    save_figure(fig, activation_line_path)
    register_output(
        "figure",
        "reaction_activity",
        activation_line_path,
        "Line plot of how the mean number of active reactions grows across the E/N sweep.",
    )

    # 7 reaction monotonicity histogram
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(reaction_monotonicity["mean_spearman_rho"], bins=30, color="#2a9d8f", ax=ax)
    ax.set_title("Reaction Mean Spearman Correlation With E/N")
    ax.set_xlabel("Mean Spearman Rho Across Density Groups")
    reaction_monotonicity_plot_path = args.figure_dir / "reaction_monotonicity_histogram.png"
    save_figure(fig, reaction_monotonicity_plot_path)
    register_output(
        "figure",
        "monotonicity_trends",
        reaction_monotonicity_plot_path,
        "Distribution of mean reaction monotonicity with respect to E/N.",
    )

    # 8 density-group clustering PCA plot
    density_group_pca_scores = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(
        group_compositions_scaled
    )
    density_group_plot_df = density_group_assignments.copy()
    density_group_plot_df["pc1"] = density_group_pca_scores[:, 0]
    density_group_plot_df["pc2"] = density_group_pca_scores[:, 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=density_group_plot_df,
        x="pc1",
        y="pc2",
        hue="cluster_id",
        palette="tab10",
        s=120,
        ax=ax,
    )
    for _, row in density_group_plot_df.iterrows():
        ax.text(row["pc1"], row["pc2"], str(int(row["density_group_id"])), fontsize=9)
    ax.set_title("Density-Group Composition Clusters in PCA Space")
    density_group_cluster_plot_path = args.figure_dir / "density_group_clustering_pca.png"
    save_figure(fig, density_group_cluster_plot_path)
    register_output(
        "figure",
        "clustering",
        density_group_cluster_plot_path,
        "PCA projection of density-group compositions colored by learned cluster.",
    )

    # 9 case clustering target PCA plot
    case_plot_df = case_cluster_assignments.copy()
    case_plot_df["pc1"] = case_target_pca[:, 0]
    case_plot_df["pc2"] = case_target_pca[:, 1]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=case_plot_df,
        x="pc1",
        y="pc2",
        hue="cluster_id",
        palette="tab20",
        s=50,
        ax=ax,
    )
    ax.set_title("Case Clusters in Target PCA Space")
    case_cluster_plot_path = args.figure_dir / "case_clustering_target_pca.png"
    save_figure(fig, case_cluster_plot_path)
    register_output(
        "figure",
        "clustering",
        case_cluster_plot_path,
        "PCA projection of the 609 cases using target-space coordinates, colored by case cluster.",
    )

    # 10 output PCA scree
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=target_pca_explained.head(25),
        x="component_index",
        y="cumulative_explained_variance_ratio",
        marker="o",
        ax=ax,
    )
    ax.set_title("Target PCA Cumulative Explained Variance")
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    target_pca_plot_path = args.figure_dir / "target_pca_scree.png"
    save_figure(fig, target_pca_plot_path)
    register_output(
        "figure",
        "low_rank_structure",
        target_pca_plot_path,
        "Cumulative explained-variance curve for PCA applied to log-transformed targets.",
    )

    # 11 baseline model comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=baseline_random_split_metrics,
        x="model_name",
        y="overall_r2",
        color="#6d597a",
        ax=ax,
    )
    ax.set_title("Baseline Model Overall R2 on Random Split")
    ax.set_xlabel("Model")
    ax.set_ylabel("Overall R2")
    baseline_plot_path = args.figure_dir / "baseline_model_comparison.png"
    save_figure(fig, baseline_plot_path)
    register_output(
        "figure",
        "baseline_predictability",
        baseline_plot_path,
        "Bar chart comparing baseline model performance on the random split.",
    )

    # 12 split strategy comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=split_strategy_metrics,
        x="split_strategy",
        y="overall_r2",
        hue="model_name",
        ax=ax,
    )
    ax.set_title("Split Strategy Comparison for Baseline Models")
    ax.set_xlabel("Split Strategy")
    ax.set_ylabel("Overall R2")
    split_plot_path = args.figure_dir / "split_strategy_comparison.png"
    save_figure(fig, split_plot_path)
    register_output(
        "figure",
        "split_strategy_analysis",
        split_plot_path,
        "Model comparison under random case split, density-group holdout, and local-case holdout.",
    )

    # 13 anomaly scatter
    anomaly_plot_df = case_plot_df.merge(
        case_anomaly_scores[["global_case_id", "combined_anomaly_rank_score"]],
        on="global_case_id",
        how="left",
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        anomaly_plot_df["pc1"],
        anomaly_plot_df["pc2"],
        c=anomaly_plot_df["combined_anomaly_rank_score"],
        cmap="magma",
        s=40,
    )
    ax.set_title("Case Anomalies in Target PCA Space")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, label="Combined anomaly rank score")
    anomaly_plot_path = args.figure_dir / "case_anomaly_scatter.png"
    save_figure(fig, anomaly_plot_path)
    register_output(
        "figure",
        "outlier_anomaly_analysis",
        anomaly_plot_path,
        "PCA scatter of cases colored by combined anomaly score.",
    )

    # 14 feature set comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=feature_set_comparison,
        x="feature_set",
        y="overall_r2",
        color="#bc6c25",
        ax=ax,
    )
    ax.set_title("Feature Set Comparison Using Ridge Baseline")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Overall R2")
    ax.tick_params(axis="x", rotation=20)
    feature_set_plot_path = args.figure_dir / "feature_set_comparison.png"
    save_figure(fig, feature_set_plot_path)
    register_output(
        "figure",
        "feature_engineering_analysis",
        feature_set_plot_path,
        "Bar chart comparing engineered feature-set choices under a Ridge baseline.",
    )

    manifest = pd.DataFrame(manifest_rows, columns=OUTPUT_MANIFEST_COLUMNS)
    manifest_path = args.output_dir / "advanced_output_manifest.csv"
    save_csv(manifest, manifest_path)

    print("Advanced analysis completed.")
    print(f"CSV output directory: {args.output_dir}")
    print(f"Figure output directory: {args.figure_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Generated {len(manifest_rows)} tracked outputs.")


if __name__ == "__main__":
    main()
