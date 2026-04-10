from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.models import ModelConfig
from global_kin_ml.pipeline import run_training_experiment


def infer_power_label(raw_file: Path) -> tuple[float, str]:
    match = re.search(r"(\d+(?:[p\.]\d+)?)mJ", raw_file.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not parse power label from filename: {raw_file.name}")
    token = match.group(1).replace("p", ".").replace("P", ".")
    value = float(token)
    return value, f"{value:g}mJ"


def resolve_holdout_power_labels(raw_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    candidates = [infer_power_label(path) for path in raw_dir.glob("*.out")]
    if not candidates:
        raise FileNotFoundError(f"No .out files found in {raw_dir}")
    _value, label = max(candidates, key=lambda item: item[0])
    return [label]


def build_super_rate_curated_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            model_key="direct__ridge__full_nonconstant_plus_log_en_log_power__alpha_10.0",
            model_family="ridge",
            feature_set="full_nonconstant_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"alpha": 10.0},
        ),
        ModelConfig(
            model_key="direct__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
        ),
        ModelConfig(
            model_key="direct__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
        ),
        ModelConfig(
            model_key="two_stage__extra_trees__composition_pca_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="two_stage_extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
        ),
        ModelConfig(
            model_key="two_stage__extra_trees__all_inputs_plus_log_en_log_power__n_200__d_12__leaf_1",
            model_family="two_stage_extra_trees",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
        ),
        ModelConfig(
            model_key="direct__mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={
                "hidden_width": 256,
                "hidden_layers": 3,
                "dropout": 0.0,
                "weight_decay": 1e-5,
            },
        ),
        ModelConfig(
            model_key="latent__extra_trees__composition_pca_plus_log_en_log_power__k_8__n_200__d_12__leaf_1",
            model_family="extra_trees",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=8,
            hyperparameters={"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 1},
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run curated SUPER RATE baseline training experiments on the NERS590 v02 dataset."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "NERS590_data_v02",
        help="Directory containing the power-specific .out files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v02_super_rate",
        help="Root directory for SUPER RATE results.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=None,
        help=(
            "Power labels to hold out for the power_holdout split (example: 5mJ or 10mJ). "
            "If omitted, the highest power found in --raw-dir is used."
        ),
    )
    args = parser.parse_args()

    configs = build_super_rate_curated_configs()
    feature_sets = sorted({config.feature_set for config in configs})
    holdout_power_labels = resolve_holdout_power_labels(args.raw_dir, args.holdout_power_labels)
    holdout_suffix = "_".join(label.replace(".", "p") for label in holdout_power_labels)

    random_results = args.results_root / "training_random_case"
    random_report = (
        REPO_ROOT
        / "docs"
        / "archive_generated_reports"
        / "ners590_v02_super_rate_training_random_case_report.md"
    )
    run_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=random_results,
        report_output_path=random_report,
        feature_sets=feature_sets,
        split_strategy="random_case",
        configs=configs,
        target_name="super_rate",
        drop_constant_targets=True,
    )
    print(f"Completed SUPER RATE random-case training run in {random_results}")

    power_holdout_results = args.results_root / f"training_power_holdout_{holdout_suffix}"
    power_holdout_report = (
        REPO_ROOT
        / "docs"
        / "archive_generated_reports"
        / f"ners590_v02_super_rate_training_power_holdout_{holdout_suffix}_report.md"
    )
    run_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=power_holdout_results,
        report_output_path=power_holdout_report,
        feature_sets=feature_sets,
        split_strategy="power_holdout",
        holdout_power_labels=holdout_power_labels,
        configs=configs,
        target_name="super_rate",
        drop_constant_targets=True,
    )
    print(f"Completed SUPER RATE power-holdout training run in {power_holdout_results}")


if __name__ == "__main__":
    main()
