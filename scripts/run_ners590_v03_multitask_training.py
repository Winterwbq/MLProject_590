from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from global_kin_ml.models import ModelConfig
from global_kin_ml.multitask_pipeline import run_multitask_training_experiment


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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


def build_multitask_configs() -> list[ModelConfig]:
    common = {
        "hidden_layers": 3,
        "dropout": 0.0,
        "weight_decay": 1e-5,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "max_epochs": 120,
        "patience": 15,
    }
    return [
        ModelConfig(
            model_key="joint__single_head_mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_single_head_mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, **common},
        ),
        ModelConfig(
            model_key="joint__single_head_mlp__composition_pca_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_single_head_mlp",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, **common},
        ),
        ModelConfig(
            model_key="joint__two_head_mlp__all_inputs_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_two_head_mlp",
            feature_set="all_inputs_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, "rate_loss_weight": 0.5, "super_loss_weight": 0.5, **common},
        ),
        ModelConfig(
            model_key="joint__two_head_mlp__composition_pca_plus_log_en_log_power__w_256__layers_3__drop_0.0__wd_1e-05",
            model_family="joint_two_head_mlp",
            feature_set="composition_pca_plus_log_en_log_power",
            latent_k=None,
            hyperparameters={"hidden_width": 256, "rate_loss_weight": 0.5, "super_loss_weight": 0.5, **common},
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multitask v03 experiments that predict RATE CONST and SUPER RATE jointly."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "NERS590_data_V03",
        help="Directory containing the v03 raw .out files.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ners590_v03_multitask",
        help="Canonical results root for multitask v03 outputs.",
    )
    parser.add_argument(
        "--holdout-power-labels",
        nargs="+",
        default=None,
        help="Optional explicit holdout power labels. Defaults to the highest available power.",
    )
    args = parser.parse_args()

    configs = build_multitask_configs()
    feature_sets = sorted({config.feature_set for config in configs})
    holdout_power_labels = resolve_holdout_power_labels(args.raw_dir, args.holdout_power_labels)
    holdout_suffix = "_".join(label.replace(".", "p") for label in holdout_power_labels)
    run_start = time.perf_counter()

    print(
        "[runner] multitask v03 start | "
        f"raw_dir={args.raw_dir} | results_root={args.results_root} | holdout={holdout_power_labels}",
        flush=True,
    )

    random_results = args.results_root / "training_random_case"
    split_start = time.perf_counter()
    print(f"[runner] start split=random_case -> {random_results}", flush=True)
    run_multitask_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=random_results,
        feature_sets=feature_sets,
        configs=configs,
        split_strategy="random_case",
    )
    print(
        f"[runner] completed split=random_case | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={random_results}",
        flush=True,
    )

    holdout_results = args.results_root / f"training_power_holdout_{holdout_suffix}"
    split_start = time.perf_counter()
    print(
        f"[runner] start split=power_holdout ({','.join(holdout_power_labels)}) -> {holdout_results}",
        flush=True,
    )
    run_multitask_training_experiment(
        raw_dataset_path=args.raw_dir,
        results_dir=holdout_results,
        feature_sets=feature_sets,
        configs=configs,
        split_strategy="power_holdout",
        holdout_power_labels=holdout_power_labels,
    )
    print(
        f"[runner] completed split=power_holdout | elapsed={format_seconds(time.perf_counter() - split_start)} | "
        f"path={holdout_results}",
        flush=True,
    )

    print(
        f"[runner] multitask v03 done | total_elapsed={format_seconds(time.perf_counter() - run_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

