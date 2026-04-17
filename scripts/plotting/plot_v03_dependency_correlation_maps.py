from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EPS = 1e-30
N_REACTIONS = 204


def _init_stats() -> dict[str, np.ndarray]:
    zeros = np.zeros(N_REACTIONS, dtype=float)
    return {
        "n_rate_all": zeros.copy(),
        "sx_rate_all_en": zeros.copy(),
        "sx2_rate_all_en": zeros.copy(),
        "sy_rate_all": zeros.copy(),
        "sy2_rate_all": zeros.copy(),
        "sxy_rate_all_en": zeros.copy(),
        "sx_rate_all_pow": zeros.copy(),
        "sx2_rate_all_pow": zeros.copy(),
        "sxy_rate_all_pow": zeros.copy(),
        "n_rate_pos": zeros.copy(),
        "sx_rate_pos_en": zeros.copy(),
        "sx2_rate_pos_en": zeros.copy(),
        "sy_rate_pos": zeros.copy(),
        "sy2_rate_pos": zeros.copy(),
        "sxy_rate_pos_en": zeros.copy(),
        "sx_rate_pos_pow": zeros.copy(),
        "sx2_rate_pos_pow": zeros.copy(),
        "sxy_rate_pos_pow": zeros.copy(),
        "n_super_all": zeros.copy(),
        "sx_super_all_en": zeros.copy(),
        "sx2_super_all_en": zeros.copy(),
        "sy_super_all": zeros.copy(),
        "sy2_super_all": zeros.copy(),
        "sxy_super_all_en": zeros.copy(),
        "sx_super_all_pow": zeros.copy(),
        "sx2_super_all_pow": zeros.copy(),
        "sxy_super_all_pow": zeros.copy(),
        "n_super_pos": zeros.copy(),
        "sx_super_pos_en": zeros.copy(),
        "sx2_super_pos_en": zeros.copy(),
        "sy_super_pos": zeros.copy(),
        "sy2_super_pos": zeros.copy(),
        "sxy_super_pos_en": zeros.copy(),
        "sx_super_pos_pow": zeros.copy(),
        "sx2_super_pos_pow": zeros.copy(),
        "sxy_super_pos_pow": zeros.copy(),
    }


def _update_group_stats(
    *,
    stats: dict[str, np.ndarray],
    rid_idx: np.ndarray,
    x_en: np.ndarray,
    x_pow: np.ndarray,
    y: np.ndarray,
    prefix: str,
) -> None:
    np.add.at(stats[f"n_{prefix}"], rid_idx, 1.0)
    np.add.at(stats[f"sx_{prefix}_en"], rid_idx, x_en)
    np.add.at(stats[f"sx2_{prefix}_en"], rid_idx, x_en * x_en)
    np.add.at(stats[f"sx_{prefix}_pow"], rid_idx, x_pow)
    np.add.at(stats[f"sx2_{prefix}_pow"], rid_idx, x_pow * x_pow)
    np.add.at(stats[f"sy_{prefix}"], rid_idx, y)
    np.add.at(stats[f"sy2_{prefix}"], rid_idx, y * y)
    np.add.at(stats[f"sxy_{prefix}_en"], rid_idx, x_en * y)
    np.add.at(stats[f"sxy_{prefix}_pow"], rid_idx, x_pow * y)


def _corr_from_stats(n: np.ndarray, sx: np.ndarray, sx2: np.ndarray, sy: np.ndarray, sy2: np.ndarray, sxy: np.ndarray) -> np.ndarray:
    num = n * sxy - sx * sy
    den = np.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy))
    corr = np.full_like(num, np.nan, dtype=float)
    valid = (n > 2.0) & np.isfinite(den) & (den > 0.0)
    corr[valid] = num[valid] / den[valid]
    return corr


def _build_correlation_frames(parsed_dir: Path, chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inputs = pd.read_csv(parsed_dir / "training_inputs.csv", usecols=["local_case_id", "e_over_n_v_cm2"])
    local_case_map = inputs.groupby("local_case_id", as_index=False)["e_over_n_v_cm2"].first()
    local_to_en = np.zeros(local_case_map["local_case_id"].max() + 1, dtype=float)
    local_to_en[local_case_map["local_case_id"].to_numpy(dtype=int)] = local_case_map["e_over_n_v_cm2"].to_numpy(dtype=float)

    reaction_map = pd.read_csv(parsed_dir / "reaction_map.csv", usecols=["reaction_id", "reaction_label"])
    reaction_map = reaction_map.sort_values("reaction_id").reset_index(drop=True)

    stats = _init_stats()

    global_rows = {
        "rate_all": {"n": 0.0, "sx_en": 0.0, "sx2_en": 0.0, "sx_pow": 0.0, "sx2_pow": 0.0, "sy": 0.0, "sy2": 0.0, "sxy_en": 0.0, "sxy_pow": 0.0},
        "super_all": {"n": 0.0, "sx_en": 0.0, "sx2_en": 0.0, "sx_pow": 0.0, "sx2_pow": 0.0, "sy": 0.0, "sy2": 0.0, "sxy_en": 0.0, "sxy_pow": 0.0},
    }

    usecols = ["reaction_id", "local_case_id", "power_mj", "rate_const", "super_rate_cc_per_s"]
    for chunk in pd.read_csv(parsed_dir / "rate_constants_long.csv", usecols=usecols, chunksize=chunksize):
        rid_idx = chunk["reaction_id"].to_numpy(dtype=int) - 1
        local_case_id = chunk["local_case_id"].to_numpy(dtype=int)
        x_en = np.log10(local_to_en[local_case_id])
        x_pow = np.log10(chunk["power_mj"].to_numpy(dtype=float))

        rate = chunk["rate_const"].to_numpy(dtype=float)
        super_rate = chunk["super_rate_cc_per_s"].to_numpy(dtype=float)
        y_rate_all = np.log10(rate + EPS)
        y_super_all = np.log10(super_rate + EPS)

        _update_group_stats(stats=stats, rid_idx=rid_idx, x_en=x_en, x_pow=x_pow, y=y_rate_all, prefix="rate_all")
        _update_group_stats(stats=stats, rid_idx=rid_idx, x_en=x_en, x_pow=x_pow, y=y_super_all, prefix="super_all")

        rate_pos_mask = rate > 0.0
        if np.any(rate_pos_mask):
            _update_group_stats(
                stats=stats,
                rid_idx=rid_idx[rate_pos_mask],
                x_en=x_en[rate_pos_mask],
                x_pow=x_pow[rate_pos_mask],
                y=np.log10(rate[rate_pos_mask]),
                prefix="rate_pos",
            )

        super_pos_mask = super_rate > 0.0
        if np.any(super_pos_mask):
            _update_group_stats(
                stats=stats,
                rid_idx=rid_idx[super_pos_mask],
                x_en=x_en[super_pos_mask],
                x_pow=x_pow[super_pos_mask],
                y=np.log10(super_rate[super_pos_mask]),
                prefix="super_pos",
            )

        for key, y_all in (("rate_all", y_rate_all), ("super_all", y_super_all)):
            g = global_rows[key]
            g["n"] += float(len(y_all))
            g["sx_en"] += float(np.sum(x_en))
            g["sx2_en"] += float(np.sum(x_en * x_en))
            g["sx_pow"] += float(np.sum(x_pow))
            g["sx2_pow"] += float(np.sum(x_pow * x_pow))
            g["sy"] += float(np.sum(y_all))
            g["sy2"] += float(np.sum(y_all * y_all))
            g["sxy_en"] += float(np.sum(x_en * y_all))
            g["sxy_pow"] += float(np.sum(x_pow * y_all))

    rate_frame = reaction_map.copy()
    rate_frame["corr_log_en_all"] = _corr_from_stats(
        stats["n_rate_all"], stats["sx_rate_all_en"], stats["sx2_rate_all_en"], stats["sy_rate_all"], stats["sy2_rate_all"], stats["sxy_rate_all_en"]
    )
    rate_frame["corr_log_power_all"] = _corr_from_stats(
        stats["n_rate_all"], stats["sx_rate_all_pow"], stats["sx2_rate_all_pow"], stats["sy_rate_all"], stats["sy2_rate_all"], stats["sxy_rate_all_pow"]
    )
    rate_frame["corr_log_en_positive_only"] = _corr_from_stats(
        stats["n_rate_pos"], stats["sx_rate_pos_en"], stats["sx2_rate_pos_en"], stats["sy_rate_pos"], stats["sy2_rate_pos"], stats["sxy_rate_pos_en"]
    )
    rate_frame["corr_log_power_positive_only"] = _corr_from_stats(
        stats["n_rate_pos"], stats["sx_rate_pos_pow"], stats["sx2_rate_pos_pow"], stats["sy_rate_pos"], stats["sy2_rate_pos"], stats["sxy_rate_pos_pow"]
    )
    rate_frame["positive_count"] = stats["n_rate_pos"].astype(int)

    super_frame = reaction_map.copy()
    super_frame["corr_log_en_all"] = _corr_from_stats(
        stats["n_super_all"], stats["sx_super_all_en"], stats["sx2_super_all_en"], stats["sy_super_all"], stats["sy2_super_all"], stats["sxy_super_all_en"]
    )
    super_frame["corr_log_power_all"] = _corr_from_stats(
        stats["n_super_all"], stats["sx_super_all_pow"], stats["sx2_super_all_pow"], stats["sy_super_all"], stats["sy2_super_all"], stats["sxy_super_all_pow"]
    )
    super_frame["corr_log_en_positive_only"] = _corr_from_stats(
        stats["n_super_pos"], stats["sx_super_pos_en"], stats["sx2_super_pos_en"], stats["sy_super_pos"], stats["sy2_super_pos"], stats["sxy_super_pos_en"]
    )
    super_frame["corr_log_power_positive_only"] = _corr_from_stats(
        stats["n_super_pos"], stats["sx_super_pos_pow"], stats["sx2_super_pos_pow"], stats["sy_super_pos"], stats["sy2_super_pos"], stats["sxy_super_pos_pow"]
    )
    super_frame["positive_count"] = stats["n_super_pos"].astype(int)

    global_summary_rows = []
    for target_name, g in global_rows.items():
        corr_en = _corr_from_stats(
            np.array([g["n"]]), np.array([g["sx_en"]]), np.array([g["sx2_en"]]), np.array([g["sy"]]), np.array([g["sy2"]]), np.array([g["sxy_en"]])
        )[0]
        corr_pow = _corr_from_stats(
            np.array([g["n"]]), np.array([g["sx_pow"]]), np.array([g["sx2_pow"]]), np.array([g["sy"]]), np.array([g["sy2"]]), np.array([g["sxy_pow"]])
        )[0]
        global_summary_rows.append(
            {
                "target": target_name,
                "entry_count": int(g["n"]),
                "corr_log_en_all": float(corr_en),
                "corr_log_power_all": float(corr_pow),
            }
        )
    global_summary = pd.DataFrame(global_summary_rows)
    return rate_frame, super_frame, global_summary


def _plot_control_corr_heatmap(
    *,
    frame: pd.DataFrame,
    title: str,
    sort_col: str,
    output_path: Path,
    top_n: int = 80,
) -> None:
    ranked = frame.copy()
    ranked["abs_sort"] = ranked[sort_col].abs().fillna(0.0)
    ranked = ranked.sort_values("abs_sort", ascending=False).head(top_n).reset_index(drop=True)
    labels = ranked["reaction_id"].map(lambda x: f"R{int(x):03d}")

    matrix = np.vstack(
        [
            ranked["corr_log_en_all"].to_numpy(dtype=float),
            ranked["corr_log_power_all"].to_numpy(dtype=float),
            ranked["corr_log_en_positive_only"].to_numpy(dtype=float),
            ranked["corr_log_power_positive_only"].to_numpy(dtype=float),
        ]
    )

    fig, ax = plt.subplots(figsize=(16, 4.8))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_yticklabels(
        [
            "corr(logE/N, log(target+eps)) [all]",
            "corr(logPower, log(target+eps)) [all]",
            "corr(logE/N, log(target)) [positive-only]",
            "corr(logPower, log(target)) [positive-only]",
        ],
        rotation=0,
    )
    tick_positions = np.arange(0, len(labels), 5)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(labels.iloc[tick_positions], rotation=90)
    ax.set_xlabel("Reactions (top by |corr with logE/N|)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v03 correlation maps showing dependence of RATE/SUPER outputs on E/N and power."
    )
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("results/ners590_v03_analysis/parsed"),
        help="Directory containing parsed tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ners590_v03_analysis/presentation_figures"),
        help="Directory for generated dependence maps and CSV summaries.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=600000,
        help="CSV chunk size for long-table processing.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="talk")

    print(f"[corr-map] building control dependence maps from {args.parsed_dir}", flush=True)
    rate_frame, super_frame, global_summary = _build_correlation_frames(args.parsed_dir, args.chunksize)

    rate_csv = args.output_dir / "rate_control_dependence_correlations.csv"
    super_csv = args.output_dir / "super_control_dependence_correlations.csv"
    global_csv = args.output_dir / "global_control_dependence_summary.csv"
    rate_frame.to_csv(rate_csv, index=False)
    super_frame.to_csv(super_csv, index=False)
    global_summary.to_csv(global_csv, index=False)

    _plot_control_corr_heatmap(
        frame=rate_frame,
        title="RATE CONST Reaction Dependence on log(E/N) and log(Power)",
        sort_col="corr_log_en_all",
        output_path=args.output_dir / "rate_control_dependence_heatmap_top80.png",
    )
    _plot_control_corr_heatmap(
        frame=super_frame,
        title="SUPER RATE Reaction Dependence on log(E/N) and log(Power)",
        sort_col="corr_log_en_positive_only",
        output_path=args.output_dir / "super_control_dependence_heatmap_top80.png",
    )

    print(f"[corr-map] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
