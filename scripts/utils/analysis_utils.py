from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 220) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def numeric_summary(values: np.ndarray, prefix: str = "", *, include_negative: bool = False) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    summary = {
        f"{prefix}count": float(array.size),
        f"{prefix}min": float(np.min(array)),
        f"{prefix}max": float(np.max(array)),
        f"{prefix}mean": float(np.mean(array)),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}std": float(np.std(array)),
        f"{prefix}p01": float(np.quantile(array, 0.01)),
        f"{prefix}p05": float(np.quantile(array, 0.05)),
        f"{prefix}p25": float(np.quantile(array, 0.25)),
        f"{prefix}p75": float(np.quantile(array, 0.75)),
        f"{prefix}p95": float(np.quantile(array, 0.95)),
        f"{prefix}p99": float(np.quantile(array, 0.99)),
        f"{prefix}zero_count": float(np.sum(array == 0.0)),
        f"{prefix}positive_count": float(np.sum(array > 0.0)),
    }
    if include_negative:
        summary[f"{prefix}negative_count"] = float(np.sum(array < 0.0))
    return summary


def normalize_v03_analysis_path(path: Path, leaf_name: str, *, log_prefix: str) -> Path:
    if path.parent.name == "ners590_v03":
        canonical = path.parent.parent / "ners590_v03_analysis" / leaf_name
        print(
            f"{log_prefix} remapping ambiguous v03 path | requested={path} -> canonical={canonical}",
            flush=True,
        )
        return canonical
    return path
