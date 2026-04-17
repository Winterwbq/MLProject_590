from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_src_on_path(repo_root: Path | None = None, *, include_scripts: bool = False) -> Path:
    root = repo_root or REPO_ROOT
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if include_scripts:
        scripts_dir = root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
    return root


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


def build_holdout_suffix(labels: list[str]) -> str:
    return "_".join(label.replace(".", "p") for label in labels)


def remap_ambiguous_v03_results_root(
    *,
    raw_dir: Path,
    results_root: Path,
    canonical_name: str,
    log_prefix: str = "[runner]",
) -> Path:
    if raw_dir.name == "NERS590_data_V03" and results_root.name == "ners590_v03":
        canonical = results_root.parent / canonical_name
        print(
            f"{log_prefix} remapping ambiguous v03 results root | "
            f"requested={results_root} -> canonical={canonical}",
            flush=True,
        )
        return canonical
    return results_root


def run_subprocess_step(*, cmd: list[str], cwd: Path, log_prefix: str) -> None:
    print(f"{log_prefix} running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)
