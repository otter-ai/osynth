"""VERSION file generation for output provenance tracking."""
from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path

from . import fileio


def _package_version() -> str:
    try:
        return version("osynth")
    except PackageNotFoundError:
        return "unknown"


def _git_head_hash() -> str:
    """Commit of the osynth source tree, when running from a git checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_version_file(
    output_dir: str,
    *,
    config_name: str = "",
    run_id: str = "",
    extra: dict | None = None,
) -> Path:
    """Write a VERSION file at the root of the output directory."""
    version_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "osynth_version": _package_version(),
        "git_hash": _git_head_hash(),
        "config_name": config_name,
        "run_id": run_id,
    }
    if extra:
        version_info.update(extra)

    fileio.makedirs(output_dir)
    path = Path(output_dir) / "VERSION"
    fileio.write_text(path, json.dumps(version_info, indent=2) + "\n")
    return path
