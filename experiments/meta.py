"""Per-run provenance record (README Section 11): git commit, config, seeds,
library versions -> results/<exp_id>/meta.json."""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = [
    "numpy", "scipy", "pandas", "networkx", "simpy", "highspy",
    "gymnasium", "stable-baselines3", "sb3-contrib", "matplotlib", "pyyaml",
]


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _library_versions() -> dict[str, str]:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def write_meta(out_dir: str | Path, exp_id: str, config: dict[str, Any], seeds: list[int]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "exp_id": exp_id,
        "git_commit": _git_commit_hash(),
        "config": config,
        "seeds": seeds,
        "library_versions": _library_versions(),
    }
    path = out_dir / "meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
