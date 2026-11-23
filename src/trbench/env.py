"""Hardware and software version record."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def git_info(repo: Path | None = None, ignore: Sequence[str] = ("results",)) -> dict[str, Any]:
    """commit + dirty flag. `ignore` paths (relative to the repo) do not count as dirty: the
    benchmark's own output directory is not a change to the code under test."""
    cwd = repo or Path(__file__).resolve().parents[2]
    spec = [".", *(f":(exclude){p}" for p in ignore)]

    def run(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        dirty = bool(run("status", "--porcelain", "--", *spec))
        return {"commit": run("rev-parse", "HEAD"), "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "", "dirty": False}


def hardware_info() -> dict[str, Any]:
    devs = jax.devices()
    d = devs[0]
    return {
        "platform": d.platform,
        "device_kind": d.device_kind,
        "n_devices": len(devs),
        "host": platform.node(),
        "jax_version": jax.__version__,
        "jaxlib_version": _ver("jaxlib"),
        "libtpu_version": _ver("libtpu") or _ver("libtpu-nightly"),
        "python": sys.version.split()[0],
    }
