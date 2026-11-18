"""Hardware and software version record."""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

import jax


def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def git_info(repo: Path | None = None) -> dict:
    cwd = repo or Path(__file__).resolve().parents[2]

    def run(*a):
        return subprocess.run(
            ["git", *a], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "", "dirty": False}


def hardware_info() -> dict:
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
