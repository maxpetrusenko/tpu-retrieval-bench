"""JSON result records (schema: docs/presearch/benchmarks.md section 10)."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from trbench import roofline

# 0.2: spread fields (stdev, bootstrap CIs) required, headline `qps` is the median pass (with
# `qps_best` kept for the fastest pass). 0.1 records lack the spread and their `qps` is the best pass.
SCHEMA_VERSION = "0.2"
MEDIAN_QPS_SCHEMA = (0, 2)  # first schema whose headline `qps` is the median-pass QPS


def schema_tuple(rec: dict[str, Any]) -> tuple[int, ...] | None:
    """schema_version "0.2" -> (0, 2); None when missing or unparsable."""
    try:
        return tuple(int(x) for x in str(rec.get("schema_version")).split("."))
    except ValueError:
        return None


def qps_basis(rec: dict[str, Any]) -> str:
    """Statistic behind the headline `qps`: "median" (schema >= 0.2) or "best" (older)."""
    ver = schema_tuple(rec)
    return "median" if ver is not None and ver >= MEDIAN_QPS_SCHEMA else "best"


def qps_best(rec: dict[str, Any]) -> float:
    """QPS of the fastest pass: `qps_best` (0.2+), else the legacy headline `qps`."""
    m = rec["metrics"]
    return float(m["qps_best"] if qps_basis(rec) == "median" else m["qps"])


def comparison_basis(recs: Iterable[dict[str, Any]]) -> str:
    """Basis on which `recs` are comparable: "median" only when every record is median-based,
    else "best" (the one statistic every schema carries)."""
    return "median" if all(qps_basis(r) == "median" for r in recs) else "best"


def qps_on(rec: dict[str, Any], basis: str) -> float:
    """QPS of `rec` on a common basis from comparison_basis(); never mixes min with median."""
    return qps_best(rec) if basis == "best" else float(rec["metrics"]["qps"])


def peak_flops(device_kind: str) -> float | None:
    """Peak bf16 FLOP/s from the single table in roofline.PEAKS; None if unknown."""
    p = roofline.peak_for(device_kind)
    return p.bf16_flops if p else None


def build_record(
    *,
    dataset: dict[str, Any],
    algorithm: dict[str, Any],
    query: dict[str, Any],
    hardware: dict[str, Any],
    metrics: dict[str, Any],
    timing_extra: dict[str, Any] | None = None,
    git: dict[str, Any],
    matmul: bool = True,
    compute_dtype: str = "bf16",
) -> dict[str, Any]:
    """matmul=False (select_only): no matmul FLOP/s or percent_of_peak, all set to null.

    The peak table is bf16 only, so f32 compute gets flops_peak / percent_of_peak = null.
    """
    best_s = metrics["best_search_time_s"]
    m, n, d = dataset["n_test"], dataset["n_train"], dataset["dim"]
    flops = 2.0 * m * n * d / best_s if matmul else None
    peak = peak_flops(hardware["device_kind"]) if matmul and compute_dtype == "bf16" else None
    metrics = dict(metrics)
    metrics.update(
        flops_achieved=flops,
        flops_peak=peak,
        percent_of_peak=(100.0 * flops / peak) if flops and peak else None,
    )
    timing = {
        "clock": "perf_counter",
        "synchronized": "block_until_ready",
        "includes_h2d_query_transfer": False,
        "includes_d2h_result": False,
    }
    timing.update(timing_extra or {})
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{now}-{git.get('commit', '')[:7]}",
        "dataset": dataset,
        "algorithm": algorithm,
        "query": query,
        "hardware": hardware,
        "metrics": metrics,
        "timing": timing,
        "git": git,
        "created_at": now,
    }


def result_path(
    root: str | Path, platform: str, dataset: str, k: int, algo: str, params: str
) -> Path:
    return Path(root) / platform / dataset / f"k{k}" / algo / f"{params}.json"


def write_record(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
