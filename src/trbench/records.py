"""JSON result records (schema: docs/presearch/benchmarks.md section 10)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from trbench import roofline

SCHEMA_VERSION = "0.2"  # 0.2: spread fields (stdev, bootstrap CIs) required; 0.1 records lack them


def peak_flops(device_kind: str):
    """Peak bf16 FLOP/s from the single table in roofline.PEAKS; None if unknown."""
    p = roofline.peak_for(device_kind)
    return p.bf16_flops if p else None


def build_record(*, dataset, algorithm, query, hardware, metrics, timing_extra=None, git,
                 matmul=True, compute_dtype="bf16"):
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


def result_path(root, platform, dataset, k, algo, params: str) -> Path:
    return Path(root) / platform / dataset / f"k{k}" / algo / f"{params}.json"


def write_record(path, record) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
