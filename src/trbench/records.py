"""JSON result records (schema: docs/presearch/benchmarks.md section 10)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

PEAK_BF16_FLOPS = {"v5e": 197e12, "v6e": 920e12}
SCHEMA_VERSION = "0.1"


def peak_flops(device_kind: str):
    """Peak bf16 FLOP/s from a small table keyed on device_kind text; None if unknown."""
    kind = device_kind.lower()
    if "v5 lite" in kind or "v5e" in kind or "v5litepod" in kind:
        return PEAK_BF16_FLOPS["v5e"]
    if "v6" in kind:
        return PEAK_BF16_FLOPS["v6e"]
    return None


def build_record(*, dataset, algorithm, query, hardware, metrics, timing_extra=None, git):
    best_s = metrics["best_search_time_s"]
    m, n, d = dataset["n_test"], dataset["n_train"], dataset["dim"]
    flops = 2.0 * m * n * d / best_s
    peak = peak_flops(hardware["device_kind"])
    metrics = dict(metrics)
    metrics.update(
        flops_achieved=flops,
        flops_peak=peak,
        percent_of_peak=(100.0 * flops / peak) if peak else None,
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
