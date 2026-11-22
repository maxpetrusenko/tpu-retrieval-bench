"""Plausibility guard for result JSON records (anti reward-hack, KernelBench style).

A record that claims more than physics or its own definition allows is rejected: peak above
100 percent, recall outside [0, 1], an exact variant that is not exact, non-positive QPS,
non-monotone latency quantiles, QPS or latency that disagree with each other, or missing keys.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

TOP_KEYS = ("schema_version", "run_id", "dataset", "algorithm", "query", "hardware", "metrics",
            "timing", "git", "created_at")
DATASET_KEYS = ("name", "n_train", "n_test", "dim", "distance", "dtype_compute")
ALGO_KEYS = ("name", "params", "exact")
QUERY_KEYS = ("k", "batch_size", "mode", "n_queries_timed")
HW_KEYS = ("platform", "device_kind", "n_devices")
METRIC_KEYS = ("recall", "qps", "best_search_time_s", "search_time_s_all",
               "latency_ms_per_batch", "compile_time_s", "index_size_bytes", "flops_achieved",
               "flops_peak", "percent_of_peak")
LATENCY_KEYS = ("min", "p50", "p95", "p99")
# variants that time a stage only and carry no result ids to score
NO_RECALL_ALGOS = ("score_only", "select_only")
SPREAD_SCHEMA = (0, 2)  # schema_version at which the spread fields became required
SPREAD_KEYS = ("n_repetitions", "latency_stdev_ms", "latency_p50_ci95_ms", "qps_median",
               "qps_stdev", "qps_ci95")
EXACT_RECALL_TOL = 1e-9
MIN_REPS = 5
REL_TOL = 0.05  # qps vs best time, latency min vs best time


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _missing(obj, keys, where, errs):
    if not isinstance(obj, dict):
        errs.append(f"{where}: expected object")
        return False
    for k in keys:
        if k not in obj:
            errs.append(f"{where}: missing key {k!r}")
    return True


def _check_recall(rec, errs):
    m, algo = rec["metrics"], rec["algorithm"]
    recall = m.get("recall")
    stage_only = str(algo.get("name")) in NO_RECALL_ALGOS
    if recall is None:
        if algo.get("exact") and not stage_only:
            errs.append("metrics.recall: null for an exact variant that returns results")
        return
    if not _num(recall) or not 0.0 <= recall <= 1.0:
        errs.append(f"metrics.recall: {recall!r} not in [0, 1]")
        return
    if algo.get("exact") and not stage_only:
        # dtype rule (docs/BENCHMARK_METHOD.md section 3): bf16 rounding can drop recall at
        # eps 1e-3 on dataset runs, so bf16 may fall back to the eps 0.01 recall; f32 may not.
        best = recall
        if rec["dataset"].get("dtype_compute") == "bf16":
            eps01 = m.get("recall_epsilon_0.01")
            if _num(eps01):
                best = max(best, eps01)
        if best < 1.0 - EXACT_RECALL_TOL:
            errs.append(f"metrics.recall: exact variant has recall {recall} < 1.0")


def _check_timing(rec, errs):
    m = rec["metrics"]
    qps, best = m.get("qps"), m.get("best_search_time_s")
    if not _num(qps) or qps <= 0:
        errs.append(f"metrics.qps: {qps!r} must be > 0")
    if not _num(best) or best <= 0:
        errs.append(f"metrics.best_search_time_s: {best!r} must be > 0")
    times = m.get("search_time_s_all")
    if not isinstance(times, list) or not times or not all(_num(t) and t > 0 for t in times):
        errs.append("metrics.search_time_s_all: must be a non-empty list of positive numbers")
    elif _num(best) and best > 0 and abs(min(times) - best) > 1e-12 + 1e-6 * best:
        errs.append("metrics.best_search_time_s: not equal to min(search_time_s_all)")
    lat = m.get("latency_ms_per_batch")
    if _missing(lat, LATENCY_KEYS, "metrics.latency_ms_per_batch", errs) and all(
            k in lat for k in LATENCY_KEYS):
        vals = [lat[k] for k in LATENCY_KEYS]
        if not all(_num(v) and v > 0 for v in vals):
            errs.append("metrics.latency_ms_per_batch: quantiles must be positive numbers")
        elif not all(a <= b for a, b in pairwise(vals)):
            errs.append(f"metrics.latency_ms_per_batch: not monotone min<=p50<=p95<=p99: {vals}")
        elif _num(best) and best > 0 and abs(lat["min"] - best * 1e3) > REL_TOL * best * 1e3:
            errs.append("metrics.latency_ms_per_batch.min: disagrees with best_search_time_s")
    nq = rec["query"].get("n_queries_timed")
    ok = _num(qps) and qps > 0 and _num(best) and best > 0 and _num(nq) and nq > 0
    if ok and abs(qps - nq / best) > REL_TOL * (nq / best):
        errs.append(f"metrics.qps: {qps} disagrees with n_queries_timed/best_search_time_s")


def _check_peak(rec, errs):
    m = rec["metrics"]
    pct = m.get("percent_of_peak")
    if pct is not None and (not _num(pct) or not 0.0 <= pct <= 100.0):
        errs.append(f"metrics.percent_of_peak: {pct!r} not in [0, 100]")
    fa, fp = m.get("flops_achieved"), m.get("flops_peak")
    if _num(fa) and _num(fp) and fp > 0 and fa > fp:
        errs.append("metrics: flops_achieved exceeds flops_peak")
    if pct is not None and not (_num(fa) and _num(fp)):
        errs.append("metrics.percent_of_peak: set without flops_achieved and flops_peak")


def schema_tuple(rec):
    """schema_version "0.2" -> (0, 2); None when missing or unparsable."""
    try:
        return tuple(int(x) for x in str(rec.get("schema_version")).split("."))
    except ValueError:
        return None


def is_dirty(rec) -> bool:
    return bool(isinstance(rec, dict) and isinstance(rec.get("git"), dict)
                and rec["git"].get("dirty"))


def _interval(x, where, errs):
    ok = (isinstance(x, list) and len(x) == 2 and all(_num(v) and v > 0 for v in x)
          and x[0] <= x[1])
    if not ok:
        errs.append(f"{where}: expected [lo, hi] positive numbers with lo <= hi, got {x!r}")


def _check_spread(rec, errs):
    """Schema >= 0.2 records must carry stdev and bootstrap CIs; 0.1 records are exempt."""
    ver = schema_tuple(rec)
    if ver is None or ver < SPREAD_SCHEMA:
        return
    m = rec["metrics"]
    for k in SPREAD_KEYS:
        if k not in m:
            errs.append(f"metrics: missing key {k!r} (required from schema_version 0.2)")
    if any(k not in m for k in SPREAD_KEYS):
        return
    n = m["n_repetitions"]
    if not isinstance(n, int) or n < MIN_REPS:
        errs.append(f"metrics.n_repetitions: {n!r} must be an int >= {MIN_REPS}")
    times = m.get("search_time_s_all")
    if isinstance(times, list) and n != len(times):
        errs.append("metrics.n_repetitions: differs from len(search_time_s_all)")
    for k in ("latency_stdev_ms", "qps_stdev"):
        if not _num(m[k]) or m[k] < 0:
            errs.append(f"metrics.{k}: {m[k]!r} must be a number >= 0")
    _interval(m["latency_p50_ci95_ms"], "metrics.latency_p50_ci95_ms", errs)
    _interval(m["qps_ci95"], "metrics.qps_ci95", errs)
    if not _num(m["qps_median"]) or m["qps_median"] <= 0:
        errs.append("metrics.qps_median: must be > 0")


def validate_record(rec) -> list[str]:
    """Return a list of problems; empty means the record is plausible."""
    errs: list[str] = []
    if not _missing(rec, TOP_KEYS, "record", errs):
        return errs
    ok = [
        _missing(rec.get("dataset"), DATASET_KEYS, "dataset", errs),
        _missing(rec.get("algorithm"), ALGO_KEYS, "algorithm", errs),
        _missing(rec.get("query"), QUERY_KEYS, "query", errs),
        _missing(rec.get("hardware"), HW_KEYS, "hardware", errs),
        _missing(rec.get("metrics"), METRIC_KEYS, "metrics", errs),
    ]
    if not all(ok) or any(k not in rec for k in TOP_KEYS):
        return errs
    _check_recall(rec, errs)
    _check_timing(rec, errs)
    _check_peak(rec, errs)
    _check_spread(rec, errs)
    return errs


def iter_records(root):
    """Yield every *.json under root (file or directory), sorted."""
    root = Path(root)
    yield from ([root] if root.is_file() else sorted(root.rglob("*.json")))


def validate_tree(root) -> dict[str, list[str]]:
    """Map path -> problems for every JSON under root that has any (empty dict = all good)."""
    bad: dict[str, list[str]] = {}
    for path in iter_records(root):
        try:
            errs = validate_record(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as ex:
            errs = [f"unreadable: {ex}"]
        if errs:
            bad[str(path)] = errs
    return bad
