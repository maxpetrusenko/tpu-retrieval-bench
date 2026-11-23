"""Record writing: turn a Measurement into the JSON record, write it, link stage records."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from bench.algos import FUSED, MATERIALIZING, BlockParams
from bench.cli import Args
from bench.execute import Measurement
from bench.plan import params_str
from trbench import records, roofline, timing
from trbench.pallas_score import resolve_blocks

Record = dict[str, Any]


def timing_metrics(res: dict[str, Any], m: int) -> dict[str, Any]:
    """Record fields for the wall / loop / serial / device-time lanes (None when not run).

    qps_wall / qps_loop are median-pass QPS like the headline `qps` (schema 0.2+); the fastest
    pass of a lane is best_search_time_s_wall / best_search_time_s_loop.
    """
    wall, loop = res.get("wall"), res.get("loop")
    if wall is None and loop is None:  # numpy_ref: wall only
        wall = res
    bw = min(wall["times_s"]) if wall else None
    bl = min(loop["times_s"]) if loop else None
    ser, dev = res.get("serial"), res.get("device") or {}
    sl = ser["latency_ms"] if ser else None
    return {
        "best_search_time_s_wall": bw,
        "best_search_time_s_loop": bl,
        "qps_wall": timing.median_qps(wall["times_s"], m) if wall else None,
        "qps_loop": timing.median_qps(loop["times_s"], m) if loop else None,
        "search_time_s_all_loop": loop["times_s"] if loop else None,
        "latency_ms_per_batch_loop": loop["latency_ms"] if loop else None,
        "serial_latency_ms": sl,
        "serial_n": len(ser["times_s"]) if ser else None,
        "serial_qps": 1e3 / sl["mean"] if sl else None,
        "device_time_ms": dev.get("device_time_ms"),
        "device_time_source": dev.get("device_time_source"),
        "device_time_note": dev.get("device_time_note"),
    }


def _roofline_fields(meas: Measurement, a: Args, device_kind: str) -> tuple[Any, Any, str]:
    """(bytes_moved, arithmetic_intensity, label); None / "unknown" off the roofline model."""
    algo, n, d, k = meas.algo, meas.n, meas.d, meas.k
    rf_algo = "pallas_fused" if algo in FUSED else algo
    if rf_algo not in roofline.VARIANTS:
        return None, None, "unknown"
    # db is stored in compute dtype after prepare; queries arrive as f32
    dt: dict[str, Any] = {"q_dtype": "f32", "db_dtype": a.dtype}
    if algo in FUSED:  # candidates written per db tile: M * n_tiles * round_up(k', 128) * 8
        bp = meas.params
        bn = resolve_blocks(meas.batch_size, n, d, bp["bm"], bp["bn"], bp["bk"], a.chunk)[1]
        dt.update(n_tiles=-(-n // bn), kprime=bp.get("kprime", bp.get("lanes")))
    bs = meas.batch_size
    rf_bytes = sum(
        roofline.bytes_moved(rf_algo, s, n, d, k, a.dtype, **dt) for s in meas.batch_sizes
    )
    return (
        rf_bytes,
        roofline.arithmetic_intensity(rf_algo, bs, n, d, k, a.dtype, **dt),
        roofline.label(rf_algo, bs, n, d, k, a.dtype, device_kind, **dt),
    )


def build_run_record(meas: Measurement, a: Args, hw: dict[str, Any], git: dict[str, Any]) -> Record:
    """The schema-0.2 JSON record for one measurement.

    Headline `qps` is the median-pass QPS and `qps_ci95` bootstraps that same statistic, so the
    headline always sits inside its CI; the fastest pass is kept as `qps_best`.
    """
    algo, m, n, d, k, res = meas.algo, meas.m, meas.n, meas.d, meas.k, meas.res
    best = min(res["times_s"])
    mode = "wall" if algo == "numpy_ref" else a.timing
    spread = timing.spread_metrics(res["times_s"], m, res["latency_ms"])
    tm = {**timing_metrics(res, m), **spread}
    rf_bytes, rf_ai, rf_label = _roofline_fields(meas, a, hw["device_kind"])
    in_factories = algo != "numpy_ref"
    bs = meas.batch_size
    return records.build_record(
        dataset={
            "name": a.dataset,
            "source_url": None,
            "n_train": n,
            "n_test": m,
            "dim": d,
            "distance": a.metric,
            "dtype_stored": a.dtype if in_factories else "float32",
            "dtype_compute": a.dtype,
            "seed": a.seed,
            "hash": meas.dataset_hash,
        },
        algorithm={
            "name": meas.name,
            "params": {"precision": a.dtype, **meas.params},
            "exact": meas.exact,
        },
        query={
            "k": k,
            "batch_size": bs,
            "n_batches": len(meas.batch_sizes),
            "mode": "batch",
            "n_queries_timed": m,
            "warmup_iters": a.warmup,
            "repetitions": a.reps,
        },
        hardware=hw,
        metrics={
            "recall": meas.recall,
            "recall_sampled_queries": meas.recall_queries,
            "recall_epsilon_0.01": meas.recall_eps2,
            "recall_compute_dtype": a.dtype,
            "qps": spread["qps_median"],
            "qps_best": m / best,
            "best_search_time_s": best,
            "search_time_s_all": res["times_s"],
            "latency_ms_per_batch": res["latency_ms"],
            "latency_ms_per_query_amortized": best * 1e3 / m,
            "compile_time_s": res["compile_time_s"],
            "build_time_s": meas.prepare_s,
            "index_size_bytes": meas.index_bytes,
            "score_matrix_bytes": 4 * bs * n if algo in MATERIALIZING else None,
            "bytes_moved": rf_bytes,
            "arithmetic_intensity": rf_ai,
            "roofline_label": rf_label,
            "selection_time_ms": best * 1e3 if algo == "select_only" else None,
            "matmul_time_ms": best * 1e3 if algo == "score_only" else None,
            "selection_over_matmul": None,
            **tm,
        },
        git=git,
        timing_extra={
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "mode": mode,
            "headline": "loop" if mode == "loop" else "wall",
            "loop_iters": a.loop_iters if mode in ("loop", "both") else None,
            "loop_checksum": timing.LOOP_CHECKSUM if mode in ("loop", "both") else None,
            "serial_n": a.serial or None,
            "device_time": bool(a.device_time),
        },
        matmul=algo != "select_only",
        compute_dtype=a.dtype,
    )


def record_path(
    a: Args, hw: dict[str, Any], algo: str, k: int, m: int, n: int, d: int, bp: BlockParams | None
) -> Path:
    return records.result_path(
        a.out, hw["platform"], a.dataset, k, algo, params_str(algo, m, n, d, a, bp)
    )


def write_run_record(
    a: Args, hw: dict[str, Any], meas: Measurement, bp: BlockParams | None, rec: Record
) -> Path:
    path = record_path(a, hw, meas.algo, meas.k, meas.m, meas.n, meas.d, bp)
    return records.write_record(path, rec)


def write_error(
    a: Args,
    algo: str,
    m: int,
    n: int,
    d: int,
    k: int,
    bp: BlockParams | None,
    hw: dict[str, Any],
    exc: BaseException,
) -> Path:
    return records.write_record(
        record_path(a, hw, algo, k, m, n, d, bp),
        {
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "algo": algo,
            "shape": {"m": m, "n": n, "d": d, "k": k},
            "batch_size": min(a.batch_size, m),
        },
    )


def link_stages(
    stage: dict[tuple[int, ...], dict[str, tuple[Path, Record]]],
    exact: dict[tuple[int, ...], list[tuple[Path, Record]]],
    a: Args,
    dkind: str,
) -> None:
    """R5: when score_only and select_only ran on the same shape, write the ratio into both.

    select_only is labelled from the ratio; score_only never is (it has no selection). Exact
    full-search variants on the same shape are relabelled VPU/COP-bound when the ratio is > 1.
    """
    dt = {"q_dtype": "f32", "db_dtype": a.dtype}
    for key, recs in stage.items():
        if "score_only" not in recs or "select_only" not in recs:
            continue
        (sp, sc), (lp, se) = recs["score_only"], recs["select_only"]
        t_mm = sc["metrics"]["best_search_time_s"]
        t_sel = se["metrics"]["best_search_time_s"]
        ratio = t_sel / t_mm
        for path, rec in ((sp, sc), (lp, se)):
            mt = rec["metrics"]
            mt["selection_time_ms"], mt["matmul_time_ms"] = t_sel * 1e3, t_mm * 1e3
            mt["selection_over_matmul"] = ratio
        ds = se["dataset"]
        se["metrics"]["roofline_label"] = roofline.label(
            "select_only",
            se["query"]["batch_size"],
            ds["n_train"],
            ds["dim"],
            key[3],
            a.dtype,
            dkind,
            select_time_s=t_sel,
            matmul_time_s=t_mm,
            **dt,
        )
        records.write_record(sp, sc)
        records.write_record(lp, se)
        if ratio > 1:
            for path, rec in exact.get(key, []):
                rec["metrics"]["roofline_label"] = "VPU/COP-bound"
                records.write_record(path, rec)


def report(algo: str, path: Path, rec: Record) -> None:
    mt = rec["metrics"]
    fl = mt["flops_achieved"]
    print(
        f"{path}\n  algo={algo} compile_time_s={mt['compile_time_s']:.4f} "
        f"warm_p50_ms={mt['latency_ms_per_batch']['p50']:.4f} recall={mt['recall']} "
        f"flops={'null' if fl is None else format(fl, '.3e')} pct_peak={mt['percent_of_peak']} "
        f"roofline={mt['roofline_label']}"
    )
