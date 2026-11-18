"""Timing harness: AOT compile measured apart from warm repetitions."""

from __future__ import annotations

import glob
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def compile_aot(fn, *args):
    """AOT lower + compile. Returns (compiled, compile_time_s)."""
    jitted = fn if hasattr(fn, "lower") else jax.jit(fn)
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    return compiled, time.perf_counter() - t0


def summarize(times_s) -> dict:
    ms = np.asarray(times_s) * 1e3
    return {
        "min": float(ms.min()),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "mean": float(ms.mean()),
    }


def time_fn(fn, *args, warmup=3, repetitions=5):
    """Return dict: compile_time_s, times_s (list), latency_ms (min/p50/p95/p99/mean), compiled."""
    if warmup < 3 or repetitions < 5:
        raise ValueError("need warmup >= 3 and repetitions >= 5")
    compiled, compile_s = compile_aot(fn, *args)
    return {"compile_time_s": compile_s, **time_compiled(compiled, *args, warmup=warmup,
                                                         repetitions=repetitions)}


def time_compiled(compiled, *args, warmup=3, repetitions=5):
    """Warm + time an already compiled fn. Returns times_s, latency_ms, compiled."""
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        out = compiled(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return {
        "times_s": times,
        "latency_ms": summarize(times),
        "compiled": compiled,
    }


def _shape_key(args):
    return tuple(tuple(x.shape) for x in jax.tree_util.tree_leaves(args))


def time_batched(fn, make_args, n_batches, *, warmup=3, repetitions=5, keep_out=True):
    """Time a full pass over n_batches query batches, `repetitions` times.

    make_args(i) builds the args for batch i untimed (its device work is synchronized before the
    clock starts). Each distinct arg-shape set is AOT-compiled once and warmed `warmup` times.
    times_s[r] = wall time of full pass r (sum of its batch times); QPS = n_queries / min.
    Returns compile_time_s, times_s, latency_ms (summary over individual batch times),
    outs (outputs of the last pass, or None when keep_out is False).
    """
    if warmup < 3 or repetitions < 5:
        raise ValueError("need warmup >= 3 and repetitions >= 5")
    compiled, compile_s = {}, 0.0
    for i in range(n_batches):
        args = make_args(i)
        key = _shape_key(args)
        if key not in compiled:
            compiled[key], dt = compile_aot(fn, *args)
            compile_s += dt
            for _ in range(warmup):
                jax.block_until_ready(compiled[key](*args))
    passes, batch_times, outs = [], [], None
    for _ in range(repetitions):
        total, outs = 0.0, []
        for i in range(n_batches):
            args = make_args(i)
            jax.block_until_ready(args)
            t0 = time.perf_counter()
            out = compiled[_shape_key(args)](*args)
            jax.block_until_ready(out)
            dt = time.perf_counter() - t0
            total += dt
            batch_times.append(dt)
            if keep_out:
                outs.append(out)
        passes.append(total)
    return {
        "compile_time_s": compile_s,
        "times_s": passes,
        "latency_ms": summarize(batch_times),
        "outs": outs if keep_out else None,
    }


# ---- in-jit repeat loop (device-amortized) ----------------------------------------------------
# Motivation: tokamax docs/benchmarking.md warns that jax.block_until_ready wall time on small
# shapes is mostly Python dispatch overhead; triton do_bench_cudagraph divides an unrolled burst.
# Here the search runs `inner` times inside one jit; the host dispatches once.

LOOP_CHECKSUM = "sum_all_output_leaves"


def _checksum(out):
    leaves = jax.tree_util.tree_leaves(out)
    return sum(jnp.sum(x.astype(jnp.float32)) for x in leaves)


def make_loop_fn(fn, inner):
    """Return jit(*args -> f32 scalar) running fn(*args) `inner` times in a lax.fori_loop.

    XLA must not hoist or CSE the repeated search, nor drop it as dead code:
    - every iteration's args pass through lax.optimization_barrier together with the running
      checksum, so iteration i+1's inputs depend on iteration i's outputs (true data dependency);
    - the carry is a checksum over every output leaf, so no output is dead.
    Cost: one reduction pass over the outputs per iteration (visible for score_only, whose output
    is the full M x N score matrix); recorded as timing.loop_checksum.
    """
    if inner < 1:
        raise ValueError("inner must be >= 1")

    def run(*args):
        def body(_, acc):
            a, _ = lax.optimization_barrier((args, acc))
            return acc + _checksum(fn(*a))

        return lax.fori_loop(0, inner, body, jnp.float32(0.0))

    return jax.jit(run)


def time_loop(fn, *args, inner=16, warmup=3, repetitions=5):
    """Per-iteration device-amortized time: each rep is one dispatch of `inner` iterations.

    times_s[r] = wall time of rep r / inner. Returns compile_time_s, times_s, latency_ms, inner.
    """
    if warmup < 3 or repetitions < 5:
        raise ValueError("need warmup >= 3 and repetitions >= 5")
    compiled, compile_s = compile_aot(make_loop_fn(fn, inner), *args)
    res = time_compiled(compiled, *args, warmup=warmup, repetitions=repetitions)
    times = [t / inner for t in res["times_s"]]
    return {"compile_time_s": compile_s, "times_s": times, "latency_ms": summarize(times),
            "inner": inner}


def time_batched_loop(fn, make_args, n_batches, *, inner=16, warmup=3, repetitions=5):
    """Loop-mode twin of time_batched: times_s[r] = sum over batches of per-iteration time."""
    if warmup < 3 or repetitions < 5:
        raise ValueError("need warmup >= 3 and repetitions >= 5")
    compiled, compile_s = {}, 0.0
    for i in range(n_batches):
        args = make_args(i)
        key = _shape_key(args)
        if key not in compiled:
            compiled[key], dt = compile_aot(make_loop_fn(fn, inner), *args)
            compile_s += dt
            for _ in range(warmup):
                jax.block_until_ready(compiled[key](*args))
    passes, batch_times = [], []
    for _ in range(repetitions):
        total = 0.0
        for i in range(n_batches):
            args = make_args(i)
            jax.block_until_ready(args)
            t0 = time.perf_counter()
            jax.block_until_ready(compiled[_shape_key(args)](*args))
            dt = (time.perf_counter() - t0) / inner
            total += dt
            batch_times.append(dt)
        passes.append(total)
    return {"compile_time_s": compile_s, "times_s": passes, "latency_ms": summarize(batch_times),
            "inner": inner}


# ---- batch-1 serial latency (VectorDBBench serial_latency_*) -----------------------------------


def time_serial(fn, make_args, n_queries, *, warmup=3):
    """One query at a time, concurrency 1: n_queries timed calls, each ended by block_until_ready.

    make_args(i) builds the args for query i (i = 0..n_queries-1; the caller may cycle its query
    pool) untimed and synchronized before the clock starts. Returns times_s (len n_queries),
    latency_ms (p50/p95/p99/min/mean), compile_time_s.
    """
    if warmup < 3 or n_queries < 1:
        raise ValueError("need warmup >= 3 and n_queries >= 1")
    first = make_args(0)
    compiled, compile_s = compile_aot(fn, *first)
    for _ in range(warmup):
        jax.block_until_ready(compiled(*first))
    times = []
    for i in range(n_queries):
        args = make_args(i)
        jax.block_until_ready(args)
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        times.append(time.perf_counter() - t0)
    return {"compile_time_s": compile_s, "times_s": times, "latency_ms": summarize(times)}


# ---- profiler device time -----------------------------------------------------------------------

DEVICE_LINE = "XLA Modules"


def device_time_from_profile(profile):
    """Sum 'XLA Modules' event durations (ms) on the first /device: plane of a ProfileData.

    Returns (total_ms, n_module_events) or (None, 0) when the trace has no device plane
    (CPU traces only carry host planes). Duck-typed on planes/lines/events/duration_ns.
    """
    for plane in profile.planes:
        if not str(plane.name).startswith("/device:"):
            continue
        durs = [e.duration_ns for ln in plane.lines if ln.name == DEVICE_LINE for e in ln.events]
        if durs:
            return float(sum(durs)) / 1e6, len(durs)
    return None, 0


def device_time(fn, *args, calls=5):
    """Trace `calls` warm calls with jax.profiler and return per-call device time.

    Returns dict(device_time_ms | None, device_time_source, device_time_note | None). Parsed
    offline from the written .xplane.pb via jax.profiler.ProfileData (no TensorBoard needed).
    Includes only the first /device: plane (one chip). Off-TPU/GPU there is no device plane:
    returns None with a note.
    """
    try:
        from jax.profiler import ProfileData
    except ImportError:
        return {"device_time_ms": None, "device_time_source": None,
                "device_time_note": "jax.profiler.ProfileData unavailable in this jax"}
    jax.block_until_ready(fn(*args))  # warm outside the trace
    with tempfile.TemporaryDirectory() as d:
        with jax.profiler.trace(d):
            for _ in range(calls):
                jax.block_until_ready(fn(*args))
        files = glob.glob(f"{d}/**/*.xplane.pb", recursive=True)
        if not files:
            return {"device_time_ms": None, "device_time_source": None,
                    "device_time_note": "profiler wrote no xplane.pb"}
        total_ms, n_events = device_time_from_profile(ProfileData.from_file(files[0]))
    if total_ms is None:
        return {"device_time_ms": None, "device_time_source": None,
                "device_time_note": f"no /device: plane with '{DEVICE_LINE}' events "
                                    f"(platform {jax.default_backend()})"}
    return {"device_time_ms": total_ms / calls, "device_time_source": "xplane_xla_modules",
            "device_time_note": f"{n_events} module events over {calls} calls"}
