"""Timing harness: AOT compile measured apart from warm repetitions."""

from __future__ import annotations

import time

import jax
import numpy as np


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
