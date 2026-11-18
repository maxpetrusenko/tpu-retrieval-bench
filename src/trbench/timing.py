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
    for _ in range(warmup):
        jax.block_until_ready(compiled(*args))
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        out = compiled(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return {
        "compile_time_s": compile_s,
        "times_s": times,
        "latency_ms": summarize(times),
        "compiled": compiled,
    }
