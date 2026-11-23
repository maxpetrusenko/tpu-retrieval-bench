"""Command-line arguments for bench.run (the sweep driver)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from bench.algos import DTYPES
from trbench import datasets, reference, timing
from trbench.pallas_fused import LANES

Args = argparse.Namespace

DESCRIPTION = """CLI sweep: runs algos over shapes, writes results/<platform>/<dataset>/k<k>/<algo>/<params>.json.

--dataset synthetic (default) sweeps --m/--n/--d. An ann-benchmarks name takes shapes and metric
from the HDF5 file (--n-train/--n-test subset it); recall uses the file ground truth.
"""


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def reps_arg(s: str) -> int:
    n = int(s)
    if n < timing.MIN_REPS:
        raise argparse.ArgumentTypeError(f"--reps must be >= {timing.MIN_REPS}, got {n}")
    return n


def parse(argv: Sequence[str] | None = None) -> Args:
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument(
        "--algos",
        default="jit_topk",
        help="comma list: numpy_ref,jit_topk,blockwise,pallas_score,"
        "pallas_fused_topk,pallas_fused_bucket,approx_max_k,score_only,select_only",
    )
    p.add_argument(
        "--kprime",
        type=int,
        default=None,
        help="pallas_fused_topk: per-tile candidates (default k = exact; < k approximate)",
    )
    p.add_argument("--lanes", type=int, default=LANES, help="pallas_fused_bucket: bins per tile")
    p.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", *datasets.DATASETS],
        help="synthetic or ann-benchmarks",
    )
    p.add_argument("--data-dir", default=None, help="dir with <dataset>.hdf5 (default repo data/)")
    p.add_argument("--n-train", type=int, default=None, help="dataset: use first N train rows")
    p.add_argument("--n-test", type=int, default=None, help="dataset: use first N test rows")
    p.add_argument("--m", type=ints, default=[16])
    p.add_argument("--n", type=ints, default=[65536])
    p.add_argument("--d", type=ints, default=[128])
    p.add_argument("--k", type=ints, default=[10])
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--metric", default="ip", choices=list(reference.METRICS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--reps",
        type=reps_arg,
        default=timing.DEFAULT_REPS,
        help=f"timed repetitions per record (default {timing.DEFAULT_REPS}, minimum "
        f"{timing.MIN_REPS}); spread and CIs in the record come from these",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run on a dirty git tree (recorded as git.dirty=true); refused otherwise",
    )
    p.add_argument("--out", default="results")
    p.add_argument("--block-size", type=int, default=65536, help="blockwise db block")
    p.add_argument("--bm", type=int, default=256)
    p.add_argument("--bn", type=int, default=512)
    p.add_argument("--bk", type=int, default=512)
    p.add_argument("--chunk", type=int, default=65536, help="pallas_score selection chunk")
    p.add_argument("--recall-target", type=float, default=0.95)
    p.add_argument(
        "--sweep-blocks",
        action="store_true",
        help="pallas_score / pallas_fused_*: sweep bm/bn/bk grid, one JSON each",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="queries per search call; a pass chunks M queries into batches so "
        "scores stay <= ~4 GB. QPS = all queries / pass time",
    )
    p.add_argument(
        "--timing",
        default="wall",
        choices=["wall", "loop", "both"],
        help="wall = per-call block_until_ready (headline); loop = in-jit fori_loop "
        "repeat, device-amortized per iteration (headline); both = wall headline, "
        "loop recorded next to it",
    )
    p.add_argument(
        "--loop-iters", type=int, default=16, help="iterations per dispatch in loop mode"
    )
    p.add_argument(
        "--serial",
        type=int,
        default=0,
        help="also time N queries one at a time (batch 1, concurrency 1); records "
        "p50/p95/p99 per query",
    )
    p.add_argument(
        "--device-time",
        action="store_true",
        help="trace a few warm calls with jax.profiler and record device time (null off-TPU/GPU)",
    )
    p.add_argument("--dump-hlo", action="store_true")
    return p.parse_args(argv)
