"""Execution: build inputs, time one (algo, shape) job, score recall. No record writing here."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import numpy.typing as npt

from bench.algos import DTYPES, FACTORIES, STAGE_ONLY, BlockParams, build
from bench.cli import Args
from trbench import data, datasets, reference, timing
from trbench.timing import block_until_ready

REF_LIMIT = 4e8  # M*N*D above this skips the numpy oracle recall check
SAMPLE_QUERIES = 64  # oracle queries when the full numpy oracle is too big


@dataclass
class Measurement:
    """Everything a record needs from one timed job."""

    algo: str
    name: str  # record algorithm name
    exact: bool
    params: dict[str, Any]
    k: int
    m: int  # queries actually timed
    n: int
    d: int
    batch_size: int
    batch_sizes: list[int]  # size of each query batch in a pass
    res: dict[str, Any]  # timing result: times_s, latency_ms, wall/loop/serial/device lanes
    prepare_s: float
    index_bytes: int
    dataset_hash: str
    recall: float | None
    recall_queries: int | None
    recall_eps2: float | None
    queries: npt.NDArray[np.float32]
    db: npt.NDArray[np.float32]


def get_inputs(a: Args, m: int, n: int, d: int, ann: dict[str, Any] | None = None) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], dict[str, Any] | None, str]:
    """(queries, db, truth, dataset_hash). truth = ann dict or None (synthetic)."""
    if ann is not None:
        return ann["test"], ann["train"], ann, ann["hash"]
    q, e, _ = data.synthetic(m, n, d, seed=a.seed)
    return q, e, None, data.data_hash(q, e)


def compute_recall(a: Args, k: int, q: Any, e: Any, ann: dict[str, Any] | None,
                   got_scores: Any) -> tuple[float | None, int | None, float | None]:
    """(recall, n_queries_used, recall_epsilon_0.01). Dataset: knn_recall vs file truth, eps 1e-3
    (`recall`) and 1e-2 (`recall_epsilon_0.01`, ann-benchmarks "epsilon" metric). bf16 scoring
    perturbs distances by ~1e-3, so exact bf16 variants can fall below 1.0 at eps 1e-3 only.
    Synthetic: threshold recall vs the numpy oracle (eps 1e-2 bf16 / 1e-4 f32), on a sampled
    subset of queries when M*N*D > REF_LIMIT; no second recall."""
    if got_scores is None:
        return None, None, None
    m = q.shape[0]
    if ann is not None:
        dist = datasets.scores_to_distances(got_scores, a.metric, q)
        return (datasets.knn_recall(dist, ann["distances"], k), m,
                datasets.knn_recall(dist, ann["distances"], k, eps=1e-2))
    idx = np.arange(m)
    if m * e.shape[0] * e.shape[1] > REF_LIMIT:
        idx = np.unique(np.linspace(0, m - 1, min(m, SAMPLE_QUERIES)).astype(int))
    ref_scores, _ = reference.numpy_ref(q[idx], e, k, metric=a.metric)
    rec = data.recall_at_k(ref_scores, np.asarray(got_scores)[idx],
                           eps=1e-2 if a.dtype == "bf16" else 1e-4)
    return rec, len(idx), None


def time_numpy(q: Any, e: Any, k: int, a: Args) -> tuple[dict[str, Any], Any]:
    times = []
    for _ in range(a.warmup + a.reps):
        t0 = time.perf_counter()
        got, _ = reference.numpy_ref(q, e, k, metric=a.metric)
        times.append(time.perf_counter() - t0)
    times = times[a.warmup :]
    res: dict[str, Any] = {"compile_time_s": 0.0, "times_s": times,
                           "latency_ms": timing.summarize(times)}
    if a.serial:
        st = []
        for i in range(a.serial):
            t0 = time.perf_counter()
            reference.numpy_ref(q[i % len(q) : i % len(q) + 1], e, k, metric=a.metric)
            st.append(time.perf_counter() - t0)
        res["serial"] = {"times_s": st, "latency_ms": timing.summarize(st)}
    return res, got


def time_jax(algo: str, q: Any, e: Any, k: int, a: Args, bp: BlockParams | None) -> tuple[
        dict[str, Any], Any, str, bool, dict[str, Any], float, int, list[int]]:
    """Prepare once (untimed index build, timed separately), then time batched search."""
    dtype = DTYPES[a.dtype]
    m = q.shape[0]
    bs = min(a.batch_size, m)
    qd, ed = jax.device_put(q), jax.device_put(e)  # db resident on device
    prep_fn, search_fn, name, exact, params = build(algo, k, dtype, a, bp, e.shape[0])
    block_until_ready(prep_fn(ed))  # compile + run once
    t0 = time.perf_counter()
    index = prep_fn(ed)
    block_until_ready(index)
    prepare_s = time.perf_counter() - t0
    qb = [qd[i : i + bs] for i in range(0, m, bs)]
    score_fn = params.pop("score_fn", None)
    if score_fn is not None:  # select_only: scores built untimed per batch
        def make_args(i: int) -> tuple[Any, ...]:
            return (score_fn(qb[i], index),)
    else:
        def make_args(i: int) -> tuple[Any, ...]:
            return (qb[i], index)
    keep = algo not in STAGE_ONLY
    wall = loop = None
    if a.timing in ("wall", "both"):
        wall = timing.time_batched(search_fn, make_args, len(qb), warmup=a.warmup,
                                   repetitions=a.reps, keep_out=keep)
    if a.timing in ("loop", "both"):
        loop = timing.time_batched_loop(search_fn, make_args, len(qb), inner=a.loop_iters,
                                        warmup=a.warmup, repetitions=a.reps)
    if wall is not None:
        res = dict(wall)
    else:  # loop headline: the loop returns a checksum, so outputs come from one untimed pass
        assert loop is not None
        outs = [search_fn(*make_args(i)) for i in range(len(qb))] if keep else None
        res = {**loop, "outs": outs}
    res["wall"], res["loop"] = wall, loop
    if a.serial:
        def serial_args(i: int) -> tuple[Any, ...]:
            q1 = qd[i % m : i % m + 1]
            return (score_fn(q1, index),) if score_fn is not None else (q1, index)
        res["serial"] = timing.time_serial(search_fn, serial_args, a.serial, warmup=a.warmup)
    if a.device_time:
        res["device"] = timing.device_time(search_fn, *make_args(0), calls=a.reps)
    got = None
    if res["outs"] is not None:
        got = np.concatenate([jax.device_get(o[0]) for o in res["outs"]], axis=0)
    size = sum(x.nbytes for x in jax.tree_util.tree_leaves(index))
    return res, got, name, exact, params, prepare_s, int(size), [len(x) for x in qb]


def measure(algo: str, m: int, n: int, d: int, k: int, a: Args, bp: BlockParams | None = None,
            ann: dict[str, Any] | None = None) -> Measurement:
    """Time one job and score its recall. Unknown algos exit with an error."""
    q, e, truth, dhash = get_inputs(a, m, n, d, ann)
    m, n = q.shape[0], e.shape[0]
    bs = min(a.batch_size, m)
    sizes, prepare_s, size = [m], 0.0, int(e.nbytes)
    params: dict[str, Any]
    if algo == "numpy_ref":
        res, got_scores = time_numpy(q, e, k, a)
        name, exact, params = "numpy_ref", True, {}
    elif algo in FACTORIES:
        res, got_scores, name, exact, params, prepare_s, size, sizes = time_jax(
            algo, q, e, k, a, bp)
    else:
        raise SystemExit(f"unknown algo {algo}")
    recall, n_recall_q, recall_eps2 = compute_recall(a, k, q, e, truth, got_scores)
    return Measurement(algo, name, exact, params, k, m, n, d, bs, sizes, res, prepare_s, size,
                       dhash, recall, n_recall_q, recall_eps2, q, e)


def compile_hlo(algo: str, q: Any, e: Any, k: int, a: Args, bp: BlockParams | None) -> str:
    """HLO text of the search step for the first batch."""
    prep_fn, search_fn, *_ = build(algo, k, DTYPES[a.dtype], a, bp, e.shape[0])
    index = prep_fn(jax.device_put(e))
    qb = jax.device_put(q[: min(a.batch_size, q.shape[0])])
    if algo == "select_only":
        score_fn = build(algo, k, DTYPES[a.dtype], a, bp, e.shape[0])[4]["score_fn"]
        return str(search_fn.lower(score_fn(qb, index)).compile().as_text())
    return str(search_fn.lower(qb, index).compile().as_text())


def is_oom(exc: BaseException) -> bool:
    return (type(exc).__name__ == "XlaRuntimeError" or "RESOURCE_EXHAUSTED" in str(exc)
            or isinstance(exc, MemoryError))
