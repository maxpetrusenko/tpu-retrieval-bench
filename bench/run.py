"""CLI sweep: runs algos over shapes, writes results/<platform>/<dataset>/k<k>/<algo>/<params>.json.

--dataset synthetic (default) sweeps --m/--n/--d. An ann-benchmarks name takes shapes and metric
from the HDF5 file (--n-train/--n-test subset it); recall uses the file ground truth.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp

from trbench import data, datasets, env, records, reference, roofline, timing
from trbench.approx import make_approx
from trbench.blockwise import make_blockwise, make_score_only, make_select_only
from trbench.jit_topk import make_jit_topk
from trbench.pallas_score import make_pallas_score

DTYPES = {"bf16": jnp.bfloat16, "f32": jnp.float32}
REF_LIMIT = 4e8  # M*N*D above this skips the numpy oracle recall check


def ints(s):
    return [int(x) for x in s.split(",")]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", default="jit_topk", help="comma list: numpy_ref,jit_topk,blockwise,pallas_score,approx_max_k,"
                   "score_only,select_only")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", *datasets.DATASETS], help="synthetic or ann-benchmarks")
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
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--block-size", type=int, default=65536, help="blockwise db block")
    p.add_argument("--bm", type=int, default=256)
    p.add_argument("--bn", type=int, default=512)
    p.add_argument("--bk", type=int, default=512)
    p.add_argument("--chunk", type=int, default=65536, help="pallas_score selection chunk")
    p.add_argument("--recall-target", type=float, default=0.95)
    p.add_argument("--sweep-blocks", action="store_true",
                   help="pallas_score: sweep bm/bn/bk grid, one JSON each")
    p.add_argument("--dump-hlo", action="store_true")
    return p.parse_args(argv)


MATERIALIZING = roofline.MATERIALIZING | {"score_only", "select_only"}
SWEEP = {"bm": [128, 256, 512], "bn": [256, 512, 1024], "bk": [128, 256, 512]}


def _jit(k, dtype, a, bp):
    return make_jit_topk(k, metric=a.metric, compute_dtype=dtype), "jit_matmul_topk", True, {}


def _blockwise(k, dtype, a, bp):
    f = make_blockwise(k, metric=a.metric, compute_dtype=dtype, block_size=a.block_size)
    return f, "blockwise_topk", True, {"block_size": a.block_size}


def _pallas(k, dtype, a, bp):
    bm, bn, bk = bp
    f = make_pallas_score(k, metric=a.metric, compute_dtype=dtype, bm=bm, bn=bn, bk=bk,
                          chunk=a.chunk)
    return f, "pallas_score_topk", True, {"bm": bm, "bn": bn, "bk": bk, "chunk": a.chunk,
                                          "interpret": jax.default_backend() != "tpu"}


def _approx(k, dtype, a, bp):
    f = make_approx(k, recall_target=a.recall_target, metric=a.metric, compute_dtype=dtype)
    return f, "approx_max_k", False, {"recall_target": a.recall_target}


def _score_only(k, dtype, a, bp):
    return make_score_only(metric=a.metric, compute_dtype=dtype), "score_only", True, {}


def _select_only(k, dtype, a, bp):
    return make_select_only(k), "select_only", True, {}


FACTORIES = {"jit_topk": _jit, "blockwise": _blockwise, "pallas_score": _pallas,
             "approx_max_k": _approx, "score_only": _score_only, "select_only": _select_only}


def get_inputs(a, m, n, d, ann=None):
    """(queries, db, truth, dataset_hash). truth = ann dict or None (synthetic)."""
    if ann is not None:
        return ann["test"], ann["train"], ann, ann["hash"]
    q, e, _ = data.synthetic(m, n, d, seed=a.seed)
    return q, e, None, data.data_hash(q, e)


def compute_recall(a, k, q, ann, got_scores, ref_scores):
    """Dataset: ann-benchmarks knn_recall vs file truth; synthetic: threshold recall vs oracle."""
    if got_scores is None:
        return None
    if ann is not None:
        dist = datasets.scores_to_distances(got_scores, a.metric, q)
        return datasets.knn_recall(dist, ann["distances"], k)
    if ref_scores is None:
        return None
    return data.recall_at_k(ref_scores, got_scores, eps=1e-2 if a.dtype == "bf16" else 1e-4)


def run_one(algo, m, n, d, k, a, hw, git, bp=None, ann=None):
    q, e, truth, dhash = get_inputs(a, m, n, d, ann)
    m, n = q.shape[0], e.shape[0]
    dtype = DTYPES[a.dtype]
    ref_scores = None
    if truth is None and m * n * d <= REF_LIMIT:
        ref_scores, _ = reference.numpy_ref(q, e, k, metric=a.metric)
    hlo = None
    if algo == "numpy_ref":
        import time

        times = []
        for _ in range(a.warmup + a.reps):
            t0 = time.perf_counter()
            got, _ = reference.numpy_ref(q, e, k, metric=a.metric)
            times.append(time.perf_counter() - t0)
        times = times[a.warmup :]
        res = {"compile_time_s": 0.0, "times_s": times, "latency_ms": timing.summarize(times)}
        got_scores, exact, name, params = got, True, "numpy_ref", {}
    elif algo in FACTORIES:
        qd, ed = jax.device_put(q), jax.device_put(e)  # db resident on device
        f, name, exact, params = FACTORIES[algo](k, dtype, a, bp)
        args = (qd, ed)
        if algo == "select_only":  # precomputed scores resident on device
            args = (make_score_only(metric=a.metric, compute_dtype=dtype)(qd, ed),)
        res = timing.time_fn(f, *args, warmup=a.warmup, repetitions=a.reps)
        out = jax.device_get(res["compiled"](*args))
        got_scores = None if algo == "score_only" else out[0]
        if a.dump_hlo:
            hlo = res["compiled"].as_text()
    else:
        raise SystemExit(f"unknown algo {algo}")
    recall = compute_recall(a, k, q, truth, got_scores, ref_scores)
    best = min(res["times_s"])
    in_roofline = algo in roofline.VARIANTS
    dkind = hw["device_kind"]
    rf_bytes = roofline.bytes_moved(algo, m, n, d, k, a.dtype) if in_roofline else None
    rf_ai = roofline.arithmetic_intensity(algo, m, n, d, k, a.dtype) if in_roofline else None
    rf_label = roofline.label(algo, m, n, d, k, a.dtype, dkind) if in_roofline else "unknown"
    has_matmul = algo != "select_only"
    rec = records.build_record(
        dataset={
            "name": a.dataset, "source_url": None, "n_train": n, "n_test": m, "dim": d,
            "distance": a.metric, "dtype_stored": "float32", "dtype_compute": a.dtype,
            "seed": a.seed, "hash": dhash,
        },
        algorithm={"name": name, "params": {"precision": a.dtype, **params}, "exact": exact},
        query={"k": k, "batch_size": m, "mode": "batch", "n_queries_timed": m,
               "warmup_iters": a.warmup, "repetitions": a.reps},
        hardware=hw,
        metrics={
            "recall": recall,
            "qps": m / best,
            "best_search_time_s": best,
            "search_time_s_all": res["times_s"],
            "latency_ms_per_batch": res["latency_ms"],
            "latency_ms_per_query_amortized": res["latency_ms"]["p50"] / m,
            "compile_time_s": res["compile_time_s"],
            "index_size_bytes": int(e.nbytes),
            "score_matrix_bytes": 4 * m * n if algo in MATERIALIZING else None,
            "bytes_moved": rf_bytes,
            "arithmetic_intensity": rf_ai,
            "roofline_label": rf_label,
            "selection_time_ms": best * 1e3 if algo == "select_only" else None,
            "matmul_time_ms": best * 1e3 if algo == "score_only" else None,
            "selection_over_matmul": None,
        },
        git=git,
        timing_extra={"xla_flags": __import__("os").environ.get("XLA_FLAGS", "")},
        matmul=has_matmul,
    )
    if has_matmul:
        rec["metrics"]["percent_of_peak"] = roofline.percent_of_peak(
            rec["metrics"]["flops_achieved"], dkind
        )
    params = f"m{m}_n{n}_d{d}_{a.metric}_{a.dtype}"
    if algo == "blockwise":
        params += f"_B{a.block_size}"
    elif algo == "pallas_score":
        params += f"_bm{bp[0]}_bn{bp[1]}_bk{bp[2]}"
    elif algo == "approx_max_k":
        params += f"_r{a.recall_target}"
    path = records.write_record(
        records.result_path(a.out, hw["platform"], a.dataset, k, algo, params), rec
    )
    if hlo is not None:
        (path.parent / f"{params}.hlo.txt").write_text(hlo)
    return path, rec


def load_dataset(a):
    """Load the ann-benchmarks file once; metric comes from the dataset, not --metric."""
    name = a.dataset
    ann = datasets.load_ann(
        Path(a.data_dir) / f"{name}.hdf5" if a.data_dir else name,
        n_train=a.n_train, n_test=a.n_test,
    )
    ann["hash"] = data.data_hash(ann["train"], ann["test"])
    a.metric = ann["metric"]
    return ann


def link_stages(stage, a, dkind):
    """R5: when score_only and select_only ran on the same shape, write the ratio into both."""
    for recs in stage.values():
        if "score_only" not in recs or "select_only" not in recs:
            continue
        (sp, sc), (lp, se) = recs["score_only"], recs["select_only"]
        t_mm = sc["metrics"]["best_search_time_s"]
        t_sel = se["metrics"]["best_search_time_s"]
        for path, rec, algo in ((sp, sc, "score_only"), (lp, se, "select_only")):
            mt = rec["metrics"]
            mt["selection_time_ms"], mt["matmul_time_ms"] = t_sel * 1e3, t_mm * 1e3
            mt["selection_over_matmul"] = t_sel / t_mm
            ds = rec["dataset"]
            mt["roofline_label"] = roofline.label(
                algo, ds["n_test"], ds["n_train"], ds["dim"], rec["query"]["k"], a.dtype, dkind,
                select_time_s=t_sel, matmul_time_s=t_mm,
            )
            records.write_record(path, rec)


def main(argv=None):
    a = parse(argv)
    hw, git = env.hardware_info(), env.git_info(Path(__file__).resolve().parents[1])
    ann = load_dataset(a) if a.dataset != "synthetic" else None
    shapes = [(len(ann["test"]), len(ann["train"]), ann["train"].shape[1])] if ann else None
    stage = {}
    for algo, k in itertools.product(a.algos.split(","), a.k):
        for m, n, d in shapes or itertools.product(a.m, a.n, a.d):
            bps = [(a.bm, a.bn, a.bk)]
            if algo == "pallas_score" and a.sweep_blocks:
                bps = list(itertools.product(SWEEP["bm"], SWEEP["bn"], SWEEP["bk"]))
            for bp in bps:
                path, rec = run_one(algo, m, n, d, k, a, hw, git, bp, ann)
                report(algo, path, rec)
                if algo in ("score_only", "select_only"):
                    stage.setdefault((m, n, d, k), {})[algo] = (path, rec)
    link_stages(stage, a, hw["device_kind"])
    for recs in stage.values():
        if len(recs) == 2:
            r = recs["select_only"][1]["metrics"]["selection_over_matmul"]
            print(f"selection_over_matmul={r:.3f}")


def report(algo, path, rec):
    mt = rec["metrics"]
    fl = mt["flops_achieved"]
    print(
        f"{path}\n  algo={algo} compile_time_s={mt['compile_time_s']:.4f} "
        f"warm_p50_ms={mt['latency_ms_per_batch']['p50']:.4f} recall={mt['recall']} "
        f"flops={'null' if fl is None else format(fl, '.3e')} pct_peak={mt['percent_of_peak']} "
        f"roofline={mt['roofline_label']}"
    )


if __name__ == "__main__":
    main()
