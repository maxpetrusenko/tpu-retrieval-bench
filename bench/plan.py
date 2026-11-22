"""Sweep planning: which (algo, k, shape, block config) jobs run, and their record file names."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.algos import PALLAS_ALGOS, BlockParams
from bench.cli import Args
from trbench import data, datasets
from trbench.pallas_score import resolve_blocks

SWEEP = {"bm": [128, 256, 512], "bn": [256, 512, 1024], "bk": [128, 256, 512]}


@dataclass(frozen=True)
class Job:
    """One benchmark record to produce. m, n, d are the requested shape (dataset runs: file)."""

    algo: str
    k: int
    m: int
    n: int
    d: int
    bp: BlockParams | None


def load_dataset(a: Args) -> dict[str, Any]:
    """Load the ann-benchmarks file once; metric comes from the dataset, not --metric."""
    name = a.dataset
    ann = datasets.load_ann(
        Path(a.data_dir) / f"{name}.hdf5" if a.data_dir else name,
        n_train=a.n_train, n_test=a.n_test,
    )
    ann["hash"] = data.data_hash(ann["train"], ann["test"])
    a.metric = ann["metric"]
    return ann


def block_params(algo: str, a: Args, bm_cap: int, n: int, d: int) -> list[BlockParams | None]:
    """Block configs to run. Sweep configs are deduped after clamping to the shape."""
    if algo not in PALLAS_ALGOS:
        return [None]
    grid: Any = [(a.bm, a.bn, a.bk)]
    if a.sweep_blocks:
        grid = itertools.product(SWEEP["bm"], SWEEP["bn"], SWEEP["bk"])
    out: list[BlockParams | None] = []
    for bm, bn, bk in grid:
        cl = resolve_blocks(bm_cap, n, d, bm, bn, bk, a.chunk)[:3]
        if cl not in out:
            out.append(cl)
    return out


def params_str(algo: str, m: int, n: int, d: int, a: Args, bp: BlockParams | None) -> str:
    """File stem of a record: shape, metric, dtype and the algorithm's own knobs."""
    params = f"m{m}_n{n}_d{d}_{a.metric}_{a.dtype}"
    if algo == "blockwise":
        params += f"_B{a.block_size}"
    elif algo in PALLAS_ALGOS:
        assert bp is not None
        params += f"_bm{bp[0]}_bn{bp[1]}_bk{bp[2]}"
        if a.sweep_blocks:
            params += "_sweep"  # keep sweep records apart from default-config records
    elif algo == "approx_max_k":
        params += f"_r{a.recall_target}"
    return params


def plan_jobs(a: Args, ann: dict[str, Any] | None) -> Iterator[Job]:
    """Jobs in run order: algo x k outer, then shape, then block config."""
    shapes: Any = None
    if ann:
        shapes = [(len(ann["test"]), len(ann["train"]), ann["train"].shape[1])]
    for algo, k in itertools.product(a.algos.split(","), a.k):
        for m, n, d in shapes or itertools.product(a.m, a.n, a.d):
            for bp in block_params(algo, a, min(a.batch_size, m), n, d):
                yield Job(algo, k, m, n, d, bp)
