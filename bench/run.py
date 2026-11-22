"""CLI sweep: runs algos over shapes, writes results/<platform>/<dataset>/k<k>/<algo>/<params>.json.

--dataset synthetic (default) sweeps --m/--n/--d. An ann-benchmarks name takes shapes and metric
from the HDF5 file (--n-train/--n-test subset it); recall uses the file ground truth.

The driver is split by concern: bench.cli (arguments), bench.algos (algorithm registry),
bench.plan (job planning and file names), bench.execute (timing and recall), bench.emit (record
building and writing). This module wires them together.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # `python bench/run.py`: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import emit, execute, plan
from bench.algos import EXACT_ALGOS, FACTORIES, STAGE_ONLY, BlockParams
from bench.cli import Args, parse
from trbench import env

Record = dict[str, Any]


def repo_git_info(a: Args) -> dict[str, Any]:
    """git_info for this repo; --out inside the repo does not count as dirty."""
    repo = Path(__file__).resolve().parents[1]
    ignore = ["results"]
    out = Path(a.out).resolve()
    if out.is_relative_to(repo) and out != repo:
        ignore.append(str(out.relative_to(repo)))
    return env.git_info(repo, ignore=ignore)


def run_one(algo: str, m: int, n: int, d: int, k: int, a: Args, hw: dict[str, Any],
            git: dict[str, Any], bp: BlockParams | None = None,
            ann: dict[str, Any] | None = None) -> tuple[Path, Record]:
    """Measure one job, write its record (and HLO with --dump-hlo); return (path, record)."""
    meas = execute.measure(algo, m, n, d, k, a, bp, ann)
    rec = emit.build_run_record(meas, a, hw, git)
    path = emit.write_run_record(a, hw, meas, bp, rec)
    if a.dump_hlo and algo in FACTORIES:
        hlo = execute.compile_hlo(algo, meas.queries, meas.db, k, a, bp)
        stem = plan.params_str(algo, meas.m, meas.n, d, a, bp)
        (path.parent / f"{stem}.hlo.txt").write_text(hlo)
    return path, rec


def main(argv: Sequence[str] | None = None) -> None:
    a = parse(argv)
    git = repo_git_info(a)
    if git["dirty"] and not a.allow_dirty:
        raise SystemExit("refusing to run: git tree is dirty (uncommitted changes outside the "
                         "output dir), so records would not match the commit. Commit or stash, "
                         "or pass --allow-dirty (records git.dirty=true).")
    hw = env.hardware_info()
    ann = plan.load_dataset(a) if a.dataset != "synthetic" else None
    stage: dict[tuple[int, ...], dict[str, tuple[Path, Record]]] = {}
    exact: dict[tuple[int, ...], list[tuple[Path, Record]]] = {}
    for job in plan.plan_jobs(a, ann):
        algo, m, n, d, k, bp = job.algo, job.m, job.n, job.d, job.k, job.bp
        try:
            path, rec = run_one(algo, m, n, d, k, a, hw, git, bp, ann)
        except Exception as exc:
            if not execute.is_oom(exc):
                raise
            path = emit.write_error(a, algo, m, n, d, k, bp, hw, exc)
            print(f"{path}\n  algo={algo} error={type(exc).__name__} (continuing)")
            continue
        emit.report(algo, path, rec)
        if algo in STAGE_ONLY:
            stage.setdefault((m, n, d, k), {})[algo] = (path, rec)
        elif algo in EXACT_ALGOS:
            exact.setdefault((m, n, d, k), []).append((path, rec))
    emit.link_stages(stage, exact, a, hw["device_kind"])
    for recs in stage.values():
        if len(recs) == 2:
            r = recs["select_only"][1]["metrics"]["selection_over_matmul"]
            print(f"selection_over_matmul={r:.3f}")


if __name__ == "__main__":
    main()
