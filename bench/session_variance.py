"""Between-session variance of two result directories (same grid, different Colab sessions).

    python -m bench.session_variance results/tpu/run3 results/tpu/run4 \
        [--json OUT.json] [--md OUT.md] [--claim pallas_fused_topk] [--baseline jit_matmul_topk]

Records are joined on (dataset, n_train, dim, k, M, algorithm, params, sweep), where `sweep` is
true for files that end in `_sweep.json`, so a block-sweep record never collides with the
default-config record of the same params. Per joined cell it reports the headline median QPS of
each run, the ratio B/A, whether the two bootstrap 95% CIs (`metrics.qps_ci95`) overlap, and the
absolute percent change |B/A - 1| * 100. The summary gives median, p90 (linear interpolation) and
max of that change for small shapes (M <= 16) and large shapes (M >= 128); cells between the two
bounds (none in the committed grid) are reported as "mid". The large class is also split by N
(>= 1,000,000 against smaller), because small N is where the CI undercounts session spread.

The claim check compares one algorithm against a baseline inside each run, per shape, for the
default config and for the best sweep config (picked per run, after the fact). A claim is
"win" in a run when the claim CI lies above the baseline CI, "loss" when it lies below and
"tie" when they overlap. For an algorithm with several non-sweep records in one shape (the
approx_max_k recall targets) the "default" pick is the fastest of them, so read it with care.
It is robust when both runs give the same non-tie verdict and fragile otherwise (a tie or a
flip in either run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SMALL_M = 16
LARGE_M = 128

Key = tuple[str, int, int, int, int, str, str, bool]


def load_run(root: Path) -> dict[Key, dict[str, Any]]:
    """Read every record under root; value has qps, ci (qps_ci95), file (relative path)."""
    cells: dict[Key, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
            ds, q, alg, met = rec["dataset"], rec["query"], rec["algorithm"], rec["metrics"]
            qps = float(met["qps"])
            ci = met.get("qps_ci95")
            key: Key = (ds["name"], int(ds["n_train"]), int(ds["dim"]), int(q["k"]),
                        int(q["batch_size"]), alg["name"],
                        json.dumps(alg.get("params", {}), sort_keys=True),
                        path.name.endswith("_sweep.json"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"unreadable record {path}: {e}") from e
        if key in cells:
            raise ValueError(f"duplicate cell {key} in {root}: {path}")
        if not (isinstance(ci, list) and len(ci) == 2):
            raise ValueError(f"{path}: missing qps_ci95 (schema 0.2 required)")
        cells[key] = {"qps": qps, "ci": [float(ci[0]), float(ci[1])],
                      "file": str(path.relative_to(root))}
    return cells


def overlaps(a: list[float], b: list[float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def size_class(m: int) -> str:
    return "small" if m <= SMALL_M else "large" if m >= LARGE_M else "mid"


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]) of a non-empty list."""
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def join(a: dict[Key, dict[str, Any]], b: dict[Key, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per cell present in both runs, sorted by key."""
    rows = []
    for key in sorted(set(a) & set(b)):
        ra, rb = a[key], b[key]
        ratio = rb["qps"] / ra["qps"]
        rows.append({
            "dataset": key[0], "n": key[1], "d": key[2], "k": key[3], "m": key[4],
            "algo": key[5], "params": json.loads(key[6]), "sweep": key[7],
            "size": size_class(key[4]),
            "qps_a": ra["qps"], "qps_b": rb["qps"], "ci_a": ra["ci"], "ci_b": rb["ci"],
            "ratio": ratio, "ci_overlap": overlaps(ra["ci"], rb["ci"]),
            "abs_pct_delta": abs(ratio - 1.0) * 100.0,
            "file": rb["file"],
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, p90 and max of abs_pct_delta per size class, plus the all-cells line."""
    out: dict[str, Any] = {}
    big_n = 1_000_000
    groups = {
        "small": lambda r: r["size"] == "small",
        "mid": lambda r: r["size"] == "mid",
        "large": lambda r: r["size"] == "large",
        "large, N>=1M": lambda r: r["size"] == "large" and r["n"] >= big_n,
        "large, N<1M": lambda r: r["size"] == "large" and r["n"] < big_n,
        "all": lambda r: True,
    }
    for name, pick in groups.items():
        sel = [r for r in rows if pick(r)]
        if not sel:
            continue
        d = [r["abs_pct_delta"] for r in sel]
        out[name] = {
            "cells": len(sel),
            "median": percentile(d, 50), "p90": percentile(d, 90), "max": max(d),
            "ci_overlap": sum(1 for r in sel if r["ci_overlap"]),
            "max_cell": max(sel, key=lambda r: r["abs_pct_delta"])["file"],
        }
    return out


def verdict(claim: dict[str, Any], base: dict[str, Any]) -> str:
    if claim["ci"][0] > base["ci"][1]:
        return "win"
    if claim["ci"][1] < base["ci"][0]:
        return "loss"
    return "tie"


def _shape(key: Key) -> tuple[str, int, int, int, int]:
    return key[:5]


def claims(a: dict[Key, dict[str, Any]], b: dict[Key, dict[str, Any]],
           claim_algo: str, baseline: str) -> list[dict[str, Any]]:
    """Per shape and config kind (default, tuned): verdict vs baseline in each run."""
    runs = {"a": a, "b": b}
    shapes = sorted({_shape(k) for run in runs.values() for k in run if k[5] == claim_algo})
    out = []
    for shape in shapes:
        base: dict[str, dict[str, Any]] = {}
        for tag, run in runs.items():
            cand = [v for k, v in run.items() if _shape(k) == shape and k[5] == baseline
                    and not k[7]]
            if len(cand) == 1:
                base[tag] = cand[0]
        if len(base) != 2:
            continue
        for kind in ("default", "tuned"):
            picked: dict[str, dict[str, Any]] = {}
            cfg: dict[str, str] = {}
            for tag, run in runs.items():
                cand_items = [(k, v) for k, v in run.items()
                              if _shape(k) == shape and k[5] == claim_algo
                              and (k[7] == (kind == "tuned"))]
                if not cand_items:
                    continue
                k, v = max(cand_items, key=lambda kv: kv[1]["qps"])
                picked[tag] = v
                p = json.loads(k[6])
                cfg[tag] = f"{p['bm']}x{p['bn']}" if "bm" in p and "bn" in p else "-"
            if len(picked) != 2:
                continue
            va, vb = verdict(picked["a"], base["a"]), verdict(picked["b"], base["b"])
            robust = va == vb and va != "tie"
            out.append({
                "dataset": shape[0], "n": shape[1], "m": shape[4], "kind": kind,
                "cfg_a": cfg["a"], "cfg_b": cfg["b"],
                "x_jit_a": picked["a"]["qps"] / base["a"]["qps"],
                "x_jit_b": picked["b"]["qps"] / base["b"]["qps"],
                "verdict_a": va, "verdict_b": vb,
                "status": ("robust " + va) if robust else "fragile",
            })
    return out


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def render_md(name_a: str, name_b: str, rows: list[dict[str, Any]], summ: dict[str, Any],
              clm: list[dict[str, Any]], claim_algo: str, baseline: str) -> str:
    lines = [f"# Between-session variance: {name_a} vs {name_b}", "",
             (f"Ratio = {name_b} / {name_a} (median QPS). Delta = |ratio - 1| * 100. "
              "CI overlap = the two bootstrap 95% CIs of the median intersect."), "",
             "| cell | M | " + f"{name_a} QPS | {name_b} QPS | ratio | CIs overlap | abs delta % |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        label = f"{r['dataset']} N={r['n']:,} {r['algo']}"
        if r["algo"].startswith("pallas") or r["algo"] == "blockwise":
            label += " " + "/".join(str(r["params"].get(x)) for x in ("bm", "bn"))
        if r["sweep"]:
            label += " (sweep)"
        lines.append(f"| {label} | {r['m']} | {_fmt(r['qps_a'])} | {_fmt(r['qps_b'])} | "
                     f"{r['ratio']:.3f} | {'yes' if r['ci_overlap'] else 'no'} | "
                     f"{r['abs_pct_delta']:.1f} |")
    lines += ["", "| shapes | cells | median % | p90 % | max % | CIs overlap |",
              "|---|---|---|---|---|---|"]
    for k, s in summ.items():
        lines.append(f"| {k} | {s['cells']} | {s['median']:.1f} | {s['p90']:.1f} | "
                     f"{s['max']:.1f} | {s['ci_overlap']} of {s['cells']} |")
    lines += ["", f"## Claims: {claim_algo} vs {baseline}", "",
              (f"| shape | config | cfg {name_a} | cfg {name_b} | x jit {name_a} "
               f"| x jit {name_b} | {name_a} | {name_b} | status |"),
              "|---|---|---|---|---|---|---|---|---|"]
    for c in clm:
        lines.append(f"| {c['dataset']} N={c['n']:,} M={c['m']} | {c['kind']} | {c['cfg_a']} | "
                     f"{c['cfg_b']} | {c['x_jit_a']:.2f} | {c['x_jit_b']:.2f} | "
                     f"{c['verdict_a']} | {c['verdict_b']} | {c['status']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--json", type=Path, dest="json_out")
    ap.add_argument("--md", type=Path, dest="md_out")
    ap.add_argument("--claim", default="pallas_fused_topk")
    ap.add_argument("--baseline", default="jit_matmul_topk")
    a = ap.parse_args(argv)
    try:
        ra, rb = load_run(a.run_a), load_run(a.run_b)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    rows = join(ra, rb)
    only_a, only_b = len(set(ra) - set(rb)), len(set(rb) - set(ra))
    if not rows or only_a or only_b:
        print(f"ERROR: runs do not cover the same cells ({len(rows)} joined, "
              f"{only_a} only in A, {only_b} only in B)", file=sys.stderr)
        return 1
    summ = summarize(rows)
    clm = claims(ra, rb, a.claim, a.baseline)
    name_a, name_b = a.run_a.name, a.run_b.name
    md = render_md(name_a, name_b, rows, summ, clm, a.claim, a.baseline)
    if a.json_out:
        a.json_out.write_text(json.dumps(
            {"run_a": name_a, "run_b": name_b, "cells": rows, "summary": summ, "claims": clm},
            indent=1) + "\n")
    if a.md_out:
        a.md_out.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
