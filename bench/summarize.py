"""Markdown tables from results/**/*.json (reusable; reads only recorded numbers).

One table per (platform, dataset, N, D, k, M) shape with algo, compile time, warm p50/p95 batch
latency, QPS, recall, recall at epsilon 0.01, percent of peak, roofline label and
selection_over_matmul. Pallas records collapse to the best config per shape; a second table
shows best vs worst block config per shape (block sensitivity). A last section lists records
that look wrong (recall below 1 for exact variants, peak above 100 percent, errors).
A KernelBench-style fast_p table gives, per exact variant, the fraction of shapes where it is
correct (recall 1 at eps 0.01) AND its QPS beats jit_matmul_topk by more than p (p in 1.0, 1.25).
An exact variant with speedup above 10x is flagged for manual review (anti reward-hack check).
Filters: --platform --dataset. Output: stdout or --out FILE.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from bench.plot import load

EXACT_RECALL_FLOOR = 1.0
BLOCK_ALGOS = ("pallas_score_topk", "pallas_fused_topk", "pallas_fused_bucket")
BASELINE = "jit_matmul_topk"
FAST_P = (1.0, 1.25)
SUSPECT_SPEEDUP = 10.0


def fmt(x, nd=2):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def shape_key(r):
    ds, q = r["dataset"], r["query"]
    return (r["hardware"]["platform"], ds["name"], ds["n_train"], ds["dim"], q["k"],
            q["batch_size"])


def row(r):
    m, a = r["metrics"], r["algorithm"]
    lat = m["latency_ms_per_batch"]
    p = a["params"]
    algo = a["name"]
    if a["name"] in BLOCK_ALGOS:
        algo += f" (bm{p.get('bm')},bn{p.get('bn')},bk{p.get('bk')})"
    elif a["name"] == "approx_max_k":
        algo += f" (target {p.get('recall_target')})"
    return [algo, fmt(m["compile_time_s"], 3), fmt(lat["p50"], 3), fmt(lat["p95"], 3),
            fmt(m["qps"], 0), fmt(m.get("recall"), 4), fmt(m.get("recall_epsilon_0.01"), 4),
            fmt(m.get("percent_of_peak")), m.get("roofline_label") or "-",
            fmt(m.get("selection_over_matmul"))]


HEAD = ["algo", "compile_s", "p50 ms", "p95 ms", "QPS", "recall", "recall eps0.01",
        "% peak", "roofline", "sel/matmul"]


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def best_per_algo(recs):
    best = {}
    for r in recs:
        a = r["algorithm"]
        key = (a["name"], a["params"].get("recall_target"))
        if key not in best or r["metrics"]["qps"] > best[key]["metrics"]["qps"]:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["algorithm"]["name"],
                                                r["algorithm"]["params"].get("recall_target") or 0))


def shape_title(k):
    plat, ds, n, d, kk, m = k
    return f"{plat} / {ds} / N={n} D={d} k={kk} M={m}"


def main_tables(recs):
    by = defaultdict(list)
    for r in recs:
        by[shape_key(r)].append(r)
    parts = []
    for k in sorted(by, key=lambda x: (x[0], x[1], x[2], x[5])):
        rows = [row(r) for r in best_per_algo(by[k])]
        parts.append(f"### {shape_title(k)}\n\n{md_table(HEAD, rows)}\n")
    return "\n".join(parts)


def block_table(recs):
    by = defaultdict(list)
    for r in recs:
        if r["algorithm"]["name"] in BLOCK_ALGOS:
            by[(*shape_key(r), r["algorithm"]["name"])].append(r)
    rows = []
    for k in sorted(by, key=lambda x: (x[0], x[1], x[2], x[5], x[-1])):
        rs = sorted(by[k], key=lambda r: r["metrics"]["qps"])
        lo, hi = rs[0], rs[-1]

        def cfg(r):
            p = r["algorithm"]["params"]
            return f"{p.get('bm')}x{p.get('bn')}x{p.get('bk')}"

        ratio = hi["metrics"]["qps"] / lo["metrics"]["qps"]
        rows.append([k[-1], shape_title(k[:-1]), str(len(rs)), cfg(hi), fmt(hi["metrics"]["qps"], 0),
                     cfg(lo), fmt(lo["metrics"]["qps"], 0), fmt(ratio, 3)])
    head = ["algo", "shape", "configs", "best bm x bn x bk", "best QPS", "worst config", "worst QPS",
            "best/worst"]
    return md_table(head, rows) if rows else "no pallas records"


def is_correct(r):
    """Exact variant with full recall (eps 0.01 when recorded: bf16 perturbs eps 1e-3)."""
    m = r["metrics"]
    rec = m.get("recall_epsilon_0.01")
    rec = m.get("recall") if rec is None else rec
    return bool(r["algorithm"].get("exact")) and rec is not None and rec >= EXACT_RECALL_FLOOR


def speedups(recs):
    """[(algo name, shape key, correct, speedup over jit_topk)] using best QPS per algo+shape."""
    by = defaultdict(dict)
    for r in recs:
        name, key = r["algorithm"]["name"], shape_key(r)
        cur = by[key].get(name)
        if cur is None or r["metrics"]["qps"] > cur["metrics"]["qps"]:
            by[key][name] = r
    out = []
    for key, algos in by.items():
        base = algos.get(BASELINE)
        if base is None:
            continue
        for name, r in algos.items():
            if name in (BASELINE, "numpy_ref") or not r["algorithm"].get("exact"):
                continue
            out.append((name, key, is_correct(r), r["metrics"]["qps"] / base["metrics"]["qps"]))
    return out


def fast_p_table(recs):
    """fast_p per exact variant: fraction of shapes correct and faster than jit_topk by > p."""
    sp = speedups(recs)
    names = sorted({n for n, *_ in sp})
    rows = []
    for name in names:
        mine = [(ok, x) for n, _, ok, x in sp if n == name]
        cells = [fmt(sum(1 for ok, x in mine if ok and x > p) / len(mine), 3) for p in FAST_P]
        rows.append([name, str(len(mine)), *cells])
    head = ["algo", "shapes", *[f"fast_{p}" for p in FAST_P]]
    return md_table(head, rows) if rows else f"no exact variants with a {BASELINE} baseline"


def suspect_speedups(recs):
    return [f"exact variant speedup {x:.1f}x > {SUSPECT_SPEEDUP:g}x over jit_topk, review by hand: "
            f"{n} {shape_title(k)}" for n, k, _, x in speedups(recs) if x > SUSPECT_SPEEDUP]


def anomalies(recs):
    out = []
    for r in recs:
        a, m = r["algorithm"], r["metrics"]
        tag = f"{a['name']} {shape_title(shape_key(r))}"
        if a.get("exact") and m.get("recall") is not None and m["recall"] < EXACT_RECALL_FLOOR:
            out.append(f"exact variant recall {m['recall']:.4f} < 1: {tag}")
        pp = m.get("percent_of_peak")
        if pp is not None and pp > 100:
            out.append(f"percent_of_peak {pp:.1f} > 100: {tag}")
        if r.get("error") or m.get("error"):
            out.append(f"error recorded: {tag}: {r.get('error') or m.get('error')}")
    return out + suspect_speedups(recs)


def render(recs):
    notes = anomalies(recs)
    return "\n".join([
        f"{len(recs)} records\n", main_tables(recs), "## Pallas block sensitivity\n",
        block_table(recs), "\n## fast_p vs jit_topk (exact variants)\n", fast_p_table(recs),
        "\n## Flags\n",
        "\n".join(f"- {n}" for n in notes) if notes else "none"]) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--platform", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    recs = [r for r in load(a.results)
            if a.platform in (None, r["hardware"]["platform"])
            and a.dataset in (None, r["dataset"]["name"])]
    if not recs:
        raise SystemExit(f"no result JSON under {a.results} matching the filters")
    text = render(recs)
    if a.out:
        Path(a.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
