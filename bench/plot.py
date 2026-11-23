"""Plots from results/**/*.json -> docs/img/*.png.

One figure set per (platform, dataset, n, d, k, dtype) group; CPU and TPU never share a plot.
Only the best-QPS config per (algo, batch size) is drawn, so a Pallas block sweep shows its best.
(a) recall vs QPS, log y: exact variants as points, approx_max_k as a curve.
(b) QPS and p99 batch latency vs batch size per algo.
(c) percent of chip peak per algo (roofline view).
QPS is the record headline `qps`: the median pass for schema 0.2+, the best pass for 0.1. A group
whose records mix the two is drawn on the best pass (`qps_best`), and axes name the basis.
Filters: --platform --dataset --n --d --k.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trbench.records import comparison_basis, qps_basis, qps_on

PALETTE = ["#1f6f8b", "#d1495b", "#edae49", "#2e7d32", "#6a4c93", "#444444"]
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 110,
    }
)


def load(root):
    recs = []
    for p in sorted(Path(root).rglob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(r, dict) and "metrics" in r and "algorithm" in r:
            recs.append(r)
    return recs


def color_map(recs):
    names = sorted({r["algorithm"]["name"] for r in recs})
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def group_key(r):
    """Plots never mix platforms or shapes: one group per (platform, dataset, n, d, k, dtype)."""
    ds = r["dataset"]
    return (
        r["hardware"]["platform"],
        ds["name"],
        ds["n_train"],
        ds["dim"],
        r["query"]["k"],
        r["algorithm"]["params"].get("precision", ds.get("dtype_compute")),
    )


def group_tag(key, kind):
    """File stem: platform first (tpu_..., cpu_...), then plot kind, then the rest of the key."""
    return "_".join([str(key[0]), kind, *(str(x) for x in key[1:])])


def group_title(key):
    plat, ds, n, d, k, dt = key
    return f"{plat} {ds} N={n} D={d} k={k} {dt}"


def keep(r, a):
    ds = r["dataset"]
    checks = (
        (a.platform, r["hardware"]["platform"]),
        (a.dataset, ds["name"]),
        (a.n, ds["n_train"]),
        (a.d, ds["dim"]),
        (a.k, r["query"]["k"]),
    )
    return all(want is None or want == got for want, got in checks)


def best_configs(recs, basis=None):
    """Per (algo, recall_target, batch size) keep the highest-QPS record.

    Collapses the Pallas block sweep (and any other tuning knobs) to the best config per shape.
    QPS is compared on `basis` (default: the common basis of recs), never median against best.
    """
    basis = basis or comparison_basis(recs)
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in recs:
        p = r["algorithm"]["params"]
        key = (r["algorithm"]["name"], p.get("recall_target"), r["query"]["batch_size"])
        if key not in best or qps_on(r, basis) > qps_on(best[key], basis):
            best[key] = r
    return list(best.values())


def grouped(recs):
    """Records per group_key, unreduced (so the QPS basis sees every schema in the group)."""
    out = defaultdict(list)
    for r in recs:
        out[group_key(r)].append(r)
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def groups(recs):
    return {k: best_configs(v) for k, v in grouped(recs).items()}


def qps_label(basis, mixed):
    """Axis text naming the statistic behind plotted QPS."""
    return f"QPS, {basis} pass" + (" (mixed 0.1/0.2 records)" if mixed else "")


def plot_recall_qps(key, recs, cm, out, basis="best", mixed=False):
    recs = [r for r in recs if r["metrics"].get("recall") is not None]
    if not recs:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    curves = defaultdict(list)
    for r in recs:
        a = r["algorithm"]
        if a.get("exact", True):
            ax.scatter(
                r["metrics"]["recall"],
                qps_on(r, basis),
                color=cm[a["name"]],
                s=36,
                label=a["name"],
                zorder=3,
            )
        else:
            curves[a["name"]].append((r["metrics"]["recall"], qps_on(r, basis)))
    for name, pts in curves.items():
        pts.sort()
        ax.plot(*zip(*pts, strict=True), marker="o", color=cm[name], label=f"{name} (curve)")
    ax.set_yscale("log")
    ax.set_xlabel("recall@k (ann-benchmarks distance threshold)")
    ax.set_ylabel(f"{qps_label(basis, mixed)} (log)")
    ax.set_title(f"Recall vs QPS: {group_title(key)}", fontsize=9)
    h, lb = ax.get_legend_handles_labels()
    uniq = dict(zip(lb, h, strict=True))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
    return save(fig, out, f"{group_tag(key, 'recall_qps')}.png")


def plot_batch(key, recs, cm, out, basis="best", mixed=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    series: dict[str, dict[int, tuple[float, Any]]] = defaultdict(dict)
    for r in recs:
        m, b = r["metrics"], r["query"]["batch_size"]
        series[r["algorithm"]["name"]][b] = (qps_on(r, basis), m["latency_ms_per_batch"].get("p99"))
    for name, pts in sorted(series.items()):
        xs = sorted(pts)
        a1.plot(xs, [pts[x][0] for x in xs], marker="o", color=cm[name], label=name)
        a2.plot(xs, [pts[x][1] for x in xs], marker="o", color=cm[name], label=name)
    for ax, yl in ((a1, qps_label(basis, mixed)), (a2, "p99 batch latency (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size M")
        ax.set_ylabel(yl)
    a1.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Throughput and tail latency vs batch size: {group_title(key)}", fontsize=9)
    return save(fig, out, f"{group_tag(key, 'batch_qps_p99')}.png")


def plot_peak(key, recs, cm, out):
    vals = defaultdict(list)
    for r in recs:
        pct = r["metrics"].get("percent_of_peak")
        if pct is not None:
            vals[r["algorithm"]["name"]].append(pct)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if vals:
        names = sorted(vals)
        ax.bar(names, [max(vals[n]) for n in names], color=[cm[n] for n in names])
        ax.set_ylabel("best % of chip peak bf16 FLOP/s")
    else:
        ax.text(
            0.5,
            0.5,
            "no peak for this group\n(CPU, unknown chip, or f32 compute)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    ax.set_title(f"Achieved FLOP/s as % of peak: {group_title(key)}", fontsize=9)
    return save(fig, out, f"{group_tag(key, 'percent_of_peak')}.png")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="docs/img")
    p.add_argument("--platform", default=None, help="filter: cpu | tpu | ...")
    p.add_argument("--dataset", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--d", type=int, default=None)
    p.add_argument("--k", type=int, default=None)
    a = p.parse_args(argv)
    recs = [r for r in load(a.results) if keep(r, a)]
    if not recs:
        raise SystemExit(f"no result JSON under {a.results} matching the filters")
    cm, out = color_map(recs), Path(a.out)
    raw = grouped(recs)
    paths = []
    for key, rs in groups(recs).items():
        basis = comparison_basis(raw[key])
        mixed = len({qps_basis(r) for r in raw[key]}) > 1
        paths += [
            plot_recall_qps(key, rs, cm, out, basis, mixed),
            plot_batch(key, rs, cm, out, basis, mixed),
            plot_peak(key, rs, cm, out),
        ]
    for p_ in filter(None, paths):
        print(p_)


if __name__ == "__main__":
    main()
