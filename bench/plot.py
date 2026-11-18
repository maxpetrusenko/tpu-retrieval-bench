"""Plots from results/**/*.json -> docs/img/*.png.

(a) recall vs QPS, log y: exact variants as points, approx_max_k as a curve.
(b) QPS and p99 batch latency vs batch size per algo.
(c) percent of chip peak per algo (roofline view).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f6f8b", "#d1495b", "#edae49", "#2e7d32", "#6a4c93", "#444444"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True,
    "grid.color": "#e3e3e3", "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "savefig.dpi": 110,
})


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


def plot_recall_qps(recs, cm, out):
    by_ds = defaultdict(list)
    for r in recs:
        if r["metrics"].get("recall") is not None:
            by_ds[r["dataset"]["name"]].append(r)
    paths = []
    for ds, rs in by_ds.items():
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        curves = defaultdict(list)
        for r in rs:
            a = r["algorithm"]
            if a.get("exact", True):
                ax.scatter(r["metrics"]["recall"], r["metrics"]["qps"], color=cm[a["name"]],
                           s=36, label=a["name"], zorder=3)
            else:
                curves[a["name"]].append((r["metrics"]["recall"], r["metrics"]["qps"]))
        for name, pts in curves.items():
            pts.sort()
            ax.plot(*zip(*pts, strict=True), marker="o", color=cm[name], label=f"{name} (curve)")
        ax.set_yscale("log")
        ax.set_xlabel("recall@k (ann-benchmarks distance threshold)")
        ax.set_ylabel("QPS (log)")
        ax.set_title(f"Recall vs QPS: {ds}")
        h, lb = ax.get_legend_handles_labels()
        uniq = dict(zip(lb, h, strict=True))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
        paths.append(save(fig, out, f"recall_qps_{ds}.png"))
    return paths


def plot_batch(recs, cm, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    series = defaultdict(dict)
    for r in recs:
        m, b = r["metrics"], r["query"]["batch_size"]
        p99 = m["latency_ms_per_batch"].get("p99")
        series[r["algorithm"]["name"]][b] = (m["qps"], p99)
    for name, pts in sorted(series.items()):
        xs = sorted(pts)
        a1.plot(xs, [pts[x][0] for x in xs], marker="o", color=cm[name], label=name)
        a2.plot(xs, [pts[x][1] for x in xs], marker="o", color=cm[name], label=name)
    for ax, yl in ((a1, "QPS"), (a2, "p99 batch latency (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size M")
        ax.set_ylabel(yl)
    a1.legend(frameon=False, fontsize=8)
    fig.suptitle("Throughput and tail latency vs batch size")
    return save(fig, out, "batch_qps_p99.png")


def plot_peak(recs, cm, out):
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
        ax.text(0.5, 0.5, "no TPU runs: peak unknown on this platform\n(CPU results have no peak)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    ax.set_title("Achieved FLOP/s as % of peak")
    return save(fig, out, "percent_of_peak.png")


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
    a = p.parse_args(argv)
    recs = load(a.results)
    if not recs:
        raise SystemExit(f"no result JSON under {a.results}")
    cm, out = color_map(recs), Path(a.out)
    paths = [*plot_recall_qps(recs, cm, out), plot_batch(recs, cm, out), plot_peak(recs, cm, out)]
    for p_ in paths:
        print(p_)


if __name__ == "__main__":
    main()
