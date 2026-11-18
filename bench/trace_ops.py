"""Summarize device ops from a jax.profiler *.trace.json.gz (small text, safe to commit).

Prints the XLA module and XLA op events on the TPU device: duration, bytes_accessed, HLO
category and the HLO long_name. This is COMPILED-program evidence (what ran on the chip).
Usage: python -m bench.trace_ops <trace.json.gz> [--out FILE]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def device_events(trace):
    ev = trace["traceEvents"]
    dev = {e["pid"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"
           and str(e["args"].get("name", "")).startswith("/device:TPU")}
    names = {(e["pid"], e["tid"]): e["args"]["name"] for e in ev
             if e.get("ph") == "M" and e.get("name") == "thread_name"}
    return [(names.get((e["pid"], e["tid"])), e) for e in ev
            if e.get("ph") == "X" and e.get("pid") in dev]


def summarize(path):
    with gzip.open(path, "rt") as f:
        trace = json.load(f)
    lines = []
    for thread, e in sorted(device_events(trace), key=lambda te: te[1]["ts"]):
        if thread not in ("XLA Modules", "XLA Ops"):
            continue
        a = e.get("args", {})
        lines.append(f"[{thread}] {e['name']}: {e['dur']:.1f} us"
                     f" | bytes_accessed={a.get('bytes_accessed', '-')}"
                     f" | category={a.get('hlo_category', '-')}\n    {a.get('long_name', '')}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    text = summarize(a.trace)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
