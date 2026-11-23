"""Compare TPU smoke records against committed reference records (used by scripts/tpu_smoke.sh).

    python -m bench.smoke_compare --smoke OUT --ref results/tpu/run3 --algos a,b [--drift-warn 0.5]

Exit 1 when an expected algorithm has no record in the smoke directory or a record cannot be
read. Drift outside [1 / (1 + x), 1 + x] of the reference QPS only prints a warning: a different
chip, host load or jax version moves small shapes by tens of percent (docs/RESULTS.md 9.7).
Records are matched by relative path, so the smoke shape must be a committed Run 3 cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _qps(path: Path) -> float:
    q = json.loads(path.read_text())["metrics"]["qps"]
    if not isinstance(q, (int, float)) or q <= 0:
        raise ValueError(f"non-positive qps {q!r}")
    return float(q)


def compare(
    smoke: Path, ref: Path, algos: list[str], drift_warn: float
) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    files = sorted(smoke.rglob("*.json"))
    seen: set[str] = set()
    for f in files:
        rel = f.relative_to(smoke)
        seen.update(rel.parts)
        try:
            got = _qps(f)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            errors.append(f"unreadable record {rel}: {e}")
            continue
        r = ref / rel
        if not r.is_file():
            warnings.append(f"no committed reference for {rel}")
            continue
        try:
            want = _qps(r)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            errors.append(f"unreadable reference {r}: {e}")
            continue
        ratio = got / want
        line = f"{rel}: {got:,.0f} QPS vs Run 3 {want:,.0f} ({ratio:.2f}x)"
        print(line)
        if ratio > 1 + drift_warn or ratio < 1 / (1 + drift_warn):
            warnings.append(f"large drift: {line}")
    for a in algos:
        if a not in seen:
            errors.append(f"missing algorithm: no record for {a}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--algos", required=True, help="comma list of algorithm directory names")
    ap.add_argument("--drift-warn", type=float, default=0.5)
    a = ap.parse_args(argv)
    errors, warnings = compare(a.smoke, a.ref, [x for x in a.algos.split(",") if x], a.drift_warn)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
