"""Validate every result JSON under the given paths (default results/).

Exit 0 when all records are plausible, 1 otherwise. See trbench.validate for the rules.
"""

from __future__ import annotations

import argparse
import sys

from trbench.validate import iter_records, validate_tree


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["results"], help="files or directories")
    ap.add_argument("--allow-empty", action="store_true",
                    help="do not fail when no JSON records are found")
    args = ap.parse_args(argv)
    total, failures = 0, {}
    for p in args.paths:
        total += sum(1 for _ in iter_records(p))
        failures.update(validate_tree(p))
    for path, errs in failures.items():
        print(f"FAIL {path}")
        for e in errs:
            print(f"  - {e}")
    print(f"checked {total} records, {len(failures)} failed")
    if total == 0 and not args.allow_empty:
        print("no records found", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
