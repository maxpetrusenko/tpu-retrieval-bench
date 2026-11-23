"""Validate every result JSON under the given paths (default results/).

Exit 0 when all records are plausible, 1 otherwise. See trbench.validate for the rules.
"""

from __future__ import annotations

import argparse
import json
import sys

from trbench.validate import is_dirty, iter_records, validate_tree


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=["results"], help="files or directories")
    ap.add_argument(
        "--allow-empty", action="store_true", help="do not fail when no JSON records are found"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also fail records with git.dirty=true (default: warn with a count)",
    )
    args = ap.parse_args(argv)
    total, dirty, failures = 0, [], {}
    for p in args.paths:
        for path in iter_records(p):
            total += 1
            try:
                if is_dirty(json.loads(path.read_text())):
                    dirty.append(str(path))
            except (OSError, json.JSONDecodeError):
                pass  # reported by validate_tree
        failures.update(validate_tree(p))
    if args.strict:
        for dirty_path in dirty:
            failures.setdefault(dirty_path, []).append("git.dirty is true (--strict)")
    for fail_path, errs in failures.items():
        print(f"FAIL {fail_path}")
        for e in errs:
            print(f"  - {e}")
    if dirty and not args.strict:
        print(
            f"warning: {len(dirty)} of {total} records have git.dirty=true "
            "(use --strict to fail on them)"
        )
    print(f"checked {total} records, {len(failures)} failed")
    if total == 0 and not args.allow_empty:
        print("no records found", file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
