"""Fail if a notebook has stored outputs or execution counts (nbstripout-like check)."""

import json
import sys


def main(paths: list[str]) -> int:
    bad = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                print(f"{path}: cell {i} has outputs or execution_count; strip before commit")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
