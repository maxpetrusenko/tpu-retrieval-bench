"""Print the CHANGELOG.md section for one version (release notes for the GitHub release).

    python3 scripts/release_notes.py 0.2.0 [CHANGELOG.md]

Takes the text under `## [VERSION] - date` up to the next `## ` heading. Exit 1 when the
section is missing or empty, so a release cannot go out without notes.
"""

import re
import sys
from pathlib import Path


def section(text: str, version: str) -> str:
    m = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## |^\[[^\]]+\]: |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    notes = section(path.read_text(), argv[0])
    if not notes:
        print(f"no CHANGELOG section for {argv[0]} in {path}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
