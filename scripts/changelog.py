"""CHANGELOG.md checks and the release cut.

    python3 scripts/changelog.py structure [CHANGELOG.md]        headings and link references
    python3 scripts/changelog.py release VERSION [CHANGELOG.md]  tag-time gate (release.yml)
    python3 scripts/changelog.py cut VERSION DATE [CHANGELOG.md] move Unreleased to [VERSION]

`structure` runs in the normal test suite and allows a non-empty Unreleased section (work in
progress). `release` runs in release.yml on the tag: it fails when the tagged version has no
non-empty section or when Unreleased still has entries, so a tag cannot go out with notes
still parked under Unreleased. `cut` is the maintainer step before tagging: it renames the
pending entries into `## [VERSION] - DATE`, leaves an empty Unreleased, fixes the link
references and sets `date-released` in CITATION.cff. Exit 0 ok, 1 check failed, 2 usage.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/maxpetrusenko/tpu-retrieval-bench"
HEADING = re.compile(r"^## (.+)$", re.MULTILINE)
VERSION_HEADING = re.compile(r"^\[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})$")
LEGACY_HEADING = re.compile(r"^\d+\.\d+\.\d+$")  # `## 0.1.0`, written before the format settled
LINK_REF = re.compile(r"^\[([^\]]+)\]: (\S+)$", re.MULTILINE)
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for every `## ` section, in file order; bodies stop before link refs."""
    body_end = LINK_REF.search(text)
    main = text[: body_end.start()] if body_end else text
    marks = list(HEADING.finditer(main))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(main)
        out.append((m.group(1).strip(), main[m.end():end].strip()))
    return out


def _vkey(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def structure_problems(text: str) -> list[str]:
    errs: list[str] = []
    secs = sections(text)
    if not text.startswith("# Changelog"):
        errs.append("first line must be '# Changelog'")
    if not secs or secs[0][0] != "Unreleased":
        errs.append("first section must be '## Unreleased' (it may be empty)")
    versions: list[str] = []
    for heading, body in secs[1:]:
        m = VERSION_HEADING.match(heading)
        if m:
            versions.append(m.group(1))
        elif LEGACY_HEADING.match(heading):
            versions.append(heading)
        else:
            errs.append(f"bad heading '## {heading}': want '## [X.Y.Z] - YYYY-MM-DD'")
            continue
        if not body:
            errs.append(f"section {heading} is empty")
    if versions != sorted(versions, key=_vkey, reverse=True):
        errs.append(f"version sections must be newest first, got {versions}")
    if len(set(versions)) != len(versions):
        errs.append("duplicate version section")
    refs = dict(LINK_REF.findall(text))
    if "Unreleased" not in refs:
        errs.append("missing link reference [Unreleased]: ...")
    for v in versions:
        bracketed = any(h.startswith(f"[{v}]") for h, _ in secs)
        if bracketed and v not in refs:
            errs.append(f"missing link reference [{v}]: ...")
    if versions and "Unreleased" in refs and f"v{versions[0]}...HEAD" not in refs["Unreleased"]:
        errs.append(f"[Unreleased] must compare v{versions[0]}...HEAD (the newest release)")
    for name, url in refs.items():
        if not url.startswith(REPO_URL + "/"):
            errs.append(f"link reference [{name}] does not point at {REPO_URL}")
    return errs


def release_problems(text: str, version: str) -> list[str]:
    if not VERSION_RE.match(version):
        return [f"version {version!r} is not X.Y.Z"]
    errs: list[str] = []
    secs = dict(sections(text))
    match = [h for h in secs if h.startswith(f"[{version}]")]
    if not match:
        errs.append(f"no '## [{version}] - DATE' section: run scripts/changelog.py cut "
                    f"{version} YYYY-MM-DD and commit before tagging")
    elif not secs[match[0]]:
        errs.append(f"section [{version}] is empty")
    if secs.get("Unreleased"):
        errs.append("Unreleased still has entries at tag time: move them into the version "
                    "section (scripts/changelog.py cut) before tagging")
    return errs


def cut(text: str, version: str, date: str) -> str:
    if not VERSION_RE.match(version) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("want VERSION as X.Y.Z and DATE as YYYY-MM-DD")
    secs = dict(sections(text))
    if any(h.startswith(f"[{version}]") for h in secs):
        raise ValueError(f"section [{version}] already exists")
    pending = secs.get("Unreleased", "")
    if not pending:
        raise ValueError("Unreleased is empty: nothing to cut")
    refs = dict(LINK_REF.findall(text))
    prev = re.search(r"/compare/v([\d.]+)\.\.\.HEAD$", refs.get("Unreleased", ""))
    prev_v = prev.group(1) if prev else None
    out = re.sub(r"^## Unreleased\n.*?(?=^## )",
                 f"## Unreleased\n\n## [{version}] - {date}\n\n{pending}\n\n", text, count=1,
                 flags=re.MULTILINE | re.DOTALL)
    new_unreleased = f"[Unreleased]: {REPO_URL}/compare/v{version}...HEAD"
    out = re.sub(r"^\[Unreleased\]: \S+$", new_unreleased, out, flags=re.MULTILINE)
    link = (f"{REPO_URL}/compare/v{prev_v}...v{version}" if prev_v
            else f"{REPO_URL}/releases/tag/v{version}")
    return out.replace(new_unreleased, f"{new_unreleased}\n[{version}]: {link}", 1)


def set_citation(path: Path, version: str, date: str) -> None:
    text = path.read_text()
    text = re.sub(r"^version: .+$", f"version: {version}", text, flags=re.MULTILINE)
    text = re.sub(r"^date-released: .+$", f"date-released: {date}", text, flags=re.MULTILINE)
    path.write_text(text)


def main(argv: list[str]) -> int:
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    default = ROOT / "CHANGELOG.md"
    if cmd == "structure" and len(args) <= 1:
        path = Path(args[0]) if args else default
        errs = structure_problems(path.read_text())
    elif cmd == "release" and 1 <= len(args) <= 2:
        path = Path(args[1]) if len(args) == 2 else default
        errs = release_problems(path.read_text(), args[0])
    elif cmd == "cut" and 2 <= len(args) <= 3:
        path = Path(args[2]) if len(args) == 3 else default
        try:
            path.write_text(cut(path.read_text(), args[0], args[1]))
        except ValueError as ex:
            print(f"cut: {ex}", file=sys.stderr)
            return 1
        citation = path.parent / "CITATION.cff"
        if citation.is_file():
            set_citation(citation, args[0], args[1])
        print(f"cut [{args[0]}] - {args[1]} in {path}")
        return 0
    else:
        print(__doc__, file=sys.stderr)
        return 2
    for e in errs:
        print(f"changelog: {e}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
