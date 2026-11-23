#!/usr/bin/env bash
# Run the release logic locally without tagging or publishing anything.
#
#   bash scripts/release_dry_run.sh [OUT_DIR]      # OUT_DIR defaults to a fresh temp directory
#
# Steps (the ones release.yml runs on a tag, minus attestation and the GitHub release):
#   1. build the sdist and wheel into OUT_DIR/dist
#   2. write SHA256SUMS and the CycloneDX SBOM (scripts/release_assets.sh) and verify the sums
#   3. run the CHANGELOG release gate on a COPY of CHANGELOG.md and CITATION.cff: when the
#      real file still has entries under Unreleased, cut them into the pyproject version on the
#      copy first, which is what the maintainer commits before tagging
#   4. extract the release notes from that copy
# The working tree is never modified. Exit 0 means a tag push would pass its checks.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
# tomllib needs Python 3.11+; fall back to a uv-managed interpreter when python3 is older.
if python3 -c 'import tomllib' 2>/dev/null; then
  PY=(python3)
else
  PY=(uv run --no-project --python ">=3.12" python)
fi
VERSION=$("${PY[@]}" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
echo "dry run for v$VERSION into $OUT"

uv build --out-dir "$OUT/dist" >/dev/null
test -f "$OUT/dist/trbench-$VERSION-py3-none-any.whl" || { echo "wheel for $VERSION missing" >&2; exit 1; }
test -f "$OUT/dist/trbench-$VERSION.tar.gz" || { echo "sdist for $VERSION missing" >&2; exit 1; }

bash scripts/release_assets.sh "$OUT/dist" "$OUT/assets"
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUT/dist" && sha256sum -c "$OUT/assets/SHA256SUMS")
else
  (cd "$OUT/dist" && shasum -a 256 -c "$OUT/assets/SHA256SUMS")
fi

mkdir -p "$OUT/changelog"
cp CHANGELOG.md CITATION.cff "$OUT/changelog/"
if python3 scripts/changelog.py release "$VERSION" "$OUT/changelog/CHANGELOG.md" 2>/dev/null; then
  :
elif python3 scripts/changelog.py release "$VERSION" "$OUT/changelog/CHANGELOG.md" --allow-pending 2>/dev/null; then
  :  # version already cut; Unreleased holds the next release's entries
else
  python3 scripts/changelog.py cut "$VERSION" "$(date -u +%Y-%m-%d)" "$OUT/changelog/CHANGELOG.md"
  python3 scripts/changelog.py release "$VERSION" "$OUT/changelog/CHANGELOG.md"
fi
python3 scripts/changelog.py structure "$OUT/changelog/CHANGELOG.md"
python3 scripts/release_notes.py "$VERSION" "$OUT/changelog/CHANGELOG.md" > "$OUT/notes.md"
test -s "$OUT/notes.md"
echo "release dry run ok: v$VERSION"
