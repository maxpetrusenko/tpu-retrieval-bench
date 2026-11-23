#!/usr/bin/env bash
# Write the release checksums and the SBOM for a directory of built distributions.
#
#   bash scripts/release_assets.sh DIST_DIR OUT_DIR
#
# OUT_DIR/SHA256SUMS   sha256 of every file in DIST_DIR, names relative, so that
#                      `sha256sum -c SHA256SUMS` works in a directory holding the downloads
# OUT_DIR/sbom.cdx.json CycloneDX 1.5 JSON of the locked environment (uv.lock): the project, its
#                      runtime dependencies and the data and plot extras, no dev tools
#
# release.yml and scripts/release_dry_run.sh both call this, so the dry run exercises the exact
# logic that ships. Needs uv (SBOM) and sha256sum or shasum.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 DIST_DIR OUT_DIR" >&2
  exit 2
fi
DIST=$1
OUT=$2
cd "$(dirname "$0")/.."
ROOT=$PWD

shopt -s nullglob
files=("$DIST"/*.whl "$DIST"/*.tar.gz)
if [ "${#files[@]}" -lt 2 ]; then
  echo "expected a wheel and an sdist in $DIST, found ${#files[@]} file(s)" >&2
  exit 1
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

if command -v sha256sum >/dev/null 2>&1; then
  sum() { sha256sum "$@"; }
else
  sum() { shasum -a 256 "$@"; }
fi
(cd "$DIST" && sum -- *.whl *.tar.gz) > "$OUT/SHA256SUMS"

(cd "$ROOT" && uv export --frozen --no-dev --all-extras --format cyclonedx1.5 \
  --preview-features sbom-export -o "$OUT/sbom.cdx.json" >/dev/null)
python3 - "$OUT/sbom.cdx.json" <<'PY'
import json
import sys

bom = json.load(open(sys.argv[1]))
assert bom["bomFormat"] == "CycloneDX", bom.get("bomFormat")
assert bom["specVersion"] == "1.5", bom.get("specVersion")
names = {c["name"] for c in bom["components"]}
assert {"jax", "numpy"} <= names, f"runtime dependencies missing from the SBOM: {sorted(names)[:10]}"
print(f"sbom: {len(bom['components'])} components")
PY
echo "wrote $OUT/SHA256SUMS and $OUT/sbom.cdx.json"
