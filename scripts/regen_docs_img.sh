#!/usr/bin/env bash
# Regenerate the tracked plots in docs/img from the committed Run 3 results.
#
#   bash scripts/regen_docs_img.sh
#
# A maintainer step, run on purpose. The Colab notebook never writes to docs/img (it plots into
# <RUN_ID>/plots, untracked), so a notebook session leaves a clean checkout. Command and run
# match docs/RESULTS.md section 3.
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -n "${PYTHON:-}" ]; then
  RUN=("$PYTHON")
elif command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  RUN=(uv run --extra plot python)
else
  RUN=(python3)
fi
"${RUN[@]}" -m bench.plot --results results/tpu/run3 --platform tpu --out docs/img
echo "docs/img regenerated from results/tpu/run3; review the diff before committing"
