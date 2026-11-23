#!/usr/bin/env bash
# TPU smoke check: one tiny shape per algorithm on a real TPU, then validate the records and
# compare their QPS against the committed Run 3 numbers.
#
#   bash scripts/tpu_smoke.sh [OUT_DIR]       # default results/smoke_<UTC stamp>
#   PYTHON=python3 bash scripts/tpu_smoke.sh  # interpreter override, as in reproduce_all.sh
#   DRY_RUN=1 bash scripts/tpu_smoke.sh OUT   # print the commands, run nothing
#   DRIFT_WARN=0.5 ...                        # warn when QPS is outside [1/(1+x), 1+x] of Run 3
#
# Exit status: non-zero when the TPU is missing, any algorithm failed to produce a record, or
# validation fails. Large drift against Run 3 only warns: a different chip, host load or jax
# version moves small shapes by tens of percent (docs/RESULTS.md section 9.7).
#
# Run it by hand on any TPU VM, or in Colab from a `!bash` cell (never import jax in the
# notebook kernel). The shape is N=65,536, M=128, D=128, k=10, bf16, synthetic, the same cell
# as Run 3, so each record has a committed counterpart to compare with.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-results/smoke_$(date -u +%Y%m%d-%H%M%S)}"
REF_DIR="${REF_DIR:-results/tpu/run3}"
DRIFT_WARN="${DRIFT_WARN:-0.5}"

if [ -n "${PYTHON:-}" ]; then
  PY=("$PYTHON")
elif command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  PY=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  PY=(python)
fi

ALGOS=(jit_topk blockwise pallas_score pallas_fused_topk pallas_fused_bucket approx_max_k \
  score_only select_only)

run() {
  echo "+ $*"
  [ "${DRY_RUN:-0}" = "1" ] || "$@"
}

if [ "${DRY_RUN:-0}" != "1" ]; then
  if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "refusing: $OUT exists and is not empty (use a fresh directory)" >&2
    exit 1
  fi
  "${PY[@]}" -c "import jax; d = jax.devices(); assert d[0].platform == 'tpu', f'not a TPU: {d}'; print(d[0].device_kind, len(d), 'device(s)')"
fi

# --reps 5 is the minimum the harness accepts; --allow-dirty because a TPU VM checkout may carry
# local edits (records say so). Failures are collected, not fatal, so every algorithm is tried.
failed=0
for a in "${ALGOS[@]}"; do
  run "${PY[@]}" -m bench.run --allow-dirty --algos "$a" --m 128 --n 65536 --d 128 --k 10 \
    --dtype bf16 --reps 5 --out "$OUT" || { echo "FAIL: $a did not run" >&2; failed=1; }
done

run "${PY[@]}" -m bench.validate_results "$OUT" || failed=1
run "${PY[@]}" -m bench.smoke_compare --smoke "$OUT" --ref "$REF_DIR" \
  --algos "$(IFS=,; echo "${ALGOS[*]}")" --drift-warn "$DRIFT_WARN" || failed=1

if [ "$failed" -ne 0 ]; then
  echo "tpu smoke: FAILED (see messages above)" >&2
  exit 1
fi
echo "tpu smoke: ok ($OUT)"
