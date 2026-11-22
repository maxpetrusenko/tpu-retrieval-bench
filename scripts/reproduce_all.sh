#!/usr/bin/env bash
# Single entry point: the full Run-2 grid with the default 20 repetitions, into a fresh results
# directory, then validate (--strict) and summarize.
#
#   bash scripts/reproduce_all.sh [OUT_DIR]        # default results/run_<UTC stamp>
#   DRY_RUN=1 bash scripts/reproduce_all.sh OUT    # print the commands, run nothing
#   ALLOW_DIRTY=1 ...                              # pass --allow-dirty (records git.dirty=true)
#   PYTHON=/path/to/python ...                     # interpreter override (see below)
#
# Interpreter, first match wins: $PYTHON; `uv run python` (uv installed and pyproject.toml
# present, so the locked environment is used); python3; python. On a TPU VM whose jax[tpu]
# lives in the system Python, set PYTHON=python3 so uv does not build a separate CPU env.
#
# Layout: OUT/<platform>/... holds the main grid, SIFT, block sweeps (file names end in _sweep)
# and the approx_max_k recall-target sweep; OUT-devtime/ holds the --device-time records, kept
# apart because they use the same file names as the main grid. Run on the TPU host; in Colab
# call it from a `!bash` cell so the notebook kernel never imports jax (it would lock the TPU).
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-results/run_$(date -u +%Y%m%d-%H%M%S)}"
DEV="${OUT}-devtime"

if [ -n "${PYTHON:-}" ]; then
  PY=("$PYTHON")
elif command -v uv >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  PY=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  PY=(python)
fi
EXTRA=()
[ "${ALLOW_DIRTY:-0}" = "1" ] && EXTRA+=(--allow-dirty)

run() {
  echo "+ $*"
  [ "${DRY_RUN:-0}" = "1" ] || "$@"
}

if [ "${DRY_RUN:-0}" != "1" ]; then
  for d in "$OUT" "$DEV"; do
    if [ -e "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
      echo "refusing: $d exists and is not empty (use a fresh directory)" >&2
      exit 1
    fi
  done
fi

ALGOS=jit_topk,blockwise,pallas_score,pallas_fused_topk,pallas_fused_bucket,approx_max_k,score_only,select_only
RUN=("${PY[@]}" -m bench.run ${EXTRA[@]+"${EXTRA[@]}"} --timing both)

# 1. main grid: synthetic N x M, then SIFT-1M
run "${RUN[@]}" --algos "$ALGOS" --m 1,16,128,1024 --n 65536,1000000 --d 128 --k 10 \
  --dtype bf16 --out "$OUT"
run "${PY[@]}" -m bench.download sift
run "${RUN[@]}" --dataset sift-128-euclidean --algos "$ALGOS" --k 10 --dtype bf16 \
  --n-test 1000 --batch-size 1000 --out "$OUT"

# 2. block sweeps (records get the _sweep suffix, so they never overwrite the main grid)
run "${RUN[@]}" --algos pallas_score,pallas_fused_topk,pallas_fused_bucket --sweep-blocks \
  --m 128,1024 --n 1000000 --d 128 --k 10 --dtype bf16 --out "$OUT"

# 3. approx_max_k recall-target sweep
for r in 0.80 0.90 0.95 0.99; do
  run "${RUN[@]}" --algos approx_max_k --recall-target "$r" --m 1024 --n 1000000 --d 128 \
    --k 10 --dtype bf16 --out "$OUT"
done

# 4. device time (profiler) into its own directory
run "${RUN[@]}" --device-time --algos jit_topk,pallas_score,pallas_fused_topk,pallas_fused_bucket,approx_max_k \
  --m 128,1024 --n 1000000 --d 128 --k 10 --dtype bf16 --out "$DEV"

# 5. validate (dirty records fail) and summarize
run "${PY[@]}" -m bench.validate_results --strict "$OUT" "$DEV"
run "${PY[@]}" -m bench.summarize --results "$OUT" --out "$OUT/summary.md"
run "${PY[@]}" -m bench.summarize --results "$DEV" --out "$DEV/summary.md"
echo "done: $OUT (summary $OUT/summary.md), $DEV"
