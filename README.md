# tpu-retrieval-bench

Goal: measure when a custom Pallas kernel beats stock XLA for exact top-k embedding retrieval on TPU, and when XLA is already fast enough. Contract: `docs/SPEC.md`.

Status: S1-S3. Implemented: numpy oracle (V0), `jax.jit` matmul + `lax.top_k` (V1), blockwise scan (V2) with score-only / select-only stage split, Pallas tiled score kernel (V3, interpret on CPU, lowers for TPU), `approx_max_k` (A1), synthetic data, timing harness, JSON records, CLI. See `docs/tasks.md`.

## Tests

```
uv sync
JAX_PLATFORMS=cpu uv run pytest -q
uv run ruff check .
```

## Bench

```
uv run python bench/run.py --algos jit_topk --m 16 --n 65536 --d 128 --k 10 [--dump-hlo]
# algos: numpy_ref jit_topk blockwise pallas_score approx_max_k score_only select_only
uv run python bench/run.py --algos pallas_score --n 8192 --bm 16 --bn 512 --bk 128 [--sweep-blocks]
```

Writes `results/<platform>/synthetic/k<k>/<algo>/<params>.json` with compile time and warm latency (min/p50/p95/p99/mean) reported separately.
