# tpu-retrieval-bench

Goal: measure when a custom Pallas kernel beats stock XLA for exact top-k embedding retrieval on TPU, and when XLA is already fast enough. Contract: `docs/SPEC.md`.

Status: S1 (walking skeleton). Implemented: numpy oracle (V0), `jax.jit` matmul + `lax.top_k` (V1), synthetic data, timing harness, JSON records, CLI.

## Tests

```
uv sync
JAX_PLATFORMS=cpu uv run pytest -q
uv run ruff check .
```

## Bench

```
uv run python bench/run.py --algos jit_topk --m 16 --n 65536 --d 128 --k 10 [--dump-hlo]
```

Writes `results/<platform>/synthetic/k<k>/<algo>/<params>.json` with compile time and warm latency (min/p50/p95/p99/mean) reported separately.
