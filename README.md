# tpu-retrieval-bench

Benchmark for exact top-k embedding retrieval on TPU. Question: when does a hand-written Pallas kernel beat stock XLA, and when is XLA already fast enough? Contract and sources: `docs/SPEC.md`, method: `docs/BENCHMARK_METHOD.md`.

## Variants

| id | name | what |
|---|---|---|
| V0 | `numpy_ref` | float64 NumPy oracle |
| V1 | `jit_topk` | `jax.jit` matmul then `lax.top_k` (baseline) |
| V2 | `blockwise` | scan over db blocks, per-block top-k, running merge |
| V3 | `pallas_score` | Pallas tiled score kernel (interpret mode on CPU), selection in JAX |
| A1 | `approx_max_k` | `lax.approx_max_k`, own recall-vs-QPS curve |
| stage | `score_only`, `select_only` | matmul and selection timed apart (selection over matmul ratio) |
| V4 | `pallas_fused_topk` | Pallas kernel fuses matmul and per-tile top-k (k <= 128); stage 2 `lax.top_k` over candidates. Exact. `kprime` option |
| V4b | `pallas_fused_bucket` | per-tile bucketed max over L lanes (TPU-KNN style), approximate, recall measured |

## Support matrix

What the tests and committed results show today. "CPU interpret" = pytest on CPU (Pallas in interpret mode: numerics only). "TPU v5e measured" = a committed record in `results/tpu/` from the Colab v5e-1 run. "GPU untested" = nothing was run on GPU. Not measured means not measured.

| variant | CPU interpret | TPU v5e measured | GPU |
|---|---|---|---|
| `jit_topk` | ip, l2, cos; mask; tested | ip (synthetic), l2 (SIFT-1M); no mask, no cos | untested |
| `blockwise` | ip, l2, cos; mask; tested | ip (synthetic), l2 (SIFT-1M); no mask, no cos | untested |
| `pallas_score` | ip, l2, cos; mask; tested; TPU lowering test from CPU | ip (synthetic), l2 (SIFT-1M); no mask, no cos; block sweep at N=1M, M in {128, 1024} | untested |
| `approx_max_k` | ip, no mask; recall tested at targets 0.9, 0.95, 0.99 | ip (synthetic), l2 (SIFT-1M); recall targets 0.80 to 0.99 (synthetic), 0.95 (SIFT) | untested |
| `pallas_fused_topk` (V4) | ip, l2, cos; mask; ties; non-aligned N/D; k <= 128; TPU lowering test from CPU | ip (synthetic), l2 (SIFT-1M); block sweep at N=1M | untested |
| `pallas_fused_bucket` (V4b) | recall tested (approximate); TPU lowering test from CPU | ip (synthetic), l2 (SIFT-1M); block sweep at N=1M | untested |

Mask support (`False` = excluded): `jit_topk`, `blockwise` and `pallas_score` are tested with a 30% mask on CPU; `bench/run.py` has no mask option, so no TPU record uses one. GPU: Pallas TPU kernels do not target GPU, and the XLA variants were never run there.

## Quickstart (CPU)

```
uv sync --extra hdf5 --extra plot
JAX_PLATFORMS=cpu uv run pytest -q
uv run ruff check .
uv run python bench/run.py --algos jit_topk,blockwise,pallas_score,score_only,select_only --m 16 --n 65536 --d 128 --k 10
uv run python bench/download.py sift
uv run python bench/run.py --dataset sift-128-euclidean --n-train 100000 --n-test 1000 --algos jit_topk,blockwise
uv run python bench/plot.py
```

Results go to `results/<platform>/<dataset>/k<k>/<algo>/<params>.json`. CPU numbers only exercise the harness; they are not comparable to TPU numbers.

## Colab TPU

[Open the notebook in Colab](https://colab.research.google.com/github/maxpetrusenko/tpu-retrieval-bench/blob/main/notebooks/colab_tpu.ipynb) and pick a TPU v5e-1 runtime.

## Results

TPU v5e-1 (free Colab, jax 0.12.0.1), k=10, bf16, N up to 1M, D=128. Two runs: Run 1 (unfused kernels, 2026-11-18) and Run 2 (fused kernels, block sweeps, loop and device timing, 2026-11-21). Full report: [docs/RESULTS.md](docs/RESULTS.md).

- Stock XLA (`jit_matmul_topk`) beat the unfused Pallas scoring kernel at 8 of 9 shapes (0.51x to 1.03x; one 3% win at N=1M, M=128).
- Selection dominates: `top_k` took 1.25x to 8.2x the matmul time at M >= 16, 85% to 89% of exact-path time at M=1024 and SIFT-1M. The 27x claim did not reproduce at k=10. No exact variant passed 3.9% of peak.
- Fusing selection into the kernel helped: exact `pallas_fused_topk` was 1.31x jit at N=1M, M=1024 with a tuned block config (tuned, bn=1024: 29,816 vs 22,738 QPS); at the default config (bn=512) it was 1.03x (23,436 QPS), and it was slower at M <= 16 (small-M ratios mostly reflect per-call dispatch cost). The approximate `pallas_fused_bucket` was 1.28x to 2.39x jit at M >= 128 (2.39x is tuned, 54,394 QPS at N=1M, M=1024, recall 1.0; 1.77x at the default config; 0.9997 recall on SIFT-1M). The default-config cells at N=1M, M in {128, 1024} need a rerun because the block sweep overwrote them.
- Block size: a 6% effect for the scoring kernel, 1.5x to 4.9x (best over worst) for the fused kernels; bn=1024 was best.
- `approx_max_k` at recall target 0.95: 567,116 QPS on SIFT-1M at recall 0.980, 23.5x the exact baseline (24,084 QPS), 77% of peak on synthetic N=1M, M=1024. The recall-target sweep (0.80 to 0.99) showed no effect on synthetic data: recall 1.0 at every target, QPS within 5.6%. It is not a recall curve.
- Wall timing understates throughput by up to 5.9x (jit, N=65,536, M=1) against in-jit loop timing; device time sits 0.28 to 0.70 ms under wall time.
- Not measured yet: matmul-only block sweep, recall-versus-QPS curve on real data, GloVe and GIST.

![SIFT-1M recall vs QPS on TPU v5e-1](docs/img/tpu_recall_qps_sift-128-euclidean_1000000_128_10_bf16.png)
