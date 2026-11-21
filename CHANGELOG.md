# Changelog

All notable changes. Format: Keep a Changelog; versions follow `pyproject.toml`.

## Unreleased

### Added
- Run 2 TPU v5e-1 results (`results/tpu/run2`, `results/tpu/run2-devtime`, `docs/RESULTS.md` section 8): fused kernels, block sweeps, `approx_max_k` recall targets, loop and device timing. Plots regenerated from Run 2.
- V4 `pallas_fused_topk` (`src/trbench/pallas_fused.py`): Pallas matmul with in-kernel per-tile top-k (k <= 128, `kprime` option), stage 2 `lax.top_k`; exact. Optional `pallas_fused_bucket` (approximate, TPU-KNN style bins). Registered in `bench/run.py`, CPU interpret and TPU-lowering tests, roofline candidate bytes (k' padded to 128 lanes).
- `bench/summarize.py`: KernelBench-style fast_p table (p 1.0, 1.25 vs `jit_topk`) and a flag for exact variants with speedup above 10x.
- Colab notebook: fused variant in the sweeps, block sweep for `pallas_score` and `pallas_fused_topk`, `approx_max_k` recall targets 0.80, 0.90, 0.95, 0.99.
- `CONTRIBUTING.md`, `docs/KERNEL_REVIEW.md` (Pallas TPU kernel review checklist), `NOTICE`.
- `.pre-commit-config.yaml` with ruff, file hygiene and a notebook-output check (`scripts/check_notebook_outputs.py`).
- Issue templates and a pull request template.
- Support matrix in `README.md`.

## 0.1.0

- V0 `numpy_ref`, V1 `jit_topk`, V2 `blockwise`, V3 `pallas_score`, A1 `approx_max_k`, stage-split timing.
- ann-benchmarks HDF5 loader, k-nn recall, roofline labels, plots.
- TPU v5e-1 results and `docs/RESULTS.md`.
