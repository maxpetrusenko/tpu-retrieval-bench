# Changelog

All notable changes. Format: Keep a Changelog; versions follow `pyproject.toml`.

## Unreleased

### Added
- `CONTRIBUTING.md`, `docs/KERNEL_REVIEW.md` (Pallas TPU kernel review checklist), `NOTICE`.
- `.pre-commit-config.yaml` with ruff, file hygiene and a notebook-output check (`scripts/check_notebook_outputs.py`).
- Issue templates and a pull request template.
- Support matrix in `README.md`.

## 0.1.0

- V0 `numpy_ref`, V1 `jit_topk`, V2 `blockwise`, V3 `pallas_score`, A1 `approx_max_k`, stage-split timing.
- ann-benchmarks HDF5 loader, k-nn recall, roofline labels, plots.
- TPU v5e-1 results and `docs/RESULTS.md`.
