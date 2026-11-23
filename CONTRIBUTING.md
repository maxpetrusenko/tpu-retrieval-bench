# Contributing

Read `docs/SPEC.md` (contract) and `docs/BENCHMARK_METHOD.md` (timing, recall, what not to compare) before changing code or quoting a number.

## Setup

```
uv sync --extra hdf5 --extra plot
uv run pre-commit install        # optional, runs ruff and file hygiene locally
```

Python >= 3.12, jax >= 0.12. Use `uv`; do not swap the package manager.

## Test and bench commands

```
JAX_PLATFORMS=cpu uv run pytest -q
uv run ruff check .
uv run mypy
uv run python bench/run.py --algos jit_topk,blockwise,pallas_score --m 16 --n 65536 --d 128 --k 10
```

CI on GitHub-hosted runners: `lint-test` (ruff, mypy, pytest with a coverage threshold, `validate_results --strict` on `results/tpu/run3` and `run3-devtime`, `doc_stats --check`, the `reproduce_all.sh` dry run, the notebook constraints check; Python 3.12 and 3.13), `package` (`uv build`, wheel installed into a fresh venv, import, `py.typed`) and `audit` (`pip-audit` on the locked dependencies). CodeQL and OpenSSF Scorecard run as their own workflows. A self-hosted job (`[self-hosted, vps-builder]`: ruff, pytest, the result validator) runs only on pushes to `main` and manual dispatch, never for pull requests; see `SECURITY.md` for the runner's isolation. Pull requests from forks run the hosted jobs only, and workflows from first-time contributors wait for a maintainer to approve them. Run the commands above locally first.

Coverage: `uv run pytest -q --cov` fails below the `fail_under` threshold in `pyproject.toml` (the measured value minus 2, rounded down). Raise it when coverage rises; do not lower it to land a change.

## TPU smoke check

`scripts/tpu_smoke.sh` runs one tiny shape per algorithm (N=65,536, M=128, D=128, k=10, bf16, 5 repetitions) on a real TPU, validates the records, and compares QPS with the committed Run 3 records in `results/tpu/run3`. It exits non-zero when the TPU is missing, an algorithm fails to run or its record is missing, or validation fails, and prints a warning when QPS drifts outside 1/1.5x to 1.5x of Run 3 (`DRIFT_WARN` changes the tolerance). Run it by hand on any TPU VM or Colab:

```
PYTHON=python3 bash scripts/tpu_smoke.sh            # TPU VM with jax[tpu] installed
!PYTHON=python3 bash scripts/tpu_smoke.sh           # Colab cell, never import jax in the kernel
```

The `tpu-smoke` workflow (`workflow_dispatch` only) runs the same script on a runner labelled `[self-hosted, tpu]`. No such runner is registered today, and nothing triggers the workflow automatically.

## Dependency constraints for the notebook

`notebooks/constraints.txt` pins the Colab install to `uv.lock`. After changing dependencies run `uv lock` and then `bash scripts/export_constraints.sh`; CI fails when the file is out of date.

## CPU interpret mode versus TPU claims

- Pallas kernels run in interpret mode on CPU. That checks numerics and index maps only. It does not check Mosaic tiling, (8,128) layout, VMEM limits or speed.
- `test_tpu_lowering_from_cpu` lowers for TPU without a chip. It catches some lowering errors, not all.
- Any statement that a kernel "works on TPU" or is "faster" needs a record from a real TPU run (`results/tpu/...`). Say which chip and jax version.
- Never compare CPU numbers with TPU numbers, or interpret-mode runs with TPU runs (`docs/BENCHMARK_METHOD.md` section 5).

## Publishing results

- A number goes into `README.md` or `docs/RESULTS.md` only if it comes from a committed result JSON under `results/`. Quote the file, not memory or a notebook cell.
- The result validator must pass on every JSON you add or change. If the validator is not on your branch yet, check by hand: `schema_version`, recall 1.0 for exact variants, `percent_of_peak <= 100`, a full `timing` and `hardware` block.
- Suspiciously good numbers (> 100% of peak, > 2x over XLA with no explanation) are bugs until shown otherwise. Re-check timing for `block_until_ready` and constant folding.
- Do not edit a result JSON by hand. Re-run the benchmark.
- Mark anything not measured as not measured.

## Kernel changes

Open kernel PRs against the checklist in `docs/KERNEL_REVIEW.md` and tick it in the PR template. A kernel PR without an interpret-mode test against the NumPy oracle, and a TPU lowering test, is incomplete.

## Code and commits

- Files under 500 LOC. No emojis in code or docs.
- Conventional Commits (`feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`).
- Bug fixes come with a regression test where one fits. Test names: `test_<feature>_<condition>`.
- Small PRs. You are responsible for every line you submit, including generated code, and must be able to explain it.
- Dataset files are never committed (`*.hdf5`, `data/` are ignored). Link licenses, do not redistribute.
- New dependencies: check recent releases and adoption first.
- Copying code from another project: Apache-2.0, MIT or BSD only, keep the original notice in the file, and add an entry to `NOTICE`. Projects without a license (for example tallax) are ideas only.
