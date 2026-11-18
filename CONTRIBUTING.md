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
uv run python bench/run.py --algos jit_topk,blockwise,pallas_score --m 16 --n 65536 --d 128 --k 10
```

CI runs the same ruff and pytest on a self-hosted runner (`[self-hosted, vps-builder]`). Pull requests from forks do not run CI automatically: the runner is ours, so a maintainer must read the diff and approve the run. Run the commands above locally first.

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
