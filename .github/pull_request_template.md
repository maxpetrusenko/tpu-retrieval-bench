## What and why

## Checks

- [ ] `JAX_PLATFORMS=cpu uv run pytest -q` and `uv run ruff check .` pass
- [ ] Every number in docs comes from a committed result JSON, and the validator passes on it
- [ ] CPU or interpret-mode results are not presented as TPU results
- [ ] Docs updated (`README.md`, `docs/RESULTS.md`, `CHANGELOG.md`) where behavior changed
- [ ] Copied code (Apache-2.0, MIT or BSD only) carries its notice and a `NOTICE` entry

## Kernel PRs only (`docs/KERNEL_REVIEW.md`)

- [ ] BlockSpec and index_map, (8,128) alignment, VMEM budget
- [ ] Accumulator init and finalize, padding and mask cannot win selection
- [ ] Interpret-mode test against `numpy_ref` and a TPU lowering test
- [ ] Roofline expectation stated; benchmark follows `docs/BENCHMARK_METHOD.md`
