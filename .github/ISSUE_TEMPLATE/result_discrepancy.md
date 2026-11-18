---
name: Result discrepancy
about: A published number does not reproduce
labels: results
---

## Which number

File and line in `README.md` or `docs/RESULTS.md`, plus the result JSON path.

## What you measured

Command, chip kind, jax and libtpu versions, your JSON (attach), and the difference.

## Checked

- [ ] Same shape, dtype, metric, k, batch size
- [ ] Same timing protocol (`docs/BENCHMARK_METHOD.md` section 2)
- [ ] Not comparing CPU or interpret mode with TPU
