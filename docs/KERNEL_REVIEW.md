# Pallas TPU kernel review checklist

read_when: writing or reviewing any change under `src/trbench/pallas_*.py` or any new Pallas kernel.

Adapted from the Pallas kernel review command in MaxText (`.gemini/commands/maxtext-review-kernels.md`, Apache-2.0, AI-Hypercomputer/maxtext), with practices from openxla/tokamax (Apache-2.0; `docs/benchmarking.md`, per-op `reference.py` and tests) and the JAX Pallas TPU docs. Rewritten for this repo; see `NOTICE`.

Rule for findings: keep one only if you can name the shape, dtype, metric or TPU generation that triggers it. "Looks risky" is not a finding.

## 1. Grid, BlockSpec, index_map

- [ ] `index_map` returns block indices, not element offsets.
- [ ] The grid covers every output tile exactly once; `cdiv` is used, not floor division.
- [ ] Block shape on the last two dims is a multiple of (8, 128) for 32-bit data, or equals the full array dim. bf16 packs 16 sublanes: prefer multiples of 16 on the second-to-last dim.
- [ ] Grid dimension semantics are right: `parallel` only where iterations are independent, `arbitrary` for the reduction axis (the D/bk loop). A wrong `parallel` on the reduction axis is a silent wrong answer on TPU that interpret mode will not show.

## 2. VMEM budget

- [ ] Sum of live block buffers (inputs double-buffered, outputs, scratch, f32 accumulator) fits VMEM with margin. v5e and v6e have 128 MiB per core in the Pallas reference; compare against `pltpu.get_tpu_info()` where available rather than a constant.
- [ ] `vmem_limit_bytes` is set only if the default is too small, and the reason is in a comment.
- [ ] The default block config is the one that was run on TPU, and the block sweep covers (bm, bn, bk) from SPEC R9.

## 3. Accumulator init and finalize

- [ ] The accumulator is zeroed on the first reduction step (`pl.when(k_idx == 0)`) and written out on the last (`pl.when(k_idx == last)`).
- [ ] Scratch is not read before it is written. Nothing depends on scratch contents surviving between different output tiles.
- [ ] The l2 norm term and the mask are applied once, in finalize, not once per reduction step.

## 4. Padding and masking

- [ ] N and D that are not multiples of the block (600 and 100 in `tests/test_pallas.py`) are padded, and the padding cannot win selection.
- [ ] Padded columns score -inf (or a documented floor), not 0. A 0 can beat negative real scores for `ip` and `cos`.
- [ ] Padded D columns are zero so they add nothing to the dot product; for l2 the padded norm term is also zero.
- [ ] A row with fewer valid candidates than k, and a fully masked row, return the -inf and sentinel-id convention from `docs/SPEC.md`, with sentinels only at the tail.
- [ ] Ties: lower id wins, matching the oracle. Out-of-bounds ids never appear.

## 5. Dtype and accumulation

- [ ] Inputs may be bf16; the matmul accumulates in f32 (`preferred_element_type=jnp.float32`).
- [ ] The score tile, norms and mask comparison use f32. No silent downcast between the accumulator and the merge.
- [ ] Output ids are int32 and cannot overflow for N up to the largest sweep value.

## 6. Tests (both are required)

- [ ] Interpret-mode test on CPU against `numpy_ref`: metrics ip, l2, cos; with and without mask; N and D not multiples of the block; k in {1, 10, 100}; f32 compare ids exactly, bf16 compare recall and scores within tolerance. Blocks small enough to force a multi-tile grid and a multi-step reduction.
- [ ] TPU lowering test from CPU (`lower(lowering_platforms=("tpu",))`) asserting `tpu_custom_call`. Skip only when the host cannot lower for TPU at all.
- [ ] Interpret mode is not evidence of TPU correctness or speed. A TPU claim needs a result JSON from a real chip.
- [ ] Compare selections, not raw id order, where bf16 rounding can reorder near-ties (Tokamax top-k tests do the same).

## 7. Roofline expectation

- [ ] State, before running, what the kernel should reach: FLOPs 2*M*N*D, bytes moved, arithmetic intensity, and which ceiling (MXU, HBM, VPU/COP) should bind. v5e: 197 TFLOP/s bf16, 820 GB/s, ridge about 240 FLOP/B.
- [ ] After running, compare with `roofline.label` and `percent_of_peak`. A kernel that matches XLA is a valid result; say so.
- [ ] Anything above 100% of peak, or bandwidth above the HBM peak, is a measurement bug.

## 8. Benchmark protocol

- [ ] Follow `docs/BENCHMARK_METHOD.md` section 2: AOT compile timed apart, warmup >= 3, >= 5 repetitions, `block_until_ready` before stopping the clock, db resident on device.
- [ ] Time the stages apart (`score_only`, `select_only`) when the claim is about selection.
- [ ] Run few iterations in several spaced experiments rather than one long burst: long bursts can throttle the chip (Tokamax benchmarking docs). Record chip kind and jax/libtpu versions.
- [ ] Compare against the stock `jit_topk` baseline on the same shapes, dtype and metric. Report losses as well as wins.
- [ ] Check at least one adversarial input: tie-heavy data, or spikes in a few columns, besides random Gaussian.
- [ ] Results come from committed JSON only (`CONTRIBUTING.md`).
