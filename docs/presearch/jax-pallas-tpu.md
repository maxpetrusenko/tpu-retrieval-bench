---
read_when: designing or implementing tpu-retrieval-bench kernels, CI, or TPU benchmark runs
date: 2026-11-18
---
# JAX + Pallas TPU presearch (verified 2026-11-18)

Legend: [V] verified this session (source/command given); [U] unverified or secondary source.

## 1. Versions and Pallas TPU API

- [V] PyPI (JSON API, https://pypi.org/pypi/jax/json): jax and jaxlib latest = 0.12.0.1 (2026-10-21); 0.12.0 on 2026-10-19; 0.11.2 on 2026-09-17. libtpu latest 0.0.53.1 (2026-11-17). xprof 2.24.2.
- [V] jax >= 0.11.0 requires Python >= 3.12 (0.10.2 and earlier: >=3.11). numpy >= 2.2, scipy >= 1.15. Pin `python_requires>=3.12` or `jax<0.11` if a host is 3.11.
- [V] `jax[tpu]` extra at 0.12.0.1 pins `libtpu==0.0.50.*` and `jaxlib==0.12.0.1` (PyPI requires_dist). Install: `pip install -U "jax[tpu]"` on TPU, `pip install -U "jax[cpu]"` on CI.
- [V] Installed jax 0.12.0.1 CPU in a venv and ran the snippets below; signatures printed with `inspect.signature`.
- [V] Rename: old `pltpu.TPUCompilerParams` is gone from current docs; Pallas changelog mentions it only in old entries (https://docs.jax.dev/en/latest/pallas/CHANGELOG.html). Use `pltpu.CompilerParams` (used in the official matmul guide https://docs.jax.dev/en/latest/pallas/tpu/matmul.html). `pltpu.KernelType` renamed `CoreType` then removed; `pl.dot` removed (use `jnp.dot`/`@`/`jax.lax.dot_general`); `pltpu.ANY` removed (use `pl.ANY`); `pltpu.HOST` removed (use `pl.HOST`); `pl.reciprocal` moved to `pltpu.reciprocal`; `pallas_call_p` deprecated. `jax.pallas` alias exists only in the Unreleased changelog, NOT in 0.12.0.1 (`hasattr(jax,"pallas")` is False). Keep `from jax.experimental import pallas as pl`.
- [V] `jax.jit` supports decorator factory since 0.8.1: `@jax.jit(static_argnames=["bm"])` (changelog; used in matmul guide). `jax.jit` fun must be positional.
- [V] Signatures in 0.12.0.1:
```python
pl.BlockSpec(block_shape=None, index_map=None, pipeline_mode=None, *, memory_space=None)   # block_shape FIRST, index_map second
pl.pallas_call(kernel, out_shape, *, grid_spec=None, grid=(), in_specs=..., out_specs=..., scratch_shapes=(),
               input_output_aliases={}, debug=False, interpret=False, name=None, compiler_params=None,
               cost_estimate=None, metadata=None)
pltpu.CompilerParams(dimension_semantics=None, allow_input_fusion=None, vmem_limit_bytes=None, ...)
pltpu.InterpretParams(*, detect_races=False, out_of_bounds_reads='raise', uninitialized_memory='nan', ...)
```
- [V] index_map receives one arg per grid axis and returns BLOCK indices (not element offsets). Matmul guide pattern (bf16 in, f32 accumulate, scratch accumulator, `parallel,parallel,arbitrary`):
```python
pl.pallas_call(
  functools.partial(matmul_kernel, nsteps=k // bk),
  grid_spec=pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=0,
    in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
              pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
    out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
    scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
    grid=(m // bm, n // bn, k // bk)),
  out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
  compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")))(x, y)
# kernel body
acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
@pl.when(pl.program_id(2) == nsteps - 1)
def _(): z_ref[...] = acc_ref[...].astype(z_ref.dtype)
```
  Plain `grid=`/`in_specs=`/`scratch_shapes=` kwargs on `pallas_call` also exist (signature above); I ran a no-PrefetchScalarGridSpec 2D-grid kernel with `interpret=True` successfully. The guide still says scratch via `PrefetchScalarGridSpec` is "the only way" (older text); `scratch_shapes` on `pallas_call` is in the 0.12.0.1 signature.
- [V] Fused RHS transpose (our Q @ E.T, E stored (N, D)): `jax.lax.dot_general(q, e, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)` with E BlockSpec `(bn, bk)` and `lambda i,j,k: (j,k)` (matmul guide). Guide says v5e MXU supports fused transpose natively.
- [V] Tested on CPU (interpret): scoring kernel with fused mask, no D-tiling:
```python
def kern(q_ref, e_ref, m_ref, o_ref):
    s = jax.lax.dot_general(q_ref[...], e_ref[...], (((1,),(1,)),((),())), preferred_element_type=jnp.float32)
    o_ref[...] = jnp.where(m_ref[...] != 0, s, -jnp.inf)     # m block (1, bn) int32 broadcasts over rows
pl.pallas_call(kern, grid=(Q//bq, N//bn),
  in_specs=[pl.BlockSpec((bq,D), lambda i,j:(i,0)), pl.BlockSpec((bn,D), lambda i,j:(j,0)), pl.BlockSpec((1,bn), lambda i,j:(0,j))],
  out_specs=pl.BlockSpec((bq,bn), lambda i,j:(i,j)), out_shape=jax.ShapeDtypeStruct((Q,N), jnp.float32),
  compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel","parallel")), interpret=True)(q,e,mask)
```
  Matched jnp reference (allclose). `(1, bn)` mask block was OK in interpret mode; [U] on real TPU Mosaic a leading dim of 1 on int32 may need (8,128) tiling alignment; unverified, test on hardware (alternative: mask as bf16/f32 or int8 broadcast to (8, bn)).
- [V] Block-size guidance from matmul guide (v5e, bf16, f32 acc): bm=bk=bn=128 got 3.7%-6% of 197 TFLOP/s; `bm=512, bk=1024, bn=1024` got 78% at 4096^3 and 91.5% at 8192^3 vs XLA 84% / 92%. Rule: make blocks big, constrained by VMEM and pipeline bubbles; sweep. Trailing dims should be multiples of 128 (lanes) and second-last multiples of 8 (f32) / 16 (bf16) [U for bf16 16 rule; standard Mosaic tiling, not re-verified this session].
- [V] Per-TensorCore specs from `pltpu` hardware ref (https://docs.jax.dev/en/latest/pallas/tpu/hardware.html), VMEM capacity:
  v2 16 MiB, v3 16 MiB, v4 16 MiB, v5e 128 MiB, v5p 64 MiB, v6e 128 MiB, TPU7x 64 MiB, TPU8i 192 MiB, TPU8t 128 MiB. The docs expose `pltpu.get_tpu_info()`-style info (`vmem_capacity_bytes`, `mem_bw_bytes_per_second`, `bf16_ops_per_second` fields appear in the page's code cell).
  [U] Default scoped VMEM limit is smaller than capacity; raise with `CompilerParams(vmem_limit_bytes=...)`. The default value was not found in the docs I fetched; confirm on hardware before relying on >16 MiB blocks.
- [V] `dimension_semantics` entries: "parallel" or "arbitrary"; reduction axis (D tiling, or sequential top-k merge axis) must be "arbitrary" and last in grid.
- [V] Interpret mode: `interpret=True` works on CPU (ran it). New richer mode `interpret=pltpu.InterpretParams()` also ran on CPU in 0.12.0.1 (detects races, OOB; pads with NaN). TPU kernel docs: "compare against a kernel run with `interpret=pltpu.InterpretParams()`" (https://docs.jax.dev/en/latest/pallas/tpu/details.html). Global toggles: `pltpu.force_tpu_interpret_mode`, `pltpu.set_tpu_interpret_mode` exist.

## 2. top_k and approx_max_k

- [V] 0.12.0.1: `jax.lax.top_k(operand, k, *, axis=-1, is_stable=True) -> (values, int32 indices)`; works on bf16 (ran). `axis` arg is new vs older docs that only had last axis; fixed INT_MIN bug in 0.11-0.12 changelog. `jax.numpy.top_k` added 0.12.0 (numpy 2.6 API).
- [V] `jax.lax.approx_max_k(operand, k, reduction_dimension=-1, recall_target=0.95, reduction_input_size_override=-1, aggregate_to_topk=True) -> (values, indices)`; also `approx_min_k`. Ran on CPU (falls back to exact there; TPU gives the approximate partial-reduce). [U] CPU recall behavior not measured.
- Use as exact baseline: `lax.top_k`; as approx baseline: `approx_max_k(recall_target=0.95/0.99)`. Paper: "TPU-KNN" (Chern et al. 2022) is the origin of approx_max_k [U citation not fetched].
- Blockwise merge: concat running (Q,k) values with block's top_k and re-top_k, or pass `-inf` init. Pure JAX, jittable with `lax.scan` over N blocks.

## 3. Benchmarking and profiling

- [V] Async dispatch: always `.block_until_ready()` (https://docs.jax.dev/en/latest/profiling.html, older page https://docs.jax.dev/en/latest/benchmarking.html shows `%time f_jit(x).block_until_ready()` for compile time then `%timeit` for runtime). `benchmarking.html` is marked "older version"; maintained material is under "Benchmarking and profiling" (profiling.html). Warm up once; exclude compile; use `jax.device_put(x).block_until_ready()` before timing so transfers are not measured.
- [V] AOT (https://docs.jax.dev/en/latest/aot.html): `lowered = jax.jit(f).trace(*args).lower(); print(lowered.as_text()); compiled = lowered.compile(); compiled.cost_analysis()['flops']`. Old form `jax.jit(f).lower(*args).compile().as_text()` still works in 0.12.0.1 (ran; printed compiled HLO header). `cost_analysis()` may be None on some backends.
- [V] Profiler: `jax.profiler.start_trace(logdir)` / `stop_trace()` or `with jax.profiler.trace(logdir):`; view with XProf (`pip install xprof`, `tensorboard --logdir`), Perfetto via `create_perfetto_link=True`. `y.block_until_ready()` inside trace so device work is captured. Programmatic server: `jax.profiler.start_server(port)`, `python -m jax.collect_profile <port> <ms>`.
- [V] Compile/cache debug flags: `jax.config.update("jax_log_compiles", True)`; `jax.config.update("jax_explain_cache_misses", True)` (config_options page; persistent cache doc). Persistent cache: `jax_compilation_cache_dir`.
- Roofline plan: scoring FLOPs = 2*Q*N*D; bytes = (Q*D + N*D)*2 (bf16) + output bytes if materialized (4*Q*N f32) — materializing the (Q,N) score matrix dominates HBM traffic, which is exactly why fused-mask + blockwise top-k kernel matters. Intensity crossover (peak FLOPs/BW): v5e ~240 FLOP/B (guide).

## 4. Free TPU access and peaks

- Colab: [V] Colab announcement (https://github.com/googlecolab/colabtools/issues/5566, 2025-08-30): TPU v2-8 retired from free tier, replaced by TPU v5e-1 (select Runtime > Change runtime type > TPU v5e-1). [U] Whether v6e-1 is free: unconfirmed (Colab FAQ says types "vary over time" and limits are unpublished, https://research.google.com/colaboratory/faq.html). Free session max 12 h "depending on availability". A search-result summary says Colab CLI can provision v5e/v6e per plan [U, secondary]. Python version on Colab not verified: need >=3.12 for jax 0.11+; if not, `pip install "jax[tpu]<0.11"` [U]. Install in notebook: `!pip install -U "jax[tpu]"` then `!pip install git+https://github.com/<owner>/tpu-retrieval-bench`, restart runtime if JAX was already imported.
- Kaggle: [U] secondary sources (https://hyperautomationlabs.co/free/tpu ; repo https://github.com/YLiu95/kaggle-tpu-inference with JAX 0.10.2 measured on "Kaggle TPU v5e-8", 8 chips, 16 GB HBM each) say phone-verified accounts pick "TPU VM v5e-8", ~20 h/week quota. An older tutorial says "TPU VM v3-8" (GeeksforGeeks). Another snippet claims v6e-8 as of 2026-05 [U, source unclear]. Kaggle docs page (https://www.kaggle.com/docs/tpu) is JS-rendered and returned empty to browse.sh; needs `browser` lane to confirm. Session limit 9 h TPU [U]. Kaggle notebook has internet when enabled (needed for pip from GitHub); kaggle repo README notes libtpu metrics server doesn't start and TPU may need polling `jax.devices()` after allocation [U, single repo].
- TRC: [V] https://sites.research.google/trc/about/ : free Cloud TPUs for accepted researchers, 1,000+ devices, JAX/TF/PyTorch supported; expected to share work openly (papers, code, blogs). Generations/duration not stated on the About page; FAQ page fetch returned nothing useful [U]. Apply via the site.
- Peaks per TensorCore, from JAX docs hardware ref (https://docs.jax.dev/en/latest/pallas/tpu/hardware.html) [V]:

| Gen | VMEM | HBM BW | bf16 peak | HBM cap |
|---|---|---|---|---|
| v2 | 16 MiB | 358 GB/s | 23 TF | 8 GB |
| v3 | 16 MiB | 412.5 GB/s | 70 TF | 17 GB |
| v4 | 16 MiB | 615 GB/s | 138 TF | 17 GB |
| v5e | 128 MiB | 820 GB/s | 197 TF | 17 GB |
| v5p | 64 MiB | 1230 GB/s | 230 TF | 51 GB |
| v6e | 128 MiB | 1640 GB/s | 920 TF | 34 GB |
| TPU7x | 64 MiB | 3700 GB/s | 1155 TF | 103 GB |

  Notes: table is per TensorCore; v2/v3/v4/v5p chips have 2 TCs. Matmul guide uses v5e 197 TF / 819 GB/s (rounding). Docs table v3 70 TF differs from Google's commonly cited 123 TF/chip; trust the table for kernel math, label chip-level numbers [U] for v2/v3.
  Pallas on v2/v3 (Colab old, Kaggle v3-8 if still offered) [U]: Mosaic/Pallas TPU support for pre-v4 chips is not guaranteed; target v5e/v6e. Not verified.

## 5. GitHub Actions CPU testing

- [V] `pip install "jax[cpu]"` + `pl.pallas_call(..., interpret=True)` runs TPU-style kernels (BlockSpec, grid, scratch VMEM, `pl.when`, `program_id`) on CPU; verified with jax 0.12.0.1 on macOS CPU (Python 3.13). Both `interpret=True` and `pltpu.InterpretParams()` worked for my kernels.
- Limitations [V where noted]: interpret mode is slow (keep test shapes tiny, e.g. Q<=128, N<=1024); padding in interpret is NaN so OOB bugs surface (changelog); `dimension_semantics`, `vmem_limit_bytes`, Mosaic tiling/alignment errors, dtype layout limits and real VMEM overflow are NOT checked on CPU [U, inference: compile-only path not exercised]. So CPU CI proves numerics and index_map logic only; compile-first-time on TPU still needs a Colab/Kaggle smoke test.
- Mark tests: `@pytest.mark.tpu` skipped unless `jax.default_backend()=="tpu"`. Make `interpret` a module flag auto-set from backend.
- Also testable on CPU: top_k, blockwise merge, approx_max_k (exact fallback, so recall assertion trivial), AOT `lower().as_text()` (StableHLO text) for the jnp path. `.lower(lowering_platforms=("tpu",))` AOT lowering on CPU for Pallas TPU kernels may work to catch Mosaic errors [U, not tested; changelog mentions `trace(...).lower(lowering_platforms=("tpu",))` for HLO].

## Open items for parent
1. Confirm Kaggle TPU type and weekly quota in a live account (browser lane).
2. Confirm Colab Python version and v6e availability in a live runtime.
3. Test `(1,bn)` int32 mask BlockSpec and default scoped VMEM on real TPU.
4. Scratch artifacts (venv, fetched docs): /private/tmp/claude-501/-Users-maxpetrusenko/bc554082-b779-456b-94ab-6cc37bbbcf02/scratchpad/
