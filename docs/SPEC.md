# tpu-retrieval-bench: PRD + Spec

read_when: before any code change, benchmark run, or README claim.
Sources: `docs/presearch/papers.md`, `docs/presearch/benchmarks.md`, `docs/presearch/jax-pallas-tpu.md`.

## 1. Problem

Exact top-k embedding retrieval (the scoring step inside RAG) is a matmul followed by a selection.
Question: on a TPU, when does a custom Pallas kernel beat stock XLA, and when is XLA already fast enough?

Prior findings this bench must reproduce or refute:
- TPU-KNN (Chern et al., NeurIPS 2022, arXiv 2206.14286): exact k-selection cannot reach peak FLOP/s; a third roofline ceiling (vector/COP ops) bounds it. `approx_max_k` recall model ((L-1)/L)^(K-1).
- Samaga et al. (arXiv 2506.04165, TMLR 2026): on TPU v5e exact `lax.top_k` cost ~27x the matmul that produced the scores.
- Faiss GPU (arXiv 1702.08734): fusing distance with k-selection is >= 25% faster than unfused.
- JAX Pallas TPU matmul guide: 128^3 blocks ~6% of peak; 512x1024x1024 ~78-92% on v5e.
- Gap found: no published fused matmul + top-k Pallas TPU kernel. This repo's kernel is new work.

Target user: an engineer deciding whether to hand-write kernels for retrieval on TPU. Also Max's portfolio evidence for JAX/XLA/Pallas/TPU roles.

Non-goals (v1): API, UI, vector DB, ANN indexes (IVF/graph), multi-chip sharding.

## 2. Variants (one interface)

```python
def search(queries: Array[M, D], db: Array[N, D], k: int, *,
           metric: Literal["ip", "l2", "cos"] = "ip",
           mask: Array[N] bool | None = None,   # False = excluded candidate
           compute_dtype = jnp.bfloat16) -> tuple[Array[M, k] f32, Array[M, k] int32]
```
Scores: ip = q.e; cos = ip on L2-normalized rows; l2 = 2 q.e - ||e||^2 (same ranking as -||q-e||^2). Accumulate in f32 (`preferred_element_type=jnp.float32`). Masked candidates score -inf.

| id | name | what |
|---|---|---|
| V0 | `numpy_ref` | float64 NumPy oracle. Ties: lower id wins. |
| V1 | `jit_topk` | `jax.jit`: full `Q @ E.T` then `lax.top_k`. Baseline. |
| V2 | `blockwise` | scan over db blocks of size B; per-block `lax.top_k`; merge running top-k. Holds one M x B score block at a time, never the full M x N; modeled HBM traffic still counts each block's score write + read. |
| V3 | `pallas_score` | Pallas TPU kernel: tiled matmul + fused mask (+ l2 norm term) writing a score tile; selection stays in JAX (blockwise merge). Isolates "kernel helped?" from "blocking helped?". |
| V4 | `pallas_fused_topk` (stretch) | Pallas kernel also emits per-tile partial top-k (k <= 128) so only M x (N/bn) x k leaves VMEM. The novel piece. |
| A1 | `approx_max_k` | `lax.approx_max_k` at recall_target {0.90, 0.95, 0.99}. Reported separately, never ranked with exact. |

## 3. Requirements (ranked, each traces to a source)

R1 Correctness oracle: V1-V4 match V0 top-k ids and scores (bf16 tolerance) with the tie policy; recall@k = 1.0 for exact variants; any deviation fails CI. (Faiss 2024 recall definition: use ann-benchmarks k-nn recall, distance-threshold.)
R2 FLOP/s achieved = 2*M*N*D / device_time, reported as % of chip peak from a cited table (v5e 197 TFLOP/s bf16, 820 GB/s).
R3 Roofline label per run: MXU-bound, HBM-bound, or VPU/COP-bound (TPU-KNN 3-ceiling model).
R4 Fused vs unfused ablation: modeled HBM bytes of the score matrix write + re-read (V1-V3) vs none (V4), and timing delta.
R5 Stage split: time scoring and selection separately; report selection time as a multiple of matmul time (compare with the 27x claim).
R6 Approx baseline A1 with measured recall@10 against V0, on its own recall-vs-QPS curve.
R7 Harness hygiene: compile time measured separately via AOT `jit(f).lower(...).compile()`; warmup >= 3; `block_until_ready` before stopping the clock; repetitions >= 5; report min (ann-benchmarks QPS), p50, p95, p99; db resident on device; optional device time from `jax.profiler` trace (tallax style).
R8 Sweep axes: N in {2^16, 1e6 (non power of 2), 1,183,514}, D in {96, 100, 128, 768}, k in {10, 100}, M in {1, 16, 128, 1024}, dtype {bf16, f32}.
R9 Pallas block sweep (bm, bn, bk), (8,128) alignment, padding for non-multiples; record best per chip; VMEM per core 128 MiB on v5e/v6e.
R10 Reproducibility record per run: chip kind/count, jax/jaxlib/libtpu versions, python, dataset + hash, seed, XLA flags, git commit/dirty.

## 4. Datasets

- Synthetic Gaussian (seeded), any shape; used in CI.
- ann-benchmarks HDF5 (`train/test/neighbors/distances`): `sift-128-euclidean` (1M x 128, 10K queries, 525 MB), `glove-100-angular` (1,183,514 x 100, 10K queries, 485 MB). Same pair TPU-KNN used. GIST-960 optional (3.8 GB).
- Ground truth: use the file's `neighbors` for recall; also cross-check V0 on a sample.
- Do not commit dataset files; download script with size + sha check. Licenses: link, do not redistribute.

## 5. Results format

One JSON per (dataset, algo, params, batch, k) at `results/<platform>/<dataset>/k<k>/<algo>/<params>.json`, schema in `docs/presearch/benchmarks.md` section 10 (ann-benchmarks superset; `schema_version`, dataset, algorithm, query, hardware, metrics, timing, git). Plots: (a) recall vs QPS (log y) with exact as points, A1 as curve; (b) QPS and p99 vs batch size; (c) roofline scatter (% peak per variant).

## 6. Repo layout

```
src/trbench/  reference.py jit_topk.py blockwise.py pallas_score.py pallas_fused.py approx.py
              data.py timing.py roofline.py records.py env.py
bench/        run.py (driver) cli.py algos.py plan.py execute.py emit.py (the split run driver:
              arguments, registry, job plan, timing and recall, record writing)
              validate_results.py summarize.py plot.py doc_stats.py smoke_compare.py ...
scripts/      reproduce_all.sh tpu_smoke.sh export_constraints.sh release_notes.py
tests/        test_correctness.py test_timing.py test_records.py
notebooks/    colab_tpu.ipynb (pinned ref, constraints install, verify jax.devices(), reproduce_all.sh)
docs/         SPEC.md presearch/ RESULTS.md (findings + HLO/XProf excerpt)
.github/workflows/  ci.yml (CPU: Pallas interpret mode, ruff, mypy, pytest with coverage, audit, package)
                    codeql.yml scorecard.yml release.yml tpu-smoke.yml (manual, TPU runner)
```
Python >= 3.12, jax 0.12.x, `uv`. Pallas imports from `jax.experimental.pallas`, `pltpu.CompilerParams`. Files < 500 LOC.

## 7. Slices + proof gates

| slice | scope | proof gate |
|---|---|---|
| S1 walking skeleton | V0, V1, synthetic data, timing + records, `bench/run.py` smallest sweep, CI | pytest green on CPU; one JSON result with compile vs warm split; CI green on GitHub |
| S2 blockwise | V2 + stage split timing | V2 == V0 across shapes incl. non-multiple N, masks, k=100 |
| S3 Pallas score | V3 in interpret mode (CPU) + TPU path, block sweep params | V3 == V0 on CPU interpret; `lower(...)` text dumped |
| S4 harness + data | ann-benchmarks HDF5 loader, A1, roofline, plots | recall@10 on SIFT sample == 1.0 for exact; A1 recall curve |
| S5 TPU run | Colab v5e-1 notebook run (free), results committed | `jax.devices()` shows TPU; results/tpu/*.json; RESULTS.md with one HLO or XProf excerpt explaining a result |
| S6 stretch | V4 fused top-k kernel | V4 == V0; faster or explained-why-not |

Paid compute (GCP TPU, RunPod GPU) only after Max approves a $ estimate.

## 8. Risks / open

- Free Colab tier type and Python version (needs >= 3.12) unverified live.
- CPU interpret mode does not catch Mosaic tiling or VMEM overflow; S5 needed for real claims.
- `lax.top_k` on very wide rows may dominate (expected; that is the finding, not a bug).
- If stock XLA wins most shapes, publish that with the HLO explanation.
