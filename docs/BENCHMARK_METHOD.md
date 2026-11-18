# Benchmark method

read_when: before reading or publishing any number from this repo, or adding a variant, dataset or plot.
Sources: `docs/presearch/benchmarks.md` (section numbers below), `docs/presearch/papers.md`, `docs/presearch/jax-pallas-tpu.md`.

## 1. What we copy from whom

| Practice | Source | Here |
|---|---|---|
| HDF5 `train/test/neighbors/distances`, 100 stored neighbors, attr `distance` | ann-benchmarks (benchmarks.md s1) | `datasets.load_ann`; euclidean -> `l2`, angular -> `cos` |
| k-nn recall with distance threshold (kth true distance + 1e-3, ties count) | ann-benchmarks `plotting/metrics.py` | `datasets.knn_recall` |
| QPS = n_queries / best (minimum) time of >= 5 repetitions; p50/p95/p99 | ann-benchmarks `runner.py`, VDBBench | `timing.py`, results JSON `metrics` |
| Batch mode: queries chunked by `--batch-size` (default 1024); batch size is a sweep axis; QPS = all queries / pass time | ann-benchmarks `--batch`, cuVS `--batch-size` | M in {1, 16, 128, 1024}; plot (b) |
| Warmup before timing, device time from a profiler trace | Faiss `bench_gpu_sift1m.py`, tallax | warmup >= 3; `jax.profiler` trace in the notebook |
| Datasets SIFT-1M (L2) and GloVe-100 (cosine), 10K queries | TPU-KNN (arXiv 2206.14286) | `bench/download.py sift|glove`; GIST-960 as a high-dim stress |
| Three ceilings: MXU FLOP/s, HBM B/s, VPU/COP ops/s | TPU-KNN s4 (papers.md s1) | `roofline.label` |
| Exact top-k time as a multiple of matmul time | Samaga et al. (27x on v5e, papers.md s2) | stage-split timing (S2) |
| Fused vs unfused: score matrix never written to HBM | Faiss GPU (>= 25% faster fused), TPU-KNN | `roofline.bytes_moved` materialization term; `pallas_fused_topk` / `pallas_fused_bucket` (V4) are unfused-free (they add only the candidate write, M * n_tiles * round_up(k', 128) * 8 B) |
| "Peak possible" = max(GEMM at peak, tile traffic at peak BW) | Faiss GPU paper (papers.md s3) | report % of that bound next to % of FLOP/s peak |

## 2. Timing rules

1. Compile is measured separately (AOT `jit(f).lower(...).compile()`), never in the timed loop.
2. Warmup >= 3 calls, then >= 5 timed repetitions; every call ends with `block_until_ready` before the clock stops.
3. Database and queries are resident on device. Host-to-device query transfer and device-to-host result copy are excluded; each record states this in its `timing` block.
4. Headline QPS = M / min(time). Latency fields come from per-batch repetitions (p50/p95/p99), so with 5 repetitions p99 is effectively the max. Say so when quoting it.
5. Device time next to wall time: `--device-time` traces `--reps` warm calls with `jax.profiler` and sums the `XLA Modules` events of the first `/device:` plane (parsed offline with `jax.profiler.ProfileData`, no TensorBoard). First batch only. Off TPU/GPU the trace has no device plane: `device_time_ms` is null and `device_time_note` says why. Wall time includes dispatch.
6. One chip, one process. Record chip kind, jax/jaxlib/libtpu versions, XLA flags, seed, dataset hash, git commit and dirty flag (R10).

### 2.1 Timing lanes (`bench/run.py`)

`--timing wall|loop|both` (default `wall`). Record fields: `timing.mode`, `timing.headline`, `timing.loop_iters`, `timing.loop_checksum`; metrics `best_search_time_s_wall`, `best_search_time_s_loop`, `qps_wall`, `qps_loop`, `latency_ms_per_batch_loop`, `search_time_s_all_loop`. `best_search_time_s` and `qps` follow the headline: wall for `wall` and `both`, loop for `loop`.

- **wall**: one call, `block_until_ready`, clock stops. Includes Python dispatch; on small shapes that is most of the number (tokamax `docs/benchmarking.md`).
- **loop**: the search runs `--loop-iters` (default 16) times inside one jitted `lax.fori_loop`, one host dispatch per rep; reported time = rep time / iters, summed over batches per pass. Anti-hoisting: each iteration's inputs pass through `lax.optimization_barrier` together with the running checksum (iteration i+1 depends on iteration i), and the carry is the sum of every output leaf, so XLA can neither hoist/CSE the body nor drop it. Cost: one reduction over the outputs per iteration, visible for `score_only` (output is the full M x N matrix), so compare stage splits within one lane only. Tests assert the per-dispatch time grows with the iteration count. Pattern: triton `do_bench_cudagraph` (unrolled burst, divided); no code copied. Loop and wall numbers are the same compute; the gap is dispatch overhead. Long bursts can thermally throttle; keep iters modest.
- **serial** (`--serial N`): VectorDBBench `serial_latency_*` style. N queries (cycling the query pool), batch 1, concurrency 1, each ended by `block_until_ready`. Records `serial_latency_ms` (p50/p95/p99/min/mean), `serial_n`, `serial_qps` (1 / mean). Use N >= 100 so p99 is meaningful; p99 of N < 100 is the max. Separate from batched QPS; never compare the two.
- Device time lane: see rule 5. Not usable on the Mac mini or CI (CPU); verify on the TPU before quoting it.

## 2.2 Invalid slots (k larger than the valid candidates)

When k exceeds the number of valid candidates (unmasked and not padding), every variant
(`numpy_ref`, `jit_topk`, `blockwise`, `pallas_score`, `approx_max_k`) fills the tail slots
with id `-1` and score `-inf`. Valid slots come first, sorted by score descending. Masked
candidates never appear with a finite score. Tests: `tests/test_padding.py`.

## 3. Recall definitions

- `knn_recall(found_dists, true_dists, k, eps=1e-3)`: ann-benchmarks definition, the primary number. A returned item counts if its distance is <= the kth true distance + eps. Needs distances in the ann-benchmarks convention (l2: Euclidean distance, cos: 1 - cos). `scores_to_distances` converts our scores (l2 score = 2q.e - |e|^2, so distance = sqrt(|q|^2 - score)).
- `id_set_recall(found_ids, true_ids, k)`: intersection of id sets over k. Stricter: with ties or bf16 rounding an exact variant can score < 1.0 here while scoring 1.0 on `knn_recall`. Report both for exact variants; gate CI on `knn_recall` and on V0 score agreement within bf16 tolerance.
- Dataset runs record `recall` (eps 1e-3, ann-benchmarks default) and `recall_epsilon_0.01` (ann-benchmarks "epsilon" metric), plus `recall_compute_dtype`. Measured on fake HDF5 (5000x64 L2, k=10): exact variants give 0.974 with bf16 and 1.0 with f32 at eps 1e-3; bf16 rounding moves distances by ~1e-3, so a bf16 `recall` < 1.0 is precision, not a bug. Read `recall_epsilon_0.01` for bf16, `recall` for f32. Synthetic runs keep eps 1e-2 (bf16) / 1e-4 (f32).
- `bench/run.py --dataset <name>` computes recall as `knn_recall` against the file's `distances` (or recomputed truth when `--n-train` subsets); the metric comes from the dataset attribute, overriding `--metric`.
- Subsetting the corpus (`n_train`) invalidates stored ground truth; `load_ann` recomputes it by brute force on the subset.
- Faiss `bench_gpu_*` recall is 1-recall@r against the single nearest neighbor. Do not mix it with k-nn recall.
- Exact variants must reach 1.0; `approx_max_k` is reported on its own recall-vs-QPS curve over `recall_target`, never ranked against exact points.

## 4. Roofline labels

`roofline.label` returns MXU-bound, HBM-bound, VPU/COP-bound or unknown.

- Peaks (per TensorCore, JAX Pallas hardware reference, jax-pallas-tpu.md s4): v5e (`TPU v5 lite`) 197 TFLOP/s bf16, 820 GB/s; v6e (`TPU v6 lite`) 920 TFLOP/s, 1640 GB/s. Other chips return `unknown`; they are not keyed because the table is per TensorCore and multi-TC chips were not verified at chip level.
- FLOPs = 2*M*N*D. Bytes = (M*D + N*D) * itemsize + M*k*8. `jit_topk`, `approx_max_k`, `blockwise` and `pallas_score` all add 2*4*M*N: they write all M*N f32 scores to HBM (blockwise and the Pallas score kernel do it in chunks) and read them back for `top_k`. `pallas_fused_topk` and `pallas_fused_bucket` (V4, per-tile top-k kept in VMEM) skip that traffic and instead pay M * n_tiles * round_up(k', 128) * 8 bytes of per-tile candidates (the kernel pads each output block to 128 lanes; f32 value + int32 id) (`kprime` = k for top-k, L lanes for bucket; `bench/run.py` passes n_tiles and kprime). The R4 ablation compares them with `pallas_score` on TPU.
- Stage helpers: `score_only` = reads Q, E and writes M*N*4 bytes; `select_only` = reads M*N*4 and writes the top-k.
- Every record carries `roofline_label`, `percent_of_peak`, `bytes_moved` and `arithmetic_intensity` in `metrics`. `select_only` does no matmul, so its `flops_achieved` and `percent_of_peak` are null; it records `selection_time_ms` instead.
- R5 ratio: when one `bench/run.py` invocation runs `score_only` and `select_only` on the same shape, both records get `selection_time_ms`, `matmul_time_ms` and `selection_over_matmul` (best selection time divided by best matmul time), and their label is recomputed from it.
- Rule: if measured selection time exceeds matmul time, label VPU/COP-bound (TPU-KNN: exact k-selection cannot run at peak FLOP/s). Otherwise the larger of ideal MXU time and ideal HBM time wins.
- Limits: ideal-time comparison ignores VPU cost unless stage times are passed in; f32 compute uses the bf16 peak, so f32 %-of-peak is a lower bound on efficiency. Ridge point v5e ~240 FLOP/B.

## 5. What not to compare

- Exact vs approximate QPS at different recall. `approx_max_k` points belong on their own curve.
- CPU numbers (CI, Mac mini) with TPU numbers. CPU results have no peak and exist only to exercise the harness.
- bf16 compute vs f32 compute, or l2 vs ip vs cos runs, without stating dtype and metric.
- Wall time including host transfer vs device-only time, or our batch-resident protocol vs ann-benchmarks single-thread CPU per-query latency.
- Our QPS vs published TPU-KNN or Faiss GPU figures: different chips, batch sizes and warmup protocols (TPU-KNN batch and warmup are not stated in the text we extracted).
- Pallas interpret-mode runs with TPU runs. Interpret mode checks numerics only, not Mosaic tiling or VMEM limits.
- Different recall definitions (section 3).
- Dataset licenses: link only, never redistribute (SIFT/GIST via IRISA texmex, GloVe via Stanford).
