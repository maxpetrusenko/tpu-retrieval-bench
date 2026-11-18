# tpu-retrieval-bench: papers and technical reports presearch

Date: 2026-11-18. Scope: exact top-k embedding retrieval on TPU (jit JAX `Q @ E.T` + `lax.top_k`, blockwise top-k merge, Pallas scoring kernel).
Method: PDFs fetched from arXiv and read with pdftotext; JAX docs fetched from docs.jax.dev. Items marked (VERIFIED) were read in source text. Items marked (DERIVED) are my arithmetic from verified numbers. Items marked (UNVERIFIED) were not confirmed in source text.

---

## 1. TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s

- Citation: Chern, Hechtman, Davis, Guo, Majnemer, Kumar. NeurIPS 2022. arXiv 2206.14286 (submitted 2022-06-28).
- Link: https://arxiv.org/abs/2206.14286 (PDF https://arxiv.org/pdf/2206.14286)

Key findings (VERIFIED from PDF text):
1. Extended roofline with a third axis: peak COP/s (gamma, non-matmul vector ops) in addition to FLOP/s (pi) and bytes/s (beta). Table 1: V100 125 TFLOP/s, 900 GB/s, 15.7 TCOP/s; A100 312, 1555, 19.5; TPU v3 126, 858, 4.0; TPU v4 274, 1144, 4.3.
2. Brute-force scoring (BLAS level 3) has memory arithmetic intensity I_MEM = 2MND / (4MN) ~= D/2 FLOP/byte if the M x N fp32 score matrix is written to HBM. Hence materializing scores is memory-bound. Their fix: aggregate inside the kernel so O(MN) bytes are never written.
3. COP budget per dot product: C <= 2D * gamma/pi. For D=128 this is only 4 COPs per score on TPU v4 and 16 on A100. Their conclusion (Remark 2): exact generic k-selection cannot be done with elementwise ops at peak FLOP/s on V100/A100/TPUv3/TPUv4. So exact top-k is VPU/COP-bound, not MXU-bound.
4. PartialReduce kernel uses C = 3 COPs per score (compare, conditional move value, conditional move index). C goes up by 1 when D is not a multiple of 128 and up by 1 when N is not a power of 2. Euclidean search hits the COP wall on TPU v4 (compute bound) but reaches peak on TPU v3; MIPS reaches peak.
5. Recall model (balls-in-bins): E[recall] = ((L-1)/L)^(K-1) with L bins; to reach target r need L >= 1/(1 - r^(1/(K-1))) ~= (K-1)/(1-r). For K=10, r=0.95 this gives L ~= 176 (DERIVED). Recall depends only on order statistics, not on the data distribution, so it is dataset-independent.
6. `jax.lax.approx_max_k(scores, k=10, recall_target=0.95)` is the exposed op (Listing 1: `einsum('ik,jk->ij')` then `approx_max_k`). Options: `reduction_dimension`, `recall_target` (default 0.95), `reduction_input_size_override`, `aggregate_to_topk` (default True, runs the ExactRescoring kernel). Achieved in-cache aggregation intensity ~4,700 FLOP/byte, far above ridge point.
7. Benchmarks: Glove 1.2M (cosine) and Sift 1M (L2), recall@10, compared with Faiss GPU Flat, IVF-Flat, IVF-PQ (IVF size 16384), on GPUs matched in peak to TPU. Claims better speed-recall trade-off at high recall. Exact QPS numbers not extracted (figure only). Dense single-chip only; multi-chip via pmap left as future work.

Requirements this implies for our bench:
- R-a: Report achieved FLOP/s (2*M*N*D / time) as a percent of peak for each variant, with chip-specific peak table (v5e 197 TFLOP/s bf16 and 819 GB/s per JAX Pallas docs; others to be sourced).
- R-b: Report a three-ceiling roofline per variant: MXU FLOP/s, HBM bytes/s, and VPU/COP-bound estimate. Flag "COP-bound" separately from "memory-bound".
- R-c: Track bytes of materialized score matrix (M*N*4) per variant; the unfused jit baseline writes it, blockwise and Pallas variants should not.
- R-d: Include `approx_max_k` as a baseline only with measured recall@10 vs exact at recall_target in {0.90, 0.95, 0.99}; label it approximate, never mix into the exact leaderboard.
- R-e: Sweep D in {128, 256, 768, 1024}, and include a non-multiple-of-128 D and a non-power-of-2 N case to test the +1 COP effects.
- R-f: Ties: JAX docs note approx_max_k is not stable on ties (VERIFIED, jax.lax.approx_max_k page); define a tie-handling policy for exactness checks.

---

## 2. A Faster Generalized Two-Stage Approximate Top-K

- Citation: Samaga B L, Yerram, Babbula, Jain, Netrapalli. TMLR 05/2026. arXiv 2506.04165 (submitted 2025-06-04).
- Link: https://arxiv.org/abs/2506.04165

Key findings (VERIFIED):
1. Motivation numbers: on TPUv5e and A100, `jax.lax.top_k` of the top-2% of Gemma 2 9B FFN activations takes 27x and 4.8x longer than the matmul that produced them. `approx_max_k` at 95% recall target still takes 9x the matmul time on v5e.
2. `jax.lax.top_k` is "the only exact Top-K implementation available for TPUs" at time of writing; `approx_max_k` is the only approximate one.
3. Generalization: stage 1 picks top-K' per bucket (K' in [1,K]). With fewer buckets and K' > 1 the stage-2 sort input (B*K') shrinks at equal expected recall. Their recall bound for the original algorithm is tighter by a factor of 2 than Chern et al.
4. Table 2: top-1024 of a 262,144-element array on TPUv5e at 99% recall target: baseline 305 us vs 27 us with K'=4 (11x). Implementation is an unfused Pallas first-stage kernel plus `jax.lax.sort_key_val`.
5. TPUv5e ridge-point framing: MXUs hold more than 95% of FLOPS, so only MXU-bound programs reach peak. They also estimate VPU gamma for v5e by timing VPU-bound kernels (Table 1; exact v5e gamma value not extracted).

Requirements:
- R-g: Time stage 1 (reduction) and stage 2 (sort/merge) separately in the blockwise and Pallas variants; stage 2 is the known bottleneck.
- R-h: Report "top-k time / matmul time" ratio per variant (this paper's headline metric: 27x for exact `lax.top_k` on v5e in their shape).
- R-i: Sweep k in {10, 100, 1024} since cost of exact k-selection scales with k.

---

## 3. Approximate Top-k for Increased Parallelism

- Citation: Key, Ribar, Cattaneo, Hudlass-Galley, Orr (Graphcore). arXiv 2412.04358 (2024-12-05).
- Link: https://arxiv.org/abs/2412.04358

Findings (VERIFIED, partial; abstract plus intro/figure text):
1. Bucketed top-k trades exactness for parallelism; TPU-KNN is the kb=1 instance with a bitonic sort in stage 2.
2. TPU-KNN's design choices target k << n (n ~ millions, k < 100); for k = 5-20% of n (sparsity), other (b, kb) settings are better. Retrieval is the first regime.
3. Compared against PyTorch top-k and RAFT exact top-k on H100 PCIe, batch m=128, fp32: speedups above 4x at good accuracy (figure caption). Specific totals not extracted.
4. Note the RAFT exact top-k baseline: a useful GPU exact reference point for what a tuned exact selector looks like.

Requirements:
- R-j: Define the exact baseline set clearly: `lax.top_k` (exact), `approx_max_k` aggregate (approx), blockwise merge (exact). Only exact ones may claim "exact".

---

## 4. Billion-scale similarity search with GPUs (FAISS GPU)

- Citation: Johnson, Douze, Jegou. IEEE Trans. Big Data 2019 (arXiv 1702.08734, 2017-02-28).
- Link: https://arxiv.org/abs/1702.08734

Findings (VERIFIED from PDF text; GPUs are Maxwell Titan X, CUDA 8):
1. WarpSelect: k-selection with state fully in registers, one pass, no cross-warp sync, supports k <= 1024. Odd-size merge networks; thread queues merged into a warp queue.
2. WarpSelect reaches 55% of the memory-bandwidth-limited peak for k=100 but only 16% at k=1000 (largest l=128,000 per row). It is 1.62x faster than fgknn at k=100 and 2.01x at k=1000. Peak is defined as scanning input once at peak memory bandwidth.
3. Exact search = GEMM for -2<x,y> (cuBLAS) + a fused kernel adding ||y||^2 and doing k-selection in registers, tiled so the nq x l distance matrix is never fully materialized. Fusion needs only 2 passes over the tile (GEMM write, k-select read). Un-fused variant is "at least 25% slower".
4. Sift1M exact search (l=1e6, d=128, nq=1e4): the GEMM is 1.28 TFLOP, "less than one second on current GPUs". Their method reaches 85% of the peak possible (GEMM plus tile read at peak bandwidth). cuBLAS GEMM has low efficiency at small reduction size d=128. Naive full sort via thrust::sort_by_key is more than 10x slower.
5. 10-NN graph of 1B vectors in under 12 hours on 4 Maxwell Titan X (abstract).

Requirements:
- R-k: Include a "fused vs unfused" ablation, reporting the extra score-matrix HBM pass (FAISS: >= 25% slowdown on GPU; check whether TPU shows the same).
- R-l: Use the same normalization of "peak possible": time = max(GEMM at peak FLOP/s, score-tile traffic at peak bandwidth). Report percent of that bound.
- R-m: Sweep k up to 1024 and show degradation of k-selection efficiency with k (FAISS: 55% -> 16% from k=100 to 1000).
- R-n: L2 handled as GEMM plus norms; support both inner product and L2 (cosine via pre-normalized IP) in correctness tests.

---

## 5. The Faiss library

- Citation: Douze, Guzhva, Deng, Johnson, Szilvasy, Mazare, Lomeli, Hosseini, Jegou. arXiv 2401.08281 (2024-01-16, v4 read).
- Link: https://arxiv.org/abs/2401.08281

Findings (VERIFIED from text):
1. Brute force (IndexFlat): distances either directly or, when queries are batched large enough, via matrix-multiplication decomposition (`knn` / `knn_gpu`). Top-k via binary heap on CPU, sorting network on GPU; for larger k a reservoir (unordered buffer of size k' > k, shrunk to k on overflow).
2. Defines accuracy metrics: n-recall@k = fraction of true top-n found in top-k; notes "recall@k" is used inconsistently across publications (1-recall@k vs k-recall@k). 10-recall@10 is the common retrieval metric.
3. Notes brute force is exact (no recall trade-off) and describes a benchmarking framework (bench_fw) and trade-off plots of speed vs accuracy.
4. No exact-search throughput numbers or cuVS integration details were found in the text I read (UNVERIFIED claim that this paper documents cuVS use; it cites CAGRA, Ootomo 2023).

Requirements:
- R-o: State metric definition explicitly: recall@10 = |approx top-10 intersect exact top-10| / 10 (10-recall@10).
- R-p: Use batch size as a first-class axis: matmul decomposition only pays off at large enough query batches; report single-query latency and large-batch throughput separately.

---

## 6. JAX docs: Pallas TPU matmul, pipelining, benchmarking, profiling

Sources (VERIFIED, fetched 2026-11-18):
- https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
- https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html
- https://docs.jax.dev/en/latest/benchmarking.html (marked "older version" banner; current page is /201/profiling.html)
- https://docs.jax.dev/en/latest/201/profiling.html
- https://docs.jax.dev/en/latest/_autosummary/jax.lax.approx_max_k.html

Findings:
1. Matmul doc ran on TPU v5e: 197 TFLOP/s bf16 and 819 GB/s HBM; op intensity ridge ~240.5 FLOP/byte. A 1024^3 matmul is 170.67 FLOP/byte in f32 and 341.33 in bf16.
2. Block-size effect (bf16, v5e, Pallas): blocks 128x128x128 gave 3.66% / 5.93% / 6.08% of peak for 1024^3 / 4096^3 / 8192^3. Blocks bm=512, bk=1024, bn=1024 gave 8.58% / 78.47% / 91.50%. XLA `jnp.matmul` gave 9.13% at 1024^3 and 166.1 TFLOP/s at 4096^3 (= 84.3%, DERIVED). Doc says bigger blocks help until VMEM or pipeline-bubble limits.
3. bf16: native MXU path takes bf16 inputs and accumulates in f32 via `preferred_element_type=jnp.float32`, with an f32 VMEM scratch accumulator (`pltpu.VMEM((bm, bn), jnp.float32)`) and a reduction grid axis last.
4. Per-kernel bandwidth: total HBM traffic = (m*k*n/bn + k*n*m/bm + m*n) * element_size, so bm/bn determine re-reads of Q and E. For retrieval: larger bn (corpus block) reduces Q re-reads, larger bm (query block) reduces E re-reads.
5. VMEM tiles are (8, 128); minor dims should align to these. Exact VMEM capacity per chip and `vmem_limit_bytes` default were NOT found in the pages read (UNVERIFIED; must check the TPU hardware reference page).
6. Doc timing harness: compile once with `jax.block_until_ready(f(...))`, then `timeit` N=100 trials of `block_until_ready`. Doc itself warns this is an upper bound because it includes Python dispatch.
7. Benchmarking guidance: first call includes compile time; dispatch is async so `block_until_ready()` is required; JAX defaults to 32-bit dtypes; time `device_put` separately from compute.
8. Profiling: `jax.profiler.start_trace(dir)` / `stop_trace()` with XProf/TensorBoard or Perfetto; XProf offers HLO Op Stats and HLO Op Profile; look for idle gaps between ops (host-bound).
9. `approx_max_k` signature: `(operand, k, reduction_dimension=-1, recall_target=0.95, reduction_input_size_override=-1, aggregate_to_topk=True)`; docs recommend wrapping in `jit` and show the MIPS example with `jax.lax.dot(qy, db.transpose())`.

Requirements:
- R-q: Harness: warmup (compile) call excluded; N timed iterations with `block_until_ready`; report median and p95; record compile time separately.
- R-r: Prefer device-side time (XProf trace) over wall-clock `timeit` for the Pallas kernel, since wall-clock includes Python dispatch; also report wall clock for end-to-end.
- R-s: Pallas block sweep over (bm, bn, bk) with bf16 inputs + f32 accumulate; log the config that wins per chip; expect 128-sized blocks to be badly slow (~6% of peak in the doc's matmul).
- R-t: Capture one XProf trace per variant on each chip type and commit the summary (not the raw trace).

---

## 7. Related recent work (lower priority)

| Source | Link | Relevance (VERIFIED from abstract unless noted) |
|---|---|---|
| StreamIndex (Triton, GPU), arXiv 2605.02568, 2026-05-04 | https://arxiv.org/abs/2605.02568 | Chunked partition-merge top-k driver that never materializes the full score tensor; on H200 materialize path OOMs at S=65,536 while StreamIndex runs to S=1,048,576 at 6.21 GB peak (32x). Mean recall vs materialized ground truth rounds to 1.0000, min >= 0.9980 across sweeps (they report this as recall; with a deterministic merge it should be 1.0, worth checking in ours). Direct precedent for blockwise merge. |
| JAXBench, arXiv 2607.20466 (listed 2026-05-19) | https://arxiv.org/abs/2607.20466 | 50 JAX workloads on TPU v6e; Pallas kernels from Tokamax; shows LLM-generated Pallas kernels vs XLA geomean 1.28x-1.36x speedups. Template for benchmark suite design with XLA as baseline. Date vs ID mismatch (ID 2607 but published 2026-05-19) noted, not resolved. |
| Ragged Paged Attention (Pallas/Mosaic), arXiv 2604.15464, 2026-04-16 | https://arxiv.org/abs/2604.15464 | Pallas TPU kernel reporting 86% memory bandwidth utilization in decode and 73% MFU in prefill on TPU7x: a template for reporting MBU and MFU for a Pallas kernel. |
| SOAR (ScaNN), arXiv 2404.00774, 2024-03-31 | https://arxiv.org/abs/2404.00774 | ANN indexing (spilling); not exact. Only abstract read; low relevance except as the ANN comparison point. |
| RadiK, arXiv 2501.14336, 2025-01-24 | https://arxiv.org/abs/2501.14336 | GPU radix top-k, supports large k beyond register-based merge limits; up to 2.5x (non-batch) and 4.8x (batch) over prior art. Abstract only. |
| Dr. Top-k, arXiv 2109.08219, 2021 | https://arxiv.org/abs/2109.08219 | Delegate-based filtering reduces top-k workload up to >99%. Abstract only. |
| Qrita, arXiv 2602.01518, 2026-02-02 | https://arxiv.org/abs/2602.01518 | Triton pivot-based top-k/top-p, default in vLLM sampler. Abstract only. |
| CAGRA, arXiv 2308.15136, 2023-08-29 | https://arxiv.org/abs/2308.15136 | GPU graph ANN (basis of cuVS); 3.8-8.8x over GPU SOTA at large batch, 90-95% recall. ANN, not exact. Abstract only. |
| vikramtiwari/jax-search-benchmark (GitHub) | https://github.com/VikramTiwari/jax-search-benchmark | Community repo: exact KNN with JAX on TPU meshes via SPMD, SIFT and synthetic data, spot TPU orchestration. README read only; claims of "sub-millisecond latency" unverified. Prior art to differentiate from. |
| NVIDIA cuVS pairwise distances doc | https://docs.nvidia.com/cuvs/user-guide/api-guides/other-apis/pairwise-distances | Search snippet only: brute-force = exact distance stage, no recall tradeoff, runtime and output memory grow with N*M. Not opened in full (UNVERIFIED beyond snippet). |

---

## 8. Not found / unverified

- No arXiv paper found for a fused matmul + top-k Pallas TPU kernel specifically (several arXiv queries on fused/Pallas/TPU/top-k returned none). Treat our Pallas kernel as novel relative to what I could find; the closest are the two-stage approximate top-k papers (sections 2, 3) and StreamIndex (GPU, Triton).
- FlashAttention-style tiling applied to kNN: no dedicated paper found. StreamIndex and FAISS fused L2+k-select are the nearest analogues.
- TPU VMEM capacity and default scoped limit per generation: not found in pages read.
- TPU-KNN per-benchmark measured FLOP/s (Appendix A.5) and QPS numbers: figures only, not extracted.
- Whether `lax.top_k` on TPU is implemented via sort or a partial reduction: not found in sources read; check XLA HLO dump in our own bench.
- ScaNN/TPU: no TPU-specific ScaNN source read.

---

## 9. Ranked top 10 spec requirements

1. Correctness oracle first: exact variants (jit `lax.top_k`, blockwise merge, Pallas) must match a float32 reference top-10 with a stated tie policy; report recall@10 (10-recall@10) = 1.0 for exact variants and flag any deviation. (Sources: Faiss 2024, `approx_max_k` tie note, StreamIndex.)
2. Report achieved FLOP/s as a percent of chip peak (2*M*N*D / time) per variant, with the peak table cited (v5e 197 TFLOP/s bf16, 819 GB/s). (TPU-KNN, JAX Pallas matmul doc.)
3. Three-ceiling roofline per variant: MXU, HBM bandwidth, VPU/COP; label each result memory-bound, COP-bound or MXU-bound. Expect exact k-select to be COP-bound. (TPU-KNN section 4.)
4. Fused vs unfused ablation: bytes of materialized M*N score matrix and the cost of the extra HBM pass. (FAISS GPU: >= 25% slower unfused.)
5. Split timing of stage 1 (scoring/block-local select) from stage 2 (merge/sort) and report top-k time as a multiple of matmul time. (Samaga: 27x on v5e for exact `lax.top_k` in their shape.)
6. Separate approximate baseline: `approx_max_k` at recall_target in {0.90, 0.95, 0.99} with measured recall@10 vs exact, never mixed into the exact ranking. (TPU-KNN, Samaga.)
7. Benchmark harness hygiene: exclude compile from timing, `block_until_ready`, median and p95 over N iterations, device-side time via XProf for kernels, compile time reported separately. (JAX benchmarking and profiling docs.)
8. Sweep axes: corpus N (include non-power-of-2), D (128 multiples and a non-multiple), k in {10, 100, 1024}, query batch M from 1 to large, dtype bf16 with f32 accumulate vs f32. (TPU-KNN C+1 effects, FAISS k scaling, batch dependence.)
9. Pallas tuning: block-size sweep (bm, bn, bk) with record of the best per chip; (8,128) tile alignment; document VMEM limit once found; expect 128^3 blocks near 6% of peak while 512x1024x1024 reached 78-92% in the doc's matmul. (JAX Pallas matmul doc.)
10. Reproducibility metadata per run: chip type and count, JAX/jaxlib/libtpu versions, dataset (synthetic plus SIFT1M/GloVe as in TPU-KNN), seed, XLA flags, committed XProf summary. (JAXBench, TPU-KNN datasets.)
