# Presearch: how leading benchmarks measure retrieval

Date: 2026-11-18. Scope: methodology and datasets to mimic in `tpu-retrieval-bench` (exact top-k on TPU in JAX). Every fact has a source (URL or `gh api` path). "Inference" marks my own reasoning. No performance numbers are invented; none are quoted except where a source states them.

## 0. Bottom line

1. Use the ann-benchmarks conventions as the comparable baseline: HDF5 files with `train/test/neighbors/distances`, k=10 default (ann-benchmarks `--count` default 10), recall vs QPS Pareto plot, build time, batch mode.
2. The only direct TPU precedent is TPU-KNN (arXiv 2206.14286). It used ann-benchmarks Glove (1,183,514 x 100, cosine) and Sift (1M x 128, L2), 10,000 queries, and plotted QPS vs recall against Faiss GPU flat/ivf. Mimic exactly those two datasets and plot style.
3. Exact search has recall 1.0 by definition (Faiss `bench_gpu_sift1m.py` comment: "the recall should be 1 at all times"). Recall vs QPS Pareto only matters for `approx_max_k` (recall_target sweep). Exact methods appear as a single point (or a vertical line) per batch size. Add a second plot: throughput and latency vs batch size (this is the cuVS Bench and Faiss `bench_gpu_sift1m` view).
4. No JAX/TPU kNN benchmark repo exists on GitHub beyond `jax.lax.approx_max_k` itself and `oliverdutton/tallax` (Pallas top-k, a different workload). Our repo fills a gap.

## 1. ann-benchmarks (erikbern)

Repo: https://github.com/erikbern/ann-benchmarks (5,753 stars, last push 2026-07-10, via `gh repo view`). Paper: Aumuller, Bernhardsson, Faithfull 2020 (cited in TPU-KNN refs).

- Datasets (README table, `gh api repos/erikbern/ann-benchmarks/contents/README.md`): all HDF5 at `http://ann-benchmarks.com/<name>.hdf5`; 100 stored neighbors per query.
  | name | dim | train | test | dist | size (README) |
  |---|---|---|---|---|---|
  | sift-128-euclidean | 128 | 1,000,000 | 10,000 | L2 | 501MB |
  | glove-100-angular | 100 | 1,183,514 | 10,000 | angular | 463MB |
  | glove-25 / 50 / 200 | 25/50/200 | 1,183,514 | 10,000 | angular | 121MB / 235MB / 918MB |
  | gist-960-euclidean | 960 | 1,000,000 | 1,000 | L2 | 3.6GB |
  | deep-image-96-angular | 96 | 9,990,000 | 10,000 | angular | 3.6GB |
  | nytimes-256-angular | 256 | 290,000 | 10,000 | angular | 301MB |
  | fashion-mnist / mnist | 784 | 60,000 | 10,000 | L2 | 217MB |
  | lastfm-64-dot | 65 | 292,385 | 50,000 | angular | 135MB |
  | coco-i2i / coco-t2i 512 | 512 | 113,287 | 10,000 | angular | 136MB |
- Dataset build (`ann_benchmarks/datasets.py`): `write_output` stores `train`, `test`, `neighbors` (n_test x 100, int), `distances` (float), attrs `distance`, `dimension`; ground truth by `BruteForceBLAS`. GloVe: Stanford `glove.twitter.27B.zip`, random split `test_size=10000, random_state=1`. SIFT/GIST from IRISA texmex. deep-image from Yandex disk.
- Metrics (`ann_benchmarks/plotting/metrics.py`): `k-nn` recall (mean; threshold is the distance of the k-th true neighbor plus epsilon 1e-3, so ties count as hits), `epsilon` recall (eps 0.01, 0.1), `rel` (relative distance error), `qps` = `1/best_search_time`, `p50/p95/p99/p999` of per-query `times`, `index_size`, `build_time`, `candidates`, `dist_computations`. Default plot: recall x-axis, QPS y-axis log (`python plot.py --x-scale logit --y-scale log`).
- Timing (`ann_benchmarks/runner.py`): `run_count` repetitions (CLI `--runs` default 5), keep the minimum total time (`best_search_time = min(...)`). Non-batch mode: `time.time()` around each `algo.query(v, count)`, one query at a time, single thread, Docker CPU-limited. Batch mode (`--batch`): all queries handed over at once, one wall-clock total; per-query latency is then reported as `total / len(X)`. Build time = wall clock around `fit`. Duplicate candidates in a result are asserted against. Per-run timeout default 2h.
- Output: one HDF5 per (algorithm, params) at `results/<dataset>/<count>/<algo>[-batch]/<params>.hdf5`; attrs `batch_mode, best_search_time, candidates, name, run_count, distance, count, build_time, index_size`; datasets `times` (n_queries, f32), `neighbors` (n_queries x k, int, -1 padded), `distances`.
- Relevance: has `BruteForceBLAS` and `bruteforce` algorithms as the exact baseline (module in `ann_benchmarks/algorithms/bruteforce`). Note: nothing here synchronizes devices; for JAX we must `block_until_ready` ourselves (inference).

## 2. big-ann-benchmarks (NeurIPS'21 and '23)

Repo: https://github.com/harsha-simhadri/big-ann-benchmarks (453 stars, push 2026-10-30). A fork of ann-benchmarks for billion scale and hardware evaluation.

- NeurIPS'21 (`neurips21/README.md`, `neurips21/t1_t2/README.md`): T1 in-memory CPU (Azure F32s_v2, 32 vCPU, 64GB; Faiss baseline), T2 out-of-core SSD (L8s_v2; DiskANN baseline), T3 custom hardware (`neurips21/t3`). Build limit 4 days/dataset. "Queries will be supplied in one shot". Score = sum of recall improvements over baseline at fixed throughput: 10,000 QPS (T1), 1,500 QPS (T2). Metric: recall@10 (AP for SSNPP range search) vs throughput. Datasets: bigann-1B, deep-1B, text2image-1B, ssnpp-1B, msturing-1B, msspacev-1B, with 1M/10M/100M subsets (`benchmark/datasets.py` registry: `bigann-1B/100M/10M`, `ssnpp-1M/10M/100M/1B`, `text2image-1B/1M`, `msturing-*`, `msspacev-*`). Local smoke test: `random-xs`.
- NeurIPS'23 (`neurips23/README.md`): tasks Filter (YFCC-10M, uint8 x192, L2, 100K queries, CC BY 4.0), Streaming (MS Turing 10M slice, float32 x100, L2; insert/delete/search runbook, 1h, 8GB DRAM, rank by average recall at checkpoints), OOD (Yandex Text-to-Image 10M, float32 x200, IP, 100K queries, CC BY 4.0), Sparse (MS MARCO SPLADE, 8,841,823 docs, 6,980 queries, avg ~120 nnz).
- Binary format: `.fbin/.u8bin` = 8-byte header (uint32 n, uint32 d) then row-major data (also used by cuVS Bench, see cuVS datasets doc).
- Output: HDF5 per run with attrs `build_time, index_size, algo, dataset` and `neighbors` (`benchmark/results.py`); extras: power capture (`benchmark/sensors/power_capture.py`), mean SSD IOs, `search_times`, `mean_latency` metrics (`benchmark/plotting/metrics.py`).
- Relevance: 10M subsets (`bigann-10M`, `text2image-10M`) are the "scale-up" tier we can add on multi-chip TPU, not on free Colab. Not verified: `--runs` default in big-ann `run.py`.

## 3. VectorDBBench (Zilliz)

Repo: https://github.com/zilliztech/VectorDBBench (1,202 stars, push 2026-11-18). Targets vector databases, not accelerators; useful for dataset choice and result schema.

- Datasets (`vectordb_bench/backend/dataset.py`): SIFT 128d L2 (500K small, 5M medium), GIST 960d L2 (100K, 1M), GloVe 200d cosine (1M), Cohere 768d cosine (100K, 1M, 10M; Wikipedia embeddings), OpenAI 1536d cosine (50K, 500K, 5M; embeddings of C4), Bioasq 1024d cosine, LAION 768d L2; MS MARCO and HotpotQA full-text-search cases (BM25). Hosted at `assets.zilliz.com/benchmark/` (S3) with Aliyun mirror (`vectordb_bench/__init__.py`). Cohere/OpenAI shipped with ground truth (`with_gt=True`).
- Metrics (`vectordb_bench/metric.py` class `Metric`): `qps`, `serial_latency_p50/p95/p99`, `recall` (+ `recall_min/p1/p5/p10`, `recall_at`), `ndcg`, `mrr`, `load_duration`, `insert_duration`, `optimize_duration`, concurrency lists `conc_num_list, conc_qps_list, conc_latency_p50/p95/p99/avg_list`, `additional_parameters`. Default `K_DEFAULT = 100`, `NUM_CONCURRENCY = [1,5,10,20,30,40,60,80]`, `CONCURRENCY_DURATION = 30` s (from `vectordb_bench/__init__.py`).
- Timing: serial runner (`runner/serial_runner.py`) loops over test queries, `time.perf_counter()` around each search, then mean/percentile latency and recall against ground truth; concurrent runner measures QPS under N clients for fixed 30 s. `--nq` (default 1) sets query vectors per request (README).
- Output: JSON `result_{date}_{label}_{db}.json` (`TestResult`/`CaseResult`: `metrics`, `task_config`, `label`, `dataset_metadata`, sensitive fields redacted). Plots: web UI, QPS and latency bar charts and recall/QPS views.
- Relevance: gives the embedding-model-scale datasets (768d, 1536d) and the serial-latency + concurrency methodology.

## 4. FAISS benchmark scripts

Repo: https://github.com/facebookresearch/faiss (41,799 stars). Scripts in `benchs/` (listing via `gh api repos/facebookresearch/faiss/contents/benchs`): `bench_gpu_sift1m.py`, `bench_gpu_1bn.py`, `bench_index_flat.py`, `bench_hybrid_cpu_gpu.py`, `bench_ivfflat_cuvs.py`, `bench_ivfpq_cuvs.py`, `bench_fw/` framework.

- `bench_gpu_sift1m.py`: loads SIFT1M (`datasets.py::load_sift1M`: base, query, learn, groundtruth), exact `GpuIndexFlatL2`, `index.add`, one warmup `index.search(xq, 123)`, then times full-query-set search for k = 1,2,4,...,1024. Reports `ms per query` and R@1 (recall 1 for exact). Then IVF with `nprobe` sweep, prints `nprobe, ms/query, recalls (R@1,R@10,R@100)`.
- Timing helper (`benchs/datasets.py::evaluate`): `t0=time.time(); D,I = index.search(xq,k); t1=time.time()`, returns `(t1-t0)*1000/nq` ms/query. Single timed run after one warmup. Recall = `(I[:, :i] == gt[:, :1]).sum()/nq`, i.e. 1-recall@i against the single true NN (different from ann-benchmarks k-recall: do not mix).
- `bench_gpu_1bn.py`: Deep1B / SIFT1B; `-ngpu`, add batch size 32768; reports 1-R@r and "intersection measure" (`eval_intersection_measure`) vs `nprobe`; stdout only, no CSV.
- Output: stdout text; Faiss `bench_fw` has JSON. TPU-KNN compared against Faiss GPU `flat`, `ivf,flat`, `ivf,pq50` (A100/V100) with IVF size 16384 and search fractions 0.24%, 0.61%, 1.22% (arXiv 2206.14286 section 6.2 and Figure 3).

## 5. cuVS / RAFT bench (cuvs-bench)

Repo: https://github.com/NVIDIA/cuvs (900 stars; former RAFT ann bench). Docs sources: `fern/pages/cuvs_bench/{introduction,running,datasets,wiki_all_dataset}.md`.

- Brute force: `cpp/bench/ann/src/cuvs/cuvs_brute_force_knn.cu`; algo config `python/cuvs_bench/cuvs_bench/config/algos/cuvs_brute_force.yaml` (empty build/search params); also `faiss_gpu_flat`.
- Run: `python -m cuvs_bench.run --dataset deep-image-96-inner --algorithms cuvs_cagra --batch-size 10 -k 10 --build --search`; export: `--data-export` (produces CSV); plot: `python -m cuvs_bench.plot`. `get_dataset --normalize` converts angular to inner product.
- Datasets: built-in list = deep-image-96-angular (10M x96), fashion-mnist, glove-50/100-angular (1.1M), mnist, nytimes-256, sift-128-euclidean (1M); all with 100-neighbor ground truth so k <= 100. Sources: ann-benchmarks HDF5 (converted by `hdf5_to_fbin.py`), big-ann for billion scale. `wiki-all`: 88M x 768 (paper: paraphrase-multilingual-mpnet-base-v2), ~251GB; subsets wiki_all_1M (2.9GB) and wiki_all_10M (29GB) at `https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar` (and `_10M`), each with 10K queries and ground truth. License: English Kaggle part CC BY-NCSA 4.0, Cohere part Apache 2.0 (check before redistributing).
- File format: `.fbin/.f16bin/.u8bin/.i8bin` base/query + `groundtruth.neighbors.ibin` + distances; 8-byte header (uint32 rows, uint32 dims).
- Metrics (running.md output table): build time (GPU, CPU), `Time` (per iteration), GPU latency per batch, `Latency` (wall batch), `Recall`, `items_per_second` (QPS), `end_to_end`, `Iterations`, hyperparameters. Google Benchmark based; two search modes: throughput (multiple threads) and latency; batch size is a first-class sweep axis (`--batch-size`; default sweep includes small batches such as 10 as in docs). Mode and sweep details beyond the docs quoted: not verified.
- Output: JSON from Google Benchmark per config, CSV after `--data-export`, PNG plots (recall vs throughput/latency).

## 6. MTEB / BEIR (embedding sources only)

- BEIR MS MARCO on HF: `BeIR/msmarco`, license cc-by-sa-4.0, size category 1M<n<10M, ~2.7 GB repo storage (HF API `https://huggingface.co/api/datasets/BeIR/msmarco`). Raw text only; embeddings must be generated.
- Pre-embedded alternative: `Cohere/msmarco-v2.1-embed-english-v3` on HF (page returned 200; card fields not parsed, license and dims unverified). `Cohere/wikipedia-22-12-simple-embeddings` page returned 401 to curl (needs browser or token); VDBBench uses Cohere Wikipedia 768d.
- Use MTEB/BEIR only to justify embedding dims (384/768/1024/1536) in a synthetic or embedded-corpus tier; they do not define a kNN speed protocol.

## 7. JAX/TPU kNN prior art (GitHub searches)

`gh search repos` for "pallas top_k", "jax knn tpu", "approx_max_k", "tpu knn benchmark jax", "jax brute force knn", "jax nearest neighbors", "vector search TPU" returned no dedicated TPU kNN benchmark repo (only unrelated hits such as `jpsotochirinos/KNN_TPU`, 1 star, "KNN work in TPU", 2022). Code search for `approx_max_k` found:

- `jax-ml/jax` `jax/_src/lax/ann.py`: API `jax.lax.approx_max_k(operand, k, reduction_dimension=-1, recall_target=0.95, reduction_input_size_override=-1, aggregate_to_topk=True)`; docstring MIPS example `dists = lax.dot(qy, db.T); approx_max_k(dists, k, recall_target)`; cites arXiv 2206.14286. `tests/ann_test.py` has `compute_recall` helper.
- TPU-KNN paper https://arxiv.org/abs/2206.14286 (Chern, Hechtman, Davis, Guo, Majnemer, Kumar). Evaluation (from the PDF text): datasets Glove1.2M (D=100 padded to 128, N=1,183,514, M=10,000 queries, cosine) and Sift1M (D=128, N=1,000,000, M=10,000, Euclidean), "adopted the methodology of public ANN benchmarks"; baselines Faiss GPU flat, ivf-flat, ivf-pq on V100/A100 chosen for peak-FLOPs parity with TPU v3/v4; Figure 3 = QPS vs recall; Figure 2 = roofline (measured GFLOP/s Table 2). Paper does not state batch size or warmup in the text I extracted.
- `oliverdutton/tallax` (5 stars, push 2026-02-27, Pallas top-k/sort for TPU; benchmark scripts `benchmarks/topk_benchmark.py`, `sampling_benchmark.py`). Its `tax/test_utils.py::benchmark`: warmup run, then one run under `jax.profiler.trace`, parse xprof trace for device time (README: "All timings extracted from xprof traces", single v5e). Workload is logits sampling (vocab 32K to 262K), not retrieval, but the timing method is the best TPU-native one found. Compared baseline `jax.jit(jax.lax.top_k)`.
- TF equivalent: `tf.math.approx_max_k` in `tensorflow/python/ops/nn_ops.py`.

## 8. Timing methodology to adopt (synthesis)

Source-backed practices: warmup before timing (Faiss `bench_gpu_sift1m`, tallax); minimum of repeated runs (ann-benchmarks, `--runs` 5); per-query latency percentiles p50/p95/p99 (ann-benchmarks, VDBBench); fixed-duration concurrency (VDBBench 30 s); batch size as an axis (cuVS `--batch-size`); device time from profiler (tallax).

Inference (JAX specifics, not from these benchmarks):
- Compile once per (shape, k, dtype); exclude compile from timing; call `jax.block_until_ready` on outputs before stopping the clock; report both wall time (includes host dispatch + D2H of top-k ids) and device time from `jax.profiler` trace.
- Keep database resident in HBM (like Faiss GPU index) and time the query path only; report host-to-device query transfer separately or include it in an "end-to-end" flag.
- Report repetitions, median and min; ann-benchmarks uses best-of-N for QPS and per-query times for percentiles.
- Memory sizing (arithmetic): float32 score matrix = batch x N x 4 bytes. For N=1,183,514 and batch 1,000 that is about 4.7 GB; batch 10,000 would be about 47 GB, over 16GB HBM. This is why blockwise top-k merge / Pallas fused scoring matter; test batches 1, 10, 100, 1,000 and 10,000 (the last only for blockwise/Pallas).
- Recall for exact variants must equal 1.0 under the ann-benchmarks distance-threshold definition; assert it, and verify ids too (ties across bf16 may differ).

## 9. Datasets recommended for free Colab/Kaggle TPU

Free TPU memory: Colab free v5e-1 = one chip, 16 GB HBM (GDE tutorial; Google Cloud v5e docs list 16 GB/chip, https://docs.cloud.google.com/tpu/docs/v5e; MaxText note says availability not guaranteed, https://maxtext.readthedocs.io/en/latest/guides/run_python_notebook.html). Older free Colab gave v2-8 (8GB per core; Hugging Face doc, SO answer; may be stale). Kaggle "8 v5e chips" came from one Medium post, unverified. Plan for 16GB single chip.

All sizes below are verified `Content-Length` from HTTP HEAD (2026-11-18) unless noted.

| # | dataset | shape (train x dim) | queries | dist | file | URL | bytes |
|---|---|---|---|---|---|---|---|
| 1 | SIFT-1M | 1,000,000 x 128 | 10,000 | L2 | `sift-128-euclidean.hdf5` | http://ann-benchmarks.com/sift-128-euclidean.hdf5 | 525,128,288 |
| 2 | GloVe-100 | 1,183,514 x 100 | 10,000 | angular (normalize, then IP) | `glove-100-angular.hdf5` | http://ann-benchmarks.com/glove-100-angular.hdf5 | 485,413,888 |
| 3 | GIST-1M (high-dim stress) | 1,000,000 x 960 | 1,000 | L2 | `gist-960-euclidean.hdf5` | http://ann-benchmarks.com/gist-960-euclidean.hdf5 | 3,844,648,288 |

Keys inside each HDF5: `train` (N x D float32), `test` (Q x D), `neighbors` (Q x 100 int), `distances` (Q x 100 float), attr `distance`. Resident size arithmetic: SIFT 1M x 128 x 4 = 512 MB; GloVe 1,183,514 x 100 x 4 = 473 MB; GIST 1M x 960 x 4 = 3.84 GB (fits 16GB; bf16 halves it).

Alternates: deep-image-96-angular (9,990,000 x 96, 10,000 queries, 3,848,008,288 bytes; HBM resident 3.84 GB f32; http://ann-benchmarks.com/deep-image-96-angular.hdf5) for a 10M tier; glove-25-angular (127,359,688 bytes) as a smoke test; `wiki_all_1M` (1M x 768, 2.9GB per docs, cuVS format) for a modern embedding-dim tier (not downloaded/HEAD-checked). Original sources: SIFT/GIST via IRISA texmex `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` (168,280,445 bytes) with `.fvecs/.ivecs`; terms: IRISA site asks citation of Jegou et al. (license not stated in what I read). GloVe: Stanford `glove.twitter.27B.zip` (the ann-benchmarks builder; Stanford GloVe is PDDL per Stanford page, not re-verified here). Deep: Yandex/Skoltech.

Note: ann-benchmarks site hosts converted HDF5 via the HTTP link; HTTPS not tested.

## 10. Recommended results schema (ann-benchmarks-compatible superset)

One JSON record per (dataset, method, params, batch_size) run. Optional sidecar HDF5 per run mirroring ann-benchmarks (`times`, `neighbors`, `distances`) so existing `ann_benchmarks/plotting` code and cuVS `data_export` ideas can be reused. Field names in brackets are the ann-benchmarks/VDBBench/cuVS equivalents.

```json
{
  "schema_version": "0.1",
  "run_id": "2026-11-18T12:00:00Z-<shortsha>",
  "dataset": {
    "name": "sift-128-euclidean",
    "source_url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "n_train": 1000000, "n_test": 10000, "dim": 128,
    "distance": "euclidean",
    "dtype_stored": "float32", "dtype_compute": "bfloat16"
  },
  "algorithm": {
    "name": "jit_matmul_topk | blockwise_topk_merge | pallas_score_topk | approx_max_k",
    "params": {"block_size": 65536, "recall_target": null, "precision": "bf16", "pallas_tile": [256, 512]},
    "exact": true
  },
  "query": {
    "k": 10, "batch_size": 1000, "mode": "batch",
    "n_queries_timed": 10000, "warmup_iters": 3, "repetitions": 5
  },
  "hardware": {
    "platform": "tpu", "device_kind": "TPU v5 lite", "n_devices": 1,
    "hbm_gb": 16, "host": "colab", "jax_version": "", "jaxlib_version": "", "libtpu_version": ""
  },
  "metrics": {
    "recall": 1.0,                       // [k-nn] ann-benchmarks distance-threshold definition
    "recall_epsilon_0.01": 1.0,          // [epsilon]
    "qps": 0.0,                          // [qps] = n_queries / best_search_time_s
    "best_search_time_s": 0.0,           // [best_search_time] min over repetitions, batch end-to-end
    "search_time_s_all": [],             // all repetitions
    "latency_ms_per_batch": {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0},   // [p50/p95/p99, serial_latency_*]
    "latency_ms_per_query_amortized": 0.0,  // Faiss-style ms/query = batch_time/batch
    "device_time_ms_per_batch": 0.0,     // from jax.profiler trace (tallax style)
    "build_time_s": 0.0,                 // [build_time] db load/transfer/normalize (0 for exact); compile time separate
    "compile_time_s": 0.0,
    "index_size_bytes": 0,               // [index_size] db bytes in HBM
    "peak_hbm_bytes": 0,
    "flops_achieved": 0.0, "flops_peak": 0.0, "mem_bw_achieved": 0.0   // TPU-KNN roofline view (optional)
  },
  "timing": {
    "clock": "perf_counter", "synchronized": "block_until_ready",
    "includes_h2d_query_transfer": false, "includes_d2h_result": true
  },
  "git": {"commit": "", "dirty": false},
  "created_at": "2026-11-18T12:00:00Z"
}
```

Plots to produce: (a) recall vs QPS, log y, logit x (ann-benchmarks style; exact methods as points at recall 1.0, `approx_max_k` as a curve over `recall_target` 0.80..0.99); (b) QPS and p99 batch latency vs batch size (cuVS/Faiss style); (c) optional roofline (TPU-KNN Fig 2). Use `ann_benchmarks` k=10 and also k=100 (VDBBench default) as a sweep. Files: `results/<dataset>/<k>/<algo>/<params>.json` (+ `.hdf5`) mirroring ann-benchmarks `build_result_filepath`.

## 11. Open questions / not verified

- big-ann `run.py` default for `--runs`; cuVS exact gbench flags (min_time, warmup iterations, throughput-vs-latency mode names); TPU-KNN batch size and warmup (paper text extracted lacks them).
- Colab free TPU current spec and Kaggle TPU spec (only secondary sources).
- Licenses: SIFT/GIST (IRISA), GloVe (Stanford), Deep (Yandex), Cohere embeddings; confirm before redistributing any file.
- Whether to include a multi-chip tier (`pmap`/sharded DB) in v1; none of the free tiers guarantees it.

## Source index

- ann-benchmarks: https://github.com/erikbern/ann-benchmarks (README, `ann_benchmarks/{datasets,runner,results,main}.py`, `plotting/metrics.py`)
- big-ann-benchmarks: https://github.com/harsha-simhadri/big-ann-benchmarks (`neurips21/README.md`, `neurips21/t1_t2/README.md`, `neurips23/README.md`, `benchmark/{datasets,results,runner}.py`)
- VectorDBBench: https://github.com/zilliztech/VectorDBBench (`README.md`, `vectordb_bench/{metric,models,__init__}.py`, `backend/dataset.py`, `backend/runner/serial_runner.py`)
- FAISS: https://github.com/facebookresearch/faiss/tree/main/benchs
- cuVS: https://github.com/NVIDIA/cuvs (`fern/pages/cuvs_bench/*.md`, `python/cuvs_bench/...`)
- TPU-KNN: https://arxiv.org/abs/2206.14286
- JAX ann: https://github.com/jax-ml/jax/blob/main/jax/_src/lax/ann.py
- tallax: https://github.com/oliverdutton/tallax
- BEIR MS MARCO HF: https://huggingface.co/datasets/BeIR/msmarco
- Cloud TPU v5e: https://docs.cloud.google.com/tpu/docs/v5e
