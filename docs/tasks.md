# Tasks

read_when: picking the next slice or checking status.

## Done
- S1 walking skeleton: numpy oracle (V0), jit top-k (V1), synthetic data, timing, records, CLI, CI.
- S2 blockwise (V2): `src/trbench/blockwise.py`, plus `make_score_only` / `make_select_only` for the R5 stage split.
- S3 Pallas score (V3): `src/trbench/pallas_score.py`, interpret mode on CPU, block params bm/bn/bk, `--sweep-blocks`.
- A1 `approx_max_k`: `src/trbench/approx.py`, recall target flag, measured recall recorded.
- TPU lowering from CPU (`trace(...).lower(lowering_platforms=("tpu",))`) works for V3; covered by `tests/test_pallas.py`.

- S4 harness + data: ann-benchmarks loader (`datasets.py`), roofline helpers, plots, Colab notebook, method doc.
- Integration (written before the `bench/run.py` split into `cli.py`, `algos.py`, `plan.py`, `execute.py`, `emit.py`; flags unchanged): `bench/run.py --dataset/--data-dir/--n-train/--n-test`, roofline fields in every record, `select_only` without matmul FLOP/s, `selection_over_matmul` (R5), blockwise and pallas_score charged for score-matrix HBM traffic.

- S5 TPU run (2026-11-18, Colab v5e-1): results in `results/tpu/`, report in `docs/RESULTS.md`, `bench/summarize.py`, `bench/dump_lowered.py`, `bench/trace_ops.py`, `bench/profile_one.py`. Mosaic accepted the Pallas kernel; stage-split ratio measured (0.77x to 8.16x). Notebook fixes: device check and profiler run as subprocesses, browser User-Agent in `bench/download.py`.

- S6 V4 fused (code): `pallas_fused_topk` + `pallas_fused_bucket`, interpret-mode exactness vs numpy_ref, TPU lowering test, roofline candidate bytes, fast_p in `bench/summarize.py`. TPU timing not run yet.

## Next
- S5b: rerun on TPU with `--sweep-blocks` (M 128, 1024), `approx_max_k` at recall targets 0.90 and 0.99, k=100, D in {96, 768}, GloVe-100, `--dump-hlo`. The block sweep decides whether the 6% vs 90% block-size claim shows up.
- S6b: run `pallas_fused_topk` on TPU (M 128/1024, N 1M, k 10/100): faster than `pallas_score`? Cost model in the `pallas_fused.py` docstring predicts a loss at large k.
