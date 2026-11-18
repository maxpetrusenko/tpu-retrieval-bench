# Tasks

read_when: picking the next slice or checking status.

## Done
- S1 walking skeleton: numpy oracle (V0), jit top-k (V1), synthetic data, timing, records, CLI, CI.
- S2 blockwise (V2): `src/trbench/blockwise.py`, plus `make_score_only` / `make_select_only` for the R5 stage split.
- S3 Pallas score (V3): `src/trbench/pallas_score.py`, interpret mode on CPU, block params bm/bn/bk, `--sweep-blocks`.
- A1 `approx_max_k`: `src/trbench/approx.py`, recall target flag, measured recall recorded.
- TPU lowering from CPU (`trace(...).lower(lowering_platforms=("tpu",))`) works for V3; covered by `tests/test_pallas.py`.

- S4 harness + data: ann-benchmarks loader (`datasets.py`), roofline helpers, plots, Colab notebook, method doc.
- Integration: `bench/run.py --dataset/--data-dir/--n-train/--n-test`, roofline fields in every record, `select_only` without matmul FLOP/s, `selection_over_matmul` (R5), blockwise and pallas_score charged for score-matrix HBM traffic.

## Next
- S5: TPU run (Colab v5e-1). Check Mosaic acceptance of the (8, bn) aux block, default scoped VMEM, best bm/bn/bk, stage-split selection/matmul ratio.
- S6 stretch: V4 fused per-tile top-k kernel.
