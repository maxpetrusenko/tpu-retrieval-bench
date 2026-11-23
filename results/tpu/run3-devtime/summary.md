10 records

### tpu / synthetic / N=1000000 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.318 | 0.691 | 0.753 | 185233 | 0.9969 | - | 26.39 | HBM-bound | - | 9.17x |
| jit_matmul_topk | 0.943 | 6.340 | 6.474 | 20190 | 1.0000 | - | 2.67 | HBM-bound | - | - |
| pallas_fused_bucket (bm128,bn512,bk128) | 0.373 | 4.314 | 4.336 | 29670 | 1.0000 | - | 3.88 | HBM-bound | - | 1.47x |
| pallas_fused_topk (bm128,bn512,bk128) | 0.997 | 8.694 | 8.807 | 14723 | 1.0000 | - | 1.92 | HBM-bound | - | 0.73x |
| pallas_score_topk (bm128,bn512,bk128) | 0.314 | 6.156 | 6.223 | 20791 | 1.0000 | - | 2.72 | HBM-bound | - | 1.03x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.397 | 1.762 | 1.810 | 581135 | 1.0000 | - | 76.41 | HBM-bound | - | 25.57x |
| jit_matmul_topk | 0.888 | 45.061 | 45.200 | 22725 | 1.0000 | - | 2.96 | HBM-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.766 | 25.478 | 25.593 | 40192 | 1.0000 | - | 5.24 | HBM-bound | - | 1.77x |
| pallas_fused_topk (bm256,bn512,bk128) | 1.295 | 43.750 | 44.059 | 23406 | 1.0000 | - | 3.05 | HBM-bound | - | 1.03x |
| pallas_score_topk (bm256,bn512,bk128) | 0.538 | 51.056 | 51.295 | 20056 | 1.0000 | - | 2.61 | HBM-bound | - | 0.88x |

## Pallas block sensitivity

| algo | shape | configs | best bm x bn x bk | best QPS | worst config | worst QPS | best/worst | QPS basis |
|---|---|---|---|---|---|---|---|---|
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 29670 | 128x512x128 | 29670 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 14723 | 128x512x128 | 14723 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 20791 | 128x512x128 | 20791 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 40192 | 256x512x128 | 40192 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 23406 | 256x512x128 | 23406 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 20056 | 256x512x128 | 20056 | 1.000 | median |

## fast_p vs jit_topk (exact variants)

| algo | shapes | fast_1.0 | fast_1.25 |
|---|---|---|---|
| pallas_fused_topk | 2 | 0.500 | 0.000 |
| pallas_score_topk | 2 | 0.500 | 0.000 |

## Flags

none
