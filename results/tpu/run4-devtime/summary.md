10 records

### tpu / synthetic / N=1000000 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.323 | 0.678 | 0.715 | 188857 | 0.9969 | - | 25.24 | HBM-bound | - | 9.14x |
| jit_matmul_topk | 0.920 | 6.193 | 6.274 | 20668 | 1.0000 | - | 2.70 | HBM-bound | - | - |
| pallas_fused_bucket (bm128,bn512,bk128) | 0.372 | 4.316 | 4.349 | 29658 | 1.0000 | - | 3.88 | HBM-bound | - | 1.43x |
| pallas_fused_topk (bm128,bn512,bk128) | 1.046 | 8.658 | 8.694 | 14784 | 1.0000 | - | 1.93 | HBM-bound | - | 0.72x |
| pallas_score_topk (bm128,bn512,bk128) | 0.346 | 6.248 | 6.370 | 20488 | 1.0000 | - | 2.68 | HBM-bound | - | 0.99x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.389 | 1.773 | 1.843 | 577401 | 1.0000 | - | 75.84 | HBM-bound | - | 25.36x |
| jit_matmul_topk | 0.902 | 44.975 | 45.098 | 22768 | 1.0000 | - | 2.97 | HBM-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.741 | 25.572 | 25.745 | 40044 | 1.0000 | - | 5.22 | HBM-bound | - | 1.76x |
| pallas_fused_topk (bm256,bn512,bk128) | 1.323 | 43.822 | 43.944 | 23367 | 1.0000 | - | 3.05 | HBM-bound | - | 1.03x |
| pallas_score_topk (bm256,bn512,bk128) | 0.520 | 51.117 | 51.331 | 20033 | 1.0000 | - | 2.61 | HBM-bound | - | 0.88x |

## Pallas block sensitivity

| algo | shape | configs | best bm x bn x bk | best QPS | worst config | worst QPS | best/worst | QPS basis |
|---|---|---|---|---|---|---|---|---|
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 29658 | 128x512x128 | 29658 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 14784 | 128x512x128 | 14784 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 1 | 128x512x128 | 20488 | 128x512x128 | 20488 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 40044 | 256x512x128 | 40044 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 23367 | 256x512x128 | 23367 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 1 | 256x512x128 | 20033 | 256x512x128 | 20033 | 1.000 | median |

## fast_p vs jit_topk (exact variants)

| algo | shapes | fast_1.0 | fast_1.25 |
|---|---|---|---|
| pallas_fused_topk | 2 | 0.500 | 0.000 |
| pallas_score_topk | 2 | 0.000 | 0.000 |

## Flags

none
