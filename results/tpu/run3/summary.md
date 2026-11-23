111 records

### tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.677 | 1.770 | 1.901 | 565089 | 0.9802 | 0.9803 | 75.07 | HBM-bound | - | 23.43x |
| blockwise_topk | 0.435 | 53.473 | 53.646 | 18701 | 1.0000 | 1.0000 | 2.43 | VPU/COP-bound | - | 0.78x |
| jit_matmul_topk | 0.666 | 41.460 | 41.556 | 24120 | 1.0000 | 1.0000 | 3.14 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.798 | 25.598 | 25.947 | 39065 | 0.9997 | 0.9997 | 5.10 | HBM-bound | - | 1.62x |
| pallas_fused_topk (bm256,bn512,bk128) | 1.324 | 43.740 | 43.862 | 22862 | 1.0000 | 1.0000 | 2.98 | VPU/COP-bound | - | 0.95x |
| pallas_score_topk (bm256,bn512,bk128) | 0.516 | 51.049 | 51.309 | 19589 | 1.0000 | 1.0000 | 2.55 | VPU/COP-bound | - | 0.81x |
| score_only | 0.208 | 6.271 | 6.324 | 159471 | - | - | 20.95 | HBM-bound | 8.18 | 6.61x |
| select_only | 0.911 | 50.792 | 50.902 | 19688 | - | - | - | VPU/COP-bound | 8.18 | 0.82x |

### tpu / synthetic / N=65536 D=128 k=10 M=1

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.152 | 0.315 | 0.338 | 3177 | 0.9000 | - | 0.03 | HBM-bound | - | 0.82x |
| blockwise_topk | 0.638 | 0.608 | 0.668 | 1644 | 1.0000 | - | 0.01 | HBM-bound | - | 0.43x |
| jit_matmul_topk | 0.142 | 0.259 | 0.288 | 3857 | 1.0000 | - | 0.04 | HBM-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 3.494 | 0.457 | 0.469 | 2188 | 1.0000 | - | 0.02 | HBM-bound | - | 0.57x |
| pallas_fused_topk (bm16,bn512,bk128) | 1.345 | 0.661 | 0.700 | 1512 | 1.0000 | - | 0.01 | HBM-bound | - | 0.39x |
| pallas_score_topk (bm16,bn512,bk128) | 0.255 | 0.279 | 0.294 | 3587 | 1.0000 | - | 0.03 | HBM-bound | - | 0.93x |
| score_only | 0.129 | 0.208 | 0.225 | 4807 | - | - | 0.04 | HBM-bound | 0.97 | 1.25x |
| select_only | 0.053 | 0.202 | 0.219 | 4940 | - | - | - | HBM-bound | 0.97 | 1.28x |

### tpu / synthetic / N=65536 D=128 k=10 M=16

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.198 | 0.246 | 0.271 | 64982 | 1.0000 | - | 0.59 | HBM-bound | - | 0.91x |
| blockwise_topk | 0.303 | 0.243 | 0.298 | 65717 | 1.0000 | - | 0.60 | VPU/COP-bound | - | 0.92x |
| jit_matmul_topk | 0.220 | 0.224 | 0.277 | 71352 | 1.0000 | - | 0.65 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.139 | 0.277 | 0.294 | 57822 | 1.0000 | - | 0.54 | HBM-bound | - | 0.81x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.309 | 0.587 | 0.603 | 27238 | 1.0000 | - | 0.24 | VPU/COP-bound | - | 0.38x |
| pallas_score_topk (bm16,bn512,bk128) | 0.283 | 0.311 | 0.330 | 51519 | 1.0000 | - | 0.47 | VPU/COP-bound | - | 0.72x |
| score_only | 0.190 | 0.172 | 0.198 | 93191 | - | - | 0.84 | HBM-bound | 1.59 | 1.31x |
| select_only | 0.184 | 0.272 | 0.310 | 58753 | - | - | - | VPU/COP-bound | 1.59 | 0.82x |

### tpu / synthetic / N=65536 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.312 | 0.234 | 0.258 | 546603 | 0.9969 | - | 4.83 | HBM-bound | - | 2.54x |
| blockwise_topk | 0.302 | 0.669 | 0.748 | 191269 | 1.0000 | - | 1.68 | VPU/COP-bound | - | 0.89x |
| jit_matmul_topk | 0.219 | 0.594 | 0.661 | 215396 | 1.0000 | - | 1.94 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm128,bn512,bk128) | 0.129 | 0.481 | 0.529 | 265902 | 1.0000 | - | 2.33 | HBM-bound | - | 1.23x |
| pallas_fused_topk (bm128,bn512,bk128) | 0.213 | 0.739 | 0.786 | 173152 | 1.0000 | - | 1.54 | VPU/COP-bound | - | 0.80x |
| pallas_score_topk (bm128,bn512,bk128) | 0.249 | 0.677 | 0.704 | 189099 | 1.0000 | - | 1.68 | VPU/COP-bound | - | 0.88x |
| score_only | 0.166 | 0.209 | 0.221 | 611045 | - | - | 5.36 | HBM-bound | 2.90 | 2.84x |
| select_only | 0.749 | 0.608 | 0.632 | 210682 | - | - | - | VPU/COP-bound | 2.90 | 0.98x (within noise) |

### tpu / synthetic / N=65536 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.510 | 0.346 | 0.405 | 2956892 | 0.9969 | - | 26.58 | HBM-bound | - | 10.13x |
| blockwise_topk | 0.342 | 3.597 | 3.683 | 284710 | 1.0000 | - | 2.45 | VPU/COP-bound | - | 0.98x |
| jit_matmul_topk | 0.203 | 3.508 | 3.575 | 291873 | 1.0000 | - | 2.54 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.148 | 1.917 | 1.994 | 534100 | 1.0000 | - | 4.61 | HBM-bound | - | 1.83x |
| pallas_fused_topk (bm256,bn512,bk128) | 0.235 | 2.954 | 2.982 | 346694 | 1.0000 | - | 2.99 | VPU/COP-bound | - | 1.19x |
| pallas_score_topk (bm256,bn512,bk128) | 0.383 | 3.794 | 3.819 | 269864 | 1.0000 | - | 2.32 | VPU/COP-bound | - | 0.92x |
| score_only | 0.151 | 0.578 | 0.585 | 1770142 | - | - | 15.43 | HBM-bound | 6.48 | 6.06x |
| select_only | 0.721 | 3.709 | 3.743 | 276094 | - | - | - | VPU/COP-bound | 6.48 | 0.95x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.146 | 0.680 | 0.690 | 1471 | 1.0000 | - | 0.19 | HBM-bound | - | 1.20x |
| blockwise_topk | 0.223 | 0.844 | 0.873 | 1185 | 1.0000 | - | 0.16 | HBM-bound | - | 0.97x |
| jit_matmul_topk | 1.069 | 0.816 | 0.827 | 1226 | 1.0000 | - | 0.16 | HBM-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.285 | 2.087 | 2.101 | 479 | 1.0000 | - | 0.06 | HBM-bound | - | 0.39x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.638 | 6.789 | 6.943 | 147 | 1.0000 | - | 0.02 | HBM-bound | - | 0.12x |
| pallas_score_topk (bm16,bn512,bk128) | 0.320 | 1.622 | 1.651 | 617 | 1.0000 | - | 0.08 | HBM-bound | - | 0.50x |
| score_only | 0.131 | 0.587 | 0.625 | 1703 | - | - | 0.23 | HBM-bound | 0.68 | 1.39x |
| select_only | 1.148 | 0.427 | 0.454 | 2341 | - | - | - | HBM-bound | 0.68 | 1.91x |

### tpu / synthetic / N=1000000 D=128 k=10 M=16

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.143 | 0.639 | 0.750 | 25039 | 0.9938 | - | 3.38 | HBM-bound | - | 1.74x |
| blockwise_topk | 0.280 | 1.388 | 1.427 | 11527 | 1.0000 | - | 1.51 | VPU/COP-bound | - | 0.80x |
| jit_matmul_topk | 0.643 | 1.114 | 1.174 | 14365 | 1.0000 | - | 1.90 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.280 | 2.111 | 2.184 | 7580 | 1.0000 | - | 1.00 | HBM-bound | - | 0.53x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.618 | 6.761 | 6.802 | 2366 | 1.0000 | - | 0.31 | VPU/COP-bound | - | 0.16x |
| pallas_score_topk (bm16,bn512,bk128) | 0.324 | 1.648 | 1.662 | 9711 | 1.0000 | - | 1.29 | VPU/COP-bound | - | 0.68x |
| score_only | 0.130 | 0.651 | 0.665 | 24573 | - | - | 3.22 | HBM-bound | 1.30 | 1.71x |
| select_only | 0.624 | 0.867 | 0.963 | 18445 | - | - | - | VPU/COP-bound | 1.30 | 1.28x |

### tpu / synthetic / N=1000000 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.305 | 0.655 | 0.696 | 195471 | 0.9969 | - | 25.97 | HBM-bound | - | 9.35x |
| blockwise_topk | 0.295 | 5.858 | 5.919 | 21850 | 1.0000 | - | 2.86 | VPU/COP-bound | - | 1.04x |
| jit_matmul_topk | 0.858 | 6.120 | 6.220 | 20914 | 1.0000 | - | 2.73 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm128,bn1024,bk128) | 0.362 | 3.639 | 3.676 | 35175 | 1.0000 | - | 4.62 | HBM-bound | - | 1.68x |
| pallas_fused_topk (bm128,bn1024,bk128) | 1.171 | 6.287 | 6.392 | 20359 | 1.0000 | - | 2.68 | HBM-bound | - | 0.97x |
| pallas_score_topk (bm128,bn1024,bk128) | 0.310 | 6.004 | 6.198 | 21317 | 1.0000 | - | 2.79 | HBM-bound | - | 1.02x |
| score_only | 0.112 | 1.390 | 1.475 | 92093 | - | - | 12.06 | HBM-bound | 3.61 | 4.40x |
| select_only | 0.936 | 5.046 | 5.153 | 25369 | - | - | - | VPU/COP-bound | 3.61 | 1.21x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.8) | 0.439 | 1.725 | 1.742 | 593513 | 1.0000 | - | 77.67 | HBM-bound | - | 26.10x |
| approx_max_k (target 0.9) | 0.397 | 1.774 | 1.827 | 577166 | 1.0000 | - | 77.10 | HBM-bound | - | 25.38x |
| approx_max_k (target 0.95) | 0.408 | 1.768 | 1.820 | 579189 | 1.0000 | - | 75.73 | HBM-bound | - | 25.47x |
| approx_max_k (target 0.99) | 0.457 | 1.823 | 1.863 | 561709 | 1.0000 | - | 74.14 | HBM-bound | - | 24.70x |
| blockwise_topk | 0.395 | 53.749 | 54.059 | 19052 | 1.0000 | - | 2.48 | VPU/COP-bound | - | 0.84x |
| jit_matmul_topk | 0.900 | 45.032 | 45.818 | 22739 | 1.0000 | - | 2.96 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm512,bn1024,bk128) | 0.791 | 18.983 | 19.439 | 53943 | 1.0000 | - | 7.08 | HBM-bound | - | 2.37x |
| pallas_fused_topk (bm256,bn1024,bk128) | 1.490 | 34.453 | 34.578 | 29722 | 1.0000 | - | 3.88 | HBM-bound | - | 1.31x |
| pallas_score_topk (bm256,bn512,bk128) | 0.473 | 50.933 | 50.994 | 20105 | 1.0000 | - | 2.62 | HBM-bound | - | 0.88x |
| score_only | 0.135 | 6.945 | 6.981 | 147441 | - | - | 19.25 | HBM-bound | 5.49 | 6.48x |
| select_only | 0.887 | 38.094 | 38.259 | 26881 | - | - | - | VPU/COP-bound | 5.49 | 1.18x |

## Pallas block sensitivity

| algo | shape | configs | best bm x bn x bk | best QPS | worst config | worst QPS | best/worst | QPS basis |
|---|---|---|---|---|---|---|---|---|
| pallas_fused_bucket | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 39065 | 256x512x128 | 39065 | 1.000 | median |
| pallas_fused_topk | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 22862 | 256x512x128 | 22862 | 1.000 | median |
| pallas_score_topk | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 19589 | 256x512x128 | 19589 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 2188 | 16x512x128 | 2188 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 1512 | 16x512x128 | 1512 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 3587 | 16x512x128 | 3587 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 57822 | 16x512x128 | 57822 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 27238 | 16x512x128 | 27238 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 51519 | 16x512x128 | 51519 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 265902 | 128x512x128 | 265902 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 173152 | 128x512x128 | 173152 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 189099 | 128x512x128 | 189099 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 534100 | 256x512x128 | 534100 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 346694 | 256x512x128 | 346694 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 269864 | 256x512x128 | 269864 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 479 | 16x512x128 | 479 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 147 | 16x512x128 | 147 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 617 | 16x512x128 | 617 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 7580 | 16x512x128 | 7580 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 2366 | 16x512x128 | 2366 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 9711 | 16x512x128 | 9711 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 35175 | 128x256x128 | 23049 | 1.526 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 20359 | 128x256x128 | 8843 | 2.302 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 21317 | 128x256x128 | 20057 | 1.063 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 512x1024x128 | 53943 | 128x256x128 | 25198 | 2.141 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 256x1024x128 | 29722 | 512x256x128 | 6118 | 4.858 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 256x512x128 | 20105 | 128x256x128 | 18732 | 1.073 | median |

## fast_p vs jit_topk (exact variants)

| algo | shapes | fast_1.0 | fast_1.25 |
|---|---|---|---|
| blockwise_topk | 9 | 0.111 | 0.000 |
| pallas_fused_topk | 9 | 0.222 | 0.111 |
| pallas_score_topk | 9 | 0.111 | 0.000 |
| score_only | 9 | 0.000 | 0.000 |
| select_only | 9 | 0.000 | 0.000 |

## Flags

none
