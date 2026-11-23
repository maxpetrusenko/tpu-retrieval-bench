111 records

### tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.716 | 1.775 | 1.845 | 563304 | 0.9802 | 0.9803 | 73.76 | HBM-bound | - | 23.37x |
| blockwise_topk | 0.401 | 53.525 | 53.678 | 18683 | 1.0000 | 1.0000 | 2.43 | VPU/COP-bound | - | 0.77x |
| jit_matmul_topk | 0.650 | 41.479 | 41.707 | 24109 | 1.0000 | 1.0000 | 3.14 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.744 | 25.561 | 25.791 | 39122 | 0.9997 | 0.9997 | 5.10 | HBM-bound | - | 1.62x |
| pallas_fused_topk (bm256,bn512,bk128) | 1.312 | 43.930 | 44.168 | 22763 | 1.0000 | 1.0000 | 2.98 | VPU/COP-bound | - | 0.94x |
| pallas_score_topk (bm256,bn512,bk128) | 0.552 | 51.037 | 51.222 | 19594 | 1.0000 | 1.0000 | 2.55 | VPU/COP-bound | - | 0.81x |
| score_only | 0.209 | 6.206 | 6.290 | 161122 | - | - | 21.15 | HBM-bound | 8.25 | 6.68x |
| select_only | 0.880 | 50.779 | 50.907 | 19693 | - | - | - | VPU/COP-bound | 8.25 | 0.82x |

### tpu / synthetic / N=65536 D=128 k=10 M=1

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.164 | 0.235 | 0.284 | 4247 | 0.9000 | - | 0.04 | HBM-bound | - | 0.81x |
| blockwise_topk | 0.622 | 0.591 | 0.644 | 1692 | 1.0000 | - | 0.01 | HBM-bound | - | 0.32x |
| jit_matmul_topk | 0.150 | 0.192 | 0.216 | 5214 | 1.0000 | - | 0.05 | HBM-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 3.496 | 0.518 | 0.564 | 1929 | 1.0000 | - | 0.02 | HBM-bound | - | 0.37x |
| pallas_fused_topk (bm16,bn512,bk128) | 1.440 | 0.686 | 0.707 | 1458 | 1.0000 | - | 0.01 | HBM-bound | - | 0.28x |
| pallas_score_topk (bm16,bn512,bk128) | 0.264 | 0.350 | 0.397 | 2860 | 1.0000 | - | 0.03 | HBM-bound | - | 0.55x |
| score_only | 0.135 | 0.245 | 0.320 | 4086 | - | - | 0.04 | HBM-bound | 0.79 | 0.78x |
| select_only | 0.052 | 0.175 | 0.230 | 5714 | - | - | - | HBM-bound | 0.79 | 1.10x (within noise) |

### tpu / synthetic / N=65536 D=128 k=10 M=16

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.198 | 0.215 | 0.271 | 74539 | 1.0000 | - | 0.70 | HBM-bound | - | 1.01x (within noise) |
| blockwise_topk | 0.328 | 0.316 | 0.365 | 50573 | 1.0000 | - | 0.46 | VPU/COP-bound | - | 0.68x |
| jit_matmul_topk | 0.226 | 0.216 | 0.241 | 74165 | 1.0000 | - | 0.67 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.126 | 0.346 | 0.399 | 46223 | 1.0000 | - | 0.42 | HBM-bound | - | 0.62x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.289 | 0.625 | 0.645 | 25594 | 1.0000 | - | 0.23 | VPU/COP-bound | - | 0.35x |
| pallas_score_topk (bm16,bn512,bk128) | 0.259 | 0.380 | 0.448 | 42116 | 1.0000 | - | 0.39 | VPU/COP-bound | - | 0.57x |
| score_only | 0.192 | 0.174 | 0.213 | 91962 | - | - | 0.88 | HBM-bound | 1.64 | 1.24x |
| select_only | 0.188 | 0.286 | 0.313 | 55976 | - | - | - | VPU/COP-bound | 1.64 | 0.75x |

### tpu / synthetic / N=65536 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.302 | 0.258 | 0.291 | 496169 | 0.9969 | - | 4.53 | HBM-bound | - | 2.54x |
| blockwise_topk | 0.282 | 0.622 | 0.644 | 205957 | 1.0000 | - | 1.91 | VPU/COP-bound | - | 1.05x |
| jit_matmul_topk | 0.219 | 0.655 | 0.692 | 195448 | 1.0000 | - | 1.82 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm128,bn512,bk128) | 0.140 | 0.448 | 0.471 | 285849 | 1.0000 | - | 2.53 | HBM-bound | - | 1.46x |
| pallas_fused_topk (bm128,bn512,bk128) | 0.196 | 0.778 | 0.829 | 164474 | 1.0000 | - | 1.43 | VPU/COP-bound | - | 0.84x |
| pallas_score_topk (bm128,bn512,bk128) | 0.261 | 0.705 | 0.826 | 181829 | 1.0000 | - | 1.67 | VPU/COP-bound | - | 0.93x |
| score_only | 0.166 | 0.263 | 0.284 | 487480 | - | - | 4.33 | HBM-bound | 2.25 | 2.49x |
| select_only | 0.778 | 0.589 | 0.680 | 217185 | - | - | - | VPU/COP-bound | 2.25 | 1.11x |

### tpu / synthetic / N=65536 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.643 | 0.351 | 0.413 | 2915674 | 0.9969 | - | 25.96 | HBM-bound | - | 9.96x |
| blockwise_topk | 0.376 | 3.610 | 3.659 | 283662 | 1.0000 | - | 2.44 | VPU/COP-bound | - | 0.97x |
| jit_matmul_topk | 0.220 | 3.497 | 3.539 | 292846 | 1.0000 | - | 2.52 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm256,bn512,bk128) | 0.140 | 1.952 | 1.976 | 524476 | 1.0000 | - | 4.54 | HBM-bound | - | 1.79x |
| pallas_fused_topk (bm256,bn512,bk128) | 0.211 | 2.969 | 3.026 | 344869 | 1.0000 | - | 2.99 | VPU/COP-bound | - | 1.18x |
| pallas_score_topk (bm256,bn512,bk128) | 0.364 | 3.867 | 3.926 | 264838 | 1.0000 | - | 2.29 | VPU/COP-bound | - | 0.90x |
| score_only | 0.156 | 0.593 | 0.604 | 1726510 | - | - | 15.00 | HBM-bound | 6.27 | 5.90x |
| select_only | 0.761 | 3.678 | 3.720 | 278402 | - | - | - | VPU/COP-bound | 6.27 | 0.95x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.145 | 0.671 | 0.738 | 1491 | 1.0000 | - | 0.21 | HBM-bound | - | 1.16x |
| blockwise_topk | 0.221 | 0.811 | 0.917 | 1233 | 1.0000 | - | 0.16 | HBM-bound | - | 0.96x |
| jit_matmul_topk | 1.052 | 0.778 | 0.795 | 1285 | 1.0000 | - | 0.17 | HBM-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.281 | 2.145 | 2.167 | 466 | 1.0000 | - | 0.06 | HBM-bound | - | 0.36x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.623 | 6.755 | 6.796 | 148 | 1.0000 | - | 0.02 | HBM-bound | - | 0.12x |
| pallas_score_topk (bm16,bn512,bk128) | 0.334 | 1.659 | 1.685 | 603 | 1.0000 | - | 0.08 | HBM-bound | - | 0.47x |
| score_only | 0.135 | 0.604 | 0.653 | 1656 | - | - | 0.22 | HBM-bound | 0.76 | 1.29x |
| select_only | 1.095 | 0.460 | 0.476 | 2175 | - | - | - | HBM-bound | 0.76 | 1.69x |

### tpu / synthetic / N=1000000 D=128 k=10 M=16

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.130 | 0.633 | 0.674 | 25283 | 0.9938 | - | 3.41 | HBM-bound | - | 1.72x |
| blockwise_topk | 0.283 | 1.368 | 1.385 | 11697 | 1.0000 | - | 1.54 | VPU/COP-bound | - | 0.80x |
| jit_matmul_topk | 0.633 | 1.088 | 1.104 | 14702 | 1.0000 | - | 1.95 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm16,bn512,bk128) | 0.300 | 2.114 | 2.149 | 7568 | 1.0000 | - | 1.00 | HBM-bound | - | 0.51x |
| pallas_fused_topk (bm16,bn512,bk128) | 0.651 | 6.877 | 6.920 | 2327 | 1.0000 | - | 0.31 | VPU/COP-bound | - | 0.16x |
| pallas_score_topk (bm16,bn512,bk128) | 0.314 | 1.674 | 1.694 | 9557 | 1.0000 | - | 1.25 | VPU/COP-bound | - | 0.65x |
| score_only | 0.129 | 0.673 | 0.691 | 23765 | - | - | 3.14 | HBM-bound | 1.27 | 1.62x |
| select_only | 0.622 | 0.855 | 0.871 | 18714 | - | - | - | VPU/COP-bound | 1.27 | 1.27x |

### tpu / synthetic / N=1000000 D=128 k=10 M=128

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.95) | 0.340 | 0.635 | 0.700 | 201532 | 0.9969 | - | 26.85 | HBM-bound | - | 9.72x |
| blockwise_topk | 0.267 | 5.819 | 5.946 | 21996 | 1.0000 | - | 2.89 | VPU/COP-bound | - | 1.06x |
| jit_matmul_topk | 0.911 | 6.174 | 6.258 | 20730 | 1.0000 | - | 2.71 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm128,bn1024,bk128) | 0.361 | 3.723 | 3.748 | 34379 | 1.0000 | - | 4.50 | HBM-bound | - | 1.66x |
| pallas_fused_topk (bm128,bn1024,bk128) | 1.125 | 6.291 | 6.378 | 20347 | 1.0000 | - | 2.68 | HBM-bound | - | 0.98x |
| pallas_score_topk (bm128,bn1024,bk128) | 0.316 | 6.028 | 6.087 | 21233 | 1.0000 | - | 2.78 | HBM-bound | - | 1.02x |
| score_only | 0.115 | 1.426 | 1.464 | 89782 | - | - | 11.84 | HBM-bound | 3.53 | 4.33x |
| select_only | 0.852 | 4.994 | 5.033 | 25633 | - | - | - | VPU/COP-bound | 3.53 | 1.24x |

### tpu / synthetic / N=1000000 D=128 k=10 M=1024

QPS = median pass (schema 0.2)

| algo | compile_s | p50 ms | p95 ms | QPS | recall | recall eps0.01 | % peak | roofline | sel/matmul | vs jit_topk |
|---|---|---|---|---|---|---|---|---|---|---|
| approx_max_k (target 0.8) | 0.422 | 1.756 | 1.774 | 583160 | 1.0000 | - | 76.58 | HBM-bound | - | 25.65x |
| approx_max_k (target 0.9) | 0.434 | 1.759 | 1.789 | 582266 | 1.0000 | - | 76.85 | HBM-bound | - | 25.61x |
| approx_max_k (target 0.95) | 0.437 | 1.777 | 1.793 | 576129 | 1.0000 | - | 75.73 | HBM-bound | - | 25.34x |
| approx_max_k (target 0.99) | 0.447 | 1.839 | 1.864 | 556913 | 1.0000 | - | 73.31 | HBM-bound | - | 24.50x |
| blockwise_topk | 0.425 | 53.824 | 54.055 | 19025 | 1.0000 | - | 2.48 | VPU/COP-bound | - | 0.84x |
| jit_matmul_topk | 0.941 | 45.042 | 45.125 | 22734 | 1.0000 | - | 2.96 | VPU/COP-bound | - | - |
| pallas_fused_bucket (bm512,bn1024,bk128) | 0.828 | 18.761 | 18.830 | 54580 | 1.0000 | - | 7.12 | HBM-bound | - | 2.40x |
| pallas_fused_topk (bm256,bn1024,bk128) | 1.464 | 34.474 | 34.654 | 29704 | 1.0000 | - | 3.88 | HBM-bound | - | 1.31x |
| pallas_score_topk (bm256,bn512,bk128) | 0.511 | 51.045 | 51.238 | 20061 | 1.0000 | - | 2.61 | VPU/COP-bound | - | 0.88x |
| score_only | 0.140 | 6.962 | 6.998 | 147075 | - | - | 19.28 | HBM-bound | 5.49 | 6.47x |
| select_only | 0.883 | 38.011 | 38.175 | 26940 | - | - | - | VPU/COP-bound | 5.49 | 1.18x |

## Pallas block sensitivity

| algo | shape | configs | best bm x bn x bk | best QPS | worst config | worst QPS | best/worst | QPS basis |
|---|---|---|---|---|---|---|---|---|
| pallas_fused_bucket | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 39122 | 256x512x128 | 39122 | 1.000 | median |
| pallas_fused_topk | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 22763 | 256x512x128 | 22763 | 1.000 | median |
| pallas_score_topk | tpu / sift-128-euclidean / N=1000000 D=128 k=10 M=1000 | 1 | 256x512x128 | 19594 | 256x512x128 | 19594 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 1929 | 16x512x128 | 1929 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 1458 | 16x512x128 | 1458 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=1 | 1 | 16x512x128 | 2860 | 16x512x128 | 2860 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 46223 | 16x512x128 | 46223 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 25594 | 16x512x128 | 25594 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=16 | 1 | 16x512x128 | 42116 | 16x512x128 | 42116 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 285849 | 128x512x128 | 285849 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 164474 | 128x512x128 | 164474 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=128 | 1 | 128x512x128 | 181829 | 128x512x128 | 181829 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 524476 | 256x512x128 | 524476 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 344869 | 256x512x128 | 344869 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=65536 D=128 k=10 M=1024 | 1 | 256x512x128 | 264838 | 256x512x128 | 264838 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 466 | 16x512x128 | 466 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 148 | 16x512x128 | 148 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1 | 1 | 16x512x128 | 603 | 16x512x128 | 603 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 7568 | 16x512x128 | 7568 | 1.000 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 2327 | 16x512x128 | 2327 | 1.000 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=16 | 1 | 16x512x128 | 9557 | 16x512x128 | 9557 | 1.000 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 34379 | 128x256x128 | 23081 | 1.490 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 20347 | 128x256x128 | 8854 | 2.298 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=128 | 4 | 128x1024x128 | 21233 | 128x256x128 | 20040 | 1.060 | median |
| pallas_fused_bucket | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 512x1024x128 | 54580 | 128x256x128 | 25184 | 2.167 | median |
| pallas_fused_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 256x1024x128 | 29704 | 512x256x128 | 6118 | 4.855 | median |
| pallas_score_topk | tpu / synthetic / N=1000000 D=128 k=10 M=1024 | 10 | 256x512x128 | 20061 | 128x256x128 | 18813 | 1.066 | median |

## fast_p vs jit_topk (exact variants)

| algo | shapes | fast_1.0 | fast_1.25 |
|---|---|---|---|
| blockwise_topk | 9 | 0.222 | 0.000 |
| pallas_fused_topk | 9 | 0.222 | 0.111 |
| pallas_score_topk | 9 | 0.111 | 0.000 |
| score_only | 9 | 0.000 | 0.000 |
| select_only | 9 | 0.000 | 0.000 |

## Flags

none
