# Between-session variance: run3 vs run4

Ratio = run4 / run3 (median QPS). Delta = |ratio - 1| * 100. CI overlap = the two bootstrap 95% CIs of the median intersect.

| cell | M | run3 QPS | run4 QPS | ratio | CIs overlap | abs delta % |
|---|---|---|---|---|---|---|
| sift-128-euclidean N=1,000,000 approx_max_k | 1000 | 565,089 | 563,304 | 0.997 | yes | 0.3 |
| sift-128-euclidean N=1,000,000 blockwise_topk | 1000 | 18,701 | 18,683 | 0.999 | yes | 0.1 |
| sift-128-euclidean N=1,000,000 jit_matmul_topk | 1000 | 24,120 | 24,109 | 1.000 | yes | 0.0 |
| sift-128-euclidean N=1,000,000 pallas_fused_bucket 256/512 | 1000 | 39,065 | 39,122 | 1.001 | yes | 0.1 |
| sift-128-euclidean N=1,000,000 pallas_fused_topk 256/512 | 1000 | 22,862 | 22,763 | 0.996 | no | 0.4 |
| sift-128-euclidean N=1,000,000 pallas_score_topk 256/512 | 1000 | 19,589 | 19,594 | 1.000 | yes | 0.0 |
| sift-128-euclidean N=1,000,000 score_only | 1000 | 159,471 | 161,122 | 1.010 | no | 1.0 |
| sift-128-euclidean N=1,000,000 select_only | 1000 | 19,688 | 19,693 | 1.000 | yes | 0.0 |
| synthetic N=65,536 approx_max_k | 1 | 3,177 | 4,247 | 1.337 | no | 33.7 |
| synthetic N=65,536 blockwise_topk | 1 | 1,644 | 1,692 | 1.029 | yes | 2.9 |
| synthetic N=65,536 jit_matmul_topk | 1 | 3,857 | 5,214 | 1.352 | no | 35.2 |
| synthetic N=65,536 pallas_fused_bucket 16/512 | 1 | 2,188 | 1,929 | 0.882 | no | 11.8 |
| synthetic N=65,536 pallas_fused_topk 16/512 | 1 | 1,512 | 1,458 | 0.964 | no | 3.6 |
| synthetic N=65,536 pallas_score_topk 16/512 | 1 | 3,587 | 2,860 | 0.797 | no | 20.3 |
| synthetic N=65,536 score_only | 1 | 4,807 | 4,086 | 0.850 | no | 15.0 |
| synthetic N=65,536 select_only | 1 | 4,940 | 5,714 | 1.157 | no | 15.7 |
| synthetic N=65,536 approx_max_k | 16 | 64,982 | 74,539 | 1.147 | no | 14.7 |
| synthetic N=65,536 blockwise_topk | 16 | 65,717 | 50,573 | 0.770 | no | 23.0 |
| synthetic N=65,536 jit_matmul_topk | 16 | 71,352 | 74,165 | 1.039 | yes | 3.9 |
| synthetic N=65,536 pallas_fused_bucket 16/512 | 16 | 57,822 | 46,223 | 0.799 | no | 20.1 |
| synthetic N=65,536 pallas_fused_topk 16/512 | 16 | 27,238 | 25,594 | 0.940 | no | 6.0 |
| synthetic N=65,536 pallas_score_topk 16/512 | 16 | 51,519 | 42,116 | 0.817 | no | 18.3 |
| synthetic N=65,536 score_only | 16 | 93,191 | 91,962 | 0.987 | yes | 1.3 |
| synthetic N=65,536 select_only | 16 | 58,753 | 55,976 | 0.953 | yes | 4.7 |
| synthetic N=65,536 approx_max_k | 128 | 546,603 | 496,169 | 0.908 | no | 9.2 |
| synthetic N=65,536 blockwise_topk | 128 | 191,269 | 205,957 | 1.077 | no | 7.7 |
| synthetic N=65,536 jit_matmul_topk | 128 | 215,396 | 195,448 | 0.907 | no | 9.3 |
| synthetic N=65,536 pallas_fused_bucket 128/512 | 128 | 265,902 | 285,849 | 1.075 | no | 7.5 |
| synthetic N=65,536 pallas_fused_topk 128/512 | 128 | 173,152 | 164,474 | 0.950 | no | 5.0 |
| synthetic N=65,536 pallas_score_topk 128/512 | 128 | 189,099 | 181,829 | 0.962 | yes | 3.8 |
| synthetic N=65,536 score_only | 128 | 611,045 | 487,480 | 0.798 | no | 20.2 |
| synthetic N=65,536 select_only | 128 | 210,682 | 217,185 | 1.031 | yes | 3.1 |
| synthetic N=65,536 approx_max_k | 1024 | 2,956,892 | 2,915,674 | 0.986 | yes | 1.4 |
| synthetic N=65,536 blockwise_topk | 1024 | 284,710 | 283,662 | 0.996 | yes | 0.4 |
| synthetic N=65,536 jit_matmul_topk | 1024 | 291,873 | 292,846 | 1.003 | yes | 0.3 |
| synthetic N=65,536 pallas_fused_bucket 256/512 | 1024 | 534,100 | 524,476 | 0.982 | no | 1.8 |
| synthetic N=65,536 pallas_fused_topk 256/512 | 1024 | 346,694 | 344,869 | 0.995 | yes | 0.5 |
| synthetic N=65,536 pallas_score_topk 256/512 | 1024 | 269,864 | 264,838 | 0.981 | no | 1.9 |
| synthetic N=65,536 score_only | 1024 | 1,770,142 | 1,726,510 | 0.975 | no | 2.5 |
| synthetic N=65,536 select_only | 1024 | 276,094 | 278,402 | 1.008 | yes | 0.8 |
| synthetic N=1,000,000 approx_max_k | 1 | 1,471 | 1,491 | 1.014 | yes | 1.4 |
| synthetic N=1,000,000 blockwise_topk | 1 | 1,185 | 1,233 | 1.040 | no | 4.0 |
| synthetic N=1,000,000 jit_matmul_topk | 1 | 1,226 | 1,285 | 1.048 | no | 4.8 |
| synthetic N=1,000,000 pallas_fused_bucket 16/512 | 1 | 479 | 466 | 0.973 | no | 2.7 |
| synthetic N=1,000,000 pallas_fused_topk 16/512 | 1 | 147 | 148 | 1.005 | no | 0.5 |
| synthetic N=1,000,000 pallas_score_topk 16/512 | 1 | 617 | 603 | 0.978 | no | 2.2 |
| synthetic N=1,000,000 score_only | 1 | 1,703 | 1,656 | 0.972 | no | 2.8 |
| synthetic N=1,000,000 select_only | 1 | 2,341 | 2,175 | 0.929 | no | 7.1 |
| synthetic N=1,000,000 approx_max_k | 16 | 25,039 | 25,283 | 1.010 | yes | 1.0 |
| synthetic N=1,000,000 blockwise_topk | 16 | 11,527 | 11,697 | 1.015 | no | 1.5 |
| synthetic N=1,000,000 jit_matmul_topk | 16 | 14,365 | 14,702 | 1.024 | no | 2.4 |
| synthetic N=1,000,000 pallas_fused_bucket 16/512 | 16 | 7,580 | 7,568 | 0.998 | yes | 0.2 |
| synthetic N=1,000,000 pallas_fused_topk 16/512 | 16 | 2,366 | 2,327 | 0.983 | no | 1.7 |
| synthetic N=1,000,000 pallas_score_topk 16/512 | 16 | 9,711 | 9,557 | 0.984 | no | 1.6 |
| synthetic N=1,000,000 score_only | 16 | 24,573 | 23,765 | 0.967 | no | 3.3 |
| synthetic N=1,000,000 select_only | 16 | 18,445 | 18,714 | 1.015 | no | 1.5 |
| synthetic N=1,000,000 approx_max_k | 128 | 195,471 | 201,532 | 1.031 | yes | 3.1 |
| synthetic N=1,000,000 blockwise_topk | 128 | 21,850 | 21,996 | 1.007 | yes | 0.7 |
| synthetic N=1,000,000 jit_matmul_topk | 128 | 20,914 | 20,730 | 0.991 | yes | 0.9 |
| synthetic N=1,000,000 pallas_fused_bucket 128/1024 (sweep) | 128 | 35,175 | 34,379 | 0.977 | no | 2.3 |
| synthetic N=1,000,000 pallas_fused_bucket 128/256 (sweep) | 128 | 23,049 | 23,081 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_bucket 128/512 | 128 | 29,536 | 29,644 | 1.004 | yes | 0.4 |
| synthetic N=1,000,000 pallas_fused_bucket 128/512 (sweep) | 128 | 29,278 | 29,758 | 1.016 | no | 1.6 |
| synthetic N=1,000,000 pallas_fused_topk 128/1024 (sweep) | 128 | 20,359 | 20,347 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 128/256 (sweep) | 128 | 8,843 | 8,854 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 128/512 | 128 | 14,711 | 14,663 | 0.997 | yes | 0.3 |
| synthetic N=1,000,000 pallas_fused_topk 128/512 (sweep) | 128 | 14,837 | 14,663 | 0.988 | no | 1.2 |
| synthetic N=1,000,000 pallas_score_topk 128/1024 (sweep) | 128 | 21,317 | 21,233 | 0.996 | yes | 0.4 |
| synthetic N=1,000,000 pallas_score_topk 128/256 (sweep) | 128 | 20,057 | 20,040 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_score_topk 128/512 | 128 | 20,823 | 20,819 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_score_topk 128/512 (sweep) | 128 | 21,006 | 20,863 | 0.993 | yes | 0.7 |
| synthetic N=1,000,000 score_only | 128 | 92,093 | 89,782 | 0.975 | no | 2.5 |
| synthetic N=1,000,000 select_only | 128 | 25,369 | 25,633 | 1.010 | no | 1.0 |
| synthetic N=1,000,000 approx_max_k | 1024 | 593,513 | 583,160 | 0.983 | no | 1.7 |
| synthetic N=1,000,000 approx_max_k | 1024 | 579,189 | 576,129 | 0.995 | yes | 0.5 |
| synthetic N=1,000,000 approx_max_k | 1024 | 561,709 | 556,913 | 0.991 | yes | 0.9 |
| synthetic N=1,000,000 approx_max_k | 1024 | 577,166 | 582,266 | 1.009 | yes | 0.9 |
| synthetic N=1,000,000 blockwise_topk | 1024 | 19,052 | 19,025 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 jit_matmul_topk | 1024 | 22,739 | 22,734 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_bucket 128/1024 (sweep) | 1024 | 51,590 | 51,627 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_bucket 128/256 (sweep) | 1024 | 25,198 | 25,184 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_bucket 128/512 (sweep) | 1024 | 37,909 | 38,025 | 1.003 | yes | 0.3 |
| synthetic N=1,000,000 pallas_fused_bucket 256/1024 (sweep) | 1024 | 53,622 | 53,404 | 0.996 | yes | 0.4 |
| synthetic N=1,000,000 pallas_fused_bucket 256/256 (sweep) | 1024 | 26,957 | 26,956 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_bucket 256/512 | 1024 | 40,020 | 40,196 | 1.004 | yes | 0.4 |
| synthetic N=1,000,000 pallas_fused_bucket 256/512 (sweep) | 1024 | 40,116 | 39,886 | 0.994 | no | 0.6 |
| synthetic N=1,000,000 pallas_fused_bucket 512/1024 (sweep) | 1024 | 53,943 | 54,580 | 1.012 | no | 1.2 |
| synthetic N=1,000,000 pallas_fused_bucket 512/256 (sweep) | 1024 | 27,501 | 27,443 | 0.998 | yes | 0.2 |
| synthetic N=1,000,000 pallas_fused_bucket 512/512 (sweep) | 1024 | 41,069 | 41,123 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 128/1024 (sweep) | 1024 | 25,102 | 25,108 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_topk 128/256 (sweep) | 1024 | 8,972 | 8,963 | 0.999 | no | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 128/512 (sweep) | 1024 | 16,887 | 16,856 | 0.998 | no | 0.2 |
| synthetic N=1,000,000 pallas_fused_topk 256/1024 (sweep) | 1024 | 29,722 | 29,704 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 256/256 (sweep) | 1024 | 13,454 | 13,449 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_topk 256/512 | 1024 | 23,414 | 23,449 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_fused_topk 256/512 (sweep) | 1024 | 23,401 | 23,410 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_topk 512/1024 (sweep) | 1024 | 20,915 | 20,923 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_topk 512/256 (sweep) | 1024 | 6,118 | 6,118 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_fused_topk 512/512 (sweep) | 1024 | 12,309 | 12,302 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_score_topk 128/1024 (sweep) | 1024 | 20,031 | 20,029 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_score_topk 128/256 (sweep) | 1024 | 18,732 | 18,813 | 1.004 | no | 0.4 |
| synthetic N=1,000,000 pallas_score_topk 128/512 (sweep) | 1024 | 19,791 | 19,813 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_score_topk 256/1024 (sweep) | 1024 | 20,035 | 20,052 | 1.001 | yes | 0.1 |
| synthetic N=1,000,000 pallas_score_topk 256/256 (sweep) | 1024 | 19,804 | 19,787 | 0.999 | yes | 0.1 |
| synthetic N=1,000,000 pallas_score_topk 256/512 | 1024 | 20,026 | 20,061 | 1.002 | yes | 0.2 |
| synthetic N=1,000,000 pallas_score_topk 256/512 (sweep) | 1024 | 20,105 | 20,055 | 0.998 | no | 0.2 |
| synthetic N=1,000,000 pallas_score_topk 512/1024 (sweep) | 1024 | 20,028 | 20,019 | 1.000 | yes | 0.0 |
| synthetic N=1,000,000 pallas_score_topk 512/256 (sweep) | 1024 | 20,048 | 20,008 | 0.998 | no | 0.2 |
| synthetic N=1,000,000 pallas_score_topk 512/512 (sweep) | 1024 | 20,060 | 20,010 | 0.998 | yes | 0.2 |
| synthetic N=1,000,000 score_only | 1024 | 147,441 | 147,075 | 0.998 | no | 0.2 |
| synthetic N=1,000,000 select_only | 1024 | 26,881 | 26,940 | 1.002 | yes | 0.2 |

| shapes | cells | median % | p90 % | max % | CIs overlap |
|---|---|---|---|---|---|
| small | 32 | 3.8 | 20.3 | 35.2 | 7 of 32 |
| large | 79 | 0.3 | 3.1 | 20.2 | 54 of 79 |
| large, N>=1M | 63 | 0.2 | 1.2 | 3.1 | 47 of 63 |
| large, N<1M | 16 | 2.8 | 9.2 | 20.2 | 7 of 16 |
| all | 111 | 0.7 | 9.3 | 35.2 | 61 of 111 |

## Claims: pallas_fused_topk vs jit_matmul_topk

| shape | config | cfg run3 | cfg run4 | x jit run3 | x jit run4 | run3 | run4 | status |
|---|---|---|---|---|---|---|---|---|
| sift-128-euclidean N=1,000,000 M=1000 | default | 256x512 | 256x512 | 0.95 | 0.94 | loss | loss | robust loss |
| synthetic N=65,536 M=1 | default | 16x512 | 16x512 | 0.39 | 0.28 | loss | loss | robust loss |
| synthetic N=65,536 M=16 | default | 16x512 | 16x512 | 0.38 | 0.35 | loss | loss | robust loss |
| synthetic N=65,536 M=128 | default | 128x512 | 128x512 | 0.80 | 0.84 | loss | loss | robust loss |
| synthetic N=65,536 M=1024 | default | 256x512 | 256x512 | 1.19 | 1.18 | win | win | robust win |
| synthetic N=1,000,000 M=1 | default | 16x512 | 16x512 | 0.12 | 0.12 | loss | loss | robust loss |
| synthetic N=1,000,000 M=16 | default | 16x512 | 16x512 | 0.16 | 0.16 | loss | loss | robust loss |
| synthetic N=1,000,000 M=128 | default | 128x512 | 128x512 | 0.70 | 0.71 | loss | loss | robust loss |
| synthetic N=1,000,000 M=128 | tuned | 128x1024 | 128x1024 | 0.97 | 0.98 | loss | loss | robust loss |
| synthetic N=1,000,000 M=1024 | default | 256x512 | 256x512 | 1.03 | 1.03 | win | win | robust win |
| synthetic N=1,000,000 M=1024 | tuned | 256x1024 | 256x1024 | 1.31 | 1.31 | win | win | robust win |
