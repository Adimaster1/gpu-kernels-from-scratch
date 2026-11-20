GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU | 20 SMs | theoretical peak DRAM BW 131.7 GB/s | torch 2.15.0+cu132 (CUDA 13.2) | median of 100 iters, L2 flushed

| kernel | variant      |          n |  time_ms |  GB/s | % peak BW | GFLOP/s | correct | max_abs_err |
|--------|--------------|-----------:|---------:|------:|----------:|--------:|--------:|------------:|
| sum    | naive_atomic | 67,108,864 | 138.5124 |   1.9 |       1.5 |    0.48 |     yes |    1.99e-01 |
| sum    | smem_tree    | 67,108,864 |   3.6895 |  72.8 |      55.3 |   18.19 |     yes |    4.84e-02 |
| sum    | smem_tree_gs | 67,108,864 |   2.1105 | 127.2 |      96.6 |   31.80 |     yes |    3.47e-04 |
| sum    | warp_shuffle | 67,108,864 |   2.1074 | 127.4 |      96.7 |   31.84 |     yes |    5.91e-04 |
| sum    | torch        | 67,108,864 |   2.1125 | 127.1 |      96.5 |   31.77 |     yes |    1.12e-03 |
| max    | naive_atomic | 67,108,864 |   2.9071 |  92.3 |      70.1 |   23.08 |     yes |    0.00e+00 |
| max    | smem_tree    | 67,108,864 |   3.6905 |  72.7 |      55.2 |   18.18 |     yes |    0.00e+00 |
| max    | smem_tree_gs | 67,108,864 |   2.1110 | 127.2 |      96.6 |   31.79 |     yes |    0.00e+00 |
| max    | warp_shuffle | 67,108,864 |   2.1140 | 127.0 |      96.4 |   31.74 |     yes |    0.00e+00 |
| max    | torch        | 67,108,864 |   2.1166 | 126.8 |      96.3 |   31.71 |     yes |    0.00e+00 |
