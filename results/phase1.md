GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU | 20 SMs | theoretical peak DRAM BW 131.7 GB/s | torch 2.15.0+cu132 (CUDA 13.2) | median of 100 iters, L2 flushed

| kernel     | variant      |          n | time_ms |  GB/s | % peak BW | GFLOP/s | correct | max_abs_err |
|------------|--------------|-----------:|--------:|------:|----------:|--------:|--------:|------------:|
| vector_add | naive        | 16,777,216 |  1.6179 | 124.4 |      94.5 |   10.37 |     yes |    5.96e-08 |
| vector_add | float4       | 16,777,216 |  1.6599 | 121.3 |      92.1 |   10.11 |     yes |    5.96e-08 |
| vector_add | torch        | 16,777,216 |  1.6287 | 123.6 |      93.9 |   10.30 |     yes |    5.96e-08 |
| sum        | naive_atomic | 16,777,216 | 33.5232 |   2.0 |       1.5 |    0.50 |     yes |    4.74e-01 |
| sum        | smem_tree    | 16,777,216 |  0.9257 |  72.5 |      55.1 |   18.12 |     yes |    1.72e-03 |
| sum        | smem_tree_gs | 16,777,216 |  0.5325 | 126.0 |      95.7 |   31.51 |     yes |    7.21e-04 |
| sum        | warp_shuffle | 16,777,216 |  0.5335 | 125.8 |      95.5 |   31.45 |     yes |    7.21e-04 |
| sum        | torch        | 16,777,216 |  0.5356 | 125.3 |      95.2 |   31.33 |     yes |    9.66e-04 |
| max        | naive_atomic | 16,777,216 |  0.7311 |  91.8 |      69.7 |   22.95 |     yes |    0.00e+00 |
| max        | smem_tree    | 16,777,216 |  0.9298 |  72.2 |      54.8 |   18.04 |     yes |    0.00e+00 |
| max        | smem_tree_gs | 16,777,216 |  0.5335 | 125.8 |      95.5 |   31.45 |     yes |    0.00e+00 |
| max        | warp_shuffle | 16,777,216 |  0.5325 | 126.0 |      95.7 |   31.51 |     yes |    0.00e+00 |
| max        | torch        | 16,777,216 |  0.5356 | 125.3 |      95.2 |   31.33 |     yes |    0.00e+00 |
