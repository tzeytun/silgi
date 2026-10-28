# Benchmark results

Generated with `python benchmarks/bench_speed.py` on 2026-10-28.

CPU: AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD | cores: 16 | Python 3.13.2

| Method | 2 MP / 1 % | 2 MP / 5 % | 12 MP / 1 % | 12 MP / 5 % | 24 MP / 1 % | 24 MP / 5 % |
|---|---:|---:|---:|---:|---:|---:|
| telea | 0.02 s | 0.13 s | 0.12 s | 0.59 s | 0.24 s | 1.20 s |
| ns | 0.02 s | 0.09 s | 0.11 s | 0.59 s | 0.24 s | 1.19 s |
| heal | 0.06 s | 0.53 s | 0.60 s | 3.39 s | 1.59 s | 6.73 s |
| patchmatch fast | 0.07 s | 0.29 s | 0.31 s | 0.78 s | 0.44 s | 1.34 s |
| patchmatch balanced | 0.14 s | 0.58 s | 0.54 s | 1.76 s | 0.84 s | 3.19 s |
| patchmatch best | 0.32 s | 1.18 s | 1.17 s | 4.15 s | 1.87 s | 8.30 s |
| alpha (reverse blend) | 0.01 s | n/a | 0.02 s | n/a | 0.03 s | n/a |

Times are wall-clock seconds for one removal (Numba kernels pre-compiled). The mask is an ellipse covering 1 % or 5 % of the image in the centre of a textured image; reverse alpha is timed with a 120 px text watermark.
