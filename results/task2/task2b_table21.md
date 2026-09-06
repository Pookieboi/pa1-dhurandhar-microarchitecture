# Table 2.1 — Metrics by matrix size and SIMD width
Machine: i7-13700HX, P-core pinned, governor=performance.
Flags: naive/AVX2 = -O2 -fno-tree-vectorize -mavx2 -mfma ; SSE = -O2 -fno-tree-vectorize -msse4.2
Instructions: isolated single-kernel driver. Times: harness median of 3 (kernel only).
AVX-512 (512-bit): NOT SUPPORTED on this CPU (no avx512f flag) -> N/A.

| Metric                    | N=512          | N=1024          | N=2048           |
|---------------------------|----------------|-----------------|------------------|
| **No SIMD (naive)**       |                |                 |                  |
| Instructions              |    838,022,006 |   6,572,916,527 |  52,076,483,939  |
| Execution time (ms)       |         52.45  |         435.92  |        3587.96   |
| **SIMD 128-bit (SSE)**    |                |                 |                  |
| Instructions              |    166,896,117 |   1,201,946,863 |   9,104,092,474  |
| Execution time (ms)       |          6.900 |          79.310 |         575.475  |
| Speedup vs no SIMD        |         7.55x  |          5.78x  |           6.23x  |
| **SIMD 256-bit (AVX2)**   |                |                 |                  |
| Instructions              |     65,653,310 |     395,047,199 |   2,653,294,523  |
| Execution time (ms)       |          4.531 |          58.369 |         499.959  |
| Speedup vs no SIMD        |        11.58x  |          7.47x  |           7.18x  |
| **SIMD 512-bit (AVX-512)**|      N/A       |       N/A       |       N/A        |

## Instruction-count reduction factors
| N    | naive/SSE | naive/AVX2 | SSE/AVX2 |
|------|-----------|------------|----------|
| 512  |   5.02x   |   12.76x   |  2.54x   |
| 1024 |   5.47x   |   16.64x   |  3.04x   |
| 2048 |   5.72x   |   19.63x   |  3.43x   |

Observations:
1. AVX2 gives the largest instruction reduction (12.8-19.6x): 8 floats/instr x FMA fusion.
2. SSE only reaches ~5x: 4 floats/instr and NO FMA under -msse4.2 (separate mul + add).
3. SSE/AVX2 = 2.5-3.4x rather than the 2x pure width would give; the excess is FMA fusion.
4. Speedup in TIME is far below the instruction reduction and falls with size for both
   widths -> the kernel is memory-bandwidth bound; wider SIMD helps only while cached.
