# llama.cpp integration (README §4) — summary

## Correctness: PASS (this is the point of the exercise)
  [ok] student matmul_optimized was actually used by ggml
  [ok] student output matches stock output token-for-token
Confirms the kernel is correct under ggml's OWN tensor strides (non-contiguous lda/ldb/ldc)
and on the M=1 generation path — neither of which the microbenchmark exercises.

## Throughput: PARITY / within noise. Do NOT quote a win or a loss.
5 repeat runs, model stories260K (1.2 MB, F32), 1 thread, 5-token prompt + 63 generated.
Ratios = student / native:
    run   prompt   generation
     1    1.152    1.060
     2    1.210    1.064
     3    1.003    1.042
     4    0.986    0.994
     5    0.940    0.963   <- whole-system slowdown, BOTH builds ~30% down
    median 1.003   1.042      (student wins generation in 3 of 5)

WHY THIS IS NOISE, not a result:
- The NATIVE binary alone varied 1.44x across runs (6,858 -> 9,892 tok/s on generation).
  When the baseline swings 44%, a 4% build-to-build difference is meaningless.
- Total runtime ~6 ms; prompt eval is timed over FIVE tokens (0.3 ms).
- stories260K matrices are L1-resident, so the kernel's cache-blocking advantage cannot
  appear; generation runs at M=1, which takes the scalar tail, not the 4x4 fast path.
- An earlier single run showed student 13,055 vs native 16,949 tok/s (a large apparent
  loss); it never reproduced in these 5 runs and was a cold-start outlier.

CONCLUSION for the report: the llama.cpp demo validates CORRECTNESS inside a real model.
Performance claims should cite the microbenchmark (18-20x vs naive, ~96 GFLOP/s at
N=1024), where the matrices are large enough for cache blocking to matter.
