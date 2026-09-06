# Task 2B (SIMD) — Findings & Interpretation

Machine: Intel i7-13700HX (Raptor Lake). Pinned to CPU 0 (P-core),
governor=performance. Flags (pinned, graded): -O2 -fno-tree-vectorize -mavx2 -mfma.
AVX-512 NOT available on this CPU (no avx512f flag) -> width study is 128 vs 256 only.
File edited: src/matmul_simd.cpp. Reference: matmul_naive.cpp (unmodified).
Raw data: task2b_sweep.txt, task2b_perf.txt, task2b_simd_width.txt,
task2b_instr_by_width.txt, task2b_correctness.txt. Table: task2b_table21.md.

## Kernel design
1x4 register-tiled AVX2 micro-kernel, un-blocked (no cache tiling/prefetch - Stage 2's job).
- One A row loaded once per K-step, reused across 4 B rows -> amortises A traffic.
- 4 independent __m256 FMA accumulators -> hides FMA latency via ILP.
- Vectorised K-loop with _mm256_fmadd_ps (8 floats/instr); scalar tail for K%8;
  N%4 handled by a 1-wide SIMD cleanup. Arbitrary M,N,K + strides (needed for llama.cpp).
  Verified on M=1, K<8, 37x53x71.

## Q1. Instruction count: naive -> SIMD, by width and size
Measured with an ISOLATED single-kernel driver (one call, no harness), because perf on
./bin/matmul also counts the shared naive reference + warmups + 3 reps and masks the kernel.
    N      naive           128-bit SSE     256-bit AVX2   | naive/SSE  naive/AVX2  SSE/AVX2
    512      838,022,006     166,896,117      65,653,310  |   5.02x     12.76x     2.54x
    1024   6,572,916,527   1,201,946,863     395,047,199  |   5.47x     16.64x     3.04x
    2048  52,076,483,939   9,104,092,474   2,653,294,523  |   5.72x     19.63x     3.43x
WHY: 256-bit processes 8 floats/instruction and FUSES mul+add (FMA), so ~8x width x ~2x
fusion ~= 16x, matching the measured 12.8-19.6x. The 128-bit build reaches only ~5x
because SSE4.2 predates FMA: it must issue separate _mm_mul_ps + _mm_add_ps, so it gets
the 4x width benefit but NOT the fusion benefit.
NOTE the SSE/AVX2 ratio is 2.5-3.4x, NOT the 2x that pure lane-width would predict -
the extra factor is exactly the FMA fusion. This is important to state: part of AVX2's
advantage here is FMA, not register width alone.

## Q2. Speedup vs matrix size (256-bit, task2b_sweep.txt)
    size    ms       GFLOP/s  speedup
    128     0.053     79.8    14.94x
    256     0.414     81.1    14.66x
    384     1.480     76.5    16.74x  <- peak
    512     4.569     58.8    11.39x
    768    25.192     36.0     7.14x
    1024   68.541     31.3     6.96x
    1536  210.796     34.4     7.37x
    2048  520.159     33.0     6.90x
Peak ~16.7x while the working set fits cache, decaying to ~7x once it does not; GFLOP/s
falls ~81 -> ~33. Instructions dropped 16.6x at 1024 but time only ~6.4x
=> MEMORY-BANDWIDTH BOUND. B is streamed once per A row with no reuse, so faster ALUs
stop helping. LIMITING FACTOR: no cache blocking - fixed in Stage 2.

## Q3. Speedup vs SIMD width (task2b_simd_width.txt) [README plot #4]
    size   128-bit SSE   256-bit AVX2   AVX2 advantage
    128      7.13x         11.68x         1.35x
    256      8.81x         15.09x         1.80x
    512      7.55x         11.58x         1.52x
    1024     5.78x          7.47x         1.36x
    1536     6.39x          7.54x         1.18x
    2048     6.23x          7.18x         1.15x
256-bit wins at EVERY size, but its advantage SHRINKS as the matrix grows (1.80x at 256
down to 1.15x at 2048). Same reason as Q2: once the kernel is DRAM-bound, doubling the
vector width no longer buys time because the bottleneck is data movement, not arithmetic.
This is the same conclusion as Task 1C: wide SIMD pays only while the data fits in cache.
512-bit (AVX-512) could not be tested - the CPU does not support it.

## Q4. Which SIMD intrinsics, and why
- _mm256_loadu_ps: unaligned 8-float load (B row j starts at j*ldb; llama.cpp tensor rows
  are not guaranteed 32B-aligned). Unaligned penalty negligible on this core.
- _mm256_fmadd_ps: fused mul+add, one instruction and one rounding step; needs -mfma.
  Responsible for roughly half the instruction reduction (see Q1).
- _mm256_setzero_ps: initialise the 4 accumulators.
- hsum256 helper (castps256_ps128 + extractf128 + add_ps + 2x hadd_ps + cvtss_f32):
  horizontal reduction of each accumulator, done ONCE per output tile (O(M*N)), NOT in
  the K-loop. AVX2 has no native horizontal-sum instruction.
- 128-bit variant uses _mm_loadu_ps / _mm_mul_ps / _mm_add_ps / _mm_hadd_ps (no FMA
  available under -msse4.2).
Doc note: the assignment appendix shows _pd (double) add/mul examples; we use the _ps
(float) equivalents because SGEMM is single-precision, plus FMA. The Intel Intrinsics
Guide linked in the doc is the sanctioned source.

## Caveats
- The N=128 width comparison is noise-prone: the two runs' naive baselines differ ~20%
  (0.665 vs 0.808 ms) at sub-millisecond scale. Treat sizes >= 256 as the reliable range
  for the width plot, or annotate the 128 point.
- The 128-bit variant is an UNGRADED separate file (kept in /tmp, built with -msse4.2 and
  linked against the AVX2 objects); the graded src/matmul_simd.cpp is unchanged AVX2.
