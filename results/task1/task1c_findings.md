# Task 1C (SIMD) — Findings & Interpretation
Machine: Intel Raptor Lake HX, L1d 48K/12-way, pinned to CPU 0 (P-core),
governor=performance. Flags: -O2 -fno-tree-vectorize -mavx2 -mfma.
AVX-512 not available (fused off) -> only 128-bit and 256-bit tested.
Raw data: results/task1c_full.txt. 512x512 excluded (sub-ms, noise-dominated).

## Q1. Change in instruction count (naive -> SIMD), and why
Instruction-count REDUCTION FACTOR (naive_instr / simd_instr), from Section 4:
    width    K=3     K=5
    128      ~6.6    ~5.3
    256      ~12.5   ~10.4
- 256-bit halves the instruction count vs 128-bit (12.5/6.6 = 1.89x, ~2x as
  expected from 8 lanes vs 4 lanes per instruction).
- The reduction is LARGER than the raw lane count (8x for 256-bit). The extra
  ~1.5x comes from the loop structure the SIMD version is built on: the kernel
  weight is broadcast once and held in a register, removing the naive version's
  per-pixel ker[] reload and 2D index arithmetic.
- Reduction is nearly constant across matrix size (12.44/12.46/12.46/12.48),
  which is expected: instruction count is deterministic and scales with work,
  not with cache behaviour.

## Q2. Did we get speedup? How much, and what caused / limited it?
Wall-clock speedup vs naive (Section 4/7), K=3:
    size    128-bit   256-bit
    1024    2.50x     2.49x
    2048    1.92x     2.18x
    4096    1.21x     1.36x
    8192    1.04x     1.13x
KEY OBSERVATION: instruction count dropped ~12.5x but time dropped at most ~2.5x,
and 128-bit vs 256-bit are nearly identical in time despite 2x fewer instructions.
=> The kernel is MEMORY-BANDWIDTH BOUND, not compute bound. Inner op is
   o[ox] += i[ox]*a: two loads + one store (12 bytes) per 2 FLOPs. Once the ALU
   work is vectorized, the bottleneck is moving data, so wider SIMD barely helps.
LIMITING FACTOR / trend with size: speedup DECAYS as matrix grows (2.5x -> 1.1x)
   because the working set outgrows L2/L3 and the run becomes fully DRAM-bound.
   Small matrices fit in cache, so SIMD's compute reduction is visible there.

## Q3. Which SIMD intrinsics, and why
- _mm256_loadu_ps / _mm_loadu_ps: load 8 / 4 floats. UNALIGNED variant required
  because the input pointer is offset by +kx (kx=1,2,... not 32-byte aligned).
  Aligned loads would fault; unaligned penalty is negligible on this core.
- _mm256_set1_ps / _mm_set1_ps: broadcast one kernel tap into all lanes, hoisted
  out of the oy/ox loops (loop-invariant), so it runs only K*K times.
- _mm256_fmadd_ps / _mm_fmadd_ps: fused multiply-add, computes vi*va+vo in ONE
  instruction with ONE rounding step. Halves arithmetic instruction count vs
  separate mul_ps + add_ps and is why the Makefile enables -mfma.
- _mm256_storeu_ps / _mm_storeu_ps: unaligned store, same reasoning as load.

## Notes for whoever writes the report
- Prefer sp_reord over sp_naive where you need tight numbers: naive_ms drifts
  ~4% run-to-run (e.g. 8192: 239.02 vs 249.57 ms), sp_reord uses one invocation.
- conv_simd is reorder-structured + vectorized; it is NOT tiled. That is why it
  loses to tile at 4096/8192 (memory bound). Combining them is Task 1D.
- Plots to make: speedup vs matrix size (Section 5), speedup vs kernel size
  (Section 6), speedup vs SIMD width (Section 4/7).
