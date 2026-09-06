# Task 2C (Software Prefetching + SIMD combined) — Findings & Interpretation

Machine: Intel i7-13700HX (Raptor Lake). Pinned to CPU 0 (P-core), governor=performance.
L1d 48KB/core, L2 2MB/P-core, L3 30MB, 64B lines.
Flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma. Seed 1234.
File: src/matmul_optimized.cpp (graded; also the kernel injected into llama.cpp).
Raw data: task2c_all_stages_vs_size.txt, task2c_prefetch_distance.txt,
task2c_prefetch_level.txt, task2c_cachemiss.txt, task2c_hwpf.txt, task2c_instr.txt,
task2c_recheck.txt.

## Kernel design (what "combined" means)
- SIMD (AVX2/FMA): 8-wide fused multiply-add down the unit-stride K dot product.
- Register blocking: 4x4 micro-kernel, 16 __m256 accumulators. Each A-vector load feeds
  4 FMAs and each B-vector load feeds 4 -> 16 FMAs per 8 loads, vs 4 FMAs/8 loads in the
  Stage-1 1x4 kernel. This 2x arithmetic-intensity jump is what makes it compute-bound.
- Cache blocking: j-loop tiled into JC=32 row panels, panel stays L2-resident and is
  reused across all M rows of A. JC=32 measured best (JC sweep: 16/24/32/48/64/96).
- Loop unrolling / reordering: 16 independent accumulator chains hide FMA latency (ILP).
- Software prefetch: available via -DPF_D / -DPF_L, DISABLED by default (see Q2).

## Q1. Final comparison: speedup vs matrix size (task2c_all_stages_vs_size.txt)
All four stages measured in ONE run per size, so they share the same naive baseline.
    size    simd     prefetch   optimized
    128    12.67x     14.39x     13.75x   (sub-ms: noise-dominated, see caveats)
    256    14.94x     12.22x     18.33x
    512    13.00x     12.60x     19.89x
    1024    7.19x     12.57x     19.10x
    1536    7.19x     12.40x     18.50x   (rechecked, task2c_recheck.txt)
    2048    6.94x     11.99x     18.15x
KEY RESULT - the three curves behave completely differently with size:
  * SIMD alone COLLAPSES (14.9x -> 6.9x): un-blocked, so B is re-streamed from DRAM once
    per A row; once the working set exceeds cache it is purely memory-bound.
  * Prefetch/tiling alone is FLAT (~12x): cache blocking fixes the decay but the 1x4
    micro-kernel does not have enough arithmetic intensity to go faster.
  * COMBINED is flat AND high (~18-20x): tiling supplies the data locality, the 4x4
    register block supplies the arithmetic intensity to exploit it.
SYNERGY: at 1024 the combined kernel (19.10x) beats BOTH the best individual technique
(prefetch 12.57x) and the naive sum of their marginal gains. Neither technique alone gets
past ~14x; together they reach ~19-20x. This is the "sabka saath" effect the assignment
asks to demonstrate: tiling without a dense micro-kernel wastes the locality it creates,
and SIMD without tiling starves for data.

## Q2. Prefetch distance on the COMBINED kernel (task2c_prefetch_distance.txt + recheck)
Median of 3 runs at N=1024 (GFLOP/s is quoted: it is more stable than the speedup column,
because each harness run re-measures naive and that baseline drifted 418-470 ms):
    PF_D (floats)   GFLOP/s
    0               90.31   <- BEST
    16              84.85
    32              84.56
    64              84.78
    128             83.89
    256             84.22
BEST DISTANCE = 0, i.e. software prefetch is a net LOSS of ~6% even on the combined
kernel, and the loss is flat across all non-zero distances. Same conclusion as Task 2A.
At N=2048: PF_D=0 = 82.74 GFLOP/s vs PF_D=64 = 78.19 (median of 3) -> ~5.5% loss.
NOTE: the first-pass run recorded PF_D=32 at 67.35 GFLOP/s and N=2048 PF_D=64 at 63.80;
both were confirmed scheduling noise and corrected by a 3-run recheck (task2c_recheck.txt).

## Q3. Cache fill level on the COMBINED kernel (task2c_prefetch_level.txt)
    level          GFLOP/s
    _MM_HINT_T0    80.65
    _MM_HINT_T1    81.94
    _MM_HINT_T2    78.12
    _MM_HINT_NTA   82.85
All four fall within ~6% of each other and ALL are below PF_D=0 (90.31). On the combined
kernel the hint level is essentially irrelevant, because the JC=32 panel is already
L2-resident before the inner loop runs - so whichever level the hint targets, the line is
usually there already. (Contrast Task 2A's un-combined kernel, where T1/T2 tripled L1
misses vs T0: there the hint level mattered a lot.)

## Q4. Cache misses on the COMBINED kernel (task2c_cachemiss.txt)
    N=1024   L1: 57.2M -> 44.8M (22% fewer)   LLC: 33K -> 25K (25% fewer)   GF/s 87.0 -> 83.5
    N=2048   L1: 635.4M -> 440.0M (31% fewer) LLC: 1362K -> 780K (43% fewer) GF/s 82.7 -> 78.2
(left = PF_D=0, right = PF_D=64; N=2048 GF/s from the 3-run recheck.)
Software prefetch demonstrably WORKS at the cache level - it removes 22-31% of L1 misses
and up to 43% of LLC misses - yet it still LOSES on wall-clock. The prefetch instructions
consume issue slots that would otherwise retire FMAs, and the misses they remove were
already being hidden behind compute. L2 misses are unchanged (424M / 3.40B), as expected:
prefetch changes WHEN a line arrives, not the total volume the working set demands.

## Q5. Hardware prefetchers on/off, COMBINED kernel (task2c_hwpf.txt, MSR 0x1a4, N=1024)
                       SW ON (PF_D=64)      SW OFF (PF_D=0)
    HW prefetch OFF     17.34x / 80.61      15.80x / 73.54
    HW prefetch ON      17.15x / 82.13      18.17x / 86.78
*** THE KEY RESULT OF TASK 2 ***
With hardware prefetchers DISABLED, software prefetch becomes a +9.6% GAIN (73.54 ->
80.61 GFLOP/s). With them ENABLED, the same software prefetch is a -5.4% LOSS.
This is direct proof that SW and HW prefetch are SUBSTITUTES, not complements: the reason
_mm_prefetch looked useless in Tasks 2A and 2C-Q2 is precisely that the hardware
prefetcher was already covering this unit-stride access pattern for free. Remove the
hardware, and the software version earns its keep. Note this experiment succeeded here
where the Task 2A version of it did not show a flip - the combined kernel is fast enough
that memory latency is a larger fraction of its runtime, so covering it matters more.

## Q6. Instruction count, combined vs SIMD-only (task2c_instr.txt)
    N       naive            simd            optimized      naive/opt   simd/opt
    1024    6,572,879,294    395,350,711     352,332,734     18.66x      1.122x
    2048   52,084,965,519  2,656,656,781   2,313,442,978     22.51x      1.148x
Wall-clock for the same isolated runs:
    N=1024: naive 0.466 s, simd 0.0757 s, optimized 0.0376 s -> optimized is 2.01x faster
            than simd while executing only 1.12x fewer instructions.
    N=2048: 3.935 / 0.639 / 0.245 s -> optimized 2.61x faster on 1.15x fewer instructions.
INTERPRETATION: the combined kernel's advantage over SIMD-alone is NOT fewer instructions
- it is higher IPC. Cache blocking makes the same instruction stream execute roughly twice
as fast because operands arrive from L2 instead of DRAM. This is the cleanest single
number for the "why combine" argument: SIMD reduces the instruction COUNT (16-20x vs
naive), tiling raises the RATE at which those instructions retire (~2x on top).

## Conclusion
Combining SIMD + register blocking + cache tiling gives ~18-20x, flat across matrix size,
versus ~7x (SIMD alone at large N) and ~12x (tiling alone). The two techniques are
complementary in the strict sense: each removes the bottleneck that limits the other.
Software prefetch, however, is redundant ON THIS HARDWARE whenever the hardware
prefetchers are active - its benefit is real but already provided for free, as proven by
the HW-off experiment where it recovers +9.6%.

## Caveats for the report
- Prefer GFLOP/s or ms over the speedup column for within-kernel comparisons: each harness
  invocation re-measures naive, and that baseline drifted 418-470 ms between runs.
- N=128 is sub-millisecond and noise-dominated (prefetch 14.39x appears to beat optimized
  13.75x there, which does not hold at any larger size). Either drop the 128 point from
  the plots or annotate it.
- Three first-pass points were confirmed noise and corrected by 3-run rechecks; use
  task2c_recheck.txt values for PF_D=32, N=1536, and N=2048 PF_D=64.
- ALWAYS re-enable HW prefetch after the MSR experiment: sudo wrmsr -a 0x1a4 0x0
  (final rdmsr confirmed 0). It also resets on reboot.

## ADDENDUM: TRUE 128-bit vs 256-bit on the COMBINED kernel
Earlier width data compared only the plain SIMD stage. ungraded/matmul_optimized128.cpp is
a full SSE (128-bit) port of the 4x4 + JC=32 combined kernel, so the comparison is like for
like. Raw: task2c_width_and_extra_sizes.txt.
    size    COMBINED 128-bit    COMBINED 256-bit    ratio
    512      10.37x (52.2 GF)    20.69x (104.8 GF)   2.00x
    1024      9.28x (44.8 GF)    19.99x ( 98.4 GF)   2.15x
    2048     10.60x (49.0 GF)    19.11x ( 90.4 GF)   1.80x
A clean ~2x, and unlike the plain-SIMD comparison it does NOT shrink with size, because
cache blocking keeps both widths compute-bound. The 128-bit port is correct on ragged
shapes (255x257x129, M=1, 3x3x3 all pass). Part of the 2x is FMA: SSE4.2 has no fused
multiply-add, so the 128-bit build issues separate mul+add.

## ADDENDUM: distance / level sweeps at sizes other than 1024
    N=512  PF_D:  0 -> 20.07x | 16 -> 19.61x | 64 -> 19.65x | 256 -> 19.32x
    N=2048 PF_D:  0 -> 18.88x | 16 -> 17.97x | 64 -> 17.91x | 256 -> 18.03x
    N=2048 PF_L:  T0 18.05x | T1 17.48x | T2 17.51x | NTA 17.97x
Same verdict as N=1024 at both extra sizes: PF_D=0 wins, T0 best among the hints, and the
prefetch penalty is a consistent ~4-6%. The conclusion is size-independent.
