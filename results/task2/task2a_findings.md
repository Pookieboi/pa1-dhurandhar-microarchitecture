# Task 2A (Software Prefetching + Cache Blocking) — Findings & Interpretation

Machine: Intel i7-13700HX (Raptor Lake, hybrid). Pinned to CPU 0 (P-core),
governor=performance. L1d 48KB/core, L2 2MB/P-core, L3 30MB, 64B lines.
Flags (pinned): -O2 -fno-tree-vectorize -mavx2 -mfma. Seed 1234.
File edited: src/matmul_prefetch.cpp. Reference: matmul_naive.cpp (unmodified).
Raw data: task2a_sweep.txt, task2a_swpf_vs_size.txt, task2a_prefetch_distance.txt,
task2a_prefetch_level.txt, task2a_cachemiss.txt, task2a_cachemiss_vs_size.txt,
task2a_cachemiss_params.txt, task2a_hwpf.txt.

## Kernel design
Cache-blocked SIMD + optional software prefetch on the Stage-1 micro-kernel.
- j (B-row) loop tiled into PANELS of JC=64 rows; a JC x K panel of B stays L2-resident
  and is REUSED across all M rows of A, instead of being re-streamed from DRAM per A row
  (which is what made Stage-1 SIMD memory-bound).
- Inside the panel: 1x4 register-tiled AVX2 FMA micro-kernel.
- _mm_prefetch pulls B[j][p+PF_D] into cache level PF_L ahead of use.
- Knobs are COMPILE-TIME defines (-DPF_D, -DPF_L, -DJC): swept by recompiling, never by
  editing the file. PF_D=0 disables SW prefetch (pure-tiling baseline).
- Correct for arbitrary M,N,K and N not a multiple of JC (verified N=200, 37x53x71, M=1).

## Q1. Speedup vs matrix size
(a) vs NAIVE, PF_D=64 (task2a_sweep.txt):
    size    ms        GFLOP/s  speedup
    128     0.060     70.1     13.16x
    256     0.509     65.9     11.71x
    512     4.071     66.0     14.10x
    1024   35.313     60.8     12.54x
    1536  118.358     61.2     12.55x
    2048  297.416     57.8     12.06x
Speedup stays ~12-14x and GFLOP/s ~58-70 across ALL sizes, whereas Stage-1 SIMD COLLAPSED
from ~81 to ~33 GFLOP/s. Cache blocking removed the memory-bound decay.

(b) The DOC's definition, Speedup = t_without_SWpf / t_with_SWpf (task2a_swpf_vs_size.txt):
    size    without(PF_D=0)  with(PF_D=64)  speedup
    128       0.046 ms        0.067 ms      0.69x
    256       0.420           0.509         0.83x
    512       3.456           3.910         0.88x
    1024     28.699          34.644         0.83x
    1536    103.396         111.772         0.93x
    2048    246.242         280.663         0.88x
ALL BELOW 1.0 -> software prefetch is a net LOSS at every size. But the penalty SHRINKS
as size grows (0.69 -> ~0.9): at large sizes there is more real memory latency to hide,
so the fixed instruction overhead of the prefetches is relatively less costly.

## Q2. Best prefetch distance (task2a_prefetch_distance.txt + task2a_cachemiss_params.txt)
    PF_D    speedup vs naive    L1-dcache-load-misses
    0       13.80x              347.8 M
    16      12.14x               41.6 M   <- fewest misses
    32      12.21x                  -
    64      12.34x               48.3 M
    128     12.00x                  -
    256     11.90x               76.5 M   <- misses climb again
BEST DISTANCE BY TIME = 0 (no SW prefetch). BEST BY L1 MISSES = 16 floats (1 cache line
ahead). The miss curve is the classic U: too small and the line has not arrived; too far
(256) and the prefetched line is EVICTED before use, so misses rise from 41.6M to 76.5M.
The prefetches do work at the cache level (348M -> 42M misses, ~8x) yet still lose on
wall-clock, because those misses were already being hidden by the hardware prefetcher.

## Q3. Cache fill level for max speedup (task2a_prefetch_level.txt + cachemiss_params)
    level         speedup    L1 misses
    _MM_HINT_T0   11.67x      47.3 M    <- best of the hints
    _MM_HINT_T1   10.97x     141.8 M
    _MM_HINT_T2   10.91x     139.9 M
    _MM_HINT_NTA  11.01x      48.2 M
T0 is best: the data is consumed immediately, so pulling it all the way into L1 is right.
T1/T2 stop at L2/L3, so the L1 miss count TRIPLES (47M -> ~140M) - direct confirmation
that the hint level does exactly what it claims. NTA matches T0 on misses but is
semantically wrong here (data IS reused down the K loop). All hints still lose to PF_D=0.

## Q4. Cache misses vs matrix size (task2a_cachemiss_vs_size.txt)
    size   L1 (without / with)    L2 (without / with)     LLC (without / with)
    256      4.75M / 1.04M         163K / 165K              3.6K / 3.6K
    1024   348.2M / 48.4M         418.3M / 418.2M          25.7K / 16.3K
    2048  2765.2M / 389.7M       3362.5M / 3385.2M          2.35M / 1.07M
SW prefetch cuts L1 misses ~7x at every size and is neutral at L2. At 2048 it also halves
LLC misses (2.35M -> 1.07M), i.e. it DOES help at the last level once the working set far
exceeds L3 - but not enough to overcome its instruction overhead in wall-clock terms.

## Q5. Effect of hardware prefetchers (task2a_hwpf.txt, MSR 0x1a4, N=1024)
                       SW ON (PF_D=64)   SW OFF (PF_D=0)
    HW prefetch OFF       12.57x            13.99x
    HW prefetch ON        12.39x            13.37x
Disabling the HW prefetchers changes almost nothing (~0.5x), and SW prefetch does NOT
overtake no-prefetch even with HW off. Reason: the JC=64 cache BLOCKING already makes the
B panel L2-resident before the inner loops run, so there is little left for EITHER
prefetcher to fetch.

## Conclusion
On this unit-stride, cache-resident SGEMM the win comes from CACHE BLOCKING, not
prefetching. Software prefetch measurably does its job (L1 misses drop ~7-8x, LLC misses
halve at 2048, and the distance/level sweeps behave exactly as theory predicts) but it
never pays for its instruction overhead: the doc-definition speedup is 0.69-0.93x, i.e.
always < 1. Optimal distance = 0 by time (16 floats by miss count); best hint = T0.
Supported by four independent measurements: distance sweep, level sweep, miss counts
across sizes and parameters, and the HW-prefetcher on/off grid.

## Caveats
- Speedups cluster in 12-14x; run-to-run noise is real at that spread. The ROBUST signals
  are the MISS COUNTS (repeatable, order-of-magnitude effects) and the consistent
  ordering (no-prefetch >= prefetch) across every run.
- ALWAYS re-enable HW prefetch after the MSR experiment: sudo wrmsr -a 0x1a4 0x0
  (final rdmsr confirmed 0 = enabled). It also resets on reboot.
- JC=64 chosen for L2 residency; a JC sweep is optional extra analysis.

## ADDENDUM: scalar prefetch-only kernel (ungraded/matmul_prefetch_scalar.cpp)
Naive triple loop + _mm_prefetch only — no SIMD, no tiling — to isolate prefetching.
Raw: task2a_scalar_prefetch.txt, task2a_scalar_cachemiss.txt, task2a_scalar_hwpf.txt.

Doc speedup (t_without / t_with), SPF_D=64:
    N=128 0.73x | 256 0.50x | 512 0.60x | 1024 0.51x | 2048 0.51x
Distance sweep at N=1024 is FLAT: every distance 8..256 gives 0.51-0.52x.
Hint-level sweep is FLAT too: T0/T1/T2/NTA all 0.51-0.52x.

WHY IT IS FLAT — this is the key caveat, do not omit it from the report:
the scalar loop issues ONE PREFETCH PER ELEMENT, i.e. ~16 redundant prefetches per
64-byte line. sw_prefetch_access.t0 confirms 5,369,387,439 prefetches for the scalar
kernel vs 671,514,518 for the vector kernel (8x fewer, one per 8-float vector).
So this experiment measures PREFETCH-INSTRUCTION OVERHEAD, not prefetch strategy:
roughly doubling the instruction count halves throughput, and that overhead swamps any
distance or hint effect. It is the correct answer to "what does naive prefetching cost",
but it is NOT evidence that distance/level do not matter in general — the vector kernel
(above) does show level sensitivity.

Cache misses still confirm prefetch works at the cache level even here:
    N=256   L1 289,902 -> 159,217      N=512   L1 2,681,293 -> 1,544,340
    N=1024  L1 40,342,999 -> 22,642,618  N=2048  L1 235,844,967 -> 140,662,179
(~45% fewer L1 misses) while wall-clock still halves.

HW prefetcher on/off, scalar kernel (N=1024):
    HW OFF: SPF ON 0.52x | SPF OFF 1.00x
    HW ON : SPF ON 0.53x | SPF OFF 1.01x
Unlike the combined kernel, disabling HW prefetch does NOT rescue software prefetch here,
because the per-element prefetch overhead dominates regardless.

## ADDENDUM: sw_prefetch_access counters (PDF Task 2A, task2a_swpf_counters.txt)
The event correctly attributes prefetches to the requested cache level:
    hint  -> counter that fires
    T0    -> sw_prefetch_access.t0
    T1/T2 -> sw_prefetch_access.t1_t2
    NTA   -> sw_prefetch_access.nta
Counts (N=1024): scalar 5.369e9 at every distance (distance does not change HOW MANY
prefetches issue, only how far ahead) ; vector kernel 671e6 (8x fewer). PF_D=0 builds show
only ~11-20k, i.e. background noise, confirming the #if PF_D>0 guard compiles them out.
