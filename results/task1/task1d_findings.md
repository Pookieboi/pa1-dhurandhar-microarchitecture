# Task 1D (Combined) — Findings
Params: NACC=2, TH=16, TW=512, AVX2 256-bit. Machine: Raptor Lake, L1d 48K/12-way,
CPU 0 pinned, governor=performance. Flags: -O2 -fno-tree-vectorize -mavx2 -mfma.
Graded run: optimized = 8.87x at 2048x2048 K=3 -> 60/60. Raw: task1d_full.txt.

## What conv_optimized combines and why
- TILING (TH x TW output block): keeps each tile's input patch L1-resident.
- AVX2 SIMD (_mm256_fmadd_ps): 8 floats per instruction.
- ILP / unrolling: NACC=2 independent SIMD accumulators break the FMA dependency
  chain (FMA latency ~4c, throughput 2/c), so two chains run in flight.
- Register accumulation: out[] written ONCE per pixel (like conv_unroll), not
  K*K times as in the += tile/simd stages. This is the biggest single win.
NOTE ON STRUCTURE: inner order is ty,tx,oy,ox,ky,kx (kernel innermost, register-
blocked) - NOT the ky,kx,oy,ox order of conv_reorder. Describe it as register-
blocked+vectorized, not "the 1A reorder".

## Why it beats every individual technique
Each attacks a different bottleneck: tile=memory, SIMD=compute, ILP=latency.
They are independent, so combining removes multiple walls at once. Best single
stages ~3x (unroll), ~2x (simd), ~1.7x (tile); combined = 8.87x at 2048, 12x at K=5.

## Accumulator sweep result (measured, not assumed)
At 4096x4096: NACC=1 ~10ms, NACC=2 ~7.2ms, NACC=4 ~13ms, NACC=8 ~11ms.
- NACC=1: FMA latency not hidden (single dependent chain).
- NACC=2: optimal - covers latency without saturating load ports.
- NACC=4/8: WORSE - too many _mm256_loadu_ps per tap exceed the 2 load ports.
  This empirically confirms the kernel is MEMORY/LOAD bound, not FMA bound:
  adding compute parallelism past 2 hurts.
Tile: TH=16 best across 2048/4096/8192; TW 256-1024 all within noise (~30.6ms @8192).

## For the report
- Figure 1.1 plot: use Section 2 (K=3) - one bar/line per technique vs matrix size:
  reorder, unroll, tile, simd, optimized. Speedup normalized to naive.
- optimized speedup DECAYS with size (8.87x @2048 -> ~5.9x @4096 wall of DRAM bw)
  but stays far above all individual techniques at every size.
