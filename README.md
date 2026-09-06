# CS683 PA-1: Hardware-Conscious Performance Engineering

| Name | Roll number |
|------|-------------|
| Pranav | 24B1074 |
| Atharva | 24B0996 |
| Divyaansh | 24B0981 |
| Sufiyan | 24B0935 |

Two tasks. Each one starts from a naive C++ kernel and makes it faster by paying attention to
the cache hierarchy and the vector units of an x86 core. The full problem statement is in the
[assignment document](https://docs.google.com/document/d/1suURtM3WaensRvABVxlHZmhbbXe-g2v-sNi8gUfuojg/edit?usp=sharing).

## Layout

```
task1/      2D convolution: src/conv_{reorder,unroll,tile,simd,optimized}.cpp, task1_report.pdf
task2/      SGEMM: src/matmul_{simd,prefetch,optimized}.cpp, task2_report.pdf, llama/ injection
plots/      figures used in the reports
results/    raw harness and perf output for every number in the reports (task1/, task2/)
```

To build and run a task, go into its directory and run `make && ./bin/<conv|matmul>`. We kept
the Makefiles as given, so the flags are the pinned `-O2 -fno-tree-vectorize -mavx2 -mfma`.

## Results in short

We measured everything on an Intel i7-13700HX, pinned to a P-core with the governor set to
`performance`. The chip has no AVX-512.

| Task | Graded kernel | Speedup vs naive | Notes |
|------|---------------|------------------|-------|
| 1 | `conv_optimized` (tiling + AVX2 FMA + 2 accumulators, output written once) | 9.25x at 2048x2048, K=3 | 8 to 17x across sizes and kernel widths |
| 2 | `matmul_optimized` (4x4 register block, JC=32 panels, prefetch tunable) | 18 to 20x at 256 to 2048 | output matches llama.cpp's native matmul token for token |

Software prefetching on its own never beat the hardware prefetchers on this machine. By the
assignment's definition it came out between 0.5x and 0.98x, so the shipped `matmul_optimized`
has it compiled out (`PF_D 0`). With the hardware prefetchers switched off it turns into a 9.6%
gain. The reasoning is in `task2/task2_report.pdf`.

The original task instructions are in [task1/README.md](task1/README.md) and
[task2/README.md](task2/README.md).
