# cpu_moe_microbench

Standalone microbenchmark for the `_cpu_moe` CPU MoE GEMV kernels
(`python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`) with **no torch/CUDA
dependencies** — the dot kernels are transcribed verbatim, so it measures the exact
production arithmetic on any x86-64 machine.

## Build

Linux/macOS (gcc/clang):

    g++ -O3 -march=native -std=c++20 -pthread cpu_moe_microbench.cpp -o bench
    # or: clang++ -O3 -march=native ...

Windows (VS2026 / MSVC — the kernels use AVX-512 BF16/VNNI intrinsics that MSVC
exposes under /arch:AVX512):

    build.bat              # builds bench_avx2.exe + bench_avx512.exe
    bench_avx512.exe       # Zen4+ / Intel AVX-512 machines
    bench_avx2.exe         # AVX2-only machines

Note: MSVC does not support per-function `__attribute__((target(...)))`; the file
strips the attributes and gates kernels on the feature macros, so build the variant
that matches your CPU. Run the same binary 2-3 times — laptop DVFS makes absolute
numbers drift between runs (use the within-run minimum).

## What it measures

1. **Single-thread bf16 dot** per SIMD tier (scalar / avx2 / avx512f / avx512bf16) —
   the decode GEMV is DRAM-bandwidth-bound, so the number that matters is
   *weight-stream GB/s*.
2. **Prefetch-distance A/B** for the AVX-512 BF16 dot (0..8192 B). The production
   kernel uses a fixed 512 B; the optimum is hardware-dependent (measured 128 B on
   one Zen4 box, 512 B+ on Emerald Rapids), which is why
   `FREETOKEN_CPU_MOE_PF_AHEAD` was added to the executor.
3. **Accumulator-count A/B** (4 vs 8) — noisy on laptops; check on the serving box.
4. **Multi-thread pass1-style GEMV** (atomic work-grab over 32-row tiles + spin
   barrier, same shape as the executor). Confirms SMT threads *hurt* the streaming
   GEMV: keep `--moe-cpu-threads` <= physical cores.
5. **Grouped (expert-dedup) vs per-token GEMV at bs>1** — measures the payoff of
   batching tokens that route to the same expert (one weight pass, M dots per row).
   Measured 2.3-3.2x more routes/s in the same-expert case; the real win scales with
   the expert-collision rate.

## Env knobs

    BENCH_ROWS=8192 BENCH_N=4096 BENCH_THREADS=8 BENCH_ITER=20 BENCH_M=4 bench_avx512.exe

## Findings so far (Ryzen 9 7940H / RTX 4070 Laptop, 2026)

**Measured machine limits (this box):**
- Host DRAM sequential read: **~51 GB/s** at 16 threads (DDR5-5600 dual-channel,
  well below the 89.6 GB/s theoretical peak — 4K-page TLB + laptop power wall).
  Use this number, not the theoretical peak, for bandwidth models.
- bf16 dot peak (L1-resident, pure instruction ceiling): **58.7 GMAC/s single
  thread, 0.28 T MAC/s at 8 threads** — i.e. ~1/5 of the theoretical Zen4 FMA
  peak. The dot is cache-read-bandwidth limited (~4B per MAC), not FMA limited.
- NVFP4 W4A8 (AVX-512 VNNI) peak: **18.9 GMAC/s single, 0.09 T MAC/s at 8
  threads** — the per-block float dequant (gather + 2 mults + cvt per 16
  weights) dominates; VNNI is currently *3x slower* than bf16 despite the
  smaller data flow. Optimizing that dequant is a concrete kernel goal before
  W4A8 can win.
- Prefetch: pf=0-128 B beats 512 B on this box; tune per machine via
  FREETOKEN_CPU_MOE_PF_AHEAD.
- Multi-thread GEMV: saturates DRAM at 4-8 threads; 16 SMT threads regress.

**Consequence for a dense-in-RAM 27B NVFP4 (15GB) on this box:** pure-CPU decode
is ~5-7 tok/s and saturates at M≈2 (compute ceiling 0.28 T MAC/s / 36 GMAC per
token, DRAM 51 GB/s / 10GB FFN); bigger batches do NOT help on this machine.
The RTX 4070 Laptop (8GB, ~256 GB/s) fused-runs 12B NVFP4 (~6.6GB) at ~39 tok/s —
the pragmatic choice here. GPU-partitioned FFN (PCIe ~20 GB/s in parallel with
CPU DRAM reads) adds ~40% only.

**Grouped GEMV:** bs=4 grouped (expert dedup) reaches 2.3-3.2x routes/s vs
per-token in the same-expert worst case; the realistic gain at top_k=8 over
E=64-256 experts is a weight-read reduction of only 1.05-1.25x.
