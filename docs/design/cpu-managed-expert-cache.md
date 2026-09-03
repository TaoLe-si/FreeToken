# CPU-Managed Expert Cache + dGPU Compute Architecture

**Status:** Architectural exploration / Design proposal
**Date:** 2026-09-03
**Authors:** FreeToken team discussion
**Target hardware:** AMD APU (Radeon 780M iGPU) + NVIDIA dGPU (RTX 4070 Laptop) + DDR5 system RAM

---

## 1. Problem Statement

LLM inference with MoE (Mixture of Experts) layers faces a fundamental memory bandwidth problem:

- Model weights are large (17.5GB for Qwen3.6-35B-A3B NVFP4)
- Per-token, only top-K experts (e.g., 8 of 256) are needed
- GPU must read top-K weights every token (320 MB / token)

At 50 tok/s, this requires **16 GB/s sustained weight read bandwidth**.

## 2. Hardware Bandwidth Survey

| Memory path | Bandwidth | Notes |
|-------------|-----------|-------|
| dGPU VRAM (RTX 4070, GDDR6) | 256 GB/s | Local VRAM, fastest |
| dGPU <-> System RAM (PCIe 3.0 x16) | 6-12 GB/s | Encoding + WDDM overhead |
| iGPU GTT (AMD APU coherent fabric) | 26 GB/s | DDR5 shared with CPU |
| CPU <-> DDR5 (dual-channel) | 89 GB/s | With L1/L2/L3 cache help |

### 2.1 Key insight: CPU > iGPU > dGPU for system RAM access

Although iGPU and CPU both share system RAM via the AMD APU coherent fabric,
CPU has 3.4x higher effective bandwidth thanks to:
- Multi-level cache hierarchy (L1/L2/L3)
- Higher clock speed
- Wider memory subsystem
- Better prefetchers

## 3. Why "all weights in VRAM" does not work

- RTX 4070 Laptop VRAM = 8 GB
- Model size = 17.5 GB
- Deficit = 9.5 GB must live somewhere else

## 4. Why "dGPU WDDM shared pool" does not work

The WDDM shared pool is just PCIe-bridged system RAM:
- Cold page faults: 0.5-3 GB/s
- Warm pages: 6-8 GB/s
- Required for 50 tok/s: 16 GB/s -> 270% over PCIe capacity

## 5. Why "iGPU GTT" only barely works

iGPU can read system RAM at 26 GB/s via APU coherent fabric. At 50 tok/s we need
16 GB/s, leaving only 60% utilization margin. Plus iGPU compute is slow (49ms/step
in the Form-2 implementation).

## 6. Proposed Architecture: CPU-Managed Expert Cache

```
    AMD APU SoC
    [CPU cores 89GB/s] <-- coherent fabric --> [iGPU 780M 26GB/s]
              \                                /
               \           shared             /
                \------ DDR5 RAM -----------/
                       (24 GB, 89 GB/s)
                              |
                         PCIe 3.0 x16
                         6-12 GB/s real
                              |
                [NVIDIA dGPU RTX 4070 Laptop]
                [VRAM 8 GB @ 256 GB/s]
                (hot expert cache)
```

### 6.1 Data placement

| Data | Location | Why |
|------|----------|-----|
| All 17.5 GB expert weights | CPU RAM (DDR5) | CPU 89 GB/s access |
| Hot top-K experts (per layer) | dGPU VRAM | 256 GB/s for matmul |
| Routing tables / metadata | CPU RAM | cheap, fits in L3 cache |
| Hidden states (per token) | dGPU VRAM | ephemeral, small |

### 6.2 Per-token execution flow

```
Token arrives in dGPU VRAM
         |
         v
   dGPU attention + GDN (uses dGPU VRAM only)
         |
         v
   MoE routing score (small matmul on dGPU)
         |
         v
   top-K expert IDs -> PCIe -> CPU
         |
         v
   CPU checks expert table: which are missing from dGPU VRAM?
         |
         v
   CPU reads missing weights from DDR5 @ 89 GB/s
         |
         v
   CPU pushes weights to dGPU VRAM via PCIe @ 6-12 GB/s
         |
         v
   dGPU runs MoE matmul from VRAM @ 256 GB/s
         |
         v
   Result stays in dGPU VRAM for next layer
```

### 6.3 Why this is fast

At 50 tok/s with 90% cache hit rate:
- Cold expert transfer: 1 expert x 1 MB x 40 layers = 40 MB / token
- PCIe bandwidth needed: 40 MB x 50 = 2 GB/s (well under 6 GB/s)
- dGPU matmul: from VRAM at 256 GB/s (effectively instant for 40 MB)
- Total weight path latency: dominated by dGPU matmul, not memory

### 6.4 Cache management

CPU maintains an LRU table mapping expert IDs to locations:
- Hot (recently used): replicated to dGPU VRAM
- Cold (LRU evicted): only in CPU RAM
- VRAM capacity: 8 GB -> can hold ~8 experts x 40 layers = 320 MB of hot set
- Eviction policy: LRU with frequency counter (LRFU)

## 7. Comparison with Existing Paths

| Approach | Weight bandwidth | 50 tok/s? | Notes |
|----------|------------------|-----------|-------|
| All in dGPU VRAM | 256 GB/s | yes if model <= 8GB | Not for 17.5GB model |
| dGPU WDDM (PCIe) | 6 GB/s | no, 270% over | PCIe bottleneck |
| iGPU GTT only | 26 GB/s | yes barely (60% util) | iGPU compute slow |
| **CPU-managed dGPU cache** | **89 + 256 GB/s** | **yes comfortable** | New approach |

## 8. Open Questions

1. CPU<->dGPU latency for top-K ID transfer (PCIe round-trip ~10-50 us)
2. MoE kernel design for sparse top-K read from VRAM
3. Cache coherence when CPU updates weights (loading, swap, etc.)
4. Cache warm-up during prefill phase
5. Worst case when ALL top-K experts evict

## 9. Risks

- dGPU compute on cold expert is wasted (matmul result invalid after eviction)
- PCIe latency adds 10-50 us per cold expert miss
- Routing table lookups on CPU add 1-10 us per token
- LRU cache thrashing if expert distribution is uniform random
- Overhead of CPU<->dGPU synchronization per layer

## 10. Hardware Reference

```
Machine: FreeToken test rig
CPU: AMD Ryzen (8 cores, 4-5 GHz)
Memory: DDR5 dual-channel, 89.6 GB/s peak
dGPU: NVIDIA RTX 4070 Laptop
  - VRAM: 8 GB GDDR6, 256 GB/s
  - PCIe: 3.0 x16, 6-12 GB/s real
iGPU: AMD Radeon 780M
  - VRAM: 512 MB dedicated
  - GTT: ~17 GB system RAM mapped, 26 GB/s
Model: Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4
  - 256 experts, top-8
  - 40 MoE layers
  - 17.3 GB total weights in NVFP4
```

## 11. Discussion Log

Session: 2026-09-03, FreeToken team channel

User observation:
> "igpu和CPU是嵌入在一起的，我怀疑CPU也能和igpu类似直读ram"

Confirmed: CPU does have direct system RAM access via APU memory controller,
at higher bandwidth (89 GB/s) than iGPU GTT (26 GB/s).

User proposal:
> "专家层放在CPU内，CPU建立一张专家表，dgpu通过读取表来直接操作"

Captured here as a formal design proposal.
