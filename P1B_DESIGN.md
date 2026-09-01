# P1b Design Memo: In-process iGPU MoE executor (2026-09-01)

## Key feasibility confirmed
- CUDA+HIP coexist in same process OK (_coexist_test.py):
  torch CUDA dev0=RTX 4070 + HIP dev0=780M, both init, no conflict.
- No separate server process / cross-process shared memory needed.
  Executor runs in-process, same as CpuMoeExecutor.

## Architecture (mirrors CpuMoeExecutor flag-sync)
CUDA graph (dGPU stream):
  D2H copy: routing(ids/weights) + activation -> pinned IO[step]
  memop_submit: done[slot]=0, ready[slot]=1  (doorbell, no kernel, no SM)
  [graph continues...]
  memop_sync: WAIT(done[slot]>=1)  (blocks stream, no SM)
  H2D copy: pinned IO[step]["y"] -> device output

HIP worker thread (separate host thread, GIL-released):
  poll ready[slot]==1
  -> hipLaunchKernel(gemv_layer, iGPU stream, pinned bank ptrs, pinned IO ptrs)
  -> hipStreamSynchronize(iGPU stream)
  -> done[slot]=1

## Bank zero-copy
- Engine already pins 16.93GB host banks via cudaHostAlloc (_pinned_tensor pyd)
- Executor hipHostRegister(ptr, nbytes, Default) on each layer bank at startup
- HIP kernel reads pinned VA directly (780M unified memory, 35-38 GB/s measured)

## Differences vs CpuMoeExecutor
| | CpuMoeExecutor | IgpuSharedMoeExecutor |
|---|---|---|
| compute | CPU thread pool (avx512bf16) | 780M HIP kernel |
| submit | _ext.submit_with_cuda_stream | HIP worker thread polls ready |
| flag-sync | memop_submit/sync (reuse) | memop_submit/sync (reuse) |
| bank read | host RAM (51 GB/s) | pinned zero-copy (35-38 GB/s) |
| graph-safe | yes (host node) | yes (same pattern) |

## Dependencies
- P1a: HIP NVFP4 GEMV kernel (subagent running) -> provides kernel signature
- _cpu_moe pyd: memop_submit/memop_sync (existing, reuse directly)
- amdhip64_6.dll: ctypes load (verified in _coexist_test.py)
