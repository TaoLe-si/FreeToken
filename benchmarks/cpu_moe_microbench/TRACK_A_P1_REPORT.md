# Track 4 (A) P1 Report: iGPU Executor for MTP Head

**Date**: 2026-08-27
**Status**: **PASS**

## Implementation
New file: `python/freetoken/engine/mtp_igpu_executor.py`

### MtpIgpuExecutor class
- Wraps P1g v2 server as long-lived subprocess
- Constructor: takes fc_packed_u32 (M=1, K//8) uint32 + K
- Internally: starts server, sends LOAD command, waits for OK reply
- `forward(act_flat_f32) -> outv (M,)`: sends CALL command with act bytes, reads 4-byte len + M*4 bytes
- `close()`: sends QUIT, server exits cleanly
- Thread-safe: lock around stdin/stdout access

## Test Results

### Setup
- fc weight row 0: shape (1, 512) uint32, dtype=torch.uint32
- K=4096

### Correctness (4 act values)
| act  | MtpIgpuExecutor | Direct STATELESS | Match |
|------|----------------|------------------|:-----:|
| 0.05 |     4.6925     |     4.6925       |   OK  |
| 0.10 |    18.7700     |    18.7700       |   OK  |
| 0.20 |    75.0800     |    75.0800       |   OK  |
| 0.01 |     0.1877     |     0.1877       |   OK  |

### Performance
- 100 calls: 38ms = **0.38ms/call** (Python overhead included)
- Server-side GPU dispatch: ~0.2-0.5ms (from earlier benchmarks)
- First call: 2.0ms (incl server startup)

### Reproducibility
- 2 consecutive calls: max diff = 0.00e+00 (bit-exact)

## Cross-check with Existing Code
The MtpDriver class in `engine/mtp_driver.py` already has similar logic
(`IgpuFcSticky` wrapper) but uses the OLD v1 server `t_mxfp4_gemv_server.exe`.
Our new `MtpIgpuExecutor` is specifically for v2 server (`t_mxfp4_gemv_v2_server.exe`):
- Same protocol (LOAD/CALL/QUIT) but with proper sticky LOAD-once semantics
- Uses the v2 protocol where the server pre-allocates resources per LOAD

## Next: P2 (scheduler integration)
Wire MtpIgpuExecutor into the FreeToken engine so MTP head can use iGPU FC
in the actual inference loop. This requires:
1. `engine.py::create_model` to instantiate the executor
2. `engine.py::forward_batch` two-phase: main model -> last hidden -> MTP executor
3. `engine/mtp_driver.py::MtpDriver` to use the new executor
4. `core.py::Req` to add draft state fields
5. `scheduler/scheduler.py` to insert verify/rollback logic
6. `scheduler/cache.py` to add `free_partial` API
7. `engine/graph.py` to recapture decode graph with K-dim
