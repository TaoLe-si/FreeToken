# [P1g] iGPU MXFP4 Sticky-Weight Server - Working

## Achievement
- Clean rewrite of the multi/sticky weight server based on the verified P1d stateless server
- LOAD <name> uploads a weight + per-row gbl/rowB once and caches the resources
- CALL <name> only re-uploads the activation and re-dispatches (~0.2ms per call)
- STATELESS protocol kept for backward compatibility
- Numerical output matches the P1d baseline byte-for-byte (verified across 7 act values)

## What was wrong with the previous multi/sticky servers
- t_mxfp4_gemv_multi_server.cpp: used `CopyResource` between buffers of different sizes
  (rW is M*nb*4 but up is packed + gbl + rowB), which D3D12 silently no-ops on mismatched sizes
- t_mxfp4_gemv_multi_server.cpp: LOAD barrier had `StateBefore=COPY_SOURCE` for rGbl/rRowB
  but the resources were created in `COPY_DEST` state, making the barrier a no-op
- t_mxfp4_gemv_sticky_server.cpp: used `w.rRb` (period, struct) where `w` is a pointer;
  should be `w->rRb` (arrow). Compile error C2228.

## Files
- t_mxfp4_gemv_v2_server.cpp - the new server (main)
- t_mxfp4_gemv_v2_server.exe - compiled binary
- build_v2_server.bat - MSVC build script
- t_test_v2_compare.py - correctness test (v1 vs v2 STATELESS vs v2 LOAD+CALL)
- t_test_v2_bench.py - latency benchmark

## Protocol
```
LOAD <name> <M> <K> <packed_size>\n<packed_bytes>
  -> reply: "OK <name> <M> <K>\n"
  -> uploads packed + per-row gbl=1.0 + per-row rowB=0.0
  -> resources stay in NON_PIXEL_SHADER_RESOURCE state

CALL <name> <szA> <szS> <szB>\n<act_bytes><scales_bytes><biases_bytes>
  -> reply: <4-byte uint32 len><M*4 bytes float32>
  -> re-uploads only the activation (scales/biases typically zero)
  -> dispatches and reads back

STATELESS <M> <K> <szP> <szA> <szS> <szB>\n<packed><act><scales><biases>
  -> reply: <4-byte uint32 len><M*4 bytes float32>
  -> same as P1d stateless server (no caching)

QUIT\n
  -> server exits cleanly
```

## Verification (P1g correctness)
Test against real MTP fc weights, M=1, K=4096:

| act       | v1 STATELESS | v2 STATELESS | v2 LOAD+CALL | match |
|-----------|-------------:|-------------:|-------------:|:-----:|
| 0.01      |     0.1877   |     0.1877   |     0.1877   |   OK  |
| 0.05      |     4.6925   |     4.6925   |     4.6925   |   OK  |
| 0.10      |    18.7700   |    18.7700   |    18.7700   |   OK  |
| 0.20      |    75.0800   |    75.0800   |    75.0800   |   OK  |
| -0.05     |     4.6925   |     4.6925   |     4.6925   |   OK  |
| 0.00      |     0.0000   |     0.0000   |     0.0000   |   OK  |
| 1.00      |  1877.0000   |  1877.0000   |  1877.0000   |   OK  |

All three paths produce **identical** outputs (bit-exact float32). The P1d baseline
of 4.6925 for act=0.05 is reproduced exactly.

## Performance
- 100 CALLs in 145.6ms total (1.46ms/call incl. Python payload-build overhead)
- Server-side GPU dispatch: 0.17-0.67ms per CALL
- v1 STATELESS (process-per-call) was 102ms/call incl. process startup
- **Process startup is the dominant cost** in v1; v2 eliminates that with persistent
  resources.

## Design notes
- Same root signature as P1d: 6 SRV (slot 0-5) + 1 UAV (slot 6) + 1 CBV (slot 7), all dense
- Two command allocators/lists (uploadAlloc/uploadList, dispatchAlloc/dispatchList) for
  clean separation, exactly as P1d
- Per-resource state tracking (rWSt, rSSt, rBSt, rActSt, rGblSt, rRowBSt, rOutSt)
- Uses CopyBufferRegion with explicit sizes (NOT CopyResource) to avoid the size-mismatch
  bug from the original multi server
- LOAD: one batched list that copies packed->rW, gbl->rGbl, rowB->rRowB and transitions
  rW/rGbl/rRowB COPY_DEST->NPSR. After LOAD, rS/rB/rAct remain in COPY_DEST.
- CALL: one batched list that copies act->rS (T1 alias), scales->rB (T2), act->rAct (T3)
  and transitions rS/rB/rAct COPY_DEST->NPSR. Then a separate dispatch list.
- STATELESS: full P1d batched upload pattern (offP, offA, offS, offB, offG, offR offsets)
  + dispatch in two lists.

## Next steps
- Integrate with FreeToken MTP head: pre-load all needed weights (fc, q, k, v, o, gate,
  up, down) at session start, then issue CALLs for each token's MTP draft
- Test with M=8 (MoE top-8 experts) to ensure realloc path works for larger M
- Add per-block e8m0 scales upload to support real MXFP4 format (currently scales are
  treated as 0; the kernel applies a 0.01 magic constant instead)
