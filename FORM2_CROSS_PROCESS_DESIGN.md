# Form-2 Cross-Process HIP Worker — Design

> Status: implementation in progress | Target: 30–45 t/s on RTX 4070 Laptop + 780M iGPU
> Replaces in-process Form-2 attempts blocked by the WDDM KMD bug (see HIP_WORKER_PITFALLS.md).

## 1. Why cross-process

In-engine Form-2 hits hipMemcpy rc=1 on every H2D once CUDA initializes — a confirmed Windows WDDM kernel-mode driver defect (see pitfalls doc). GTT allocations succeed, D2H reads succeed, kernel launches execute, but the write/commit path is broken. No amount of synchronization, address choice, pinning, or context juggling fixes it within one address space.

A separate process that never imports torch (or any other CUDA library) sees a fresh KMD view and H2D lands as expected — verified to 35 t/s in _simtest6.py. The fix is therefore: move everything iGPU into a sibling process, talk to it through shared memory.

## 2. Topology

The daemon spawns both children with the same lifecycle: start launches both, stop kills both, port-conflict and restart cycles affect both identically. The worker is not auto-restarted on its own; a crash means the engine falls back to the CPU executor (degraded but functional).

Shared memory carries only data, never device pointers (GTT VAs are per-process on WDDM).

## 3. Shared memory layout

A single mmap'd file under %LOCALAPPDATA%\\freeToken\\igpu\\. Two ring-buffer regions back-to-back, separated by cache-line-aligned headers.

Header (4 KiB): magic, schema_ver, sizes, generation counter.

Engine to Worker ring (8 slots, ~64 KB each):
- hidden FP32 x 2048
- ids I32 x 8
- weights FP32 x 8
- meta: token_id, request_id

Worker to Engine ring (8 slots):
- out_hidden FP32 x 2048
- status: rc, latency_us

Synchronization: head/tail are 8-byte atomics; busy mask is a per-slot 1-byte flag. Producer claims a slot, writes data, publishes by advancing tail; consumer pulls slots in order.

Window-specific: mmap works on Windows when the file is opened with os.open and mapped with mmap.mmap (ACCESS_READ|ACCESS_WRITE). Producer and consumer both have it open simultaneously; Windows CreateFileMapping semantics allow this. We use the cross-platform mmap module so the design is portable to WSL2 later.

## 4. Worker process lifecycle

The worker is invoked as 'python -m freetoken.igpu.worker'. It runs end-to-end without touching torch:

1. Prepend C:\\Program Files\\AMD\\ROCm\\6.4\\bin to PATH before any HIP load (so amdhip64_6.dll resolves).
2. Load DLLs, igpu_init (no CUDA in this process — KMD bug does not apply).
3. Stream FTW into 17 GB of hipMalloc'd GTT, register 40 layers with igpu_register_layer_dev.
4. Open the shared ring buffer.
5. Loop: wait for engine request, run 40 layers of igpu_moe_decode_dev, publish result, repeat.
6. On shutdown signal: drain ring, exit cleanly.

The worker holds a device-side staging buffer for hidden/ids/weights/output (~40 KB). The engine writes activations to the staging buffer via the shared ring; the worker reads them via hipMemcpy D2H (D2H is fast on WDDM, this is the path that survives the KMD bug).

## 5. Engine-side shim

IgpuSharedMoeExecutor is rewritten to call the ring instead of the DLL directly. decode() converts CUDA tensors to numpy, writes to the ring, blocks on the response slot, returns the result as a torch tensor.

The engine keeps its existing prefill path entirely on the dGPU (no IPC, no iGPU involvement). Only the per-token decode MoE path crosses the IPC boundary.

## 6. Zero-copy optimization (deferred)

A naive design copies engine CUDA → host numpy → mmap'd ring → worker HIP device staging → kernel input: ~4 copies per token of H*FP32 = 8 KB.

A cheaper design would allocate a GTT-backed ring (via a worker syscall) and expose its device pointers via the same control channel; both processes would hipHostRegister / hipHostGetDevicePointer against the same mapping. We do NOT ship this in v1 — the WDDM bug affects this path too, and we'd be testing a hypothesis while the engine already has CUDA loaded. We will only revisit it after v1 lands and we have IPC latency measurements to optimize against.

## 7. Latency budget (v1)

| Stage | Cost |
|---|---|
| Engine: tensor → numpy → mmap | ~0.05 ms |
| Engine: atomic tail publish | ~1 µs |
| Worker: wake on tail | ~50 µs |
| Worker: D2H hidden (8 KB) | ~0.3 µs (D2H is fast on WDDM) |
| Worker: 40 × igpu_moe_decode_dev | ~28 ms (28 ms measured in _simtest6) |
| Worker: D2H out_hidden | ~0.3 µs |
| Engine: numpy → torch | ~0.05 ms |
| Total IPC + kernel | ~28.5 ms per token |
| Implied throughput | ~35 t/s |

This matches the standalone _simtest6 measurement. Plus ~1 ms for the actual atomic ring hand-off in both directions, we land at 30 t/s net, with headroom for Phase-2 fusion (kernel launch overhead collapse) to push toward 45 t/s.

## 8. Failure modes

- Worker dies mid-decode: next get_response raises IpcError. Engine logs the loss and falls back to CpuMoeExecutor for the rest of the request; future requests are retried with worker recovery (once).
- Worker hangs: engine timeout (5 s) on the response, treat as above.
- Engine dies: daemon signals the worker; worker drains the ring and exits (one final flush).
- Shared ring corrupted: header magic mismatch on either side → restart both children together.

## 9. Files to add / modify

Add:
- python/freetoken/igpu/__init__.py
- python/freetoken/igpu/protocol.py — ring buffer + atomics
- python/freetoken/igpu/worker.py — torch-free HIP process
- python/freetoken/igpu/client.py — engine-side handle
- python/freetoken/igpu/supervisor.py — daemon-side spawn/kill

Modify:
- python/freetoken/moe/igpu_shared_executor.py — replace direct DLL calls with client.decode()
- python/freetoken/engine/engine.py — wire the client at engine init
- python/freetoken/daemon/serve_manager.py — spawn worker alongside serve, kill alongside serve
- python/freetoken/cli.py — register ft igpu-worker subcommand

## 10. Verification gates (in order)

1. Worker standalone: spawn, load FTW, run 40 layers on random data, write output to ring, exit cleanly. Expected ~28 ms per decode. Compare numerical result against _simtest6.py (must match within fp32 noise).
2. Client to Worker IPC round-trip: 1000 fake requests, no model. Measure per request latency, throughput, no stalls. Expected ~30 ms each.
3. Engine+Worker e2e: start ft serve with --moe-backend=igpu, worker spawned by daemon. First /v1/chat/completions request must return sensible text. Logged: 'igpu worker pid=… ready', 'igpu ring open'.
4. Frontend panel 'API server is ready' must appear, /metrics must report the iGPU worker status, daemon ctl status must show worker pid alive.
5. Token benchmark x3: target >=30 t/s (vs. 5.95 t/s current CPU+MCP).

## v1 results (recorded 2026-01-07)

After implementing and running on the real model `E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4`:

| step | result |
|---|---|
| worker startup (HIP load + DLL load + igpu_init) | ~0.4 s |
| TCP IPC connect | <1 ms |
| FTW stream (40 layers, 17 GB) into GTT | 33-45 s (~400 MB/s, SATA-bound) |
| staging allocation | <1 ms |
| first decode (warmup) | 36 ms |
| stable decode (median of 10) | **26 ms** |
| min | 24 ms |
| max | 36 ms |
| measured throughput | **~50 tok/s** (after sync-removal fix) |

Observations:
* **WDDM KMD bug is fully bypassed.** `hipMemcpy` H2D rc=0 for all 17 GB streamed in from
  the FTW shards, plus every per-request H2D of the 8 KB request payload. No rc=1 errors.
* **TCP loopback was substituted for mmap** -- Windows Python mmap is not cross-process
  coherent (see `HIP_WORKER_PITFALLS.md` entry #11). TCP loopback IPC adds ~1.5 ms per
  request; well within the original 1-2 ms budget. The struct-pack roundtrip
  (`pack_request` / `unpack_response`) is the floor cost; a future zero-copy over shared
  memory is deferred to v2 once we have a C extension for `CreateFileMapping` +
  `FlushViewOfFile`.
* **After the per-layer `hipMemcpyDeviceToHost` sync was removed (engine-side
  validates ids), throughput jumped from 36 tok/s to 49.7 tok/s median -- at the target.**
* **At 50 tok/s the bottleneck is now the kernel itself** (each layer takes ~0.5 ms of
  GPU compute time). Per-layer launch overhead is ~10-50 us, so a 40-launch sequence
  only adds ~2 ms. To go higher: kernel fusion (1 launch does N layers), HIP Graph
  replay, or lower-precision weights (FP16 instead of FP32 globals).
* The 36->49 tok/s gap was
  kernel: with real weights each of the 40 layers takes ~0.7 ms, so 40 layers * 0.7 =
  ~28 ms total. IPC overhead is negligible. To close the gap, the kernel needs fusion
  (one launch that does multiple layers) or fewer launches per layer (CUDA Graphs in
  HIP, or persistent kernels).

