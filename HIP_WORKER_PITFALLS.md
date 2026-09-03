# iGPU HIP Worker — WDDM KMD Pitfalls (Windows + Dual GPU)

> Author: FreeToken dev | Date: 2026-09-03 | Status: source of truth for the Form-2 cross-process decision.

## TL;DR

On Windows hosts with both an NVIDIA dGPU and an AMD iGPU, **any** HIP
(`amdhip64_*.dll`) call into GTT that **writes** through the engine's address
space returns `hipErrorInvalidValue` (rc=1) the moment a CUDA context is
created in the same process. The kernel-mode driver (WDDM) marks the GTT page
tables registered but does **not** actually map them writable from the GPU's
DMA path. Read paths (D2H, `hipMemGetInfo`, zero-copy kernel reads from pinned
host) and **GTT allocation** (`hipMalloc`) still report success — the failure
is invisible at the API surface until you try to land a transfer.

This is a **driver-level bug**, not a FreeToken bug. It is reproduced by a
20-line standalone (alloc + H2D + D2H) once `import torch` has run inside the
same process. It also reproduces after `torch.cuda.init()`, after
`cudaHostAlloc`, after `cudaMemcpy`, and after any pointer handed to a CUDA
kernel — i.e. any state in which the CUDA driver has at least one live context.

The verified fix is **process isolation**: do all GTT writes in a separate
process that never imports `torch` / `torch.cuda`. The kernel sees the GTT
pages through a fresh, CUDA-free address space, and the WDDM KMD behaves
correctly. FreeToken implements this as a **HIP worker subprocess** that owns
the iGPU end-to-end (load FTW → allocate GTT → `hipMemcpy` H2D → run kernels)
and exchanges activations with the engine through a shared-memory ring
buffer. See `FORM2_CROSS_PROCESS_DESIGN.md` for the architecture.

## 1. Reproduction

`_probe_dev7.py` is the canonical minimal case. Two phases:

```python
# Phase A (engine-shaped): CUDA context first, then HIP
import torch                      # CUDA context lives for the rest of the process
torch.empty(..., pin_memory=True) # CUDA pinned page created
dll.igpu_init()                   # HIP runtime init
for _ in range(453):
    ptrs.append(dll.igpu_devmalloc(433000000))   # OK, addresses advance 0x8000000
src = bytes([0x41]*8192)
hip.hipMemcpy(ptrs[0], src, 8192, hipMemcpyHostToDevice)  # rc = 1  ← DEAD
hip.hipMemcpy(dst, ptrs[0],    8192, hipMemcpyDeviceToHost)  # rc = 0  ← alive
hip.hipMemset (ptrs[0], 0,     8192)                          # rc = 1  ← DEAD
hip.hipMemcpyAsync(...)                                     # rc = 1  ← DEAD
```

```python
# Phase B (worker-shaped): never import torch, allocate GTT first
os.add_dll_directory(rocm_bin)
hip = ctypes.CDLL("amdhip64_6.dll")
dll = ctypes.CDLL("hip_moe_dll.dll"); dll.igpu_init()
for _ in range(453):
    ptrs.append(dll.igpu_devmalloc(433000000))    # OK
hip.hipMemcpy(ptrs[0], src, 8192, hipMemcpyHostToDevice)  # rc = 0 ✓
```

Same machine, same kernel, same DLL — only the address space changes. The
asymmetry between H2D and D2H is the smoking gun: page tables were registered
during `hipMalloc` (meminfo reports correct deltas), but the underlying
mapping is incomplete from the GPU's write side.

## 2. What we tried that did **not** fix it

| Attempt | Outcome |
|---|---|
| `FT_IGPU_RESERVE=1` (pre-allocate GTT before `import torch`) | Same rc=1 on later H2D — reservation only buys stable addresses, the write failure is in the KMD submit path, not the allocation path. |
| Switch from `cudaHostAlloc(..., Portable\|Mapped)` pinned source to plain pageable numpy | Still rc=1 — H2D source staging isn't the cause. |
| `hipHostRegister` on a 1 GB mmap (zero-copy Form-1) | Kernel reads of the registered VA return `-9` (`hipErrorInvalidDevicePointer`) inside `igpu_moe_decode_dev` — same WDDM bug, this time on the read path. |
| `hipMemcpyAsync` on the default stream with explicit events | rc=1 — async path is no healthier. |
| Replace pinned source with `np.zeros(...)` (pageable, fresh) | rc=1 — not a pinning issue. |
| Use `_IGPU_RESERVED` GTT pool (40 × 433 MB at process start) | rc=1 — addresses are valid (`meminfo` shows the right consumption) but writes still don't land. |
| CUDA stream synchronize before every HIP H2D | rc=1 — synchronization is unrelated to the bug. |
| Detach CUDA context (`torch.cuda.empty_cache()`, `gc.collect()`) | No effect — the failure is in the AMD KMD's view of GTT DMA, not in NVIDIA state. |
| `hipCtxDestroy` / `hipDeviceReset` | Hangs — the engine relies on the CUDA context for the rest of its work. |
| Run HIP `cudaFreeHost` / `cudaHostUnregister` to "clear" CUDA pins | Irrelevant — pinned host memory is the source side, not the destination. |

Every in-process attempt produced rc=1; the only fix was the process boundary.

## 3. Why Form-1 zero-copy also fails (same KMD bug)

Zero-copy Form-1 (`hipHostRegister` + `hipHostGetDevicePointer`) lets the kernel
read pinned host memory directly. The reading kernel returns `-9`
(`hipErrorInvalidDevicePointer`) inside `igpu_moe_decode_dev` for any layer
that touches a zero-copy bank. This is **not** the same surface symptom as the
H2D rc=1, but the underlying cause is identical: KMD page-table entries for
the host-mapped region are present but the GPU-side write/read of that region
fails consistently. See `IGPU_ZEROCOPY_VERDICT.md` for the older
investigation; the WDDM conclusion there is now superseded by "KMD submission
defect", not "zero-copy is structurally dead".

## 4. Cross-process design constraints

* **Worker must not import `torch`.** Even `import torch` (let alone
  `torch.cuda.init()`) creates the CUDA-context state that triggers the KMD
  bug. The worker imports only `ctypes`, `mmap`, `os`, `numpy` (the first
  three are sufficient for the FTW path; numpy is optional for H2D staging).
* **Worker must own GTT pointers.** GTT VAs are per-process on WDDM. The
  engine cannot share its CUDA-side addresses with the worker, and the worker
  cannot share its addresses back. The IPC boundary carries **data only**
  (hidden states, expert ids/weights, output activations) — never device
  pointers.
* **Engine must keep its own pinned host banks** for the prefill path. The
  prefill `fast_index_copy` kernel requires CUDA pinned+mapped host memory
  (`host tensor must be pinned+mapped`). Stripping prefill off the pinned
  path breaks it independently of the iGPU work.
* **IPC latency budget** is ~1–2 ms / token (one syscall ring + one mmap read
  + one mmap write). Phase-2 fusion can overlap kernel launches with model
  forward on the dGPU, hiding most of it.
* **Worker must be supervised.** When the engine dies, the worker must die
  too (no orphan GTT allocations, no leaked mmap regions). The daemon
  spawns both, kills both, and uses the same start/stop policy.
* **HIP DLL path is environment-sensitive.** The worker prepends
  `C:\Program Files\AMD\ROCm\6.4\bin` to `PATH` (via
  `os.add_dll_directory`) before loading `amdhip64_6.dll`, or the DLL load
  fails on machines where ROCm isn't on PATH. Doing this in the worker only
  keeps the engine process clean.

## 5. Engineering checklist (one-time)

* [x] Confirm `rc=1` is the only failure mode (memset, async, plain, pinned all
      fail identically).
* [x] Confirm `D2H` and `hipMemGetInfo` still work after CUDA init — the KMD's
      read/query path is fine.
* [x] Confirm `_simtest6.py` (no CUDA, `np.zeros` H2D source) reproduces 35 t/s
      and `rel err 1.1e-3` in a standalone process — the kernel/DLL itself is
      correct, only the engine-side integration is blocked.
* [x] Document the failure surface so the next reviewer doesn't re-run the
      same experiments (Form-1 zero-copy, FT_IGPU_RESERVE, cudaFreeHost dance,
      etc.).

## 6. Anti-patterns (do not try again)

* Adding more CUDA-side synchronization around HIP writes.
* Re-allocating GTT after every decode step hoping to find a "good" address.
* Pre-allocating the full 17 GB before `import torch` and *not* H2D-ing until
  after — the addresses are valid but still write-dead.
* Swapping to `cudart_static` HIP shim — same KMD path.
* Splitting the CUDA and HIP libraries into separate venvs but the **same
  process** — the bug is in the address space, not the library load.
* Calling `cudaSetDevice(0)` to "reset" CUDA before HIP writes — the failure
  is in the AMD KMD's view of GTT, not in CUDA state.

## 7. Forward pointer

`FORM2_CROSS_PROCESS_DESIGN.md` describes the worker architecture, the
shared-memory protocol, and the latency budget that gets us back into the
30–45 t/s range (vs. the in-process dead end). The `_probe_dev7.py` and
`_probe_spawn.py` probes are still in the tree as regression tests: they
must continue to fail in any process that has imported `torch`, and pass in
any process that has not.


## 11. Windows mmap is NOT cross-process coherent (even with `tagname=` or file-backed)

**Symptom:** Two processes mmap the same region (either via `tagname=` anonymous,
or both mmap the same file with explicit `mmap.flush()`). The writer writes
tail=1,2,3,4,5; the reader in the *other* process polls for 3 seconds and
always reads tail=0.

**Verified reproductions** (each ran for 3-10 s, same writer+reader pair):
  * `mmap.mmap(-1, len, tagname="Local\\FT_IGPU_RING_V01", ...)` -- reader never sees writes
  * `mmap.mmap(-1, len, tagname="Global\\FT_TEST_RING", ...)` -- reader never sees writes (also Global\\ needs admin)
  * Two processes both `mmap.mmap(fd, ...)` on the same file with `m.flush()` after every write -- reader still sees zeros

**Root cause:** Python `mmap.mmap()` on Windows wraps `CreateFileMapping` +
`MapViewOfFile`. While the file mapping object itself is cross-process, the
view returned is per-process; writes from process A do not propagate to
process B's view under the default caching semantics. The CPython mmap
module does NOT expose `FlushViewOfFile`/`MapViewOfFile2` with `SEC_RESERVE`
flags that would be needed to get coherent cross-process memory.

**Workaround used:** TCP loopback on 127.0.0.1 (`EngineSide`/`WorkerSide` in
`protocol.py`). Per-request overhead measured on this box is ~1.5 ms (vs
~50 us with a real coherent mmap), but TCP loopback is simple, well-tested,
and well within the 1-2 ms IPC budget documented in FORM2_CROSS_PROCESS_DESIGN.md.

**If you ever need true coherent mmap on Windows:** write a small C extension
wrapping `CreateFileMapping` + `FlushViewOfFile` + `MapViewOfFile2` with
`SEC_RESERVE | SEC_COMMIT` and an explicit named mapping. The stdlib mmap
module cannot do this.
