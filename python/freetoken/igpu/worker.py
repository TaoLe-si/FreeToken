"""HIP worker process -- the iGPU side of the MoE pipeline.

This module is run as a subprocess by the engine supervisor. It deliberately
does NOT import torch or any other CUDA library: doing so would re-create the
WDDM KMD defect that makes hipMemcpy H2D return rc=1 in the engine process
(see HIP_WORKER_PITFALLS.md).

Lifecycle:
1. Set up ROCm DLL search path (os.add_dll_directory) before any HIP load.
2. Load amdhip64_6.dll + hip_moe_dll.dll.
3. Open the shared ring file created by the engine (supervisor passes the path).
4. Stream-load FTW banks into hipMalloc'd GTT (H2D succeeds here -- no CUDA).
5. Register all 40 layers with igpu_register_layer_dev.
6. Allocate per-request device-side staging buffers (hidden / ids / weights / out_hidden).
7. Loop: wait for engine request, run 40 x igpu_moe_decode_dev, publish result.
8. On SIGTERM / SIGINT: drain ring (best effort), exit 0.

The worker prints structured log lines to stdout that the daemon captures
and forwards to the engine log; it does not write to any file itself.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import signal
import socket
import struct
import sys
import time

# IMPORTANT: do not import torch at module top level. numpy is OK because it
# does not pull CUDA. See HIP_WORKER_PITFALLS.md for the rationale.

from freetoken.igpu.protocol import (
    DEFAULT_PORT,
    H_DIM,
    IpcError,
    TOPK,
    WorkerSide,
    pack_response,
    unpack_request,
)


def _setup_rocm_path() -> None:
    candidates = [
        r"C:\Program Files\AMD\ROCm\6.4\bin",
        r"C:\Program Files\AMD\ROCm\6.3\bin",
        r"C:\Program Files\AMD\ROCm\6.2\bin",
        r"C:\Program Files\AMD\ROCm\6.1\bin",
        r"C:\Program Files\AMD\ROCm\6.0\bin",
        r"C:\Program Files\AMD\ROCm\5.7\bin",
    ]
    for p in candidates:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass
            return


def _load_dlls() -> tuple:
    hip_names = ("amdhip64_6.dll", "amdhip64_5.dll", "amdhip64_4.dll")
    hip = None
    last_err = None
    for name in hip_names:
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError as e:
            last_err = e
    if hip is None:
        raise RuntimeError("failed to load any amdhip64_*.dll: " + str(last_err))

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    dll_path = os.path.join(repo_root, "benchmarks", "cpu_moe_microbench", "hip_moe_dll.dll")
    if not os.path.isfile(dll_path):
        raise RuntimeError("hip_moe_dll.dll not found at " + dll_path)
    dll = ctypes.CDLL(dll_path)

    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMemcpy.restype = ctypes.c_int

    dll.igpu_init.argtypes = []
    dll.igpu_init.restype = ctypes.c_int
    dll.igpu_meminfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    dll.igpu_meminfo.restype = ctypes.c_int
    dll.igpu_devmalloc.argtypes = [ctypes.c_size_t]
    dll.igpu_devmalloc.restype = ctypes.c_void_p
    dll.igpu_devfree.argtypes = [ctypes.c_void_p]
    dll.igpu_devfree.restype = ctypes.c_int
    dll.igpu_register_layer_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 6
    dll.igpu_register_layer_dev.restype = ctypes.c_int
    dll.igpu_moe_decode_dev.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
    dll.igpu_moe_decode_dev.restype = ctypes.c_int

    return hip, dll


def _alloc_staging(dll, slot_count=8):
    hidden_b = H_DIM * 4
    ids_b = TOPK * 4
    wts_b = TOPK * 4
    out_b = H_DIM * 4
    slots = []
    for _ in range(slot_count):
        slots.append({
            "hidden": (dll.igpu_devmalloc(ctypes.c_size_t(hidden_b)), hidden_b),
            "ids":    (dll.igpu_devmalloc(ctypes.c_size_t(ids_b)),    ids_b),
            "wts":    (dll.igpu_devmalloc(ctypes.c_size_t(wts_b)),    wts_b),
            "out":    (dll.igpu_devmalloc(ctypes.c_size_t(out_b)),    out_b),
        })
    return slots


def _stream_ftw_to_gtt(path, dll, hip):
    import numpy as np
    from freetoken.checkpoint.ftw import FTWReader
    reader = FTWReader(path)
    bank_entries = reader.entries("experts_bank")
    alpha_names = {"gate_up_alpha", "down_alpha"}
    row_entries = [e for e in bank_entries if e["name"] not in alpha_names]

    pat = re.compile(r"^(?P<base>.+)#L(?P<layer>\d{5,})$")
    by_layer = {}
    for e in row_entries:
        m = pat.match(e["name"])
        if not m:
            raise RuntimeError("FTW bank entry not per-layer: " + repr(e["name"]))
        base = m.group("base")
        layer = int(m.group("layer"))
        by_layer.setdefault(base, {})[layer] = e

    expected = ("gate_up_packed", "gate_up_scale", "gate_up_global",
                "down_packed", "down_scale", "down_global")
    for b in expected:
        if b not in by_layer:
            raise RuntimeError("FTW missing bank " + repr(b))
    for b in expected:
        if set(by_layer[b].keys()) != set(range(len(by_layer[expected[0]]))):
            raise RuntimeError("FTW bank " + repr(b) + " has wrong layers: " + str(sorted(by_layer[b])))

    num_layers = len(by_layer[expected[0]])
    ptrs = []
    for L in range(num_layers):
        layer_ptrs = []
        for b in expected:
            entry = by_layer[b][L]
            nbytes = entry["nbytes"]
            p = dll.igpu_devmalloc(ctypes.c_size_t(nbytes))
            if not p:
                raise RuntimeError("igpu_devmalloc(" + str(nbytes) + ") for " + repr(b) + " L" + str(L) + " failed")
            layer_ptrs.append(int(p))
            CHUNK = 16 * 1024 * 1024
            cursor = 0
            while cursor < nbytes:
                take = min(CHUNK, nbytes - cursor)
                # Build a fresh numpy staging buffer per chunk; release it
                # before reader.close() so its pointer doesn't pin the shard
                # mmap on Windows.
                staging = np.empty(take, dtype=np.uint8)
                pieces = list(reader._pieces(entry["global_off"] + cursor, take))
                for shard_file, file_off, _dest_off, length in pieces:
                    shard_mv = reader._map(shard_file)
                    src = shard_mv[file_off:file_off + length]
                    staging[:length] = np.frombuffer(bytes(src), dtype=np.uint8)[:length]
                rc = hip.hipMemcpy(
                    ctypes.c_void_p(layer_ptrs[-1] + cursor),
                    ctypes.c_void_p(staging.ctypes.data),
                    ctypes.c_size_t(take),
                    1,
                )
                del staging, pieces
                if rc != 0:
                    raise RuntimeError("hipMemcpy H2D failed rc=" + str(rc) + " for " + repr(b) + " L" + str(L))
                cursor += take
        ptrs.append(tuple(layer_ptrs))

    # Release any remaining memoryviews from the reader before closing.
    # On Windows, mmap.mmap.close() refuses to release while any view is
    # alive (even a parent memoryview that has been "released" can keep the
    # underlying file mapping alive). Drop _maps + _fds, force GC, then
    # close.
    try:
        for m, mv in list(reader._maps.values()):
            try:
                mv.release()
            except Exception:
                pass
        reader._maps.clear()
        for fd in list(reader._fds.values()):
            try:
                os.close(fd)
            except Exception:
                pass
        reader._fds.clear()
        import gc
        gc.collect()
        reader.close()
    except Exception as e:
        _log("WARN", "reader close raised", error=str(e))
        pass
    return ptrs


def _log(level, msg, **fields):
    rec = {"ts": time.time(), "level": level, "msg": msg}
    rec.update(fields)
    try:
        sys.stdout.write("IGPU_W " + json.dumps(rec) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _set_signal_handlers(state):
    def _handler(signum, frame):
        _log("INFO", "signal received", signum=int(signum))
        state["shutdown"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftw", required=True)
    parser.add_argument("--num-layers", type=int, default=40)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    _log("INFO", "starting", ftw=args.ftw, num_layers=args.num_layers, port=args.port)

    _setup_rocm_path()

    try:
        hip, dll = _load_dlls()
    except Exception as e:
        _log("ERROR", "dll load failed", error=str(e))
        return 2

    if dll.igpu_init() != 0:
        _log("ERROR", "igpu_init failed")
        return 2
    _log("INFO", "hip runtime ready")

    ipc = WorkerSide(port=args.port)
    try:
        ipc.connect(slot_count=8, timeout_s=120.0)
    except IpcError as e:
        _log("ERROR", "ipc connect failed", error=str(e))
        return 2
    _log("INFO", "ipc connected")

    t0 = time.perf_counter()
    try:
        ptrs = _stream_ftw_to_gtt(args.ftw, dll, hip)
    except Exception as e:
        _log("ERROR", "ftw stream failed", error=str(e))
        return 3
    _log("INFO", "ftw streamed", layers=len(ptrs), seconds=round(time.perf_counter() - t0, 2))

    for L, p in enumerate(ptrs):
        rc = dll.igpu_register_layer_dev(ctypes.c_int(L), *map(ctypes.c_void_p, p))
        if rc != 0:
            _log("ERROR", "register failed", layer=L, rc=rc)
            return 4

    staging = _alloc_staging(dll, slot_count=8)
    _log("INFO", "staging allocated", slots=len(staging))

    loop = {"shutdown": False}
    _set_signal_handlers(loop)

    H2D = 1
    D2H = 2
    requests_handled = 0
    kernel_ms_avg = 0.0

    _log("INFO", "ready for requests")

    ipc._sock.settimeout(0.5)
    while not loop["shutdown"]:
        try:
            payload = ipc.recv_request()
        except IpcError:
            break
        except socket.timeout:
            continue
        except OSError as e:
            _log("WARN", "recv error", error=str(e))
            break

        slot_idx = requests_handled % len(staging)
        hidden, ids, wts, token_id, request_id, seq = unpack_request(payload)

        s = staging[slot_idx]
        h_dev, h_b = s["hidden"]
        i_dev, i_b = s["ids"]
        w_dev, w_b = s["wts"]
        o_dev, o_b = s["out"]

        hidden_bytes = struct.pack(f"<{H_DIM}f", *hidden)
        ids_bytes = struct.pack(f"<{TOPK}i", *ids)
        wts_bytes = struct.pack(f"<{TOPK}f", *wts)
        staging_hidden = (ctypes.c_uint8 * h_b).from_buffer_copy(hidden_bytes)
        staging_ids = (ctypes.c_uint8 * i_b).from_buffer_copy(ids_bytes)
        staging_wts = (ctypes.c_uint8 * w_b).from_buffer_copy(wts_bytes)

        rc = hip.hipMemcpy(ctypes.c_void_p(h_dev), ctypes.c_void_p(ctypes.addressof(staging_hidden)),
                           ctypes.c_size_t(h_b), H2D)
        if rc != 0:
            _log("ERROR", "H2D hidden failed", rc=rc)
            break
        rc = hip.hipMemcpy(ctypes.c_void_p(i_dev), ctypes.c_void_p(ctypes.addressof(staging_ids)),
                           ctypes.c_size_t(i_b), H2D)
        if rc != 0:
            _log("ERROR", "H2D ids failed", rc=rc)
            break
        rc = hip.hipMemcpy(ctypes.c_void_p(w_dev), ctypes.c_void_p(ctypes.addressof(staging_wts)),
                           ctypes.c_size_t(w_b), H2D)
        if rc != 0:
            _log("ERROR", "H2D wts failed", rc=rc)
            break

        t_kernel = time.perf_counter()
        for L in range(args.num_layers):
            rc = dll.igpu_moe_decode_dev(
                ctypes.c_int(L),
                ctypes.c_void_p(h_dev),
                ctypes.c_void_p(i_dev),
                ctypes.c_void_p(w_dev),
                ctypes.c_void_p(o_dev),
            )
            if rc != 0:
                _log("ERROR", "moe decode failed", layer=L, rc=rc)
                break
            h_dev, o_dev = o_dev, h_dev
        kernel_ms = (time.perf_counter() - t_kernel) * 1000.0
        kernel_ms_avg = kernel_ms_avg * 0.9 + kernel_ms * 0.1

        final_out = o_dev if h_dev == staging[slot_idx]["hidden"][0] else h_dev
        out_bytes = (ctypes.c_uint8 * H_DIM * 4)()
        rc = hip.hipMemcpy(ctypes.c_void_p(ctypes.addressof(out_bytes)),
                           ctypes.c_void_p(final_out),
                           ctypes.c_size_t(H_DIM * 4), D2H)
        if rc != 0:
            _log("ERROR", "D2H out failed", rc=rc)
            break
        out_arr = (ctypes.c_float * H_DIM).from_buffer(out_bytes)
        resp_payload = pack_response(list(out_arr), rc=0, latency_us=int(kernel_ms * 1000))

        try:
            ipc.send_response(resp_payload)
        except IpcError as e:
            _log("WARN", "send response failed", error=str(e))
            break

        requests_handled += 1
        if requests_handled % 50 == 1:
            _log("INFO", "heartbeat", requests=requests_handled, kernel_ms=round(kernel_ms_avg, 2))

    try:
        ipc.close()
    except Exception:
        pass
    _log("INFO", "shutting down", handled=requests_handled)
    return 0

if __name__ == "__main__":
    sys.exit(main())
