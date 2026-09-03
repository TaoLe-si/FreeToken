"""HIP worker process -- iGPU side of the MoE pipeline (Form-2 cross-process).

Per-layer decode: engine sends (layer_id, hidden, topk_ids, topk_weights)
for one (or several) tokens; worker runs a single igpu_moe_decode_dev
for the requested layer and returns the H-dim output.

This module is run as subprocess by the engine (via IgpuSharedMoeExecutor).
It deliberately does NOT import torch -- doing so would re-create the WDDM
KMD defect (HIP_WORKER_PITFALLS.md).

Lifecycle:
  1. Set ROCm DLL search path
  2. Load amdhip64_6.dll + hip_moe_dll.dll, igpu_init
  3. Bind TCP listener on 127.0.0.1:<port>
  4. Stream FTW banks into GTT (H2D succeeds here -- no CUDA)
  5. Per-request: read length-prefixed frame, run kernel for requested
     layer, write length-prefixed response
  6. On shutdown: drain, exit 0
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

import numpy as np

from freetoken.igpu.protocol import (
    DEFAULT_PORT,
    IpcError,
    WorkerSide,
)


_H = 2048
_TOPK = 8
_REQUEST_BYTES = 1 + _H * 4 + _TOPK * 4 + _TOPK * 4  # 8257
_RESPONSE_BYTES = _H * 4  # 8192


def _setup_rocm_path() -> None:
    candidates = (
        r"C:\Program Files\AMD\ROCm\6.4\bin",
        r"C:\Program Files\AMD\ROCm\6.3\bin",
        r"C:\Program Files\AMD\ROCm\6.2\bin",
        r"C:\Program Files\AMD\ROCm\6.1\bin",
        r"C:\Program Files\AMD\ROCm\6.0\bin",
        r"C:\Program Files\AMD\ROCm\5.7\bin",
    )
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
    hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipStreamSynchronize.restype = ctypes.c_int

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


def _alloc_staging(dll, slot_count: int = 8):
    """Per-request device-side staging buffers (GTT). Reused across requests."""
    hidden_b = _H * 4
    ids_b = _TOPK * 4
    wts_b = _TOPK * 4
    out_b = _H * 4
    slots = []
    for _ in range(slot_count):
        slots.append({
            "hidden": (dll.igpu_devmalloc(ctypes.c_size_t(hidden_b)), hidden_b),
            "ids": (dll.igpu_devmalloc(ctypes.c_size_t(ids_b)), ids_b),
            "wts": (dll.igpu_devmalloc(ctypes.c_size_t(wts_b)), wts_b),
            "out": (dll.igpu_devmalloc(ctypes.c_size_t(out_b)), out_b),
        })
    return slots


def _stream_ftw_to_gtt(path, dll, hip):
    """Stream FTW banks into GTT (no host materialisation)."""
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
            raise RuntimeError("FTW bank " + repr(b) + " has wrong layers: " +
                               str(sorted(by_layer[b])))

    num_layers = len(by_layer[expected[0]])
    total_bytes = sum(by_layer[b][L]["nbytes"] for L in range(num_layers) for b in expected)
    _log("INFO", "ftw stream begin", total_bytes=total_bytes, layers=num_layers)
    ptrs = []
    bytes_done = 0
    last_log = time.perf_counter()
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
                staging = np.empty(take, dtype=np.uint8)
                pieces = list(reader._pieces(entry["global_off"] + cursor, take))
                for shard_file, file_off, _dest_off, length in pieces:
                    shard_mv = reader._map(shard_file)
                    src = shard_mv[file_off:file_off + length]
                    staging[:length] = np.frombuffer(bytes(src), dtype=np.uint8)[:length]
                rc = hip.hipMemcpy(
                    ctypes.c_void_p(layer_ptrs[-1] + cursor),
                    ctypes.c_void_p(staging.ctypes.data),
                    ctypes.c_size_t(take), 1,
                )
                del staging, pieces
                if rc != 0:
                    raise RuntimeError("hipMemcpy H2D failed rc=" + str(rc) +
                                       " for " + repr(b) + " L" + str(L))
                cursor += take
                bytes_done += take
                if time.perf_counter() - last_log > 2.0:
                    pct = 100.0 * bytes_done / total_bytes
                    _log("INFO", "ftw stream progress",
                         pct=round(pct, 1), bytes_done=bytes_done, total=total_bytes)
                    last_log = time.perf_counter()
        ptrs.append(tuple(layer_ptrs))
        if L % 5 == 4:
            _log("INFO", "ftw layers streamed", layer=L+1, total_layers=num_layers,
                 bytes_done=bytes_done, total=total_bytes)
    _log("INFO", "ftw stream end", bytes_done=bytes_done, total=total_bytes)

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


def _send_exact(sock, data):
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        n = sock.send(view[sent:])
        if n == 0:
            raise IpcError("socket closed mid-send")
        sent += n


def _recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        buf = sock.recv(remaining)
        if not buf:
            raise IpcError("socket closed mid-recv")
        chunks.append(buf)
        remaining -= len(buf)
    return b"".join(chunks)


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

    # Listen
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)
    listener.settimeout(300.0)
    _log("INFO", "listening", port=args.port)

    # Wait for connection
    try:
        conn, addr = listener.accept()
    except socket.timeout:
        _log("ERROR", "accept timeout")
        return 3
    listener.close()
    conn.settimeout(60.0)
    _log("INFO", "ipc connected", addr=str(addr))

    # Stream FTW
    t0 = time.perf_counter()
    try:
        ptrs = _stream_ftw_to_gtt(args.ftw, dll, hip)
    except Exception as e:
        _log("ERROR", "ftw stream failed", error=str(e))
        return 4
    _log("INFO", "ftw streamed", layers=len(ptrs), seconds=round(time.perf_counter() - t0, 2))

    for L, p in enumerate(ptrs):
        rc = dll.igpu_register_layer_dev(ctypes.c_int(L), *map(ctypes.c_void_p, p))
        if rc != 0:
            _log("ERROR", "register failed", layer=L, rc=rc)
            return 5

    staging = _alloc_staging(dll, slot_count=4)
    _log("INFO", "staging allocated", slots=len(staging))

    loop = {"shutdown": False}
    _set_signal_handlers(loop)

    H2D = 1
    D2H = 2

    _log("INFO", "ready for requests")

    requests_handled = 0
    kernel_us_total = 0
    while not loop["shutdown"]:
        try:
            raw_len = _recv_exact(conn, 4)
            req_len = struct.unpack("<I", raw_len)[0]
        except (IpcError, OSError) as e:
            _log("INFO", "client closed", error=str(e))
            break
        if req_len % _REQUEST_BYTES != 0:
            _log("ERROR", "bad request size", size=req_len)
            return 6
        n_tokens = req_len // _REQUEST_BYTES
        payload = _recv_exact(conn, req_len)

        # Parse each token's request
        slot_idx = requests_handled % len(staging)
        s = staging[slot_idx]
        h_dev, h_b = s["hidden"]
        i_dev, i_b = s["ids"]
        w_dev, w_b = s["wts"]
        o_dev, o_b = s["out"]

        out_buf = bytearray(n_tokens * _RESPONSE_BYTES)
        H2D_total_us = 0
        D2H_total_us = 0
        kernel_total_us = 0

        for t in range(n_tokens):
            base = t * _REQUEST_BYTES
            layer_id = payload[base]
            hidden = payload[base + 1:base + 1 + _H * 4]
            ids = payload[base + 1 + _H * 4:base + 1 + _H * 4 + _TOPK * 4]
            wts = payload[base + 1 + _H * 4 + _TOPK * 4:base + _REQUEST_BYTES]

            t0 = time.perf_counter()
            rc = hip.hipMemcpy(h_dev, ctypes.c_void_p(ctypes.addressof(
                (ctypes.c_uint8 * h_b).from_buffer_copy(hidden))),
                ctypes.c_size_t(h_b), H2D)
            if rc != 0:
                _log("ERROR", "H2D hidden failed", rc=rc, token=t)
                return 7
            rc = hip.hipMemcpy(i_dev, ctypes.c_void_p(ctypes.addressof(
                (ctypes.c_uint8 * i_b).from_buffer_copy(ids))),
                ctypes.c_size_t(i_b), H2D)
            if rc != 0:
                _log("ERROR", "H2D ids failed", rc=rc)
                return 7
            rc = hip.hipMemcpy(w_dev, ctypes.c_void_p(ctypes.addressof(
                (ctypes.c_uint8 * w_b).from_buffer_copy(wts))),
                ctypes.c_size_t(w_b), H2D)
            if rc != 0:
                _log("ERROR", "H2D wts failed", rc=rc)
                return 7
            H2D_total_us += int((time.perf_counter() - t0) * 1e6)

            # Run single layer
            t0 = time.perf_counter()
            rc = dll.igpu_moe_decode_dev(
                ctypes.c_int(layer_id),
                ctypes.c_void_p(h_dev),
                ctypes.c_void_p(i_dev),
                ctypes.c_void_p(w_dev),
                ctypes.c_void_p(o_dev),
            )
            if rc != 0:
                _log("ERROR", "moe decode failed", layer=layer_id, rc=rc)
                return 8
            kernel_total_us += int((time.perf_counter() - t0) * 1e6)

            # D2H output
            t0 = time.perf_counter()
            out_bytes = (ctypes.c_uint8 * _RESPONSE_BYTES).from_buffer(out_buf, t * _RESPONSE_BYTES)
            rc = hip.hipMemcpy(
                ctypes.c_void_p(ctypes.addressof(out_bytes)),
                ctypes.c_void_p(o_dev),
                ctypes.c_size_t(_RESPONSE_BYTES), D2H,
            )
            if rc != 0:
                _log("ERROR", "D2H out failed", rc=rc)
                return 9
            D2H_total_us += int((time.perf_counter() - t0) * 1e6)

        # Sync once per request
        stream_handle = dll.igpu_get_stream()
        hip.hipStreamSynchronize(stream_handle)

        # Send response
        try:
            _send_exact(conn, struct.pack("<I", len(out_buf)) + bytes(out_buf))
        except (IpcError, OSError) as e:
            _log("WARN", "send response failed", error=str(e))
            break

        requests_handled += 1
        if requests_handled % 50 == 1:
            _log("INFO", "heartbeat", requests=requests_handled,
                 h2d_us=H2D_total_us, kernel_us=kernel_total_us, d2h_us=D2H_total_us)

    try:
        conn.close()
    except Exception:
        pass
    _log("INFO", "shutting down", handled=requests_handled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
