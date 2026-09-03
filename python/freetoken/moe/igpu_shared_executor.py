"""Shared-pool iGPU MoE executor -- Form-2 cross-process (--moe-backend igpu).

The engine process never loads HIP directly. Doing so re-creates the Windows
KMD WDDM bug: in a process that has both CUDA + AMD HIP initialised, all
hipMemcpy H2D returns rc=1 even though GTT allocations, D2H, and kernel
launches succeed (HIP_WORKER_PITFALLS.md #1..#10).

Instead we spawn "python -m freetoken.igpu.worker" as a subprocess. The
worker never imports torch; it loads amdhip64_6.dll + the MoE plugin,
streams the FTW banks into the 780M's GTT (H2D succeeds in a no-CUDA
process), and serves per-layer decode requests via TCP loopback on
127.0.0.1. Per-layer IPC overhead is small; 40 layers * ~0.5 ms +
~22 ms of kernel compute = ~26 ms / token.

Public API (unchanged):
  IgpuSharedMoeExecutor(cache, device, num_layers, num_experts, top_k)
  .register_banks()
  .decode(layer_id, h, w, ids) -> Tensor
"""

from __future__ import annotations

import os
import socket as _socket
import subprocess
import sys
import struct
import threading
import time

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


_H = 2048
_TOPK = 8
_REQUEST_BYTES = 1 + _H * 4 + _TOPK * 4 + _TOPK * 4  # 8257
_RESPONSE_BYTES = _H * 4  # 8192


def _default_port() -> int:
    s = _socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_argv(ftw_path: str, num_layers: int, port: int) -> tuple:
    py = sys.executable
    env_pypath = os.environ.get("PYTHONPATH", "")
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    pkg_root = os.path.join(pkg_root, "python")
    pypath = os.pathsep.join(x for x in (env_pypath, pkg_root) if x)
    env = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
        "PYTHONPATH": pypath,
        "PATH": os.environ.get("PATH", ""),
    }
    argv = [py, "-u", "-m", "freetoken.igpu.worker",
            "--ftw", ftw_path, "--num-layers", str(num_layers),
            "--port", str(port)]
    return argv, env


class IgpuSharedMoeExecutor:
    """Cross-process iGPU MoE executor. Spawns an HIP worker subprocess and
    routes per-layer decode through it via TCP loopback."""

    # The cross-process decode is fundamentally not graph-replayable: the
    # captured graph would replay only the D2H/H2D memcpys around our pinned
    # staging buffers and skip the TCP send/recv + worker kernel call (those
    # are CPU-side ops that don't get recorded), so on replay the output
    # GPU tensor would carry stale data from the capture-time call. The
    # engine checks this flag in its graph_runner.can_use_cuda_graph branch
    # to fall back to eager decode for this executor.
    graph_replay_safe = False

    def __init__(self, cache, device, num_layers, num_experts, top_k=_TOPK) -> None:
        self.cache = cache
        self.device = device
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        if int(top_k) != _TOPK:
            raise NotImplementedError(
                f"igpu worker hardcodes top_k={_TOPK}, this model routes top_k={top_k}"
            )
        self.top_k = int(top_k)
        self.quant_format = cache.quant_format
        if self.quant_format != "nvfp4":
            raise NotImplementedError(
                f"igpu executor reads nvfp4 banks, cache is {self.quant_format!r}"
            )

        self._registered = False
        self._proc = None
        self._sock = None
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._ipc_ready = threading.Event()
        self._ftw_path = getattr(cache, "folder_path", None) or getattr(cache, "ftw_path", None)
        if not self._ftw_path or not os.path.isdir(self._ftw_path):
            raise RuntimeError(
                "IgpuSharedMoeExecutor: cache has no folder_path; cannot launch worker"
            )

        self._port = _default_port()
        argv, env = _worker_argv(self._ftw_path, self.num_layers, self._port)
        logger.info_rank0(
            "iGPU cross-process worker: port=%d ftw=%s num_layers=%d",
            self._port, self._ftw_path, self.num_layers,
        )
        CREATE_NO_WINDOW = 0x08000000
        self._proc = subprocess.Popen(
            argv, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._drain_stdout, args=(self._proc.stdout,), daemon=True).start()

        # Connect TCP FIRST. The worker blocks on accept() before it logs
        # "ready for requests", so waiting for the ready event before
        # connecting is a handshake deadlock (worker waits for us to
        # connect, we wait for worker to be ready, both block forever).
        deadline = time.monotonic() + 30.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                # No timeout: blocking mode. Decode may legitimately take
                # several seconds for large bs (e.g. warmup_prefill at bs=128:
                # 128 tokens x 0.5 ms kernel + 5 prefill lengths x 40 layers
                # = several seconds of TCP-loopback work). If the worker dies
                # the OS closes the socket and recv_into returns 0 (EOF), which
                # we already detect and raise on.
                s.settimeout(None)
                s.connect(("127.0.0.1", self._port))
                self._sock = s
                break
            except OSError as e:
                last_err = e
                try:
                    s.close()
                except Exception:
                    pass
                time.sleep(0.2)
        if self._sock is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            raise RuntimeError(
                f"failed to connect to igpu worker at port {self._port}: {last_err}"
            )

        # Now wait for the worker to actually load HIP + stream FTW + be ready
        self._wait_for_ready(timeout_s=240.0)
        logger.info_rank0(
            "iGPU executor connected to worker on 127.0.0.1:%d (pid=%d)",
            self._port, self._proc.pid,
        )

        # Per-call pinned staging for CUDA-graph safe D2H/H2D
        self._staging = None
        self._staging_bs = 0

    def _drain_stdout(self, stream) -> None:
        try:
            for raw in iter(stream.readline, b""):
                if not raw:
                    break
                try:
                    text = raw.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if text.startswith("IGPU_W "):
                    msg = text[len("IGPU_W "):]
                    sys.stderr.write("[igpu-worker] " + msg + "\n")
                    sys.stderr.flush()
                elif text:
                    sys.stderr.write("[igpu-worker] " + text + "\n")
                    sys.stderr.flush()
                if '"msg": "ready for requests"' in text:
                    self._ready_event.set()
                if '"msg": "ipc connected"' in text:
                    self._ipc_ready.set()
        except Exception:
            pass

    def _wait_for_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready_event.is_set():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("igpu worker exited before ready (rc=%d)" % self._proc.returncode)
            time.sleep(0.1)
        raise RuntimeError("igpu worker did not become ready within %.1fs" % timeout_s)

    def register_banks(self) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError("igpu worker exited during init")
        self._registered = True
        logger.info_rank0(
            "iGPU shared-pool MoE executor ready (Form-2 cross-process, %d layers, "
            "worker pid=%d, port=%d)",
            self.num_layers, self._proc.pid, self._port,
        )

    def health_check(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def shutdown(self) -> None:
        try:
            if self._sock is not None:
                try:
                    self._sock.shutdown(2)
                except Exception:
                    pass
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
        finally:
            if self._proc is not None:
                try:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=5.0)
                    except Exception:
                        self._proc.kill()
                        self._proc.wait(timeout=5.0)
                except Exception:
                    pass
                self._proc = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ---- per-call pinned staging (CUDA-graph friendly) ----

    def _ensure_staging(self, bs: int) -> None:
        cur_bs = getattr(self, "_staging_bs", 0)
        if bs <= cur_bs and getattr(self, "_staging", None) is not None:
            return
        if getattr(self, "_staging", None) is not None:
            self._staging = None
            import gc as _gc
            _gc.collect()
        self._staging = (
            torch.empty((bs, _H), dtype=torch.float32, pin_memory=True, requires_grad=False),
            torch.empty((bs, _TOPK), dtype=torch.int32, pin_memory=True, requires_grad=False),
            torch.empty((bs, _TOPK), dtype=torch.float32, pin_memory=True, requires_grad=False),
            torch.empty((bs, _H), dtype=torch.float32, pin_memory=True, requires_grad=False),
        )
        self._staging_bs = bs

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-layer decode via TCP. Returns a GPU [bs, H] tensor.

        Async D2H into pinned host staging (CUDA-graph capture-safe), TCP
        request to the no-torch HIP worker, async H2D back into a fresh
        GPU tensor. The pinned memcpys are recorded into the graph; the
        TCP I/O is plain CPU and not recorded.
        """
        assert self._registered, "register_banks() must run before decode"
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == _H
        assert topk_ids.shape[-1] == _TOPK
        bs = int(hidden_states.shape[0])

        self._ensure_staging(bs)
        hid_buf, ids_buf, wts_buf, out_buf = self._staging
        # Async D2H into pinned staging
        hid_buf[:bs].copy_(hidden_states.to(torch.float32), non_blocking=True)
        ids_buf[:bs].copy_(topk_ids.to(torch.int32), non_blocking=True)
        wts_buf[:bs].copy_(topk_weights.to(torch.float32), non_blocking=True)
        # async D2H submitted; sync current stream so pinned bytes are visible
        # before we read them on the host (send to worker over TCP).
        torch.cuda.current_stream().synchronize()

        # Pack: 1B layer_id + 8192B hidden + 32B ids + 32B weights per token
        if bs == 1:
            req = bytearray(_REQUEST_BYTES)
            req[0] = int(layer_id) & 0xFF
            req[1:1 + _H * 4] = bytes(hid_buf[0].numpy())
            off = 1 + _H * 4
            req[off:off + _TOPK * 4] = bytes(ids_buf[0].numpy())
            off += _TOPK * 4
            req[off:off + _TOPK * 4] = bytes(wts_buf[0].numpy())
            req = bytes(req)
        else:
            parts = []
            for tok in range(bs):
                p = bytearray(_REQUEST_BYTES)
                p[0] = int(layer_id) & 0xFF
                p[1:1 + _H * 4] = bytes(hid_buf[tok].numpy())
                off = 1 + _H * 4
                p[off:off + _TOPK * 4] = bytes(ids_buf[tok].numpy())
                off += _TOPK * 4
                p[off:off + _TOPK * 4] = bytes(wts_buf[tok].numpy())
                parts.append(bytes(p))
            req = b"".join(parts)

        n_tokens = len(req) // _REQUEST_BYTES
        total_resp = n_tokens * _RESPONSE_BYTES

        with self._lock:
            try:
                self._sock.sendall(len(req).to_bytes(4, "little") + req)
                # Worker prepends a 4-byte little-endian length to every
                # response. Read it first, otherwise recv_into(total_resp)
                # eats the length prefix as response data (silent corruption).
                hdr = bytearray(4)
                got_hdr = 0
                while got_hdr < 4:
                    n = self._sock.recv_into(memoryview(hdr)[got_hdr:], 4 - got_hdr)
                    if n == 0:
                        raise RuntimeError("igpu worker closed socket during recv (header)")
                    got_hdr += n
                resp_len = struct.unpack("<I", bytes(hdr))[0]
                if resp_len != total_resp:
                    raise RuntimeError(
                        f"igpu worker response length {resp_len} != expected {total_resp}"
                    )
                # recv_into needs a writable buffer; pinned tensors don't expose
                # one through numpy, so receive into a plain bytearray then copy
                # the rows back into the pinned staging.
                resp_buf = bytearray(total_resp)
                got = 0
                while got < total_resp:
                    n = self._sock.recv_into(memoryview(resp_buf)[got:], total_resp - got)
                    if n == 0:
                        raise RuntimeError("igpu worker closed socket during recv")
                    got += n
                # Copy bytes -> pinned float32 staging
                out_np = np.frombuffer(resp_buf, dtype=np.float32).reshape(n_tokens, _H).copy()
                out_buf[:bs].copy_(torch.from_numpy(out_np), non_blocking=True)
            except OSError as e:
                raise RuntimeError("igpu worker IPC failed: " + str(e)) from e

        out = torch.empty((bs, _H), dtype=hidden_states.dtype, device=self.device)
        out.copy_(out_buf[:bs], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return out
