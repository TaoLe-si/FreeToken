"""iGPU attn fused server client (Phase 2.4 stub binding)."""
from __future__ import annotations

import os, threading, time
from typing import Optional
import numpy as np, torch


class IgpuAttnClient:
    """Python wrapper around t_mtp_attn_server.exe (Phase 2.4 stub).

    Drives the server via IgpuService (C++) when available; falls back to subprocess.Popen.
    Protocol:
      ATTN_LOAD_QKV <bytes>\n + body
      ATTN_LOAD_O   <bytes>\n + body
      ATTN_FORWARD <pos>\n + qg(16,128) + kv_cache -> out_f32[2048]
    Phase 2.4 HLSL compiles (4 DXIL files); server currently stubs GPU dispatch.
    """
    def __init__(self, server_path=None):
        if server_path is None:
            cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "benchmarks", "cpu_moe_microbench", "t_mtp_attn_server.exe")
            server_path = os.path.abspath(cand)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU attn server not found: {server_path}")
        self.server_path = server_path
        self.server_cwd = os.path.dirname(server_path)
        self._qkv_loaded = False
        self._o_loaded = False
        self._lock = threading.Lock()
        self._cpp = None
        self._proc = None
        try:
            import freetoken.kernel._freetoken_igpu as _igpu
            self._cpp = _igpu.igpu.IgpuService(server_path, 0, 0, 0)
            time.sleep(2.0)
        except Exception:
            self._cpp = None
            import subprocess
            self._proc = subprocess.Popen(
                [server_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, cwd=self.server_cwd)
            threading.Thread(target=self._drain, daemon=True).start()
            time.sleep(2.0)

    def _drain(self):
        while True:
            try:
                l = self._proc.stderr.readline()
            except Exception:
                return
            if not l:
                return

    def load_qkv(self, weights: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor):
        body = torch.cat([weights.contiguous().view(torch.uint8),
                          q_norm.contiguous().view(torch.uint8),
                          k_norm.contiguous().view(torch.uint8)])
        cmd = f"ATTN_LOAD_QKV {body.numel()}"
        with self._lock:
            if self._cpp is not None:
                self._cpp.send_raw(cmd, body)
                self._cpp.recv_raw(3)
            else:
                self._proc.stdin.write((cmd + "\n").encode())
                self._proc.stdin.write(bytes(body.numel()))
                self._proc.stdin.flush()
                self._proc.stdout.read(3)
        self._qkv_loaded = True

    def load_o(self, weights: torch.Tensor):
        body = weights.contiguous().view(torch.uint8)
        cmd = f"ATTN_LOAD_O {body.numel()}"
        with self._lock:
            if self._cpp is not None:
                self._cpp.send_raw(cmd, body)
                self._cpp.recv_raw(3)
            else:
                self._proc.stdin.write((cmd + "\n").encode())
                self._proc.stdin.write(bytes(body.numel()))
                self._proc.stdin.flush()
                self._proc.stdout.read(3)
        self._o_loaded = True

    def forward(self, qg: torch.Tensor, kv_state: torch.Tensor, position: int) -> torch.Tensor:
        if not (self._qkv_loaded and self._o_loaded):
            raise RuntimeError("IgpuAttnClient.forward: call load_qkv() + load_o() first")
        body = torch.cat([qg.contiguous().view(torch.uint8),
                          kv_state.contiguous().view(torch.uint8)])
        cmd = f"ATTN_FORWARD {position}"
        with self._lock:
            if self._cpp is not None:
                self._cpp.send_raw(cmd, body)
                out = self._cpp.recv_raw(2048 * 4)
            else:
                self._proc.stdin.write((cmd + "\n").encode())
                self._proc.stdin.write(bytes(body.numel()))
                self._proc.stdin.flush()
                out_bytes = self._proc.stdout.read(2048 * 4)
                out = torch.frombuffer(bytearray(out_bytes), dtype=torch.uint8)
        return torch.frombuffer(bytearray(out), dtype=torch.float32).clone()

    def close(self):
        with self._lock:
            try:
                if self._cpp is not None:
                    self._cpp.send_raw("QUIT", torch.Tensor())
                    self._cpp.close()
                elif self._proc is not None:
                    self._proc.stdin.write(b"QUIT\n")
                    self._proc.stdin.flush()
                    self._proc.terminate()
            except Exception:
                pass


class IgpuAttnSticky:
    def __init__(self, server_path=None):
        self._c = IgpuAttnClient(server_path)
    def load_qkv(self, *args, **kwargs): self._c.load_qkv(*args, **kwargs)
    def load_o(self, *args, **kwargs): self._c.load_o(*args, **kwargs)
    def forward(self, *args, **kwargs): return self._c.forward(*args, **kwargs)
    def close(self): self._c.close()


__all__ = ["IgpuAttnClient", "IgpuAttnSticky"]
