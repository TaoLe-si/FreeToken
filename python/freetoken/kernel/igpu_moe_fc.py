"""iGPU MoE fused server client (Phase 2.3 stub binding).

Persistent subprocess to t_mtp_moe_server.exe, talks via ASCII+binary stdin/stdout.
Protocol:
  - Input:  "MOE_LOAD <E> <I> <H>\n" + body (bf16 weights: gate+up+down+shared+router)
            "MOE_FORWARD\n" + hidden_f32[2048]
  - Output: "OK\n" (after MOE_LOAD) or out_f32[2048] (after MOE_FORWARD)

This is the Phase 2.3 integration wrapper. Phase 2.3 HLSL kernels compile cleanly
(8 DXIL files); the server currently stubs the GPU dispatch (returns zeros for
MOE_FORWARD). Numerical alignment + perf benchmarks are follow-up PRs.

Use IgpuMoeSticky for sticky weights (load once, forward many times).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np
import torch


class IgpuMoeClient:
    """Python wrapper around t_mtp_moe_server.exe (Phase 2.3 stub).

    Drives the server via IgpuService (C++) when available; falls back to
    subprocess.Popen if the .pyd extension isn't loaded.
    """

    def __init__(self, server_path: Optional[str] = None, E: int = 256, I: int = 512, H: int = 2048):
        if server_path is None:
            cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "benchmarks", "cpu_moe_microbench", "t_mtp_moe_server.exe")
            server_path = os.path.abspath(cand)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU MoE server not found: {server_path}")
        self.server_path = server_path
        self.server_cwd = os.path.dirname(server_path)
        self.E, self.I, self.H = E, I, H
        self._loaded = False
        self._lock = threading.Lock()

        # Try the C++ IgpuService first (lower latency, ~1ms IPC); fall back to subprocess.
        self._cpp = None
        self._proc = None
        self._stderr_lines = []
        try:
            import freetoken.kernel._freetoken_igpu as _igpu  # type: ignore
            self._cpp = _igpu.igpu.IgpuService(server_path, 0, 0, 0)
            time.sleep(2.0)  # wait for server ready ("mxfp4-v3 server ready" -- MoE server prints similar)
            self._wait_ready(timeout_s=10)
        except Exception:
            # Fallback to subprocess.Popen
            self._cpp = None
            import subprocess
            self._proc = subprocess.Popen(
                [server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                cwd=self.server_cwd,
            )
            threading.Thread(target=self._drain, daemon=True).start()
            time.sleep(2.0)
            self._wait_ready(timeout_s=10)

    def _wait_ready(self, timeout_s: float = 10):
        # Heuristic: stderr logs "ready" eventually. For now just sleep -- the
        # server doesn't have a strict handshake (we send the first command and
        # see what happens).
        time.sleep(1.0)

    def _drain(self):
        while True:
            try:
                l = self._proc.stderr.readline()
            except Exception:
                return
            if not l:
                return
            self._stderr_lines.append(l.decode("utf-8", errors="replace"))
            if len(self._stderr_lines) > 4096:
                self._stderr_lines = self._stderr_lines[-4096:]

    def load(self, weights: torch.Tensor):
        """Upload sticky MoE weights.

        weights layout (bf16):
          [expert_gate (E*I*H), expert_up (E*I*H), expert_down (E*H*I),
           shared_gate (I*H), shared_up (I*H), shared_down (H*I),
           shared_gw (H), router_w (E*H)]
        """
        if weights.dtype != torch.bfloat16:
            raise ValueError(f"IgpuMoeClient.load: weights must be bfloat16, got {weights.dtype}")
        body = weights.contiguous().view(torch.uint8)
        E, I, H = self.E, self.I, self.H
        expected = (2 * E * I * H + E * H * I + 3 * I * H + H * I + H + E * H) * 2
        if body.numel() != expected:
            raise ValueError(f"IgpuMoeClient.load: size mismatch {body.numel()} != {expected}")
        cmd = f"MOE_LOAD {E} {I} {H}"
        with self._lock:
            if self._cpp is not None:
                self._cpp.send_raw(cmd, body)
                ack = self._cpp.recv_raw(3)
            else:
                self._proc.stdin.write((cmd + "\n").encode())
                self._proc.stdin.write(bytes(body.numel()))
                self._proc.stdin.flush()
                ack = self._proc.stdout.read(3)
        self._loaded = True

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run a single MTP MoE forward. Returns out_f32[2048]."""
        if not self._loaded:
            raise RuntimeError("IgpuMoeClient.forward: call load() first")
        if hidden.numel() != self.H:
            raise ValueError(f"IgpuMoeClient.forward: hidden.numel()={hidden.numel()} != H={self.H}")
        body = hidden.contiguous().to(torch.float32).view(torch.uint8)
        with self._lock:
            if self._cpp is not None:
                self._cpp.send_raw("MOE_FORWARD", body)
                out = self._cpp.recv_raw(self.H * 4)
            else:
                self._proc.stdin.write(b"MOE_FORWARD\n")
                self._proc.stdin.write(bytes(body.numel()))
                self._proc.stdin.flush()
                out_bytes = self._proc.stdout.read(self.H * 4)
                out = torch.frombuffer(bytearray(out_bytes), dtype=torch.uint8)
        # out is bytes; view as float32 tensor.
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


class IgpuMoeSticky:
    """Alias for IgpuMoeClient -- kept for naming consistency with IgpuFcSticky."""
    def __init__(self, server_path=None, E=256, I=512, H=2048):
        self._c = IgpuMoeClient(server_path, E, I, H)
    def load(self, weights): self._c.load(weights)
    def forward(self, hidden): return self._c.forward(hidden)
    def close(self): self._c.close()


__all__ = ["IgpuMoeClient", "IgpuMoeSticky"]
