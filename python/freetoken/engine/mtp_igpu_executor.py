"""MTP head iGPU executor: wraps v3 server with PROPER MXFP4 e8m0 scale bindings.

Architecture:
  - v3 server runs as a long-lived subprocess
  - At session start: LOAD mtp.fc.weight + mtp.fc.scales (one-time, pre-decoded e8m0 floats)
  - Per call: send only the activation + per-row bias, server does the GEMV
  - Returns: outv [1, 2048] float32 (the FC output for the 2048 hidden dimensions)

Differences from v2 executor:
  - Uses v3 server (correct MXFP4 semantics, not the a^2 * sum_w workaround)
  - At LOAD: also sends the pre-decoded e8m0 scales (fc.scales as float32)
  - Server bindings: t1=scales (float), t3=act (float), t5=bias (float)
  - Per CALL: only sends act bytes + bias bytes (no scales, no rowBias - those are sticky)
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


class MtpIgpuExecutor:
    """Persistent subprocess wrapper for the MTP head's FC layer on iGPU (v3 server)."""

    def __init__(
        self,
        fc_packed_u32: np.ndarray,   # shape (M, K//8) uint32
        fc_scales_f32: np.ndarray,    # shape (M, K//32) float32 (pre-decoded e8m0 scales)
        K: int = 4096,
        server_path: Optional[str] = None,
        name: str = "mtp_fc",
    ):
        if server_path is None:
            default = (
                Path(__file__).parent.parent.parent.parent
                / "benchmarks" / "cpu_moe_microbench" / "t_mxfp4_gemv_v3_server.exe"
            )
            server_path = str(default)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU v3 server not found: {server_path}")
        self.server_path = server_path
        self.server_cwd = os.path.dirname(server_path)
        self.K = K
        self.M = fc_packed_u32.shape[0]
        assert self.M == 1, f"MtpIgpuExecutor currently supports M=1, got M={self.M}"
        assert K % 32 == 0
        assert fc_packed_u32.dtype == np.uint32
        assert fc_packed_u32.shape == (1, K // 8)
        assert fc_scales_f32.dtype == np.float32
        assert fc_scales_f32.shape == (1, K // 32)
        self.packed = fc_packed_u32.copy()
        self.scales = fc_scales_f32.copy()
        self._lock = threading.Lock()
        self.stderr_lines = []
        self._open()
        self._load(name)

    def _open(self):
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=self.server_cwd,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        time.sleep(2.0)

    def _drain(self):
        while True:
            try:
                l = self.proc.stderr.readline()
            except Exception:
                return
            if not l:
                return
            text = l.decode(errors="replace")
            self.stderr_lines.append(text)

    def _load(self, name: str):
        """LOAD command: send packed bytes + scales bytes for the weight."""
        with self._lock:
            cmd = f"LOAD {name} {self.M} {self.K} {self.packed.size * 4} {self.scales.size * 4}\n".encode()
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(self.packed.tobytes())
            self.proc.stdin.write(self.scales.tobytes())
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline()
            if not reply.startswith(b"OK "):
                raise RuntimeError(f"MTP iGPU LOAD failed: {reply!r}")
            self.loaded_name = name

    def forward(self, act_flat_f32: np.ndarray, bias_flat_f32: Optional[np.ndarray] = None) -> np.ndarray:
        """Run MTP head FC on iGPU with proper MXFP4 semantics.

        act_flat_f32: shape (K,) float32 -- the concatenated (pre_fc_norm_embed || pre_fc_norm_hidden)
        bias_flat_f32: shape (M,) float32 -- per-row bias (default zeros)
        Returns: outv shape (M,) float32 -- the FC output
        """
        assert act_flat_f32.dtype == np.float32
        assert act_flat_f32.shape == (self.K,), f"act shape {act_flat_f32.shape} != ({self.K},)"
        if bias_flat_f32 is None:
            bias_flat_f32 = np.zeros(self.M, dtype=np.float32)
        assert bias_flat_f32.dtype == np.float32
        assert bias_flat_f32.shape == (self.M,), f"bias shape {bias_flat_f32.shape} != ({self.M},)"
        szA = self.K * 4
        szB = self.M * 4
        with self._lock:
            cmd = f"CALL {self.loaded_name} {szA} {szB}\n".encode()
            body = act_flat_f32.tobytes() + bias_flat_f32.tobytes()
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            rl = self._read_exact(4)
            if len(rl) < 4:
                raise RuntimeError(f"MTP iGPU CALL no len: got {len(rl)} bytes")
            sz = struct.unpack('<I', rl)[0]
            outv = self._read_exact(sz)
            if len(outv) < sz:
                raise RuntimeError(f"MTP iGPU CALL short: {len(outv)}/{sz}")
        return np.frombuffer(outv, dtype=np.float32).copy()

    def _read_exact(self, n):
        out = b''
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                return out
            out += chunk
        return out

    def close(self):
        try:
            if self.proc and self.proc.poll() is None:
                with self._lock:
                    self.proc.stdin.write(b"QUIT\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=2)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def get_log(self, last_n=20):
        return self.stderr_lines[-last_n:]


__all__ = ["MtpIgpuExecutor"]
