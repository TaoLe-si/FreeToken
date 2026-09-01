"""MTP head MoE executor using v3 server with BATCH_ALL.

Architecture:
  - One v3 server per MTP head instance (persistent subprocess)
  - On init: LOAD all 256 experts x 3 projections (768 weights, ~200MB upload)
  - Per call: 2 BATCH_ALL dispatches (16 GEMVs gate+up, 8 GEMVs down)

Key fixes vs v2/v3:
  - Server binds bias to slot 3 (per-row, M floats), act to slot 2 (per-K-element, K floats)
  - Body layout per item: act_bytes | bias_bytes (body order)
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


class MtpIgpuMoeExecutor:
    """Manages v3 server for MTP head's MoE block (gate/up/down per expert)."""

    def __init__(
        self,
        switch_gate_packed,
        switch_gate_scales,
        switch_up_packed,
        switch_up_scales,
        switch_down_packed,
        switch_down_scales,
        K_gate_up: int = 2048,
        K_down: int = 512,
        server_path: Optional[str] = None,
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
        self.K_gate_up = K_gate_up
        self.K_down = K_down
        self.I = 512
        self.H = 2048

        self.sw_gate_packed = np.asarray(switch_gate_packed, dtype=np.uint32)
        self.sw_gate_scales = np.asarray(switch_gate_scales, dtype=np.float32)
        self.sw_up_packed = np.asarray(switch_up_packed, dtype=np.uint32)
        self.sw_up_scales = np.asarray(switch_up_scales, dtype=np.float32)
        self.sw_down_packed = np.asarray(switch_down_packed, dtype=np.uint32)
        self.sw_down_scales = np.asarray(switch_down_scales, dtype=np.float32)
        self.num_experts = self.sw_gate_packed.shape[0]
        self._lock = threading.Lock()
        self.stderr_lines = []
        self._open()

        t0 = time.time()
        self._load_all()
        print(f"[MtpIgpuMoeExecutor] Loaded {self.num_experts * 3} weights in {(time.time()-t0)*1000:.0f}ms", flush=True)

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
            self.stderr_lines.append(l.decode(errors="replace"))

    def _load_weight(self, name, packed, scales, M, K):
        with self._lock:
            cmd = "LOAD {} {} {} {}\n".format(name, M, K, packed.size * 4, scales.size * 4).encode()
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(packed.tobytes())
            self.proc.stdin.write(scales.tobytes())
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline()
            if not reply.startswith(b"OK "):
                raise RuntimeError("LOAD {} failed: {}".format(name, reply))

    def _load_all(self):
        for e in range(self.num_experts):
            self._load_weight("e{}_g".format(e), self.sw_gate_packed[e:e+1], self.sw_gate_scales[e:e+1], self.I, self.K_gate_up)
            self._load_weight("e{}_u".format(e), self.sw_up_packed[e:e+1], self.sw_up_scales[e:e+1], self.I, self.K_gate_up)
            self._load_weight("e{}_d".format(e), self.sw_down_packed[e:e+1], self.sw_down_scales[e:e+1], self.H, self.K_down)

    def _read_exact(self, n):
        out = b''
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                return out
            out += chunk
        return out

    def _call(self, name, act_bytes, bias_bytes):
        """Single CALL dispatch."""
        with self._lock:
            szA = len(act_bytes)
            szB = len(bias_bytes)
            cmd = "CALL {} {} {}\n".format(name, szA, szB).encode()
            self.proc.stdin.write(cmd + act_bytes + bias_bytes)
            self.proc.stdin.flush()
            rl = self._read_exact(4)
            sz = struct.unpack('<I', rl)[0]
            out = self._read_exact(sz)
            return np.frombuffer(out, dtype=np.float32).copy()

    def forward(self, x, top_w, top_idx):
        """Run MoE on iGPU via 2 BATCH_ALL dispatches."""
        K_gu = self.K_gate_up
        I = self.I
        H = self.H

        # Stage 1+2: 16 GEMVs (8 gate + 8 up), bias=0 for now
        szA = K_gu * 4
        szB_gate = I * 4
        szB_up = I * 4
        cmd_parts = ["BATCH_ALL 16"]
        for i in range(8):
            e = int(top_idx[i])
            cmd_parts.append("e{}_g {} {}".format(e, szA, szB_gate))
            cmd_parts.append("e{}_u {} {}".format(e, szA, szB_up))
        cmd = (" ".join(cmd_parts) + "\n").encode()
        x_bytes = np.asarray(x, dtype=np.float32).tobytes()
        bias_bytes = np.zeros(I, dtype=np.float32).tobytes()
        body = b''
        for _ in range(8):
            body += x_bytes + bias_bytes + x_bytes + bias_bytes
        payload = cmd + body

        with self._lock:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            gate_outs = []
            up_outs = []
            for i in range(16):
                rl = self._read_exact(4)
                sz = struct.unpack('<I', rl)[0]
                out = self._read_exact(sz)
                arr = np.frombuffer(out, dtype=np.float32).copy()
                if i % 2 == 0:
                    gate_outs.append(arr)
                else:
                    up_outs.append(arr)

        intermediates = []
        for i in range(8):
            silu_gate = gate_outs[i] / (1.0 + np.exp(-gate_outs[i]))
            intermediates.append((silu_gate * up_outs[i]).astype(np.float32))

        # Stage 3: 8 down
        szA_down = I * 4
        szB_down = H * 4
        cmd_parts = ["BATCH_ALL 8"]
        for i in range(8):
            e = int(top_idx[i])
            cmd_parts.append("e{}_d {} {}".format(e, szA_down, szB_down))
        cmd = (" ".join(cmd_parts) + "\n").encode()
        body = b''
        for i in range(8):
            body += intermediates[i].tobytes() + np.zeros(H, dtype=np.float32).tobytes()
        payload = cmd + body

        with self._lock:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            down_outs = []
            for i in range(8):
                rl = self._read_exact(4)
                sz = struct.unpack('<I', rl)[0]
                out = self._read_exact(sz)
                down_outs.append(np.frombuffer(out, dtype=np.float32).copy())

        result = np.zeros(H, dtype=np.float32)
        for i in range(8):
            result = result + down_outs[i] * float(top_w[i])
        return result

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


__all__ = ["MtpIgpuMoeExecutor"]
