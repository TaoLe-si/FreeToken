"""iGPU MXFP4 GEMV server client (NVFP4 format).

Persistent subprocess to t_mxfp4_gemv_v3_server.exe, talks via ASCII+binary stdin/stdout.
Protocol:
  - Input  (host->server):  "STATELESS M K szP szS szA szB\\n"  (ASCII line)
                             packed (M*nb uints) | scales (M*ns float32) | act (K int32) | bias_pb (M*ns float32)
  - Output (server->host):  4 byte uint32 len + M float32 outv

The server is t_mxfp4_gemv_v3_server.exe running t_nvfp4_gemv_sk.dxil (NVFP4 format).
MXFP4 fc weights are actually NVFP4 fp16 scale + per-block bias.
"""
import os
import struct
import subprocess
import sys
import threading
import time
import numpy as np


class IgpuFcClient:
    def __init__(self, server_path=None, max_M=8, max_K=4096, max_ns=128):
        if server_path is None:
            cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "benchmarks", "cpu_moe_microbench", "t_mxfp4_gemv_v3_server.exe")
            server_path = os.path.abspath(cand)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU server not found: {server_path}")
        self.server_path = server_path
        # The server reads t_mxfp4_gemv_sk.dxil from cwd.
        self.server_cwd = os.path.dirname(server_path)
        self.proc = None
        self.stderr_lines = []
        self.max_M = max_M
        self.max_K = max_K
        self.max_ns = max_ns
        self._lock = threading.Lock()
        self._open()

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
        import sys
        while True:
            try:
                l = self.proc.stderr.readline()
            except Exception:
                return
            if not l:
                return
            text = l.decode(errors="replace")
            self.stderr_lines.append(text)
            sys.stderr.write("[SERVER] " + text)
            sys.stderr.flush()

    def close(self):
        """Graceful shutdown: send QUIT, then kill if needed."""
        try:
            if getattr(self, 'proc', None) is not None and self.proc.poll() is None:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def forward(self, packed_u32, act_int32, scales_f32=None, biases_f32=None):
        """Run one GEMV on the iGPU. Returns outv float32 [M].

        Args:
            packed_u32: (M, K//8) uint32 e2m1 nibble-packed weights
            act_int32:  (K,) int32 activations (server converts to float)
            scales_f32: (M, K//32) float32 NVFP4 fp16 scale (per-block)
            biases_f32: (M, K//32) float32 NVFP4 per-block bias
        """
        assert packed_u32.dtype == np.uint32
        assert act_int32.dtype == np.int32
        M = packed_u32.shape[0]
        K = act_int32.shape[0]
        assert K % 32 == 0
        assert packed_u32.shape[1] == K // 8
        nb = K // 8
        ns = K // 32
        if scales_f32 is None:
            scales_f32 = np.zeros((M, ns), dtype=np.float32)
        if biases_f32 is None:
            biases_f32 = np.zeros((M, ns), dtype=np.float32)
        assert scales_f32.shape == (M, ns)
        assert biases_f32.shape == (M, ns)
        sz_p = packed_u32.size * 4      # M*nb*4
        sz_s = scales_f32.size * 4      # M*ns*4 (NVFP4 scale)
        sz_a = act_int32.size * 4       # K*4 (int32)
        sz_b = biases_f32.size * 4      # M*ns*4 (per-block bias)
        cmd = ("STATELESS %d %d %d %d %d %d\n" % (M, K, sz_p, sz_s, sz_a, sz_b)).encode()
        body = (packed_u32.tobytes() + scales_f32.tobytes() +
                act_int32.tobytes() + biases_f32.tobytes())
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError(f'server dead, rc={self.proc.returncode}')
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            rl = self._read_exact(4)
            if len(rl) < 4:
                raise RuntimeError(f"server returned no len header: got {len(rl)} bytes, last stderr: {self.stderr_lines[-3:] if self.stderr_lines else []}")
            sz = struct.unpack('<I', rl)[0]
            outv = self._read_exact(sz)
            if len(outv) < sz:
                raise RuntimeError(f"server returned short: got {len(outv)}/{sz} bytes")
        return np.frombuffer(outv, dtype=np.float32)[:M]

    def _read_exact(self, n):
        out = b''
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                return out
            out += chunk
        return out

class IgpuFcSticky:
    """Sticky-weight FC via v3 server FC_LOAD/FC_CALL protocol.

    Weights (packed + scales + biases) are uploaded ONCE at construction; each
    __call__ transfers only the shared activation vector (K*4 bytes) and gets
    back M float32 outputs.

    Supports full (M, K//8) weight matrices (e.g. MTP fc: 2048 x 4096) — the
    GPU kernel is the fcbcast variant: one shared act, M output rows.

    .torch() returns a torch-callable bridge (cuda/cpu in, same-device out)
    matching the MTP head's igpu_fc contract.
    """

    def __init__(self, packed_u32, K, scales_f32=None, biases_f32=None,
                 server_path=None, client=None):
        if server_path is None:
            cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "benchmarks", "cpu_moe_microbench", "t_mxfp4_gemv_v3_server.exe")
            server_path = os.path.abspath(cand)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU v3 server not found: {server_path}")

        assert packed_u32.dtype == np.uint32
        assert K % 32 == 0
        M = packed_u32.shape[0]
        assert packed_u32.shape == (M, K // 8), f"got {packed_u32.shape}, expected ({M}, {K//8})"
        ns = K // 32
        if scales_f32 is None:
            scales_f32 = np.zeros((M, ns), dtype=np.float32)
        if biases_f32 is None:
            biases_f32 = np.zeros((M, ns), dtype=np.float32)
        scales_f32 = np.ascontiguousarray(scales_f32, dtype=np.float32)
        biases_f32 = np.ascontiguousarray(biases_f32, dtype=np.float32)
        assert scales_f32.shape == (M, ns), f"scales_f32 shape {scales_f32.shape} != ({M}, {ns})"
        assert biases_f32.shape == (M, ns), f"biases_f32 shape {biases_f32.shape} != ({M}, {ns})"

        self.M = M
        self.K = K
        self.packed_u32 = np.ascontiguousarray(packed_u32)
        self.scales_f32 = scales_f32
        self.biases_f32 = biases_f32
        self.server_path = server_path
        self.server_cwd = os.path.dirname(server_path)
        self.stderr_lines = []
        self._lock = threading.Lock()
        self._open()

        # Upload weights once (FC_LOAD)
        szP, szS, szB = M * (K // 8) * 4, M * ns * 4, M * ns * 4
        cmd = f"FC_LOAD {M} {K} {szP} {szS} {szB}\n".encode()
        body = (self.packed_u32.tobytes() + self.scales_f32.tobytes()
                + self.biases_f32.tobytes())
        with self._lock:
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            ack = self._read_exact(3)
            if ack != b"OK\n":
                raise RuntimeError(f"FC_LOAD failed: {ack!r} log={self.get_log(10)}")

    # ---- process management ----
    def _open(self):
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, cwd=self.server_cwd,
        )
        threading.Thread(target=self._drain, daemon=True).start()
        t0 = time.time()
        while time.time() - t0 < 15.0:
            log = " ".join(self.stderr_lines[-8:])
            if "psoFc ok" in log or "server ready" in log:
                return
            time.sleep(0.05)

    def _drain(self):
        while True:
            try:
                l = self.proc.stderr.readline()
            except Exception:
                return
            if not l:
                return
            self.stderr_lines.append(l.decode(errors="replace").rstrip())

    def _read_exact(self, n):
        out = b''
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk:
                raise RuntimeError(
                    f"iGPU server died (read {len(out)}/{n}). "
                    f"Log tail: {self.get_log(10)}")
            out += chunk
        return out

    def close(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                try:
                    self.proc.wait(timeout=2)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # ---- core call ----
    def __call__(self, act_flat):
        """act_flat: (K,) float32 numpy -> (M,) float32 numpy."""
        assert act_flat.dtype == np.float32
        assert act_flat.shape == (self.K,), f"act shape {act_flat.shape} != ({self.K},)"
        act_c = np.ascontiguousarray(act_flat, dtype=np.float32)
        with self._lock:
            self.proc.stdin.write(f"FC_CALL {self.K * 4}\n".encode())
            self.proc.stdin.write(act_c.tobytes())
            self.proc.stdin.flush()
            rl = self._read_exact(4)
            sz = struct.unpack('<I', rl)[0]
            outv = self._read_exact(sz)
        return np.frombuffer(outv, dtype=np.float32).copy()

    # ---- torch bridge ----
    def torch(self):
        """Return a torch-callable wrapper (cuda/cpu in, same-device out)."""
        import torch
        sticky = self

        class _TorchFc:
            def __call__(self_, x):
                if x.dim() == 2 and x.shape[0] == 1:
                    x = x.squeeze(0)
                assert x.dim() == 1 and x.shape[0] == sticky.K
                dev = x.device
                x32 = x.detach().to(torch.float32).cpu().numpy()
                out = sticky(x32)
                t = torch.from_numpy(out)
                return t.to(dev).to(x.dtype).unsqueeze(0)  # (1, M)

            def close(self_):
                sticky.close()

        return _TorchFc()

    # ---- legacy compat ----
    def update_weight(self, packed_u32, scales_f32=None, biases_f32=None):
        """Replace weights in-place (re-runs FC_LOAD)."""
        assert packed_u32.dtype == np.uint32
        assert packed_u32.shape == self.packed_u32.shape
        self.packed_u32 = np.ascontiguousarray(packed_u32)
        if scales_f32 is not None: self.scales_f32 = np.ascontiguousarray(scales_f32, dtype=np.float32)
        if biases_f32 is not None: self.biases_f32 = np.ascontiguousarray(biases_f32, dtype=np.float32)
        M, K, ns = self.M, self.K, self.K // 32
        szP, szS, szB = M * (K // 8) * 4, M * ns * 4, M * ns * 4
        cmd = f"FC_LOAD {M} {K} {szP} {szS} {szB}\n".encode()
        body = (self.packed_u32.tobytes() + self.scales_f32.tobytes()
                + self.biases_f32.tobytes())
        with self._lock:
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            ack = self._read_exact(3)
            if ack != b"OK\n":
                raise RuntimeError(f"FC_LOAD (update) failed: {ack!r}")

    def get_log(self, last_n=20):
        return self.stderr_lines[-last_n:]



class IgpuMultiClient:
    """Multi-weight iGPU server client. Pre-loads named weights, then per-call only sends activations."""
    def __init__(self, server_path=None):
        if server_path is None:
            cand = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "benchmarks", "cpu_moe_microbench", "t_mxfp4_gemv_multi_server.exe")
            server_path = os.path.abspath(cand)
        if not os.path.exists(server_path):
            raise FileNotFoundError(f"iGPU multi server not found: {server_path}")
        self.server_path = server_path
        self.server_cwd = os.path.dirname(server_path)
        self.proc = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, cwd=self.server_cwd,
        )
        self.stderr_lines = []
        self._lock = threading.Lock()
        threading.Thread(target=self._drain, daemon=True).start()
        time.sleep(2.0)
        self.loaded = {}

    def _drain(self):
        while True:
            try:
                l = self.proc.stderr.readline()
            except Exception:
                return
            if not l:
                return
            self.stderr_lines.append(l.decode(errors="replace"))

    def __del__(self):
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2)
        except Exception:
            pass

    def load(self, name, packed_u32, K):
        assert packed_u32.dtype == np.uint32
        assert K % 32 == 0
        M = packed_u32.shape[0]
        packed_bytes = packed_u32.tobytes()
        with self._lock:
            cmd = ("LOAD " + name + " " + str(M) + " " + str(K) + "\n").encode()
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(packed_bytes)
            self.proc.stdin.flush()
            reply = self.proc.stdout.readline()
            if not reply.startswith(b"OK "):
                raise RuntimeError(f"load {name} failed: {reply!r}")
            self.loaded[name] = (M, K)

    def call(self, name, act_int32, scales_f32=None, biases_f32=None):
        assert name in self.loaded, f"weight {name!r} not loaded"
        assert act_int32.dtype == np.int32
        M, K = self.loaded[name]
        ns = K // 32
        if scales_f32 is None: scales_f32 = np.zeros((M, ns), dtype=np.float32)
        if biases_f32 is None: biases_f32 = np.zeros((M, ns), dtype=np.float32)
        cmd = ("CALL " + name + "\n").encode()
        body = act_int32.tobytes() + scales_f32.tobytes() + biases_f32.tobytes()
        with self._lock:
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            rl = self._read_exact(4)
            if len(rl) < 4:
                raise RuntimeError(f"call {name} no len: {len(rl)} bytes")
            sz = struct.unpack('<I', rl)[0]
            outv = self._read_exact(sz)
            if len(outv) < sz:
                raise RuntimeError(f"call {name} short: {len(outv)}/{sz}")
        return np.frombuffer(outv, dtype=np.float32)[:M]

    def _read_exact(self, n):
        out = b''
        while len(out) < n:
            chunk = self.proc.stdout.read(n - len(out))
            if not chunk: return out
            out += chunk
        return out

    def call_batch(self, acts: dict):
        """BATCH_ALL: dispatch all pre-loaded weights with their per-weight acts.

        acts: dict mapping name -> np.int32 array of shape (K,).
        Returns dict mapping name -> np.float32 array of shape (M,).
        """
        # Find max K for shared buffer
        max_K = max(a.shape[0] for a in acts.values())
        max_act_bytes = max_K * 4
        # Each weight gets its own act payload (M*ns*4 bytes zeros for scales/biases)
        # We need per-weight act + per-weight scales/biases. Send per-weight.
        # Server: BATCH_ALL <act_K> <szA> <szS> <szB> then shared buf
        # Simplest: just call each weight with the per-weight act, server iterates
        # Actually for MTP head, the only thing that varies is K (4096 for fc/o, 2048 for q/k/v)
        # We can pad all acts to max_K (server reads szA bytes; we'd waste some, but the
        # K=2048 server will only use the first 2048 floats and ignore the rest... actually
        # the GPU will read the full M*ns*4 bytes for scales/biases, so it must be correct size).
        # Easiest: one BATCH_ALL per K-group.
        # Skip for now; users should use call() per weight.
        raise NotImplementedError("use call() per weight; BATCH_ALL coming soon")

    def call_all(self, acts):
        """ALL: dispatch all loaded weights with per-weight acts in one server frame.

        acts: ordered list of np.int32 arrays (one per loaded weight, in LOAD order).
        Returns: dict mapping name -> np.float32 output, in same order.
        """
        # Build the command line: ALL <N> <size1> <size2> ... <sizeN>\n
        # Server uses std::map which iterates sorted by key.
        # `acts` is in load order (self.loaded insertion order).
        # Build name -> act from insertion order.
        loaded_names = list(self.loaded.keys())
        name_to_act = {loaded_names[i]: acts[i] for i in range(len(acts))}
        sorted_names = sorted(name_to_act.keys())
        acts = [name_to_act[n] for n in sorted_names]
        cmd_parts = [f"ALL {len(acts)}"]
        total = 0
        for a in acts:
            assert a.dtype == np.int32
            sz = a.size * 4
            cmd_parts.append(str(sz))
            total += sz
        cmd = (' '.join(cmd_parts) + "\n").encode()
        body = b''.join(a.tobytes() for a in acts)
        assert len(body) == total
        with self._lock:
            self.proc.stdin.write(cmd)
            self.proc.stdin.write(body)
            self.proc.stdin.flush()
            results = {}
            for _ in acts:
                rl = self._read_exact(4)
                if len(rl) < 4:
                    raise RuntimeError("ALL: short len")
                sz = struct.unpack('<I', rl)[0]
                outv = self._read_exact(sz)
                if len(outv) < sz:
                    raise RuntimeError("ALL: short data")
                # Convert and store by order
                results[list(self.loaded.keys())[len(results)]] = np.frombuffer(outv, dtype=np.float32)
            return results

    def get_log(self, last_n=20):
        return self.stderr_lines[-last_n:]
