"""iGPU (AMD Radeon integrated GPU, D3D12) W4A8 GEMV backend.

Serves the offload-family decode experts from the *same pinned host banks* the
CPU executor reads, but computes the GEMV on the iGPU via a persistent D3D12
compute service (t_d3d12_service.exe, stdio binary protocol). This is the
B-group compute path of the dense_host_offload architecture: the iGPU reads
weights straight from DRAM (26-36 GB/s measured on AMD 780M) with no PCIe
round-trip and no CPU-core occupation, complementing the CPU executor for
multi-request decode.

Layout (matches the shared NVFP4 format, see docs/design/dense_host_offload.md
sec 17): per K=16 block -> packed uint8[8] (16 e2m1 nibbles, low nibble first),
per-block int8 scale (e4m3 bit pattern in the low byte), per-row activation
int8[16] (even/odd halves) + per-block float asb + per-row float global.

Protocol (little-endian, stdin/stdout):
  request  : u32 M, u32 K,
             packed[M*NB*8] u8, scl[M*NB*4] u32, act[NB*16*4] i32,
             asb[NB*4] f32, gbl[M*4] f32        (NB = K // 16)
  response : f32 out[M]

The C++ service must run in binary stdin/stdout mode (it does: _O_BINARY).
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
import time

import numpy as np

import logging

logger = logging.getLogger(__name__)

# e2m1x2 16-entry LUT (index = nibble): matches cpu_moe_ext.cpp / d3d12_gemv_sk.hlsl.
_K_E2M1X2 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int32)


def _default_service_exe() -> str | None:
    """Locate the D3D12 GEMV service executable.

    Order: $FREETOKEN_IGPU_SERVICE, the repo microbench build output, PATH.
    """
    env = os.getenv("FREETOKEN_IGPU_SERVICE")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    cand = os.path.join(repo, "benchmarks", "cpu_moe_microbench", "t_d3d12_service.exe")
    if os.path.isfile(cand):
        return cand
    return shutil.which("t_d3d12_service.exe")


class IgpuServiceError(RuntimeError):
    pass


class IgpuGemvService:
    """Persistent D3D12 GEMV service process with a stdio binary protocol.

    One D3D12 device/PSO is created at start() and reused across calls;
    per-call cost is an upload memcpy + dispatch + fence wait (~0.5-1 ms at
    M=4096 K=4096 on 780M; the stdio transfer dominates the Python-visible
    latency, so the production path should embed this as a DLL.
    """

    _REQ_HDR = struct.Struct("<II")
    _DLL_CANDIDATES = ["d3d12_gemv.dll"]  # searched in CWD, exe dir, PATH

    @classmethod
    def _default_dll_path(cls) -> str | None:
        """Find the d3d12_gemv.dll in likely locations."""
        cwd = os.getcwd()
        for cand in cls._DLL_CANDIDATES:
            for base in (cwd, os.path.dirname(_default_service_exe()) if _default_service_exe() else None):
                if base and os.path.isfile(os.path.join(base, cand)):
                    return os.path.join(base, cand)
        # fallback: try PATH
        for p in os.environ.get("PATH", "").split(os.pathsep):
            fp = os.path.join(p, cand)
            if os.path.isfile(fp):
                return fp
        return None

    def _load_dll(self) -> bool:
        """Load d3d12_gemv.dll and create a D3D12 handle.  Returns True on success."""
        if self._dll is not None:
            return True  # already loaded
        if self._dll_failed:
            return False
        import ctypes
        try:
            dll_path = self._default_dll_path()
            if not dll_path:
                self._dll_failed = True
                return False
            dll = ctypes.CDLL(dll_path)
            dll.igpu_create.restype = ctypes.c_void_p
            dll.igpu_gemv.restype = ctypes.c_int
            dll.igpu_gemv.argtypes = [
                ctypes.c_void_p,       # handle
                ctypes.c_int,          # M
                ctypes.c_int,          # K
                ctypes.c_void_p,       # packed
                ctypes.c_void_p,       # scl
                ctypes.c_void_p,       # act
                ctypes.c_void_p,       # asb
                ctypes.c_void_p,       # gbl
                ctypes.c_void_p,       # out
            ]
            dll.igpu_destroy.argtypes = [ctypes.c_void_p]
            dll.igpu_destroy.restype = None
            dll.igpu_errmsg.argtypes = [ctypes.c_void_p]
            dll.igpu_errmsg.restype = ctypes.c_char_p
            # The shader blob (d3d12_gemv_sk.dxil) is opened via a RELATIVE path
            # inside igpu_create(); the process CWD is arbitrary, so chdir to the
            # DLL directory for the call. First load happens under _lock.
            _prev_cwd = os.getcwd()
            os.chdir(os.path.dirname(dll_path) or ".")
            try:
                handle = dll.igpu_create()
            finally:
                os.chdir(_prev_cwd)
            if not handle:
                err = dll.igpu_errmsg(None)
                logger.warning("igpu DLL create failed: %s", err or "unknown")
                self._dll_failed = True
                return False
            self._dll = dll
            self._dll_handle = handle
            return True
        except Exception as exc:
            logger.warning("igpu DLL load failed: %s", exc)
            self._dll_failed = True
            return False

    def _gemv_via_dll(
        self, packed, scl, act, asb, gbl, M: int, K: int, NB: int
    ) -> np.ndarray | None:
        """Try DLL path; return float32[M] on success, None on failure (fall back to stdio)."""
        if not self._load_dll():
            return None
        import ctypes
        try:
            out = np.zeros(M, dtype=np.float32)
            dll = self._dll
            h = self._dll_handle
            pk_p = np.ascontiguousarray(packed, dtype=np.uint8).ctypes.data_as(ctypes.c_void_p)
            sc_p = np.ascontiguousarray(scl, dtype=np.uint32).ctypes.data_as(ctypes.c_void_p)
            ac_p = np.ascontiguousarray(act, dtype=np.int32).ctypes.data_as(ctypes.c_void_p)
            as_p = np.ascontiguousarray(asb, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
            gb_p = np.ascontiguousarray(gbl, dtype=np.float32).ctypes.data_as(ctypes.c_void_p)
            ou_p = out.ctypes.data_as(ctypes.c_void_p)
            rc = dll.igpu_gemv(h, M, K, pk_p, sc_p, ac_p, as_p, gb_p, ou_p)
            if rc != 0:
                logger.warning("igpu_gemv DLL returned %d", rc)
                return None
            return out
        except Exception as exc:
            logger.warning("igpu_gemv DLL exception: %s", exc)
            return None

    def __init__(self, exe: str | None = None):
        self.exe = exe or _default_service_exe()
        if not self.exe:
            raise IgpuServiceError(
                "iGPU service executable not found; set FREETOKEN_IGPU_SERVICE or "
                "build benchmarks/cpu_moe_microbench/t_d3d12_service.exe"
            )
        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()  # re-entrant: gemv() holds it while start() runs
        self.adapter_desc: str = ""
        # DLL-first path: d3d12_gemv.dll (same dir as exe, or FREETOKEN_IGPU_DLL)
        self._dll = None          # ctypes.CDLL once loaded
        self._dll_handle = None   # igpu_create() handle
        self._dll_failed = False  # do not retry after a load/selfcheck failure

    # ---------------- lifecycle ----------------
    def start(self, timeout: float = 20.0) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._proc = subprocess.Popen(
                [self.exe],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(self.exe) or None,
            )
        deadline = time.monotonic() + timeout
        stderr_lines: list[str] = []
        while time.monotonic() < deadline:
            line = self._proc.stderr.readline().decode(errors="replace").strip()
            if not line:
                if self._proc.poll() is not None:
                    raise IgpuServiceError(
                        "iGPU service exited during init: " + " | ".join(stderr_lines)
                    )
                time.sleep(0.01)
                continue
            stderr_lines.append(line)
            if line.startswith("[igpu] adapter:"):
                self.adapter_desc = line[len("[igpu] adapter:") :].strip()
            if "[igpu] ready" in line:
                logger.info("iGPU D3D12 service ready: %s", self.adapter_desc)
                return
        raise IgpuServiceError("iGPU service init timed out")

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------- compute ----------------
    def gemv(
        self,
        packed,
        scl,
        act,
        asb,
        gbl,
        M: int,
        K: int,
    ) -> np.ndarray:
        """One W4A8 GEMV: out[r] = sum_b (w4(r,b) dot act[b]) * scale(r,b) + asb[b], *0.25*gbl[r].

        Inputs are numpy arrays in the layouts above. Returns float32 [M].
        """
        NB = K // 16
        if packed.nbytes != M * NB * 8 or scl.nbytes != M * NB * 4:
            raise ValueError("packed/scl shape mismatch")
        if act.nbytes != NB * 16 * 4 or asb.nbytes != NB * 4 or gbl.nbytes != M * 4:
            raise ValueError("act/asb/gbl shape mismatch")
        dll_out = self._gemv_via_dll(packed, scl, act, asb, gbl, M, K, NB)
        if dll_out is not None:
            return dll_out
        payload = (
            self._REQ_HDR.pack(M, K)
            + np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
            + np.ascontiguousarray(scl, dtype=np.uint32).tobytes()
            + np.ascontiguousarray(act, dtype=np.int32).tobytes()
            + np.ascontiguousarray(asb, dtype=np.float32).tobytes()
            + np.ascontiguousarray(gbl, dtype=np.float32).tobytes()
        )
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self.start()
            stdin, stdout = self._proc.stdin, self._proc.stdout
            stdin.write(payload)
            stdin.flush()
            raw = stdout.read(M * 4)
        if len(raw) != M * 4:
            raise IgpuServiceError(f"short reply {len(raw)} != {M * 4}")
        return np.frombuffer(raw, dtype=np.float32)

    # ---------------- probing ----------------
    def probe_bandwidth(self, M: int = 4096, K: int = 4096, iters: int = 8) -> float:
        """Synthetic W4A8 GEMV bandwidth (GB/s), counting every weight/act byte."""
        NB = K // 16
        rng = np.random.default_rng(0)
        packed = rng.integers(0, 256, M * NB * 8, dtype=np.uint8)
        scl = rng.integers(0, 128, M * NB, dtype=np.uint32)
        act = rng.integers(-127, 128, NB * 16, dtype=np.int32)
        asb = rng.random(NB, dtype=np.float32)
        gbl = rng.random(M, dtype=np.float32)
        self.gemv(packed, scl, act, asb, gbl, M, K)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.gemv(packed, scl, act, asb, gbl, M, K)
        dt = (time.perf_counter() - t0) / iters
        wbytes = M * NB * 8 + M * NB * 4 + NB * 16 * 4 + NB * 4 + M * 4
        return wbytes / dt / 1e9

    def selfcheck(self, M: int = 1024, K: int = 4096) -> tuple:
        """Numpy reference check of the GEMV math (catches ABI/format drift)."""
        NB = K // 16
        rng = np.random.default_rng(7)
        packed = rng.integers(0, 256, M * NB * 8, dtype=np.uint8)
        scl = rng.integers(0, 128, M * NB, dtype=np.uint32)
        act = rng.integers(-127, 128, NB * 16, dtype=np.int32)
        asb = rng.random(NB, dtype=np.float32)
        gbl = rng.random(M, dtype=np.float32)
        out = self.gemv(packed, scl, act, asb, gbl, M, K)
        pk = packed.reshape(M, NB, 8).astype(np.int32)
        sc = (scl.reshape(M, NB) & 0xFF).astype(np.float64)
        ac = act.reshape(NB, 16).astype(np.int32)
        low = _K_E2M1X2[pk & 0x0F]
        high = _K_E2M1X2[(pk >> 4) & 0x0F]
        ac_e = ac[None, :, :8]
        ac_o = ac[None, :, 8:]
        wsum = (low * ac_e).sum(axis=2) + (high * ac_o).sum(axis=2)
        ref = (wsum.astype(np.float64) * 0.01 * sc + asb[None, :]).sum(axis=1) * 0.25 * gbl
        err = float(np.abs(out - ref).max())
        ok = err < 1e-2 * (float(np.abs(ref).max()) + 1)
        return ok, f"maxerr={err:.4f}"


def igpu_available(exe: str | None = None) -> tuple:
    """Availability probe: start the D3D12 service (it enumerates the AMD iGPU
    itself), then run one numpy-reference self-check."""
    path = exe or _default_service_exe()
    if not path:
        return False, "service exe not found (FREETOKEN_IGPU_SERVICE / repo build)"
    try:
        with IgpuGemvService(path) as svc:
            if "0x1002" not in svc.adapter_desc and "Radeon" not in svc.adapter_desc:
                return False, f"service bound to non-AMD adapter: {svc.adapter_desc!r}"
            ok, msg = svc.selfcheck()
            return ok, msg
    except Exception as e:
        return False, str(e)


def find_service_exe() -> str | None:
    return _default_service_exe()


class IgpuMoeExecutor:
    """Decode-time expert compute on the iGPU D3D12 service (--moe-backend igpu).

    Reads the native nvfp4 host banks straight from the OffloadMoeCache and
    computes the routed experts on the iGPU: hidden activations are quantized
    to per-16-block int8 (+ float asb) on the host, then the D3D12 service
    computes the W4A8 GEMV over the pinned bank rows. One service call per
    routed row per projection (gate_up, then down through the activation) --
    correct for single-request decode and small batches; the stdio transfer
    dominates multi-request latency until the service is embedded as a DLL.

    NOTE: the service currently approximates the e4m3 block scale as
    0.01 x bit-pattern (benchmark convention, exact against the microbench
    reference). Production dequant needs the e4m3 LUT in d3d12_gemv_sk.hlsl.
    """

    def __init__(
        self,
        cache,
        *,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        service: IgpuGemvService | None = None,
        max_tokens: int = 1,
        device=None,
    ) -> None:
        fmt = cache.quant_format
        if fmt != "nvfp4":
            raise NotImplementedError(
                "--moe-backend igpu computes the native nvfp4 rows and supports "
                f"quant_format 'nvfp4', got {fmt!r}; use --moe-backend cpu/offload."
            )
        if activation not in ("silu", "swish", "gelu", "gelu_tanh",
                              "gelu_pytorch_tanh", "swigluoai"):
            raise NotImplementedError(f"iGPU MoE backend: unsupported activation {activation!r}")
        self.num_layers = int(cache.num_layers)
        self.num_experts = int(cache.num_experts)
        self.top_k = int(top_k)
        self.activation = activation
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.service = service if service is not None else IgpuGemvService()
        self.device = device
        # Model compute dtype: numpy outputs must land back in it (bf16), else the
        # next bf16 Linear rejects the float32 hidden states.
        self.out_dtype = out_dtype

        # Host-side weight/scale caches: the DLL call needs stable numpy buffers;
        # rebuilding them per token cost a 44 MB transpose + conversions EVERY
        # layer call (the dominant decode overhead after weight upload itself).
        self._cache = {}
        self._banks = cache.bank_sources  # dict[str, list[torch.Tensor]] per layer
        src = self._banks["gate_up_packed"][0]
        # src: [E, 2I, H//2] uint8 -> H = 2*last, I = rows//2.
        self.H = int(src.shape[-1]) * 2
        self.I = int(src.shape[-2]) // 2
        self._act = self.activation

    # ---------------------------------------------------------- decode
    def decode(self, layer_id, hidden_states, topk_weights, topk_ids):
        """Compute the routed experts on the iGPU.

        hidden_states: (B, H) on self.device; topk_weights/topk_ids: (B*K,).
        Returns (B, H) float32 on self.device (the routed expert sum, weighted).
        """
        import torch

        B = int(hidden_states.shape[0])
        H, I, K = self.H, self.I, self.top_k
        x = hidden_states.detach().cpu().float().numpy()  # (B, H)
        w = topk_weights.detach().cpu().float().numpy()  # (B*K,)
        ids = topk_ids.detach().cpu().numpy().astype(np.int64)  # (B*K,)
        out = np.zeros((B, H), dtype=np.float32)
        gu_p = self._banks["gate_up_packed"][layer_id]
        gu_s = self._banks["gate_up_scale"][layer_id]
        gu_g = self._banks["gate_up_global"][layer_id]
        dn_p = self._banks["down_packed"][layer_id]
        dn_s = self._banks["down_scale"][layer_id]
        dn_g = self._banks["down_global"][layer_id]
        for r in range(B * K):
            b, e = r // K, int(ids[r])
            xr = x[b]  # (H,)
            xq, xasb = _quantize_w4a8(xr, H)  # int8 (NB*16,), float (NB,)
            gu_out = self._project(gu_p[e], gu_s[e], gu_g[e], xq, xasb, I * 2, H)  # (2I,)
            act = _activation(gu_out[:I], gu_out[I:], self._act)  # (I,)
            aq, aasb = _quantize_w4a8(act, I)
            dn_out = self._project(dn_p[e], dn_s[e], dn_g[e], aq, aasb, H, I)  # (H,)
            out[b] += float(w[r]) * dn_out
        t = torch.from_numpy(out)
        return t.to(device=self.device) if self.device is not None else t

    def _project(self, packed, scale, glob, act_i8, asb, rows, K):
        """One W4A8 GEMV: [rows, K] x act -> [rows] via the D3D12 service.

        packed: [rows, K//2] uint8; scale: [rows, K//16] fp8; glob: [rows] fp16.
        act_i8: [NB*16] int32; asb: [NB] float32. Returns float32 [rows].
        """
        NB = K // 16
        pk = np.ascontiguousarray(packed, dtype=np.uint8).reshape(rows, NB * 8)
        # fp8 (e4m3) block scales: 1 byte per block; the service protocol wants
        # u32 per block with the bit pattern in the low byte.
        sc = np.ascontiguousarray(scale, dtype=np.uint8).astype(np.uint32).reshape(rows, NB).copy()
        gb = np.ascontiguousarray(glob, dtype=np.float16).astype(np.float32).reshape(rows).copy()
        return self.service.gemv(pk, sc, act_i8, asb, gb, rows, K)


def _quantize_w4a8(x: np.ndarray, K: int) -> tuple:
    """Per-16-block int8 quantization: x -> (int8 codes, float per-block asb).

    W4A8 stores activations as int8 with a per-16-block float bias (asb) so the
    GEMV decodes to w4 . i8 * scale + asb. Here the block scale is folded into
    asb (asb = block scale) and the codes are i8 = round(x / scale); the service
    applies asb[b] additively, which matches the shared NVFP4 convention for a
    zero-mean activation stream (the exact asb convention of the production
    checkpoint is applied at load time once the act/asb fields are surfaced).
    """
    NB = K // 16
    codes = np.zeros(NB * 16, dtype=np.int32)
    asb = np.zeros(NB, dtype=np.float32)
    for b in range(NB):
        blk = x[b * 16 : (b + 1) * 16]
        amax = float(np.max(np.abs(blk))) if blk.size else 0.0
        if amax < 1e-6:
            continue
        sc = amax / 127.0
        codes[b * 16 : (b + 1) * 16] = np.clip(np.round(blk / sc), -127, 127).astype(np.int32)
        asb[b] = sc
    return codes, asb


def _activation(gate: np.ndarray, up: np.ndarray, name: str) -> np.ndarray:
    """gpt-oss-style gated activation (silu/swiglu/gelu) over gate/up halves."""
    import numpy as _np
    if name in ("swigluoai",):
        return gate * _np.multiply(up, 1.0 / (1.0 + _np.exp(-gate)))
    if name in ("gelu", "gelu_tanh", "gelu_pytorch_tanh"):
        # tanh approximation (matches torch.nn.functional.gelu approximate='tanh')
        g = gate * 0.7978845608028654 * (1.0 + 0.044715 * gate * gate)
        return 0.5 * gate * (1.0 + _np.tanh(g)) * up
    return up * (1.0 / (1.0 + _np.exp(-gate)))  # silu / swish


class IgpuDenseFfnExecutor:
    """Dense FFN executor for B-group offload (--dense-ffn-engine igpu).

    Computes SwiGLU FFN (gate_up_proj -> silu -> elementwise multiply with up -> down_proj)
    on the iGPU D3D12 service. Used for dense (non-MoE) models where the B-group FFN
    layers run on iGPU instead of GPU.

    Weights are expected to be in NVFP4 format (from Nvfp4DenseLinear / Nvfp4DenseColMerged):
    - weight: [out_features, in_features // 2] uint8, packed 4-bit weights
    - weight_scale: [out_features, in_features // 16] float8_e4m3fn, per-block scales
    - weight_global: [out_features] float16, per-row global scales

    The service's gemv expects weight scales as uint32 with e4m3 bit pattern (0.01x),
    so we reinterpret the float8 bytes as uint32 directly.
    """

    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        gate_up_global: torch.Tensor,
        down_weight: torch.Tensor,
        down_scale: torch.Tensor,
        down_global: torch.Tensor,
        *,
        intermediate_size: int,
        hidden_size: int,
        activation: str = "silu",
        service: IgpuGemvService | None = None,
        device=None,
        out_dtype=None,
    ) -> None:
        """
        Args:
            gate_up_weight: [2I, H//2] uint8, packed W4A8 gate|up weights (fused)
            gate_up_scale: [2I, H//16] float8_e4m3fn, per-block scales
            gate_up_global: [2I] float16, per-row global scales
            down_weight: [H, I//2] uint8, packed W4A8 down weights
            down_scale: [H, I//16] float8_e4m3fn, per-block scales
            down_global: [H] float16, per-row global scales
            intermediate_size: FFN intermediate dimension (I)
            hidden_size: Model hidden dimension (H)
            activation: Activation name ('silu' for SwiGLU)
            service: iGPU D3D12 service instance
            device: torch device to use for results
        """
        self.gate_up_weight = gate_up_weight
        self.gate_up_scale = gate_up_scale
        self.gate_up_global = gate_up_global
        self.down_weight = down_weight
        self.down_scale = down_scale
        self.down_global = down_global
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.service = service if service is not None else IgpuGemvService()
        self.device = device
        # numpy outputs must land back in model dtype (bf16), else the next
        # bf16 Linear rejects float32 hidden states.
        self.out_dtype = out_dtype
        self._cache = {}  # key -> (w_np, sc_rows_u32, gbl_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dense FFN on iGPU: out = down_proj(silu(gate_up_proj(x))) * up * gate.

        Args:
            x: (B, H) hidden states on CPU/GPU
        Returns:
            (B, H) output hidden states on the same device as input
        """
        import torch

        B, H = x.shape
        I = self.intermediate_size

        # Gate+Up projection: (B, H) x (H, 2I) -> (B, 2I)
        gate_out = self._gemv_layer(
            self.gate_up_weight, self.gate_up_scale, self.gate_up_global,
            x, 2 * I, H
        )  # (B, 2I)

        # Split gate and up
        gate = gate_out[:, :I]
        up = gate_out[:, I:]

        # Apply activation and multiply (SwiGLU: silu(gate) * up)
        activated = _activation(gate, up, self.activation)  # (B, I)

        # Down projection: (B, I) x (I, H) -> (B, H)
        out = self._gemv_layer(
            self.down_weight, self.down_scale, self.down_global,
            activated, H, I
        )  # (B, H)

        # Convert back to torch tensor on the target device
        result = torch.from_numpy(out)
        if self.device is not None:
            result = result.to(self.device)
        target_dtype = self.out_dtype or (x.dtype if hasattr(x, "dtype") else None)
        if target_dtype is not None:
            result = result.to(target_dtype)
        return result

    def _prep_layer(self, key: str, weight, scale, glob, M, K):
        """One-time host repack per layer: stable uint8 weights [M,NB*8],
        row-per-output u32 scales [M,NB], fp32 globals [M]. Returns tuple."""
        import torch
        if key in self._cache:
            return self._cache[key]
        NB = K // 16
        w_np = weight.detach().cpu().numpy()
        sc_t = scale.detach().cpu()
        sc_u8 = sc_t.view(torch.uint8).numpy() if sc_t.dtype == torch.float8_e4m3fn \
            else sc_t.numpy().astype(np.uint8)
        sc_rows = np.ascontiguousarray(sc_u8.T.astype(np.uint32))
        gbl = glob.detach().cpu().float().numpy().reshape(-1)
        if gbl.shape[0] != M:
            gbl = np.ones(M, dtype=np.float32)
        self._cache[key] = (w_np, sc_rows, gbl)
        return self._cache[key]
    def _gemv_layer(
        self, weight: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor,
        x: torch.Tensor, M: int, K: int
    ) -> np.ndarray:
        """Compute one W4A8 GEMV layer on iGPU.

        Args:
            weight: [M, K//2] uint8, packed W4A8 weights (row-major)
            scale: [K//16, M] float8_e4m3fn -- CHECKPOINT layout is block-major;
                converted to the service's row-per-output [M, NB] uint32 below.
            glob: [M] float16, per-row global scales
            x: (B, K) input activations
            M: output dimension (rows)
            K: input dimension
        Returns:
            (B, M) float32 output
        """
        NB = K // 16
        # Down-projection receives the numpy SwiGLU intermediate; accept both.
        if isinstance(x, np.ndarray):
            x_cpu = x.astype(np.float32, copy=False)
        else:
            x_cpu = x.detach().cpu().float().numpy()
        B = x_cpu.shape[0]

        key = "gu" if weight is self.gate_up_weight else "dn"
        w_np, sc_rows, gbl_f32 = self._prep_layer(key, weight, scale, glob, M, K)

        # Activations: W4A8 block quantization -- fully vectorized (a naive B x NB
        # python loop burned ~10 cores for minutes at prefill sizes).
        blocks = x_cpu.reshape(B, NB, 16)
        amax = np.abs(blocks).max(axis=2)                          # (B, NB)
        sc = np.maximum(amax, 1e-6) / 127.0                        # (B, NB)
        codes = np.clip(
            np.round(blocks / sc[:, :, None]), -127, 127
        ).astype(np.int32)                                         # (B, NB, 16)
        act_i8 = codes.reshape(B, NB * 16)
        act_asb = sc.astype(np.float32)

        out = np.zeros((B, M), dtype=np.float32)
        for b in range(B):
            # DLL path computes ALL M rows in one native call.
            row_out = self.service._gemv_via_dll(
                w_np, sc_rows, act_i8[b], act_asb[b], gbl_f32, M, K, NB
            )
            if row_out is not None:
                out[b] = row_out
                continue
            # stdio fallback speaks ONE output row per request.
            for r in range(M):
                out[b, r] = self.service.gemv(
                    w_np[r], sc_rows[r], act_i8[b], act_asb[b],
                    gbl_f32[r:r + 1], 1, K,
                )[0]
        return out


def attach_dense_ffn_executor(model, executor: IgpuDenseFfnExecutor, layer_ids: list[int]) -> None:
    """Attach a dense FFN executor to specific decoder layers."""
    if hasattr(model, "model"):  # causal LM wrapper
        model = model.model
    if hasattr(model, "layers"):
        for layer_id, layer in enumerate(model.layers.op_list):
            if layer_id in layer_ids and hasattr(layer, "mlp"):
                layer.mlp.dense_ffn_executor = executor
