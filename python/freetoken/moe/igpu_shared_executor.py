"""Shared-pool iGPU MoE executor over the ROCm HIP DLL (--moe-backend igpu, P2).

Runs the NVFP4 expert GEMVs on the AMD Radeon 780M (gfx1103) through
benchmarks/cpu_moe_microbench/hip_moe_dll.dll while the rest of the model stays
on the NVIDIA dGPU (verified CUDA+HIP coexistence in one process). The DLL
reads the expert weight banks ZERO-COPY over PCIe: the cache's pinned host
banks are hipHostRegister'd (they are CUDA-pinned already; HIP registers the
same pages with the AMD driver for its own device alias) and
igpu_register_layer resolves each to a HIP device pointer internally, so no
bank bytes ever move for the iGPU to stream them.

The DLL side expects the exact bank row layouts of the cache's native "nvfp4"
schema (gate_up_packed [E, 2I, H//2] u8, gate_up_scale [E, 2I, H//16] e4m3 u8,
gate_up_global [E, 2I] fp16 u16, down likewise over [E, H, I]) and hardcodes
the geometry (top_k=8 lanes, E<=8 routed experts/token, H=2048, I=512, one
token per call), so register_banks validates shapes before wiring each layer
and decode flattens the batch to per-token calls. IO hidden/ids/weights/out
buffers must also be HIP-registered: the DLL resolves them through
hipHostGetDevicePointer the same way.

Synchronous for now: each decode D2H-copies onto pinned buffers, calls
igpu_moe_decode (which blocks on the HIP stream), then H2D-copies the result
back on the current CUDA stream. A flag-sync graph bridge (the CPU executor's
stream-memop handshake) is a later step.
"""

from __future__ import annotations

import ctypes
import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_ROCM_BIN = r"C:\Program Files\AMD\ROCm\6.4\bin"
_HIP_DLL = "amdhip64_6.dll"
_DLL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "benchmarks", "cpu_moe_microbench", "hip_moe_dll.dll",
)

# The DLL's hardcoded geometry (see hip_moe_dll.hip): NVFP4 bank layout for
# H=2048 / I=512 models with top_k=8 routing, one token per decode call.
_EXPECTED_TOP_K = 8
_H = 2048
_I = 512
_MAX_LAYERS = 64  # MAX_LAYERS in hip_moe_dll.hip

_BANK_ORDER = (
    "gate_up_packed",
    "gate_up_scale",
    "gate_up_global",
    "down_packed",
    "down_scale",
    "down_global",
)


def _load_dlls(path):
    """Load amdhip64_6.dll first (the ROCm runtime the plugin links against),
    then the MoE plugin. The ROCm bin dir must be on the DLL search path before
    the plugin loads or its amdhip64_6 import fails."""
    if not os.path.isdir(_ROCM_BIN):
        raise RuntimeError(f"ROCm 6.4 HIP SDK not found at {_ROCM_BIN!r}")
    os.add_dll_directory(_ROCM_BIN)
    hip = ctypes.CDLL(os.path.join(_ROCM_BIN, _HIP_DLL))
    hip.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    hip.hipHostRegister.restype = ctypes.c_int  # hipError_t; 0 == hipSuccess
    dll = ctypes.CDLL(path or _DLL_PATH)
    dll.igpu_init.restype = ctypes.c_int
    dll.igpu_init.argtypes = []
    dll.igpu_register_layer.restype = ctypes.c_int
    dll.igpu_register_layer.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 6
    dll.igpu_moe_decode.restype = ctypes.c_int
    dll.igpu_moe_decode.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 4
    dll.igpu_hostmalloc.restype = ctypes.c_void_p
    dll.igpu_hostmalloc.argtypes = [ctypes.c_size_t]
    dll.igpu_hostfree.restype = ctypes.c_int
    dll.igpu_hostfree.argtypes = [ctypes.c_void_p]
    dll.igpu_version.restype = ctypes.c_char_p
    dll.igpu_version.argtypes = []
    return hip, dll


class IgpuSharedMoeExecutor:
    """Decode-time expert compute on the iGPU over the cache's shared pinned banks.

    Mirrors CpuMoeExecutor's engine-facing interface (a per-layer decode call
    returning a GPU [bs, H] tensor) but routes the expert GEMVs through the HIP
    plugin on the Radeon 780M.
    """

    def __init__(
        self,
        cache,
        device,
        num_layers: int,
        num_experts: int,
        top_k: int = _EXPECTED_TOP_K,
    ) -> None:
        self.cache = cache
        self.device = device
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        if int(top_k) != _EXPECTED_TOP_K:
            raise NotImplementedError(
                f"the hip_moe_dll kernels hardcode top_k={_EXPECTED_TOP_K}, "
                f"this model routes top_k={top_k}"
            )
        self.top_k = int(top_k)
        self.quant_format = cache.quant_format
        if self.quant_format != "nvfp4":
            raise NotImplementedError(
                f"IgpuSharedMoeExecutor reads the native 'nvfp4' bank schema, "
                f"but this cache's experts are {self.quant_format!r}"
            )
        if self.num_layers > _MAX_LAYERS:
            raise NotImplementedError(
                f"the hip_moe_dll bank registry holds {_MAX_LAYERS} layers, "
                f"the model has {self.num_layers}"
            )
        self.hip, self.dll = _load_dlls(os.environ.get("FREETOKEN_IGPU_MOE_DLL") or None)
        version = self.dll.igpu_version()
        rc = self.dll.igpu_init()
        if rc != 0:
            raise RuntimeError(f"igpu_init failed with HIP error code {rc}")
        logger.info_rank0(
            f"iGPU shared-pool MoE executor ready: {version.decode()} "
            f"({self.num_layers} layers x {self.num_experts} experts, zero-copy banks)"
        )
        self._registered = False
        # Per-layer host bank pointer tuples (the bank tensors themselves stay
        # alive through cache.bank_sources).
        self._bank_ptrs: list[tuple[int, ...]] = []
        # Pinned IO buffers per batch size, HIP-registered once. The DLL resolves
        # all four through hipHostGetDevicePointer, so registration is mandatory.
        self._io: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Registration: hipHostRegister the cache's pinned banks per layer, then
    # hand their pointers to igpu_register_layer (which resolves the HIP device
    # aliases internally).
    # ------------------------------------------------------------------

    def register_banks(self) -> None:
        """Migrate every layer's six banks into HIP shared (mapped) memory.

        hipHostRegister on >96MB CUDA-pinned pages silently fails on this
        ROCm/780M combo (iGPU reads zeros), while hipHostMalloc works at any
        size. So we allocate an equivalent shared buffer per bank, copy the
        bytes in, and REPLACE the cache's bank tensor with a torch view on the
        shared memory. Net steady-state host usage is unchanged: the original
        cudaHostAlloc bank is dropped with the reference.
        """
        if self._registered:
            return
        sources = self.cache.bank_sources
        missing = [n for n in _BANK_ORDER if n not in sources]
        if missing:
            raise RuntimeError(f"cache bank_sources missing banks: {missing}")
        self._validate_bank_shapes(sources)
        self._diag("register_banks entered: migration mode")
        for layer_id in range(self.num_layers):
            ptrs = []
            for name in _BANK_ORDER:
                t = sources[name][layer_id]
                assert t.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                nbytes = t.numel() * t.element_size()
                # 1) shared mapped allocation (iGPU-visible at any size)
                h = self.dll.igpu_hostmalloc(ctypes.c_size_t(nbytes))
                if not h:
                    raise RuntimeError(
                        f"igpu_hostmalloc({nbytes}) failed for {name!r} layer {layer_id}"
                    )
                # 2) copy the weight bytes in (host-to-host memcpy)
                ctypes.memmove(h, ctypes.c_void_p(t.data_ptr()), nbytes)
                # 3) replace the cache entry: old cudaHostAlloc block is freed
                #    with the dropped reference (steady-state usage unchanged)
                sources[name][layer_id] = torch.frombuffer(
                    (ctypes.c_char * nbytes).from_address(h),
                    dtype=t.dtype,
                ).view(t.shape)
                ptrs.append(h)
            rc = self.dll.igpu_register_layer(layer_id, *(ctypes.c_void_p(p) for p in ptrs))
            if rc != 0:
                raise RuntimeError(f"igpu_register_layer({layer_id}) failed with {rc}")
            self._bank_ptrs.append(tuple(ptrs))
            if layer_id == 0:
                # 别名读回校验: iGPU 视角是否看到真实权重
                chk = (ctypes.c_char * 64).from_address(self.dll.igpu_hostmalloc(ctypes.c_size_t(64)))
                dp = ctypes.c_void_p()
                self.hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(ptrs[0]), 0)
                chk_dp = ctypes.c_void_p()
                self.hip.hipHostGetDevicePointer(ctypes.byref(chk_dp), ctypes.cast(chk, ctypes.c_void_p), 0)
                self.hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                rc3 = self.hip.hipMemcpy(chk_dp, dp, ctypes.c_size_t(64), 2)
                host_bytes = ctypes.string_at(ctypes.c_void_p(ptrs[0]), 64)
                alias_bytes = chk.raw
                self._diag(f"L0 bank readback: rc={rc3} match={host_bytes == alias_bytes} host_nz={sum(1 for b in host_bytes if b)}/64 alias_nz={sum(1 for b in alias_bytes if b)}/64")
        self._registered = True

    def _validate_bank_shapes(self, sources: dict) -> None:
        """Fail loudly on a layout the hardcoded DLL geometry cannot serve."""
        head = {n: sources[n][0] for n in _BANK_ORDER}
        for name, t in head.items():
            # DLL reads raw bytes; accept any 1-byte or 2-byte dtype
            assert t.element_size() in (1, 2), (name, t.dtype, t.element_size())
            assert t.is_contiguous() and t.dim() >= 2, (name, t.shape)
            assert t.size(0) == self.num_experts, (name, t.shape, self.num_experts)
        gu_pack = head["gate_up_packed"]
        gu_scale = head["gate_up_scale"]
        gu_global = head["gate_up_global"]
        dn_pack = head["down_packed"]
        dn_scale = head["down_scale"]
        dn_global = head["down_global"]
        # gate_up: [E, 2I, H//2] u8 + [E, 2I, H//16] u8 + [E, 2I] u16
        assert tuple(gu_pack.shape) == (self.num_experts, 2 * _I, _H // 2), gu_pack.shape
        assert tuple(gu_scale.shape) == (self.num_experts, 2 * _I, _H // 16), gu_scale.shape
        assert tuple(gu_global.shape) == (self.num_experts, 2 * _I), gu_global.shape
        assert gu_global.element_size() == 2, (gu_global.dtype, gu_global.element_size())
        # down: [E, H, I//2] u8 + [E, H, I//16] u8 + [E, H] u16
        assert tuple(dn_pack.shape) == (self.num_experts, _H, _I // 2), dn_pack.shape
        assert tuple(dn_scale.shape) == (self.num_experts, _H, _I // 16), dn_scale.shape
        assert tuple(dn_global.shape) == (self.num_experts, _H), dn_global.shape
        assert dn_global.element_size() == 2, (dn_global.dtype, dn_global.element_size())

    # ------------------------------------------------------------------
    # Decode: D2H -> DLL (blocks on the HIP stream) -> H2D, per token.
    # ------------------------------------------------------------------

    def _io_for(self, bs: int):
        io = self._io.get(bs)
        if io is None:
            # hipHostRegister on CUDA-pinned pages is silently unreadable on this
            # ROCm/780M combo; hipHostMalloc (via the DLL) is verified readable,
            # so all decode IO lives in shared mapped memory too.
            def _shared(shape, dtype):
                t = torch.empty(shape, dtype=dtype)
                nbytes = t.numel() * t.element_size()
                h = self.dll.igpu_hostmalloc(ctypes.c_size_t(nbytes))
                if not h:
                    raise RuntimeError(f"igpu_hostmalloc({nbytes}) failed on decode IO")
                view = torch.frombuffer(
                    (ctypes.c_char * nbytes).from_address(h), dtype=dtype
                ).view(shape)
                return h, view
            h_hid, hidden = _shared((bs, _H), torch.float32)
            h_ids, ids = _shared((bs, _EXPECTED_TOP_K), torch.int32)
            h_w, weights = _shared((bs, _EXPECTED_TOP_K), torch.float32)
            h_out, out = _shared((_H,), torch.float32)
            self._io[bs] = (hidden, ids, weights, out)
            self._io_raw = getattr(self, "_io_raw", {})
            self._io_raw[bs] = (h_hid, h_ids, h_w, h_out)
        return self._io[bs]

    _dump_once: bool = True

    def _dump(self, layer_id, pinned_hidden, pinned_ids, pinned_weights, out_rows) -> None:
        if not self._dump_once:
            return
        self._dump_once = False
        try:
            torch.save({
                "hidden": pinned_hidden.clone(),
                "ids": pinned_ids.clone(),
                "weights": pinned_weights.clone(),
                "out": torch.stack(out_rows).clone(),
            }, "E:/FreeToken/igpu_layer_dump.pt")
        except OSError:
            pass

    def _diag(self, msg: str) -> None:
        try:
            with open("E:/FreeToken/igpu_dll_diag.log", "a", encoding="utf-8") as f:
                f.write("[py] " + msg + "\n")
        except OSError:
            pass

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the iGPU. Returns a GPU [bs, H] tensor.

        Synchronous today: the DLL blocks on its HIP stream per call, so the
        batch is flattened to per-token invocations and the results ship back
        with one H2D copy on the current CUDA stream."""
        assert self._registered, "register_banks() must run before decode"
        assert hidden_states.shape[1] == _H, hidden_states.shape
        assert topk_ids.shape[-1] == _EXPECTED_TOP_K, topk_ids.shape
        bs = int(hidden_states.shape[0])
        self._diag(f"decode enter layer={layer_id} bs={bs} hid={hidden_states.data_ptr():#x} dtype={hidden_states.dtype} ids={topk_ids.data_ptr():#x} tkw={topk_weights.data_ptr():#x} wdtype={topk_weights.dtype}")
        pinned_hidden, pinned_ids, pinned_weights, pinned_out = self._io_for(bs)
        stream = torch.cuda.current_stream()

        # D2H: activations + routing onto the HIP-registered pinned buffers.
        # float32 converts from whatever compute dtype the model runs (bf16/fp16).
        pinned_hidden.copy_(hidden_states, non_blocking=True)
        pinned_ids.copy_(topk_ids.to(torch.int32), non_blocking=True)
        pinned_weights.copy_(topk_weights.to(torch.float32), non_blocking=True)
        # The copies must be complete before the host hands the pointers to the
        # HIP stream (the DLL reads them via its own device alias + host sync).
        stream.synchronize()

        out_rows = []
        for i in range(bs):
            self._diag(f"call igpu_moe_decode tok={i} hid={pinned_hidden[i].data_ptr():#x} ids={pinned_ids[i].data_ptr():#x} tkw={pinned_weights[i].data_ptr():#x} out={pinned_out.data_ptr():#x}")
            rc = self.dll.igpu_moe_decode(
                int(layer_id),
                ctypes.c_void_p(pinned_hidden[i].data_ptr()),
                ctypes.c_void_p(pinned_ids[i].data_ptr()),
                ctypes.c_void_p(pinned_weights[i].data_ptr()),
                ctypes.c_void_p(pinned_out.data_ptr()),
            )
            if rc != 0:
                raise RuntimeError(
                    f"igpu_moe_decode(layer={layer_id}, token={i}) failed with {rc}"
                )
            out_rows.append(pinned_out.clone())

        self._dump(layer_id, pinned_hidden, pinned_ids, pinned_weights, out_rows)
        if layer_id == 0 and not getattr(self, "_hid_checked", False):
            self._hid_checked = True
            dp = ctypes.c_void_p()
            self.hip.hipHostGetDevicePointer(ctypes.byref(dp), ctypes.c_void_p(pinned_hidden[0].data_ptr()), 0)
            chk = self.dll.igpu_hostmalloc(ctypes.c_size_t(64))
            chk_dp = ctypes.c_void_p()
            self.hip.hipHostGetDevicePointer(ctypes.byref(chk_dp), ctypes.c_void_p(chk), 0)
            self.hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.hip.hipMemcpy(chk_dp, dp, ctypes.c_size_t(64), 2)
            import struct
            host_v = struct.unpack("4f", ctypes.string_at(ctypes.c_void_p(pinned_hidden[0].data_ptr()), 16))
            alias_v = struct.unpack("4f", ctypes.string_at(ctypes.c_void_p(chk), 16))
            self._diag(f"hidden alias check: hid_host={hex(pinned_hidden[0].data_ptr())} dp={hex(dp.value or 0)} host={host_v} alias={alias_v}")
        # H2D back: stack the per-token rows host-side, then one async copy onto
        # the current CUDA stream.
        out_host = torch.stack(out_rows)  # [bs, H] host
        out = torch.empty((bs, _H), dtype=hidden_states.dtype, device=self.device)
        out.copy_(out_host, non_blocking=True)  # H2D f32 -> cast to model dtype
        return out

    def health_check(self) -> bool:
        """Cheap liveness probe: the DLL answers with its version string."""
        try:
            return bool(self.dll.igpu_version())
        except Exception:  # noqa: BLE001 - a probe must never raise
            return False
