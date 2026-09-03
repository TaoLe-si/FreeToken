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
import gc
import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

from freetoken.engine.engine import _IGPU_RESERVED

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
    hip.hipStreamCreate.argtypes = [ctypes.c_void_p]
    hip.hipStreamCreate.restype = ctypes.c_int
    hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
    hip.hipStreamDestroy.restype = ctypes.c_int
    hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
    hip.hipStreamSynchronize.restype = ctypes.c_int
    hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    hip.hipMemcpyAsync.restype = ctypes.c_int
    hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    hip.hipMemset.restype = ctypes.c_int
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
        """边复制边删除: stream-load FTW directly into 780M GTT, never holding
        full bank in host memory. Peak host mem = one 64 MB pinned chunk.

        两种模式:
        1) Streaming (empty cache.bank_sources): call stream_ftw_to_gtt
           to populate GTT directly from FTW file. No host banks materialized.
        2) Legacy (cache.bank_sources populated): per-bank H2D from pinned host.
        """
        if self._registered:
            return
        sources = self.cache.bank_sources
        # Detect streaming path: bank_sources empty (no set_bank_sources call)
        streaming_path = len(sources) == 0
        if not streaming_path:
            missing = [n for n in _BANK_ORDER if n not in sources]
            if missing:
                raise RuntimeError(f"cache bank_sources missing banks: {missing}")
            self._validate_bank_shapes(sources)
        logger.info_rank0("iGPU register_banks: streaming=%s reserved=%d", streaming_path, len(_IGPU_RESERVED))
        self.hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.hip.hipMemcpy.restype = ctypes.c_int
        H2D = 1
        f_b = ctypes.c_size_t(0); t_b = ctypes.c_size_t(0)
        if self.dll.igpu_meminfo(ctypes.byref(f_b), ctypes.byref(t_b)) == 0:
            logger.info("GTT meminfo before: free=%.2f GB total=%.2f GB", f_b.value / 2**30, t_b.value / 2**30)
        t0 = time.perf_counter()

        if streaming_path:
            # Streaming path: call stream_ftw_to_gtt if not already done
            ftw_path = getattr(self.cache, "folder_path", None) or getattr(self, "ftw_path", None) or getattr(self.cache, "ftw_path", None)
            if ftw_path is None:
                raise RuntimeError(
                    "register_banks: bank_sources is empty and no ftw_path available; "
                    "cannot stream-load"
                )
            from freetoken.checkpoint.ftw import stream_ftw_to_gtt
            logger.info_rank0("iGPU stream-load: %s", ftw_path)
            gtt = stream_ftw_to_gtt(
                ftw_path,
                num_layers=self.num_layers,
                dll=self.dll,
                hip=self.hip,
                bank_order=list(_BANK_ORDER),
            )
            if gtt is None:
                raise RuntimeError("stream_ftw_to_gtt returned None")
            for name in _BANK_ORDER:
                for layer_id in range(self.num_layers):
                    rc = self.dll.igpu_register_layer_dev(
                        layer_id,
                        *[ctypes.c_void_p(gtt[n][layer_id][0]) for n in _BANK_ORDER],
                    )
                    if rc != 0:
                        raise RuntimeError(f"igpu_register_layer_dev(layer={layer_id}) failed with {rc}")
                    self._bank_ptrs.append(tuple(gtt[n][layer_id][0] for n in _BANK_ORDER))
            self._registered = True
            self._migrate_s = time.perf_counter() - t0
            logger.info_rank0("iGPU edge-map-gc: stream-load done (%.1fs); only 64 MB pinned at peak", self._migrate_s)
            if self.dll.igpu_meminfo(ctypes.byref(f_b), ctypes.byref(t_b)) == 0:
                logger.info("GTT meminfo after stream-load: free=%.2f GB total=%.2f GB", f_b.value / 2**30, t_b.value / 2**30)
            return

        # Legacy path: per-bank H2D from pinned host memory
        for layer_id in range(self.num_layers):
            dev_ptrs = []
            for name in _BANK_ORDER:
                t = sources[name][layer_id]
                if t is None:
                    raise RuntimeError(f"bank {name!r} layer {layer_id} is None (not loaded)")
                nbytes = t.numel() * t.element_size()
                if _IGPU_RESERVED:
                    d = _IGPU_RESERVED.pop(0)
                else:
                    d = self.dll.igpu_devmalloc(ctypes.c_size_t(nbytes))
                    if not d:
                        raise RuntimeError(f"igpu_devmalloc({nbytes}) failed for {name!r} layer {layer_id}")
                CHUNK = 64 * 1024 * 1024
                src_base = t.data_ptr()
                rc = 0
                for off in range(0, nbytes, CHUNK):
                    n = min(CHUNK, nbytes - off)
                    rc = self.hip.hipMemcpy(d + off, ctypes.c_void_p(src_base + off), ctypes.c_size_t(n), H2D)
                    if rc != 0:
                        break
                if rc != 0:
                    raise RuntimeError(f"hipMemcpy H2D failed ({rc}) for {name!r} layer {layer_id}")
                dev_ptrs.append(d)
                # 边映射边 gc: 显式释放 host pin 配额
                try:
                    from freetoken.kernel.pinned import free_pinned_addr
                    free_pinned_addr(t.data_ptr())
                except Exception:
                    pass
                sources[name][layer_id] = None
            rc = self.dll.igpu_register_layer_dev(layer_id, *(ctypes.c_void_p(p) for p in dev_ptrs))
            if rc != 0:
                raise RuntimeError(f"igpu_register_layer_dev({layer_id}) failed with {rc}")
            self._bank_ptrs.append(tuple(dev_ptrs))
        self._registered = True
        self._migrate_s = time.perf_counter() - t0
        logger.info_rank0("iGPU edge-map-gc: legacy register_banks done; prefill routes to iGPU decode")

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
        """One MoE layer of decode on the 780M. Returns a GPU [bs, H] tensor.

        Form-2 data path: banks live in 780M device memory (GTT). Per token:
        hidden/ids/weights go host->780M via hipMemcpy (8KB + 64B), the kernel
        reads device banks natively, and the result comes back host->dGPU.
        """
        assert self._registered, "register_banks() must run before decode"
        assert hidden_states.shape[1] == _H, hidden_states.shape
        assert topk_ids.shape[-1] == _EXPECTED_TOP_K, topk_ids.shape
        bs = int(hidden_states.shape[0])
        stream = torch.cuda.current_stream()
        self.hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        H2D, D2H = 1, 2

        # Phase 1 integration: pinned host buffers (allocated once), async D2H,
        # single hipStreamSynchronize at end (no per-layer sync).
        # One-time allocation of pinned staging buffers for D2H.
        if getattr(self, "_host_staging", None) is None or self._host_staging[0] < bs:
            hidden_cpu = torch.empty((bs, _H), dtype=torch.float32, pin_memory=True)
            ids_cpu = torch.empty((bs, _EXPECTED_TOP_K), dtype=torch.int32, pin_memory=True)
            weights_cpu = torch.empty((bs, _EXPECTED_TOP_K), dtype=torch.float32, pin_memory=True)
            out_host = torch.empty((bs, _H), dtype=torch.float32, pin_memory=True)
            self._host_staging = (bs, hidden_cpu, ids_cpu, weights_cpu, out_host)
        _, hidden_cpu, ids_cpu, weights_cpu, out_host = self._host_staging
        # Async D2H: pinned + non_blocking=True; does NOT sync CPU.
        hidden_cpu.copy_(hidden_states.to(torch.float32), non_blocking=True)
        ids_cpu.copy_(topk_ids.to(torch.int32), non_blocking=True)
        weights_cpu.copy_(topk_weights.to(torch.float32), non_blocking=True)

        # one-time device IO staging buffers (780M side)
        if getattr(self, "_dev_io", None) is None or self._dev_io[0] < bs:
            if getattr(self, "_dev_io", None) is not None:
                for p in self._dev_io[1]:
                    self.dll.igpu_devfree(p)
            d_h = self.dll.igpu_devmalloc(ctypes.c_size_t(bs * _H * 4))
            d_i = self.dll.igpu_devmalloc(ctypes.c_size_t(bs * _EXPECTED_TOP_K * 4))
            d_w = self.dll.igpu_devmalloc(ctypes.c_size_t(bs * _EXPECTED_TOP_K * 4))
            d_o = self.dll.igpu_devmalloc(ctypes.c_size_t(bs * _H * 4))
            if not all((d_h, d_i, d_w, d_o)):
                raise RuntimeError("igpu_devmalloc failed on decode IO staging")
            self._dev_io = (bs, (d_h, d_i, d_w, d_o))
            logger.info('iGPU dev IO staging: d_h=%s d_i=%s d_w=%s d_o=%s bs=%d', hex(d_h or 0), hex(d_i or 0), hex(d_w or 0), hex(d_o or 0), bs)
        _, (d_h, d_i, d_w, d_o) = self._dev_io
        # Async H2D to 780M staging; HIP stream queue accepts in parallel.
        self.hip.hipMemcpy(d_h, ctypes.c_void_p(hidden_cpu.data_ptr()), ctypes.c_size_t(bs * _H * 4), H2D)
        self.hip.hipMemcpy(d_i, ctypes.c_void_p(ids_cpu.data_ptr()), ctypes.c_size_t(bs * _EXPECTED_TOP_K * 4), H2D)
        self.hip.hipMemcpy(d_w, ctypes.c_void_p(weights_cpu.data_ptr()), ctypes.c_size_t(bs * _EXPECTED_TOP_K * 4), H2D)

        # Per-layer kernel enqueue (1 kernel per layer). Phase 3 will batch all 40.
        for i in range(bs):
            rc = self.dll.igpu_moe_decode_dev(
                int(layer_id),
                ctypes.c_void_p(d_h + i * _H * 4),
                ctypes.c_void_p(d_i + i * _EXPECTED_TOP_K * 4),
                ctypes.c_void_p(d_w + i * _EXPECTED_TOP_K * 4),
                ctypes.c_void_p(d_o),
            )
            if rc != 0:
                raise RuntimeError(
                    f"igpu_moe_decode_dev(layer={layer_id}, token={i}) failed with {rc}"
                )
        # ONE sync at end (Phase 1 change).
        self.dll.igpu_get_stream.argtypes = []
        self.dll.igpu_get_stream.restype = ctypes.c_void_p
        hip_stream = self.dll.igpu_get_stream()
        self.hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.hip.hipStreamSynchronize.restype = ctypes.c_int
        self.hip.hipStreamSynchronize(hip_stream)
        # Async D2H output (HIP is done, just memcpy back).
        self.hip.hipMemcpy(ctypes.c_void_p(out_host.data_ptr()), d_o, ctypes.c_size_t(bs * _H * 4), D2H)
        out = torch.empty((bs, _H), dtype=hidden_states.dtype, device=self.device)
        out.copy_(out_host, non_blocking=True)
        return out

    def health_check(self) -> bool:
        """Cheap liveness probe: the DLL answers with its version string."""
        try:
            return bool(self.dll.igpu_version())
        except Exception:  # noqa: BLE001 - a probe must never raise
            return False
