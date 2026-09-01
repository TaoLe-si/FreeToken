from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # iGPU MoE backend (--moe-backend igpu): path to the persistent D3D12 GEMV
    # service executable (t_d3d12_service.exe). None -> auto-locate (repo build
    # or FREETOKEN_IGPU_SERVICE).
    igpu_service: str | None = None
    # When the iGPU service cannot start (no AMD iGPU / missing exe / self-check
    # failure): True falls back to --moe-backend cpu for the decode experts,
    # False raises. Never silently computes on the GPU offload path (the bank
    # layout would not match the native rows).
    igpu_fallback: bool = True
    # Dense-model FFN engine (dense_host_offload architecture, B-group compute):
    # "cpu" (default) runs FFN GEMVs on the CPU executor from DRAM, "igpu" runs
    # them on the iGPU D3D12 service, "gpu" keeps everything on the GPU (resident).
    # Only consulted for dense (non-MoE) models; MoE models ignore it.
    dense_ffn_engine: str = "cpu"
    # KV cache placement. "cuda" = VRAM-only with the strict startup budget (current
    # policy); "shared" = allow the pool to exceed free VRAM -- Windows WDDM places the
    # overflow in the driver-managed shared pool (dGPU reads it over GTT; this machine's
    # 8GB card already runs 22GB+ this way) -- boot derives pages from context needs
    # instead of asserting on a negative budget; "cpu" = host-resident KV (experimental,
    # requires an attention backend that can read host pages).
    kv_device: str = "cuda"
    # KV cache value format. "bf16" = plain bf16 (default); "q8_0" = llama.cpp-style
    # block-wise 8-bit quant (32 elem/block: int8 data + fp16 scale, ~1.06 byte/elem,
    # ~1/2 of bf16); "q4_0" is accepted but currently falls back to bf16 with a warning
    # at startup (q4_0 / subnormal handling is not yet implemented in this build).
    kv_quant: str = "bf16"
    # MTP (Multi-Token Prediction) speculative decoding. ``mtp=True`` enables the head and
    # the per-step draft hook in the scheduler; the head only fires on Qwen3.5/3.6 MoE
    # checkpoints (the only ones that ship an ``mtp.*`` block today). ``mtp_k`` is the
    # number of speculative drafts per step (the head runs ``mtp_k`` autoregressive steps
    # per accepted-batch round trip). ``mtp_igpu_fc`` routes the head's MXFP4 fc GEMV
    # through the iGPU D3D12 service (slower than dGPU F.linear due to IPC + GPU fence
    # sync; legacy path).
    mtp: bool = False
    # Default K=3: with --mtp-igpu-verify-graph enabled (P0), K=3 gives 25.9 t/s vs
    # off-MTP 20 t/s (+30%); K=2 gives 21.2 t/s (~baseline), K=4 gives 23.2 t/s.
    mtp_k: int = 3
    # P2.1 (2026-09-02): default dGPU bf16 F.linear FC executor. The iGPU sticky bridge
    # is ~1056 us/call (D3D12 stdin/stdout + fence sync); dGPU bf16 F.linear is ~75 us/call
    # (14x faster) and keeps the result on the dGPU (no CPU/GPU sync hop). Set True to
    # force the legacy iGPU sticky path.
    mtp_igpu_fc: bool = False
    # G.3: enable CUDA-graph capture of the 24-layer Qwen3_5Model.forward on
    # MTP verify batches. Collapses ~265 kernel launches into a single
    # dispatch for ~50-200x launch overhead reduction. Requires mtp=True and
    # mtp_igpu_fc=True (the GPU side of the graph capture). Disabled by default
    # -- the capture takes ~1-2s on first verify batch and adds ~50-100 MB of
    # device-side cached graph memory per cached bs.
    mtp_igpu_verify_graph: bool = False

    # compressed-tensors FP8 policy: "native" keeps float8 weights on the W8A16 kernel;
    # "bf16" dequantizes every FP8 dense weight at load (larger footprint, reference path).
    ct_fp8: str = "native"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    def __post_init__(self) -> None:
        # q4_0 is implemented in MHAKVCache (llama.cpp Q4_0 blocks + shared bf16
        # staging per layer); no fallback needed.
        pass

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        return parse_config(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
