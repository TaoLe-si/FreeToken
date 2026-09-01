from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from freetoken.distributed import DistributedInfo
from freetoken.scheduler import SchedulerConfig
from freetoken.utils import init_logger

_ZMQ_ADDR_CACHE: dict = {}


_ZMQ_ADDR_CACHE: dict = {}


def _zc_set(idx, suffix, addr):
    _ZMQ_ADDR_CACHE[(idx, suffix)] = addr
    return addr

_zmq_transport_addr_patched = True  # 源码自带探测+缓存；shim 无需再包

def _zmq_transport_addr(idx: int, suffix: str) -> str:
    """Windows lacks the ZMQ ipc transport; use loopback TCP, deterministic port.

    厂商常驻软件（华硕管家/应用商店等）会恰好监听在公式窗口内——连接它们将得到
    无限期的无应答（10060），detokenizer 因此在装载期静默死亡。此处对候选端口做
    bind 探测，被占用即向上避让。

    [ft-zmq-env-sync] 探测结果经环境变量跨进程同步：主进程（前端）先分配并导出，
    spawn 子进程继承环境后必须原样复用。否则子进程自己的 bind 探测会把前端真实
    绑定视为"被占用"而避让，把回复推到无人监听的端口上。"""
    import os
    _k = (idx, suffix)
    _c = globals().get('_ZMQ_ADDR_CACHE')
    if _c is not None and _k in _c:
        return _c[_k]
    env_key = "FREETOKEN_ZMQ_ADDR_%d" % idx
    inherited = os.environ.get(env_key)
    if inherited:
        if _c is not None:
            _c[_k] = inherited
        return inherited
    addr = None
    if os.name == "nt":
        import socket as _socket
        seed = 0
        for ch in suffix:
            if ch.isdigit():
                seed = seed * 10 + int(ch)
        start = 34000 + idx * 10 + (seed % 400)
        for offset in range(0, 500):
            candidate = start + offset
            try:
                with _socket.socket() as _sk:
                    _sk.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            addr = "tcp://127.0.0.1:%d" % candidate
            break
        if addr is None:
            addr = "tcp://127.0.0.1:%d" % start
    else:
        addr = "ipc:///tmp/freetoken_%d%s" % (idx, suffix)
    if _c is not None:
        _c[_k] = addr
    os.environ[env_key] = addr
    return addr





@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    num_tokenizer: int = 0
    silent_output: bool = False
    # The terminal shell is attached to this server (ft shell --model / ft serve --shell-mode).
    # The workers read it to leave the shell's foreground process group, so the ^C that cancels
    # a turn cannot also kill the engine — see server/launch.py:_detach_process_group.
    shell_mode: bool = False
    served_model_name: str | None = None
    tool_call_parser: str = "llama3"
    # Reasoning parser that splits <think> reasoning from content for OpenAI
    # responses. None disables it (default for models without a reasoning protocol).
    reasoning_parser: str | None = None
    # "model": fill unspecified request sampling params from generation_config.json
    # (temperature/top_k/top_p), like sglang. "none": use framework defaults only.
    sampling_defaults: str = "model"
    # Default max output (decode) tokens for a request that omits one. None falls back to the
    # adapter's built-in default (32k).
    max_output_tokens: int | None = None
    # Report the prefix-cache hit in each response's usage block (OpenAI
    # prompt_tokens_details.cached_tokens, Anthropic cache_read_input_tokens, Responses
    # input_tokens_details.cached_tokens). Mirrors sglang's --enable-cache-report.
    enable_cache_report: bool = False
    # Comma-separated CORS allow-list for browser/webview clients (e.g. the desktop
    # app). Empty string disables CORS headers entirely; "*" allows any origin.
    cors_origins: str = "tauri://localhost,http://tauri.localhost,http://localhost:1420"
    # --gpu entries in TP-rank order, empty = not given
    gpu: tuple[str, ...] = ()
    # full UUIDs resolved from --gpu, entry i = TP rank i; None = NVML unavailable, each worker then resolves its raw entry against CUDA's own enumeration
    gpu_assigned: "tuple[str, ...] | None" = None
    # Bounded request lifetime: a generation that produces no terminal reply within this
    # many seconds is aborted engine-side and the API layer surfaces a generation error.
    # Guards against a lost/stuck backend reply stranding the client stream forever.
    request_timeout_s: float = 600.0

    @property
    def share_tokenizer(self) -> bool:
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        return _zmq_transport_addr(3, self._unique_suffix)

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = _zmq_transport_addr(4, self._unique_suffix)
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(
    args: List[str],
    run_shell: bool = False,
    prog: str | None = None,
) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    from freetoken.attention import validate_attn_backend
    from freetoken.kvcache import SUPPORTED_CACHE_MANAGER
    from freetoken.moe import SUPPORTED_MOE_BACKENDS

    def _parse_moe_cache_rate(value: str) -> float:
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a number in [0, 1]") from exc
        if not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1]")
        return rate

    def _positive_int(value: str) -> int:
        try:
            n = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be a positive integer") from exc
        if n < 1:
            raise argparse.ArgumentTypeError("must be >= 1")
        return n

    def _lazy_gpu_arg(value: str) -> tuple[str, ...]:
        from freetoken.gpu_select import gpu_arg

        return gpu_arg(value)

    def _infer_tool_call_parser(model_path: str) -> str:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        # M3 first: its marker also contains the bare "minimax" substring, but the
        # namespaced tool grammar is a different protocol from M2's.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        if (
            "qwen3_5" in marker
            or "qwen3.5" in marker
            or ("qwen3" in marker and "coder" in marker)
        ):
            return "qwen3_coder"
        if "qwen" in marker:
            return "qwen25"
        if "deepseek" in marker and ("v4" in marker or "deepseek_v4" in marker):
            return "deepseekv32"
        if "deepseek" in marker and ("v3.2" in marker or "v32" in marker):
            return "deepseekv32"
        if "glm" in marker:
            return "glm47"
        if "mistral" in marker:
            return "mistral"
        return "llama3"

    def _infer_reasoning_parser(model_path: str) -> str | None:
        try:
            from freetoken.utils import cached_load_hf_config

            cfg = cached_load_hf_config(model_path).to_dict()
        except Exception:
            cfg = {}

        text_cfg = cfg.get("text_config") or {}
        candidates = [
            model_path,
            str(cfg.get("model_type", "")),
            str(text_cfg.get("model_type", "")),
            " ".join(str(v) for v in cfg.get("architectures", []) or []),
            " ".join(str(v) for v in text_cfg.get("architectures", []) or []),
        ]
        marker = " ".join(candidates).lower()
        if "gpt_oss" in marker or "gpt-oss" in marker or "gptoss" in marker:
            return "gpt_oss"
        if "deepseek" in marker and any(
            tag in marker for tag in ("v4", "deepseek_v4", "v3.2", "v32")
        ):
            return "deepseekv32"
        if "qwen3" in marker or "qwen3.5" in marker or "qwen3_5" in marker:
            return "qwen3"
        if "glm" in marker:
            return "glm"
        # M3 first ("minimax" is a substring): <mm:think> tags + 3 thinking gears,
        # not M2's always-on implicit <think>.
        if "minimax_m3" in marker or "minimax-m3" in marker or "minimaxm3" in marker:
            return "minimax_m3"
        if "minimax" in marker:
            return "minimax"
        if "muse_glimmer" in marker or "muse-glimmer" in marker or "museglimmer" in marker:
            return "muse_glimmer"
        if "gemma4" in marker:
            return "gemma4"
        return None

    parser = argparse.ArgumentParser(prog=prog, description="FreeToken Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--gpu",
        type=_lazy_gpu_arg,
        default=ServerArgs.gpu,
        help=(
            "GPU(s) to run on, comma-separated; entry i is TP rank i. Each entry is a GPU "
            "UUID (GPU-xxxx..., as nvidia-smi -L prints) or an nvidia-smi index"
        ),
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=_positive_int,
        default=ServerArgs.max_output_tokens,
        help="Default max output tokens for requests that omit one (default 32k).",
    )

    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help=(
            "Fraction of total GPU free memory the engine may use for weights + MoE "
            "cache + KV cache combined; the remainder is reserved runtime headroom."
        ),
    )

    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        "--request-timeout-s",
        type=float,
        dest="request_timeout_s",
        default=ServerArgs.request_timeout_s,
        help="Abort a generation that produces no terminal reply within this many seconds.",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        "--decode-log-interval",
        type=_positive_int,
        default=ServerArgs.decode_log_interval,
        help="Print one decode scheduler status line every N decode forwards.",
    )

    kv_capacity_group = parser.add_mutually_exclusive_group()
    kv_capacity_group.add_argument(
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    kv_capacity_group.add_argument(
        "--num-tokens",
        dest="num_token_override",
        type=int,
        default=ServerArgs.num_token_override,
        help=(
            "Total KV-cache capacity in tokens; must be a multiple of the resolved page "
            "size (DSV4: 128 window page, TRTLLM backend: 64). Mutually exclusive with "
            "--num-pages."
        ),
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="KV cache strategy (naive | radix). For hybrid GDN models 'radix' is materialized "
        "as a GDN-aware radix (cross-request GDN-state prefix reuse); pass 'naive' to opt out.",
    )

    parser.add_argument(
        "--enable-cache-report",
        action="store_true",
        default=ServerArgs.enable_cache_report,
        help=(
            "Return the number of prefix-cached prompt tokens in each response's usage block "
            "(OpenAI usage.prompt_tokens_details.cached_tokens, Anthropic "
            "usage.cache_read_input_tokens, Responses usage.input_tokens_details.cached_tokens). "
            "On /v1/messages this also makes input_tokens EXCLUDE the cached prefix, matching "
            "Anthropic billing semantics."
        ),
    )

    parser.add_argument(
        "--sampling-defaults",
        type=str,
        default=ServerArgs.sampling_defaults,
        choices=["model", "none"],
        help=(
            "Source for unspecified request sampling params. 'model' fills "
            "temperature/top_k/top_p from the checkpoint's generation_config.json "
            "(recommended for reasoning models to avoid greedy repetition loops); "
            "'none' uses framework defaults only."
        ),
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=ServerArgs.served_model_name,
        help="Model id returned by /v1/models. Defaults to the basename of --model.",
    )

    parser.add_argument(
        "--tool-call-parser",
        type=str,
        default="auto",
        choices=[
            "auto",
            "llama3",
            "qwen",
            "qwen25",
            "qwen3_coder",
            "mistral",
            "deepseekv32",
            "gemma4",
            "glm47",
            "minimax",
            "minimax_m3",
            "muse_glimmer",
            "gpt_oss",
            "gpt-oss",
        ],
        help="Tool-call parser format for OpenAI-compatible tool responses.",
    )

    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default="auto",
        choices=[
            "auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm",
            "minimax", "minimax_m3", "muse_glimmer", "gemma4",
        ],
        help=(
            "Reasoning parser that splits chain-of-thought into reasoning_content "
            "for OpenAI responses. 'auto' selects per model family (gpt-oss Harmony, "
            "<think> for qwen3/glm/minimax, <mm:think> for minimax-m3, ATEM to=self "
            "channels for muse-glimmer, gemma thought, dsv4); 'off' disables it."
        ),
    )

    parser.add_argument(
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help=(
            "The MoE backend to use. 'auto' resolves a MoE model to the offload family "
            "(offload, or hybrid when a `ft bench bw` profile recommends it); resident "
            "'fused' experts must be requested explicitly."
        ),
    )

    parser.add_argument(
        "--nvfp4-backend",
        default=ServerArgs.nvfp4_backend,
        choices=["auto", "marlin", "flashinfer", "triton"],
        help=(
            "NVFP4 routed-expert GEMM backend (default: triton, the portable inline-dequant "
            "kernel). auto picks by GPU (marlin on sm80-99 + vLLM; flashinfer b12x on sm120+ "
            "& CUDA>=13; else triton). Force one to override; it fails loudly if it cannot run."
        ),
    )

    parser.add_argument(
        "--expert-load",
        default=ServerArgs.expert_load,
        choices=["auto", "serial", "parallel"],
        help=(
            "How MoE expert banks are read into host RAM. 'auto' (default) reads scattered "
            "experts in parallel (fast) but falls back to serial when free RAM can't cover "
            "the banks + the parallel reader's extra whole-shard buffer; 'serial' forces the "
            "low-memory reclaimable read (slower); 'parallel' forces the fast read."
        ),
    )

    moe_cache_group = parser.add_mutually_exclusive_group()
    moe_cache_group.add_argument(
        "--moe-cache-size",
        type=int,
        default=ServerArgs.moe_cache_size,
        help="The number of unified MoE expert slots on GPU.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-rate",
        type=_parse_moe_cache_rate,
        default=ServerArgs.moe_cache_rate,
        help="The fraction of all MoE experts to keep in GPU cache.",
    )
    moe_cache_group.add_argument(
        "--moe-cache-auto",
        action="store_true",
        default=ServerArgs.moe_cache_auto,
        help=(
            "Auto-pick --moe-cache-size from free VRAM and expert size, MoE-priority "
            "(KV gets --kv-reserve-tokens as a floor). Not supported for owned-KV models."
        ),
    )

    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=ServerArgs.kv_reserve_tokens,
        help="KV-cache token floor reserved before --moe-cache-auto fills experts.",
    )

    parser.add_argument(
        "--ct-fp8",
        choices=["native", "bf16"],
        default=ServerArgs.ct_fp8,
        help=(
            "compressed-tensors FP8 handling: native = keep weights float8 and run the"
            " W8A16 kernel (half the decode bandwidth); bf16 = dequantize to BF16 at load."
        ),
    )

    parser.add_argument(
        "--kv-device",
        choices=["cuda", "shared", "cpu"],
        default=ServerArgs.kv_device,
        help=(
            "KV cache placement: cuda = VRAM with strict startup budget; shared = allow the"
            " pool to exceed free VRAM (Windows WDDM places overflow in the driver-managed"
            " shared pool); cpu = host-resident KV (experimental)."
        ),
    )
    parser.add_argument(
        "--kv-quant",
        choices=["bf16", "q8_0", "q4_0"],
        default=ServerArgs.kv_quant,
        help=(
            "KV cache value format: bf16 (default) = plain bf16; q8_0 = llama.cpp-style"
            " block-wise 8-bit quant (32 elem/block: int8 data + fp16 scale, ~1/2 the bf16"
            " footprint); q4_0 = reserved, currently falls back to bf16 with a warning at"
            " startup (not yet implemented in this build)."
        ),
    )

    parser.add_argument(
        "--mtp",
        action="store_true",
        dest="mtp",
        default=ServerArgs.mtp,
        help=(
            "Enable MTP (Multi-Token Prediction) speculative decoding. Loads the MTP head"
            " from a Qwen3.5/3.6 checkpoint's mtp.* block (no-op on other architectures)"
            " and runs K autoregressive draft steps per decode round; the main model"
            " verifies the K+1 candidates in one prefill-style forward and accepts up to"
            " K+1. Greedy sampling only (temperature=0 or top_k=1)."
        ),
    )
    parser.add_argument(
        "--mtp-k",
        type=int,
        default=ServerArgs.mtp_k,
        help=(
            "Number of speculative drafts per MTP step (default 3; range 1..5). The MTP"
            " head runs K autoregressive steps before the main model verifies, so each"
            " accepted batch is up to K+1 tokens."
        ),
    )
    parser.add_argument(
        "--mtp-igpu-fc",
        action="store_true",
        dest="mtp_igpu_fc",
        default=ServerArgs.mtp_igpu_fc,
        help=(
            "Route the MTP head's MXFP4 fc GEMV through the iGPU D3D12 service when"
            " available (sticky full-weight upload). Falls back to the dGPU torch path"
            " if the iGPU service is absent."
        ),
    )
    parser.add_argument("--no-mtp-igpu-fc", dest="mtp_igpu_fc", action="store_false", help="Force the dGPU torch reference fc (debug/parity check against the iGPU path)")
    # G.3: MTP verify CUDA-graph capture. Collapses ~265 kernel launches into
    # 1 dispatch for ~50-200x launch overhead reduction on the MTP verify path.
    parser.add_argument(
        "--mtp-igpu-verify-graph",
        action="store_true",
        dest="mtp_igpu_verify_graph",
        default=ServerArgs.mtp_igpu_verify_graph,
        help=(
            "Enable CUDA-graph capture of the 24-layer Qwen3_5Model.forward on MTP verify"
            " batches (G.3). Requires --mtp-igpu-fc. Disabled by default (capture takes"
            " ~1-2s on first verify batch)."
        ),
    )
    parser.add_argument("--no-mtp-igpu-verify-graph", dest="mtp_igpu_verify_graph", action="store_false", help="Disable the MTP verify CUDA-graph capture (fall back to eager forward).")

    parser.add_argument(
        "--moe-cache-policy",
        default=ServerArgs.moe_cache_policy,
        choices=["lru"],
        help="The unified MoE cache eviction policy.",
    )

    parser.add_argument(
        "--moe-cpu-threads",
        type=int,
        default=ServerArgs.moe_cpu_threads,
        help=(
            "Number of CPU worker threads for --moe-backend cpu decode experts. "
            "0 = auto (physical cores)."
        ),
    )

    parser.add_argument(
        "--moe-cpu-layers",
        type=str,
        default=ServerArgs.moe_cpu_layers,
        help=(
            "With --moe-backend offload/hybrid: which MoE layers compute on the "
            "CPU executor instead of the GPU offload/PCIe path (where CUDA pinning "
            "is quota-capped, e.g. WSL, their banks are OS-locked instead of pinned). Explicit id list ('3,7,11'), a count ('8' = 8 "
            "layers evenly strided), or a fraction ('0.5'). Unset = automatic where "
            "CUDA pinning is quota-capped, e.g. WSL (locks just enough head+tail "
            "layers when the banks exceed the pin budget, none otherwise); '0' "
            "forces all layers on GPU."
        ),
    )

    parser.add_argument(
        "--moe-hybrid-max-fetch",
        type=int,
        default=ServerArgs.moe_hybrid_max_fetch,
        help=(
            "For --moe-backend hybrid: max experts fetched over PCIe per (layer, decode "
            "step); the rest of that step's misses are computed on the CPU, overlapped. "
            "-1 (default) = auto: fetch the benched pcie/cpu bandwidth fraction of each "
            "step's misses (perfect overlap; needs an `ft bench bw` profile, else 1). "
            "0 = never fetch (all misses on CPU); large = behaves like plain offload."
        ),
    )

    parser.add_argument(
        "--disable-moe-prefill-overlap",
        action="store_false",
        dest="moe_prefill_overlap",
        default=ServerArgs.moe_prefill_overlap,
        help=(
            "Disable two-buffer overlap for prefill MoE expert copies. "
            "By default, prefill overlap is enabled and requires "
            "--moe-cache-size >= 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--enable-special-token-ckpt",
        action="store_true",
        dest="special_token_ckpt",
        default=ServerArgs.special_token_ckpt,
        help=(
            "Checkpoint decode state at special tokens (currently the tool-call opener). "
            "When a GDN-hybrid or SWA model samples its tool-call opener token, the "
            "scheduler preserves a reuse point just after it (GDN: a state snapshot "
            "donated to the prefix cache; SWA: the trailing window is kept resumable), so "
            "a client that rewrites the echoed tool call only invalidates the call body, "
            "not the turn."
        ),
    )

    parser.add_argument(
        "--moe-prefill-hit-d2d",
        action="store_true",
        dest="moe_prefill_hit_d2d",
        default=ServerArgs.moe_prefill_hit_d2d,
        help=(
            "During prefill prefetch, copy cache-resident experts device-side into "
            "the double buffer and stream only the misses over PCIe "
            "(cudaMemcpyBatchAsync, CUDA >= 13.0). Effective with "
            "--moe-cache-size > 2 * num_experts."
        ),
    )

    parser.add_argument(
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    parser.add_argument(
        "--cors-origins",
        type=str,
        default=ServerArgs.cors_origins,
        help=(
            "Comma-separated CORS allow-list for browser/webview clients "
            "(default: local Tauri/Vite dev origins). '' disables, '*' allows any."
        ),
    )

    # Dense (B-group)
    parser.add_argument(
        "--dense-ffn-engine",
        type=str,
        default=ServerArgs.dense_ffn_engine,
        choices=["cpu", "igpu", "gpu"],
        help="Dense model FFN engine: cpu (DRAM), igpu (iGPU D3D12), gpu (VRAM)",
    )

    # iGPU
    parser.add_argument(
        "--igpu-service",
        type=str,
        default=ServerArgs.igpu_service,
        help="iGPU D3D12 service executable path",
    )
    parser.add_argument(
        "--igpu-no-fallback",
        action="store_false",
        dest="igpu_fallback",
        help="Fail instead of falling back to CPU when iGPU is unavailable",
    )

    # Parse arguments
    kwargs = parser.parse_args(args).__dict__.copy()

    # reject a too-long list here with a clear reason, not as a dead rank later
    if len(kwargs["gpu"]) not in (0, kwargs["tensor_parallel_size"]):
        if kwargs["tensor_parallel_size"] == 1 and len(kwargs["gpu"]) > 1:
            parser.error("tensor parallelism is not supported yet: --gpu takes one entry")
        parser.error(
            f"--gpu has {len(kwargs['gpu'])} entries but --tensor-parallel-size is "
            f"{kwargs['tensor_parallel_size']}; give one entry per TP rank"
        )

    # resolve some arguments
    run_shell |= kwargs.pop("shell_mode")
    kwargs["shell_mode"] = run_shell
    if run_shell:
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["served_model_name"] is None:
        kwargs["served_model_name"] = (
            os.path.basename(os.path.normpath(kwargs["model_path"])) or kwargs["model_path"]
        )

    if kwargs["tool_call_parser"] == "auto":
        kwargs["tool_call_parser"] = _infer_tool_call_parser(kwargs["model_path"])

    if kwargs["reasoning_parser"] == "auto":
        kwargs["reasoning_parser"] = _infer_reasoning_parser(kwargs["model_path"])
    elif kwargs["reasoning_parser"] == "off":
        kwargs["reasoning_parser"] = None

    # Offload-family backends (offload/cpu/hybrid) need a slot cache; if the user gave no
    # sizing flag at all, default to --moe-cache-auto so a bare `ft serve <FTW MoE>` works
    # out of the box (the scheduler resolves the size from free VRAM). Explicit
    # size/rate/auto is preserved.
    from freetoken.moe import is_offload_moe_backend

    _no_cache_flag = (
        kwargs["moe_cache_size"] == 0
        and not kwargs["moe_cache_auto"]
        and (kwargs["moe_cache_rate"] is None or kwargs["moe_cache_rate"] == 0)
    )
    if is_offload_moe_backend(kwargs["moe_backend"]) and _no_cache_flag:
        kwargs["moe_cache_auto"] = True

    if kwargs["model_source"] == "modelscope":
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            from modelscope import snapshot_download

            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    del kwargs["model_source"]

    # "auto" (or an unspecified dtype) resolves to the checkpoint's dtype. Multimodal /
    # hybrid configs (e.g. Qwen3.5-MoE) keep it under ``text_config`` and use the newer
    # ``dtype`` key rather than top-level ``torch_dtype``, so check both; default bf16.
    if (dtype_str := kwargs["dtype"]) in ("auto", None):
        from freetoken.utils import cached_load_hf_config

        cfg = cached_load_hf_config(kwargs["model_path"]).to_dict()
        text_cfg = cfg.get("text_config") or {}
        dtype_str = (
            cfg.get("torch_dtype") or cfg.get("dtype")
            or text_cfg.get("torch_dtype") or text_cfg.get("dtype") or "bfloat16"
        )

    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    del kwargs["tensor_parallel_size"]

    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result, run_shell


def _build_option_schema() -> dict:
    """Extract the full CLI-options schema from the serve parser.
    Returns {name: {type, default, choices, help, group}}.
    """
    import argparse
    import re
    from freetoken.moe import SUPPORTED_MOE_BACKENDS

    parser = argparse.ArgumentParser(prog="ft serve", description="FreeToken Server Arguments")
    # KV cache
    kv = parser.add_argument_group("KV cache")
    kv.add_argument("--page-size", type=int, default=16, help="KV page size (tokens)")
    kv.add_argument("--max-batch-rows", type=int, default=None, help="Max KV pool rows")
    kv.add_argument("--kv-device", choices=["cuda", "shared", "cpu"], default="cuda",
                    help="KV placement: cuda (VRAM budget), shared (WDDM shared pool, may exceed VRAM), cpu (experimental)")
    kv.add_argument("--kv-quant", choices=["bf16", "q8_0", "q4_0"], default="bf16",
                    help="KV value format: bf16 (default); q8_0 (llama.cpp-style block-wise int8 + fp16 scale); q4_0 (falls back to bf16)")
    # MoE
    moe = parser.add_argument_group("MoE / expert offload")
    moe.add_argument("--moe-backend", choices=["auto"] + [n for n in SUPPORTED_MOE_BACKENDS.supported_names()], default=None, help="MoE decode backend")
    moe.add_argument("--moe-cache-size", type=int, default=None, help="Expert cache slots")
    moe.add_argument("--moe-cpu-layers", type=str, default=None, help="CPU layers (e.g. 0-5,7)")
    # Dense (B-group)
    dense = parser.add_argument_group("Dense (B-group engine)")
    dense.add_argument("--dense-ffn-engine", choices=["cpu", "igpu", "gpu"], default="cpu", help="Dense model FFN engine: cpu (DRAM), igpu (iGPU D3D12), gpu (VRAM)")
    # iGPU
    igpu = parser.add_argument_group("iGPU D3D12")
    igpu.add_argument("--igpu-service", type=str, default=None, help="iGPU D3D12 service exe")
    igpu.add_argument("--igpu-no-fallback", action="store_true", default=False, help="Fail instead of falling back to CPU")
    # MTP speculative decoding
    mtp = parser.add_argument_group("MTP speculative decoding")
    mtp.add_argument("--mtp", action="store_true", default=False, help="Enable MTP (Multi-Token Prediction) speculative decoding")
    mtp.add_argument("--mtp-k", type=int, default=3, help="Number of speculative drafts per MTP step (1..5)")
    # P2.1 (2026-09-02): default is now dGPU bf16 F.linear (DgpuBf16Fc, ~75 us/call) --
    # the iGPU D3D12 sticky bridge is ~1056 us/call (D3D12 stdin/stdout + GPU fence
    # sync) and was the main MTP-verify bottleneck. Pass --mtp-igpu-fc to force the
    # legacy iGPU path for A/B comparison or iGPU-specific workloads.
    mtp.add_argument("--mtp-igpu-fc", action="store_true", default=False, help="[P2.1 deprecated] Force the iGPU D3D12 sticky FC (legacy, ~14x slower than dGPU F.linear)")
    mtp.add_argument("--no-mtp-igpu-fc", dest="mtp_igpu_fc", action="store_false", help="[P2.1 default] Use the dGPU bf16 F.linear FC executor (DgpuBf16Fc)")
    mtp.add_argument("--mtp-igpu-verify-graph", action="store_true", default=False, help="Enable CUDA-graph capture of the 24-layer Qwen3_5Model.forward on MTP verify batches (G.3; collapses ~265 launches into 1 dispatch). Requires --mtp-igpu-fc.")
    mtp.add_argument("--no-mtp-igpu-verify-graph", dest="mtp_igpu_verify_graph", action="store_false", help="Disable the MTP verify CUDA-graph capture (fall back to eager forward).")

    out: dict[str, dict] = {}
    for group in parser._action_groups:
        gname = group.title or "misc"
        for action in group._group_actions:
            opts = action.option_strings
            if not opts:
                continue
            name = re.sub(r"^--", "", opts[0])
            out[name] = {
                "flags": opts,
                "type": "int" if action.type is int else "float" if action.type is float else
                        "bool" if isinstance(action, argparse._StoreTrueAction) else "str",
                "default": None if action.default is None else action.default,
                "choices": list(action.choices) if action.choices else None,
                "help": action.help or "",
                "group": gname,
            }
    return out
