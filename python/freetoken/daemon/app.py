"""The daemon's HTTP control plane. camelCase JSON throughout. Loopback by default; an optional
``X-FT-Token`` shared secret gates everything except the daemon's own ``/health`` liveness probe.

Handlers are ``async`` and push every blocking call to an executor so the event loop never
blocks. Two executors: a small **lifecycle** pool for start/stop/switch, kept separate from the
**proxy/metrics** pool, so a storm of health/metrics polls against a loading serve can never
starve an operator's stop."""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .accounting import AccountingOutboxError, AccountingPrepareError
from .serve_manager import Conflict
from .version import DAEMON_VERSION


class StartBody(BaseModel):
    model: str
    port: int | None = None
    args: list[str] = []


class StopBody(BaseModel):
    force: bool = False


class SwitchBody(StartBody):
    force: bool = False


class AccountingAckBody(BaseModel):
    receiptId: str


class CheckpointBody(BaseModel):
    id: str
    args: list[str] = []


class CancelBody(BaseModel):
    id: str


class BenchBody(BaseModel):
    # Raw `ft bench bw` args (e.g. ["--dtype", "nvfp4", "--threshold", "2.5"]); empty = all dtypes.
    args: list[str] = []
class ModelDownloadBody(BaseModel):
    id: str
    source: str = "hf"

def _bench_profile_path(gpu_uuid: str | None) -> str | None:
    # per-GPU profiles and no torch here: the serve's own card when its --gpu names one, else the newest file
    from freetoken.moe.bench_profile import default_profile_path, latest_profile_path  # torch-free

    if gpu_uuid:
        path = default_profile_path(gpu_uuid)
        if os.path.isfile(path):
            return path
    return latest_profile_path()


def _serve_gpu_uuid(args: list[str]) -> str | None:
    """The full UUID a serve's `--gpu` pins, or None when there is none or it cannot be resolved."""
    for i, a in enumerate(args):
        val = a[len("--gpu="):] if a.startswith("--gpu=") else (args[i + 1] if a == "--gpu" and i + 1 < len(args) else None)
        if not val:
            continue
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            resolved = resolve_gpu_uuids([val])
        except ValueError:
            return None
        if resolved:
            return resolved[0]
        # no NVML: a UUID value still keys the profile file (canonical prefix), an index cannot
        return "GPU-" + val[len("GPU-"):] if val.upper().startswith("GPU-") else None
    return None


def _read_bench_profile(path: str | None) -> dict | None:
    if path is None:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _bench_sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _dir_size(path: str) -> int:
    """Total bytes of a directory tree (best-effort)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fname in files:
                try:
                    total += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass
    except OSError:
        return 0
    return total

def _parse_ftbench(line: str) -> dict | None:
    """``FTBENCH <done> <total> <label>`` -> a progress dict (mirrors ft checkpoint's FTCONVERT)."""
    parts = line.split(maxsplit=3)
    if len(parts) < 4 or parts[0] != "FTBENCH":
        return None
    try:
        return {"done": int(parts[1]), "total": int(parts[2]), "label": parts[3]}
    except ValueError:
        return None


# ---- custom model dirs (user-added native paths) ----

def _daemon_settings_file() -> str:
    return os.path.join(os.path.expanduser("~"), ".freetoken", "daemon", "settings.json")

def _hub_mirror_choice() -> str:
    """用户设置的 HF 镜像偏好：'' 直连 / 'hf-mirror' / 'modelscope'(视为直连+自动回退)。"""
    try:
        return str((_load_daemon_settings() or {}).get("mirror", "") or "")
    except Exception:
        return ""

def _apply_hub_mirror() -> None:
    """按设置应用 HF_ENDPOINT；必须在 huggingface_hub 首次导入前调用才最可靠。"""
    import os as _os
    choice = _hub_mirror_choice()
    # 直连也默认给镜像兜底：国内直连 hf.co 几乎必超时
    endpoint = "https://hf-mirror.com" if choice in ("", "hf-mirror", "modelscope") else ""
    if endpoint:
        _os.environ["HF_ENDPOINT"] = endpoint

def _load_daemon_settings() -> dict:
    try:
        with open(_daemon_settings_file(), encoding="utf-8") as f:
            v = json.load(f)
            return v if isinstance(v, dict) else {}
    except Exception:
        pass
    return {}

def _save_daemon_settings(d) -> None:
    try:
        p = _daemon_settings_file()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

def _get_model_dir() -> str:
    return str(_load_daemon_settings().get("model_dir", "") or "")

def _set_model_dir(path) -> None:
    d = _load_daemon_settings()
    if path:
        d["model_dir"] = path
    else:
        d.pop("model_dir", None)
    _save_daemon_settings(d)

def _custom_dirs_file() -> str:
    return os.path.join(os.path.expanduser("~"), ".freetoken", "daemon", "custom_model_dirs.json")

def _load_custom_dirs() -> list:
    try:
        with open(_custom_dirs_file(), encoding="utf-8") as f:
            v = json.load(f)
            if isinstance(v, list):
                return [str(x) for x in v if x]
    except Exception:
        pass
    return []

def _save_custom_dirs(dirs) -> None:
    try:
        p = _custom_dirs_file()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(sorted(set(dirs)), f, ensure_ascii=False, indent=1)
    except Exception:
        pass

_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")

def _dir_has_weights(path) -> bool:
    try:
        for name in os.listdir(path):
            if name.lower().endswith(_WEIGHT_SUFFIXES):
                return True
    except OSError:
        pass
    return False

def _is_model_dir(path) -> bool:
    """Model root: config.json present, or weight files directly inside."""
    return os.path.isfile(os.path.join(path, "config.json")) or _dir_has_weights(path)

def _mk_entry(path) -> dict:
    norm = path.replace(os.sep, "/").rstrip("/")
    return {
        "id": norm,
        "name": norm.rsplit("/", 1)[-1] or norm,
        "path": path,
        "sizeBytes": _dir_size(path),
        "source": "local-dir",
    }

def _collect_models_in(root) -> list:
    """Models under root: itself, children, grand-children, nested HF-hub cache."""
    out = []

    def _hub_entries(hub):
        got = []
        try:
            for entry in os.listdir(hub):
                if entry.startswith("models--"):
                    repo = entry[len("models--"):].replace("--", "/")
                    got.append({
                        "id": repo,
                        "name": repo.split("/")[-1],
                        "path": os.path.join(hub, entry),
                        "sizeBytes": _dir_size(os.path.join(hub, entry)),
                        "source": "hf-cache",
                    })
        except OSError:
            pass
        return got

    if not os.path.isdir(root):
        return out
    if os.path.basename(root.rstrip(os.sep + "/")).lower() == "hub":
        out.extend(_hub_entries(root))
        return out
    if _is_model_dir(root):
        out.append(_mk_entry(root))
        return out
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for name in entries:
        p2 = os.path.join(root, name)
        if not os.path.isdir(p2):
            continue
        if os.path.basename(p2).lower() == "hub":
            out.extend(_hub_entries(p2))
            continue
        if _is_model_dir(p2):
            out.append(_mk_entry(p2))
            continue
        try:
            for sub in sorted(os.listdir(p2)):
                p3 = os.path.join(p2, sub)
                if not os.path.isdir(p3):
                    continue
                if os.path.basename(p3).lower() == "hub":
                    out.extend(_hub_entries(p3))
                elif _is_model_dir(p3):
                    out.append(_mk_entry(p3))
        except OSError:
            continue
    hub = os.path.join(root, "hub")
    if os.path.isdir(hub):
        out.extend(_hub_entries(hub))
    return out


def _SUPPORTED_ARCH_FAMILIES() -> set[str]:
    """dev tree 引擎实际能加载的架构家族（与 models/register.py 的 _MODEL_REGISTRY 同步）。

    用途：搜索/推荐结果必须按这个集合过滤，否则用户会下载 dev tree 不能加载的模型，
    启动时再被 EngineConfig 拒绝，造成"白下载"体验。
    """
    return {
        "llama", "qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe",
        "mistral", "gemma4", "gpt_oss", "minimax_m2", "minimax_m3",
        "deepseek_v4", "glm4_moe", "glm_moe_dsa", "muse_glimmer",
    }


def _arch_family_for_id(model_id: str, *, library_name: str = "", tags=None) -> str | None:
    """从 model id（必填）+ 可选的 HF 库名/tags 推断家族；不能识别返回 None。"""
    mid = (model_id or "").lower()
    tags = [str(t).lower() for t in (tags or []) if t]
    libs = (library_name or "").lower()
    blob = " ".join([mid, libs, " ".join(tags)])
    rules = [
        ("qwen3_5_moe", ["qwen3.5-moe", "qwen3_5_moe", "qwen3.5 moe"]),
        ("qwen3_5",     ["qwen3.5", "qwen3_5"]),
        ("qwen3_moe",   ["qwen3-moe", "qwen3_moe"]),
        ("qwen3",       ["qwen3"]),
        ("qwen2",       ["qwen2"]),
        ("minimax_m3",  ["minimax-m3", "minimax_m3"]),
        ("minimax_m2",  ["minimax-m2", "minimax_m2"]),
        ("deepseek_v4", ["deepseek-v4", "deepseek_v4"]),
        ("glm_moe_dsa", ["glm-moe-dsa", "moe-dsa"]),
        ("glm4_moe",    ["glm-4-moe", "glm4_moe", "chatglm"]),
        ("muse_glimmer",["muse-glimmer", "muse_glimmer"]),
        ("gpt_oss",     ["gpt-oss", "gpt_oss"]),
        ("gemma4",      ["gemma-4", "gemma4"]),
        ("mistral",     ["mistral", "codestral"]),
        ("llama",       ["llama"]),
    ]
    for fam, kws in rules:
        if any(k in blob for k in kws):
            return fam
    return None


def _is_supported_model_id(model_id: str, *, library_name: str = "", tags=None) -> bool:
    fam = _arch_family_for_id(model_id, library_name=library_name, tags=tags)
    return fam in _SUPPORTED_ARCH_FAMILIES()


def _QUANT_MULT(key):
    """bytes-per-weight multiplier for a quant key."""
    return {"bf16": 2.0, "fp8": 1.0, "q4": 0.5, "nvfp4": 0.5, "mxfp4": 0.5}.get(key, 2.0)

_FACTS_CACHE = {}

def _get_facts(model_path):
    """AutoConfig + true weight size, cached per model."""
    if model_path in _FACTS_CACHE:
        return _FACTS_CACHE[model_path]
    import threading as _th
    box = [None, None]
    def _load():
        try:
            from transformers import AutoConfig
            # 本地优先：不创建空壳缓存
            try:
                box[0] = AutoConfig.from_pretrained(model_path, local_files_only=True)
            except Exception:
                # 远端仅当 model_path 是本地路径（非 repo id）时回退
                import os as _os2
                if _os2.path.isdir(model_path):
                    box[0] = AutoConfig.from_pretrained(model_path)
                else:
                    raise  # repo id 但本地无缓存 → 不下载，避免创建空壳
        except Exception as exc:
            box[1] = exc
    t = _th.Thread(target=_load, daemon=True)
    t.start(); t.join(15)
    if box[0] is None:
        raise RuntimeError(str(box[1])[:200] if box[1] else "config load timeout")
    cfg = box[0]
    repo_total = 0
    # 本地目录直读权重体积（非标准分片名/无 index 也准）
    import os as _osp
    if _osp.isdir(model_path):
        try:
            from pathlib import Path as _Pl
            repo_total = int(sum(
                f.stat().st_size for f in _Pl(model_path).rglob("*")
                if f.is_file() and f.suffix.lower() in (".safetensors", ".bin", ".pt", ".gguf")))
        except Exception:
            pass
    if not repo_total:
      try:
        from pathlib import Path as _P
        from huggingface_hub import snapshot_download as _sd
        snap = _sd(model_path, local_files_only=True)
        wt = sum(f.stat().st_size for f in _P(snap).rglob("*")
                 if f.is_file() and f.suffix.lower() in (".safetensors", ".bin", ".pt"))
        if wt > 0:
            repo_total = int(wt)
      except Exception:
          pass
    if not repo_total:
        try:
            from huggingface_hub import HfApi as _Hf
            try:
                info_f = _Hf().model_info(model_path, files_metadata=True, timeout=10)
            except TypeError:
                info_f = _Hf().model_info(model_path, files_metadata=True)
            s = sum(int(getattr(x, "size", 0) or 0)
                    for x in (getattr(info_f, "siblings", None) or [])
                    if (getattr(x, "rfilename", "") or "").endswith(".safetensors"))
            if s > 0:
                repo_total = s
        except Exception:
            pass
    h = getattr(cfg, "hidden_size", 0) or 0
    facts = {
        "cfg": cfg, "repo_total": repo_total,
        "h": h,
        "nl": getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0)) or 0,
        "voc": getattr(cfg, "vocab_size", 0) or 0,
        "inter": getattr(cfg, "intermediate_size", getattr(cfg, "ffn_hidden_size", h * 4)) or (h * 4),
        "num_exp": getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", 0)) or 0,
        "top_k": getattr(cfg, "num_experts_per_tok", getattr(cfg, "top_k", 0)) or 0,
        "n_kv": getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None) or 8,
        "head_dim": getattr(cfg, "head_dim", None) or 128,
        "tie": bool(getattr(cfg, "tie_word_embeddings", False) or getattr(cfg, "tie_embeddings", False)),
    }
    _FACTS_CACHE[model_path] = facts
    return facts

def _kv_tokens_from(args):
    kt = 8192
    for i, a in enumerate(args or []):
        if a in ("--kv-reserve-tokens", "--max-seq-len-override") and i + 1 < len(args):
            try: kt = max(kt, int(args[i + 1]))
            except ValueError: pass
    return kt

def _estimate_core(facts, dtype_bytes, args=None):
    """Pure math for one quant variant of a model."""
    h = facts["h"]; nl = facts["nl"]; voc = facts["voc"]; inter = facts["inter"]
    num_exp = facts["num_exp"]; top_k = facts["top_k"]
    attn = nl * 4 * h * h * dtype_bytes
    ffn = nl * 3 * h * inter * dtype_bytes if num_exp == 0 else 0
    moe_ffn = nl * 3 * h * inter * num_exp * dtype_bytes if num_exp > 0 else 0
    embed = (voc * h * dtype_bytes) if facts["tie"] else (2 * voc * h * dtype_bytes)
    total = int(attn + ffn + moe_ffn + embed)
    kv = int(2 * 2 * max(nl, 1) * max(facts["n_kv"], 1) * facts["head_dim"] * _kv_tokens_from(args) * dtype_bytes)
    a_group = int(attn + embed + kv)
    b_group = int(ffn + (moe_ffn * 0.55 if num_exp > 0 else 0))
    return {
        "total": total, "a": a_group, "b": b_group, "kv": kv,
        "isMoE": num_exp > 0,
        "numExperts": num_exp if num_exp > 0 else None,
        "topK": top_k if top_k > 0 else None,
        "numLayers": nl, "hiddenSize": h, "vocabSize": voc,
    }

def _probe_mem():
    vram, gpu_name = 8 * 1024 ** 3, ""
    try:
        info = _ensure_gpu_probe()
        if info.get("ok"):
            vram = int(info.get("vramTotal", vram))
            gpu_name = str(info.get("name", ""))
    except Exception:
        pass
    if not gpu_name:
        try:
            import winreg
            kPath = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000"
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, kPath)
            vram = int(winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0])
            gpu_name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
        except Exception:
            pass
    dram = 0
    try:
        import ctypes as ct
        class MSE(ct.Structure):
            _fields_ = [("dwLength", ct.c_ulong), ("dwMemoryLoad", ct.c_ulong), ("ullTotalPhys", ct.c_ulonglong), ("ullAvailPhys", ct.c_ulonglong), ("ullTotalPageFile", ct.c_ulonglong), ("ullAvailPageFile", ct.c_ulonglong), ("ullTotalVirtual", ct.c_ulonglong), ("ullAvailVirtual", ct.c_ulonglong), ("ullAvailExtendedVirtual", ct.c_ulonglong)]
        m = MSE(); m.dwLength = ct.sizeof(MSE)
        if ct.windll.kernel32.GlobalMemoryStatusEx(ct.byref(m)):
            dram = int(m.ullTotalPhys)
    except Exception:
        pass
    return vram, dram, gpu_name



def _apply_rules(backend, c, vram, dram, gpu_name):
    """Verdict rules —— 与引擎(0.1.1 win)真实能力严格对齐：
    · 稠密模型：仅「全显存驻留」一条路（gpu）。实测 --moe-backend offload/hybrid/cpu
      对 dense 一律被忽略（引擎打印 ignoring MoE settings），无任何分层/CPU 卸载。
    · MoE 模型：专家可卸载到 DRAM（offload/hybrid 家族真实现），按 A/B 组联合预算判定。"""
    exp = c["numExperts"]
    moe_tag = "，MoE {} 专家(55%热集)".format(exp) if c["isMoE"] else ""
    is_dense = not c["isMoE"]
    if backend == "gpu":
        if (c["total"] + c["kv"]) < vram * 0.92:
            note = "全显存驻留：{:.1f}/{:.1f} GiB ({}){}".format(
                c["total"]/1024**3, vram*0.92/1024**3, gpu_name, moe_tag)
            return True, note, "full"
        budget = vram * 0.92 + dram * 0.35   # 与启动守卫同一预算公式（驱动共享内存回退）
        if (c["total"] + c["kv"]) < budget:
            note = "原生回退模式：总量 {:.1f} GiB ≤ 显存+共享预算 {:.1f} GiB；权重经 PCIe 流转，速度稍慢{}".format(
                c["total"]/1024**3, budget/1024**3, moe_tag)
            return True, note, "layered"
        return False, ("稠密 {:.1f} GiB 超出可承载上限（显存+共享预算 ≈ {:.1f} GiB）；可选 Q4/NVFP4 或 MoE 版本").format(
            c["total"]/1024**3, budget/1024**3), None
    if is_dense:
        return False, ("稠密模型当前引擎仅支持全显存驻留（本档约 {:.1f} GiB > 显存 {:.1f} GiB）；"
                       "同参数请改用 MoE 版本（专家卸载到内存）").format(
            c["total"]/1024**3, vram/1024**3), None
    if backend == "igpu":
        fit = c["a"] < vram * 0.92 and c["b"] < dram * 0.72
        mode = "full" if fit else None
        note = "A组(GPU 注意力+KV): {:.1f}/{:.1f} GiB; B组(DRAM 专家FFN): {:.1f}/{:.1f} GiB{}".format(
            c["a"]/1024**3, vram*0.92/1024**3, c["b"]/1024**3, dram*0.72/1024**3, moe_tag)
        return fit, note, mode
    if backend in ("cpu", "hybrid"):
        fit = c["a"] < vram * 0.92 and c["b"] < dram * 0.8
        mode = "full" if fit else None
        note = "A组(GPU): {:.1f}/{:.1f} GiB; B组(DRAM): {:.1f}/{:.1f} GiB{}".format(
            c["a"]/1024**3, vram*0.92/1024**3, c["b"]/1024**3, dram/1024**3, moe_tag)
        return fit, note, mode
    return False, "Unknown backend: {}".format(backend), None


def _estimate_model_fit(model_path, backend, args=None, quant=None):
    """Estimate fit for one model; quant=None means honor --dtype in args (real load)."""
    try:
        facts = _get_facts(model_path)
    except Exception as exc:
        return {"fit": False, "note": f"无法读取模型配置: {exc}",
                "breakdown": {"totalBytes": 0, "aGroupBytes": 0, "bGroupBytes": 0,
                    "vramBytes": 0, "dramBytes": 0, "gpuName": "", "numLayers": 0,
                    "hiddenSize": 0, "vocabSize": 0, "isMoE": False,
                    "numExperts": None, "topK": None}}
    # dtype 选择：显式量化档优先，否则看 --dtype 参数，默认 BF16
    if quant:
        dtype_bytes = _QUANT_MULT(quant)
    else:
        dtype_bytes = 2.0
        a_list = args or []
        for i, a in enumerate(a_list):
            if a == "--dtype" and i + 1 < len(a_list):
                d = str(a_list[i + 1]).lower()
                if "fp8" in d or "int8" in d:
                    dtype_bytes = 1.0
                elif "4" in d:
                    dtype_bytes = 0.5
    c = _estimate_core(facts, dtype_bytes, args)
    # 真实加载时用磁盘权重做下限；量化模拟不做（假设另下量化版仓库）
    if not quant and facts["repo_total"] > 0:
        c["total"] = max(c["total"], int(facts["repo_total"]))
        c["b"] = min(c["b"], max(1024 ** 3, int(facts["repo_total"] * 0.85)))
    vram, dram, gpu_name = _probe_mem()
    fit, note, mode = _apply_rules(backend, c, vram, dram, gpu_name)
    return {
        "fit": bool(fit), "note": note, "quant": quant or "auto", "mode": mode,
        "breakdown": {
            "totalBytes": c["total"], "aGroupBytes": c["a"], "bGroupBytes": c["b"],
            "vramBytes": vram, "dramBytes": dram, "gpuName": gpu_name,
            "numLayers": c["numLayers"], "hiddenSize": c["hiddenSize"],
            "vocabSize": c["vocabSize"], "isMoE": c["isMoE"],
            "numExperts": c["numExperts"], "topK": c["topK"],
        },
    }



# ---- engine arg whitelist: the panel may emit flags the official Windows
# release wheel (freetoken 0.1.1) does not implement; drop unknown flags and
# map unsupported values so `ft serve` never dies on argparse. ----
_ALLOWED_ENGINE_FLAGS = {
    "--max-running-requests", "--memory-ratio", "--moe-backend", "--moe-cpu-threads",
    "--moe-cpu-layers", "--max-seq-len-override", "--max-output-tokens", "--dtype",
    "--tensor-parallel-size", "--attention-backend", "--cache-type", "--kv-reserve-tokens",
    "--moe-cache-rate", "--moe-cache-size", "--moe-cache-auto", "--cuda-graph-max-bs",
    "--num-tokenizer", "--max-prefill-length", "--disable-pynccl", "--num-pages",
    "--page-size", "--tool-call-parser", "--reasoning-parser", "--served-model-name",
    "--model-source", "--nvfp4-backend", "--expert-load", "--moe-hybrid-max-fetch",
    "--moe-cache-policy", "--enable-cache-report", "--sampling-defaults", "--mtp", "--mtp-k", "--mtp-igpu-fc", "--no-mtp-igpu-fc", "--mtp-igpu-verify-graph", "--no-mtp-igpu-verify-graph",
    "--moe-prefill-hit-d2d", "--decode-log-interval", "--num-tokens", "--host", "--port",
    "--dense-ffn-engine", "--igpu-service", "--igpu-no-fallback",
    "--kv-device", "--kv-quant", "--ct-fp8", "--num-tokens-override",
    "--mtp", "--mtp-k", "--mtp-igpu-fc", "--mtp-igpu-verify-graph", "--disable-moe-prefill-overlap",
}

def _sanitize_engine_args(args):
    """Keep only flags the engine release accepts; rewrite unsupported values."""
    out = []
    i = 0
    argv = list(args or [])
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("-"):
            i += 1
            continue
        flag = tok.split("=")[0]
        if "=" in tok:
            val = tok.split("=", 1)[1]
            has_val = True
        else:
            has_val = False
            val = None
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]
        if flag not in _ALLOWED_ENGINE_FLAGS:
            if has_val or val is not None:
                i += 1
            i += 1
            continue
        # P2: --moe-backend igpu now supported via IgpuSharedMoeExecutor (HIP shared-pool)
        if flag in ("--dtype",) and val == "auto":
            pass
        if has_val:
            out.append(flag + "=" + str(val))
        elif val is not None:
            out.append(flag)
            out.append(str(val))
        else:
            out.append(flag)
        i += 1 if (has_val or val is not None) else 1
        if val is not None and not has_val:
            i += 1
    return out

# Offline fallback: known recommendation sizes so estimate-quants still works
# when huggingface.co is unreachable (badges must NEVER blanket-"unrunnable").
_POOL_PARAMS_B = {
    "openai/gpt-oss-20b": 20.9,
    "Qwen/Qwen2.5-Coder-7B-Instruct": 7.6,
    "Qwen/Qwen2.5-Coder-32B-Instruct": 32.8,
    "mistralai/Codestral-22B-v0.1": 22.2,
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": 30.5,
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": 15.7,
    "zai-org/GLM-4.5-Air": 106.0,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 30.5,
}

def _pool_params_b(model_path):
    if model_path in _POOL_PARAMS_B:
        return _POOL_PARAMS_B[model_path]
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)B", str(model_path), _re.IGNORECASE)
    return float(m.group(1)) if m else None

def _synth_core(params_b, dbytes):
    """Approximate core from parameter count alone (bytes-per-param=dbytes)."""
    total = int(params_b * (2 ** 29) * dbytes)
    a = int(total * 0.14)
    kv = min(int(total * 0.04), 2 * 1024 ** 3)
    return {"total": total, "a": a, "b": total - a, "kv": kv,
            "isMoE": False, "numExperts": None, "topK": None,
            "numLayers": 0, "hiddenSize": 0, "vocabSize": 0}

def build_app(
    *,
    manager,
    ring,
    probe,
    footprint_fn: Callable[[int | None], dict],
    lifecycle_pool: ThreadPoolExecutor,
    proxy_pool: ThreadPoolExecutor,
    default_serve_port: int = 1919,
    token: str | None = None,
    checkpoints=None,
    started_wall: float = 0.0,
    wall_now: Callable[[], float] | None = None,
    shutdown_hook: Callable[[], None] | None = None,
) -> FastAPI:
    import time as _time

    wall_now = wall_now or _time.time
    app = FastAPI(title="FreeToken daemon", version=DAEMON_VERSION)

    if shutdown_hook is not None:

        @app.on_event("shutdown")
        async def _on_shutdown() -> None:
            # uvicorn fires this on SIGTERM/SIGINT. Run the (blocking) hook off-loop so the grace
            # period in stop() can't wedge the event loop during shutdown.
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, shutdown_hook)
            except Exception:  # noqa: BLE001
                pass

    def require_token(x_ft_token: str | None = Header(default=None)) -> None:
        if token is not None and x_ft_token != token:
            raise HTTPException(status_code=401, detail="invalid or missing X-FT-Token")

    auth = [Depends(require_token)]

    async def run(pool: ThreadPoolExecutor, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, functools.partial(fn, *args))

    def resolve_port(explicit: int | None) -> int:
        if explicit is not None:
            return explicit
        st = manager.status()
        return st.get("port") or default_serve_port

    def accounting_error(exc: Exception) -> JSONResponse:
        code = (
            "accounting_outbox_failed"
            if isinstance(exc, AccountingOutboxError)
            else "accounting_prepare_failed"
        )
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "code": code,
                "enginePreserved": True,
            },
        )

    # ---- daemon self-health (never gated; always answers if the daemon is up) ----

    # ---- model library (download / browse / select) ----
    _HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    _apply_hub_mirror()  # boot
    _MODEL_DIRS = [
        _HF_CACHE_DIR,
        os.path.join(os.path.expanduser("~"), ".freetoken", "models"),
        os.path.join(os.path.expanduser("~"), "models"),
    ]

    @app.get("/models", dependencies=auth)
    async def list_models():
        """Scan the machine for locally-available models (HF cache + model dirs).
        Returns [{id, name, path, sizeBytes, files, family}...]."""
        def _scan() -> list[dict]:
            out = []
            seen = set()
            for root_dir in _MODEL_DIRS:
                if not os.path.isdir(root_dir):
                    continue
                # HF hub cache: models--org--name -> org/name
                if os.path.basename(root_dir) == "hub":
                    try:
                        for entry in os.listdir(root_dir):
                            if entry.startswith("models--"):
                                repo = entry[len("models--"):].replace("--", "/")
                                snap = os.path.join(root_dir, entry, "snapshots")
                                if os.path.isdir(snap):
                                    full = repo.replace("/", "_")
                                    out.append({
                                        "id": repo,
                                        "name": repo.split("/")[-1],
                                        "path": os.path.join(root_dir, entry),
                                        "sizeBytes": _dir_size(os.path.join(root_dir, entry)),
                                        "source": "hf-cache",
                                    })
                            seen.add(root_dir)
                    except OSError:
                        pass
                else:
                    try:
                        for entry in os.listdir(root_dir):
                            p2 = os.path.join(root_dir, entry)
                            if os.path.isdir(p2) and os.path.exists(os.path.join(p2, "config.json")):
                                out.append({
                                    "id": entry,
                                    "name": entry,
                                    "path": p2,
                                    "sizeBytes": _dir_size(p2),
                                    "source": "local",
                                })
                    except OSError:
                        pass
            # default storage dir (also download target)
            mdir = _get_model_dir()
            if mdir and os.path.isdir(mdir):
                try:
                    out.extend(_collect_models_in(mdir))
                except Exception:
                    pass
            # user-added native dirs
            for cdir in _load_custom_dirs():
                try:
                    out.extend(_collect_models_in(cdir))
                except Exception:
                    pass
            # dedupe by id (prefer first / hf-cache)
            uniq = {}
            for m in out:
                if m["id"] not in uniq:
                    uniq[m["id"]] = m
            return sorted(uniq.values(), key=lambda m: m["name"].lower())
        return await run(proxy_pool, _scan)

    @app.get("/models/search", dependencies=auth)
    async def search_models(q: str = "", source: str = "hf", limit: int = 20):
        """Search the HF (or ModelScope) hub for downloadable models.
        Returns [{id, title, description, downloads, likes, tags, source}]."""
        def _search_hf() -> list[dict]:
            try:
                _apply_hub_mirror()  # 尊重用户镜像设置
                from huggingface_hub import HfApi
                import signal as _sig
                # Internal timeout guard (HfApi may hang on slow network)
                class _TimeoutError(Exception): pass
                _result = [None]
                _exc = [None]
                def _do_search():
                    try:
                        api = HfApi()
                        _result[0] = api.list_models(search=q or None, limit=min(limit, 10), sort="downloads")
                    except Exception as e:
                        _exc[0] = e
                t = __import__("threading").Thread(target=_do_search, daemon=True)
                t.start()
                t.join(timeout=9)
                import os as _oe
                _ep = _oe.environ.get("HF_ENDPOINT", "https://huggingface.co")
                if t.is_alive():
                    return [{"error": "搜索超时（端点 " + _ep + "）——请检查该镜像站点可达性"}]
                if _exc[0]:
                    return [{"error": "[" + _ep + "] " + str(_exc[0])}]
                models = _result[0]
                out = []
                for m in models:
                    mid = m.modelId
                    lib = m.library_name or ""
                    tags = list(getattr(m, "tags", []) or [])
                    if not _is_supported_model_id(mid, library_name=lib, tags=tags):
                        continue
                    out.append({
                        "id": mid,
                        "title": mid,
                        "description": (m.pipeline_tag or "") + " " + lib,
                        "downloads": getattr(m, "downloads", 0) or 0,
                        "likes": getattr(m, "likes", 0) or 0,
                        "tags": tags[:6],
                        "source": "hf",
                    })
                return out
            except Exception as exc:
                return [{"error": str(exc)}]

        def _search_ms() -> list[dict]:
            """ModelScope 关键词搜索：直接打公开 REST（dolphin），绕开 HubApi 版本差异。"""
            try:
                import httpx as _hx
                r = _hx.put(
                    "https://modelscope.cn/api/v1/dolphin/models",
                    json={"Name": q or "", "PageNumber": 1, "PageSize": max(1, min(limit, 30)), "SortBy": "Default"},
                    timeout=12,
                )
                data = r.json()
                models = (((data or {}).get("Data") or {}).get("Model") or {}).get("Models") or []
                res = []
                for m in models:
                    owner = m.get("Path") or m.get("Owner") or ""
                    name = m.get("Name") or ""
                    mid = f"{owner}/{name}" if owner else name
                    if not mid:
                        continue
                    if not _is_supported_model_id(mid):
                        continue
                    res.append({
                        "id": mid,
                        "title": mid,
                        "description": (m.get("ChineseName") or m.get("Description") or "")[:80],
                        "downloads": int(m.get("Downloads") or 0),
                        "likes": int(m.get("Stars") or 0),
                        "tags": [t for t in [m.get("TaskType"), m.get("PublishType")] if t][:6],
                        "source": "ms",
                    })
                return res
            except Exception as exc:
                return [{"error": str(exc)}]

        if source == "hf":
            return {"results": await run(proxy_pool, _search_hf)}
        if source == "modelscope":
            return {"results": await run(proxy_pool, _search_ms)}
        return {"results": []}


    _RECOMMEND_POOL = [
        {"id": "Qwen/Qwen3-Coder-30B-A3B-Instruct", "label": "30B-A3B", "moe": True, "coding": True, "desc": "Qwen3 编码旗舰 MoE，agent 工具调用强"},
        {"id": "openai/gpt-oss-20b", "label": "20B-A3.6B", "moe": True, "coding": True, "nvidia": True, "desc": "MXFP4 原生量化，NVIDIA 格式优化"},
        {"id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "label": "16B-A2.4B", "moe": True, "coding": True, "desc": "DeepSeek 编码 MoE，轻量高效"},
        {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "label": "32B", "moe": False, "coding": True, "quants": ["fp8"], "desc": "稠密编码强者，有官方 FP8 版"},
        {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "label": "7B", "moe": False, "coding": True, "quants": ["fp8", "q4"], "desc": "轻量稠密编码，单卡友好"},
        {"id": "mistralai/Codestral-22B-v0.1", "label": "22B", "moe": False, "coding": True, "quants": ["q4"], "desc": "Mistral 代码专精"},
        {"id": "zai-org/GLM-4.5-Air", "label": "106B-A12B", "moe": True, "coding": False, "desc": "GLM agent 旗舰 MoE，重载场景"},
        {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "label": "30B-A3B", "moe": True, "coding": False, "quants": ["fp8"], "desc": "通用 MoE，混合思考"},
    ]


    @app.get("/models/recommend", dependencies=auth)
    async def models_recommend(limit: int = 8):
        """Coding-agent-first online recommendations, mixing MoE and dense."""
        pool = [dict(x) for x in _RECOMMEND_POOL]
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            def _meta(pid, sink):
                try:
                    info = api.model_info(pid)
                    sink["downloads"] = getattr(info, "downloads", 0) or 0
                    sink["likes"] = getattr(info, "likes", 0) or 0
                except Exception:
                    pass
            import threading as _th
            _ts = []
            for x in pool:
                t = _th.Thread(target=_meta, args=(x["id"], x), daemon=True)
                t.start(); _ts.append(t)
            for t in _ts:
                t.join(timeout=6)
        except Exception:
            pass
        pool.sort(key=lambda x: (0 if x["coding"] else 1, 0 if x.get("nvidia") or x.get("quants") else 1, 0 if x["moe"] else 1, -(x.get("downloads") or 0)))
        items = []
        for x in pool[: max(1, min(limit, len(pool)))]:
            tags = (["Coding"] if x["coding"] else []) + (["MoE"] if x["moe"] else ["稠密"])
            items.append({
                "id": x["id"], "name": x["id"].split("/")[-1],
                "sizeLabel": x["label"], "desc": x["desc"], "tags": tags,
                "downloads": x.get("downloads"), "likes": x.get("likes"),
                "nvidia": bool(x.get("nvidia")) or bool(x.get("quants")),
            })
        return {"items": items}

    @app.get("/models/dirs", dependencies=auth)
    async def models_dirs():
        """User-added native model directories."""
        return {"custom": _load_custom_dirs()}

    class _DirBody(dict):
        pass

    @app.post("/models/dir/add", dependencies=auth)
    async def models_dir_add(body: dict):
        """Register a native folder; returns models recognized inside it."""
        raw = str(body.get("path", "")).strip().strip("\"")
        if not raw:
            raise HTTPException(status_code=400, detail="path is required")
        path = os.path.expanduser(raw)
        if not os.path.isdir(path):
            raise HTTPException(status_code=400, detail="directory does not exist: " + path)
        found = _collect_models_in(path)
        dirs = _load_custom_dirs()
        norm = os.path.normpath(path)
        if not any(os.path.normpath(d) == norm for d in dirs):
            dirs.append(norm)
            _save_custom_dirs(dirs)
        return {"added": True, "path": norm, "models": found, "count": len(found)}

    @app.post("/models/dir/remove", dependencies=auth)
    async def models_dir_remove(body: dict):
        raw = str(body.get("path", "")).strip()
        norm = os.path.normpath(os.path.expanduser(raw))
        dirs = [d for d in _load_custom_dirs() if os.path.normpath(d) != norm]
        _save_custom_dirs(dirs)
        return {"removed": True, "custom": dirs}

    @app.post("/models/browse-folder", dependencies=auth)
    async def models_browse_folder():
        """Open a NATIVE Windows folder picker (runs on the daemon host)."""
        import base64 as _b64
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "[System.Windows.Forms.Application]::EnableVisualStyles();"
            "$dlg = New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$dlg.Description = '选择模型所在文件夹';"
            "$dlg.ShowNewFolderButton = $false;"
            "$r = $dlg.ShowDialog((New-Object System.Windows.Forms.Form -Property @{TopMost=$true}));"
            "if ($r -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dlg.SelectedPath }"
        )
        enc = _b64.b64encode(ps.encode("utf-16-le")).decode("ascii")
        try:
            import subprocess as _sp
            proc = _sp.run(
                ["powershell", "-NoProfile", "-STA", "-EncodedCommand", enc],
                capture_output=True, text=True, timeout=300,
            )
            picked = (proc.stdout or "").strip().splitlines()
            picked = picked[-1].strip() if picked else ""
            return {"path": picked or None}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)[:200])

    @app.post("/engine/estimate-quants", dependencies=auth)
    async def engine_estimate_quants(body: dict):
        """Evaluate BF16/FP8/Q4/NVFP4/MXFP4 variants.
        backend=auto evaluates every mode (gpu/hybrid/igpu/cpu) and reports,
        per variant, WHICH modes can run it (backends list)."""
        model_path = str(body.get("model", "")).strip()
        backend_req = str(body.get("backend", "auto")).strip().lower() or "auto"
        backend_order = ["gpu", "hybrid", "igpu", "cpu"]
        backends = backend_order if backend_req in ("auto", "all") else [backend_req]
        args = body.get("args") or []
        if not model_path:
            raise HTTPException(status_code=400, detail="model is required")
        synthetic_b = None
        try:
            facts = _get_facts(model_path)
        except Exception:
            synthetic_b = _pool_params_b(model_path)
            if not synthetic_b:
                return {"variants": [], "anyFit": False,
                        "error": "model metadata unavailable (offline)"}
        vram, dram, gpu_name = _probe_mem()
        labels = {"bf16": "BF16", "fp8": "FP8", "q4": "Q4", "nvfp4": "NVFP4", "mxfp4": "MXFP4"}
        out = []
        for key in ("bf16", "fp8", "q4", "nvfp4", "mxfp4"):
            try:
                if synthetic_b:
                    c = _synth_core(synthetic_b, _QUANT_MULT(key))
                else:
                    c = _estimate_core(facts, _QUANT_MULT(key), args)
                if not synthetic_b and facts["repo_total"] > 0:
                    scaled = int(facts["repo_total"] * _QUANT_MULT(key) / 2.0)
                    c["total"] = max(c["total"], scaled)
                    c["b"] = min(c["b"], max(1024 ** 3, int(scaled * 0.85)))
                fit_any = False; best_be = None; best_note = ""; best_mode = None; fitted = []
                for be in backends:
                    f_b, n_b, m_b = _apply_rules(be, c, vram, dram, gpu_name)
                    if f_b:
                        fit_any = True; fitted.append(be)
                        if best_be is None or backend_order.index(be) < backend_order.index(best_be):
                            best_be, best_note, best_mode = be, n_b, m_b
                out.append({
                    "key": key, "label": labels[key],
                    "nvidia": key in ("fp8", "q4", "nvfp4", "mxfp4"),
                    "fit": bool(fit_any), "note": best_note, "mode": best_mode,
                    "backends": fitted,
                    "totalBytes": c["total"],
                })
            except Exception as exc:
                out.append({"key": key, "label": key.upper(), "nvidia": False,
                            "fit": False, "note": str(exc)[:120], "mode": None,
                            "backends": [], "totalBytes": 0})
        anyFit = any(v["fit"] for v in out)
        def _pick(modeFilter):
            for k in ("bf16", "fp8", "q4", "nvfp4", "mxfp4"):
                for v in out:
                    if v["key"] == k and v["fit"] and (v.get("mode") == modeFilter):
                        return k
            return None
        bestKey = _pick("full") or _pick("layered")
        nvidiaFit = any(v["fit"] and v["nvidia"] for v in out)
        _bv = None
        for _k in ("nvfp4", "mxfp4", "q4", "fp8", "bf16"):
            _cand = next((v for v in out if v["key"] == _k and v["fit"]), None)
            if _cand:
                _bv = _cand; break
        return {"variants": out, "anyFit": anyFit, "bestKey": bestKey,
                "bestLabel": labels.get(bestKey),
                "nvidiaFit": nvidiaFit,
                "bestBackends": (_bv or {}).get("backends", []),
                "gpuName": gpu_name}

    @app.get("/settings/model-dir", dependencies=auth)
    async def settings_get_model_dir():
        """Default model storage dir (also used as download target)."""
        st = _load_daemon_settings()
        return {"path": _get_model_dir(), "mirror": st.get("mirror", "")}

    @app.post("/settings/model-dir", dependencies=auth)
    async def settings_set_model_dir(body: dict):
        raw = str(body.get("path", "")).strip().strip("\"")
        path = os.path.normpath(os.path.expanduser(raw)) if raw else ""
        if path:
            if not os.path.isdir(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="cannot create dir: " + str(exc)[:120])
        _set_model_dir(path)
        d = _load_daemon_settings()
        if "mirror" in body:
            d["mirror"] = str(body.get("mirror", "") or "")
            _save_daemon_settings(d)
            _apply_hub_mirror()   # 立即生效，无需重启
        return {"path": path, "mirror": d.get("mirror", "")}

    @app.post("/models/resolve-quant", dependencies=auth)
    async def models_resolve_quant(body: dict):
        """Resolve a quant variant key (fp8/q4/nvfp4) to a real pre-quantized repo id.

        Fix: previous version only mapped fp8/q4 and silently fell back to BF16 on miss,
        making every non-BF16 selection look broken to the user. Now also covers NVFP4
        and probes ModelScope when the body.source / settings mirror prefer it.
        """
        base = str(body.get("model", "")).strip()
        key = str(body.get("key", "")).strip().lower()
        source = str(body.get("source", "")).strip().lower()
        if not base or not key or key == "bf16":
            return {"found": False, "id": None}

        # 候选命名规则：常见 Qwen/Llama/Mistral 官方预量化仓库。
        # FreeToken: NVFP4 走项目自有名 "<id>-NVFP4" 兜底，缺则尝试通用 suffix。
        tail_map = {
            "fp8":   ["-FP8", "-fp8", "-FP8-Dynamic", "-FP8-static"],
            "nvfp4": ["-NVFP4", "-nvfp4", "-NVFP4-A3B", "-FP4", "-fp4"],
            "q4":    ["-AWQ", "-GPTQ-Int4", "-int4", "-AWQ-4bit",
                      "-GGUF-Q4_K_M", "-GGUF-Q4_0", "-Q4_K_M-GGUF"],
        }
        cands = tail_map.get(key, [])
        if not cands:
            return {"found": False, "id": None, "key": key, "reason": "unsupported-key"}

        # 决定查哪个镜像（HF 优先，失败再回退 MS；MS 优先则反过来）
        probe_hf = source != "modelscope"
        probe_ms = source == "modelscope" or True  # 双源都探，命中即返回

        # ── 内部：检查探测到的仓库是否是「纯精度」还是「混合精度」 ──
        def _is_pure_quant_repo(repo_id: str, key: str, source: str) -> bool:
            """返回 True 当仓库的量化配置严格匹配 key（无其他精度混入）。

            关键修复：Qwen/Qwen3.8-27B-NVFP4 实测是 FP8 + NVFP4 混合，名称里写 NVFP4 但
            实际不是纯 NVFP4，会让 dev tree 引擎加载时报 'Float8 promotion not supported'。
            """
            try:
                import json as _j
                import urllib.request as _ur
                if source == "hf":
                    # 拉 config.json 探测：HF 树内直读
                    raw_paths = [
                        f"https://huggingface.co/{repo_id}/resolve/main/config.json",
                        f"https://huggingface.co/{repo_id}/resolve/main/hf_quant_config.json",
                    ]
                else:
                    raw_paths = [
                        f"https://modelscope.cn/{repo_id}/resolve/main/config.json",
                        f"https://modelscope.cn/{repo_id}/resolve/main/hf_quant_config.json",
                    ]
                merged = ""
                for p in raw_paths:
                    try:
                        with _ur.urlopen(p, timeout=6) as r:
                            merged += r.read().decode("utf-8", "replace")
                    except Exception:
                        continue
                if not merged:
                    return True  # 拿不到就放行（保守）
                # 任一路径解析出 dict
                import re as _re
                cfg = None
                for blob in _re.split(r"\}\s*\{", merged):
                    try:
                        cfg = _j.loads(blob + "}" if blob.count("{")>blob.count("}") else blob)
                        if isinstance(cfg, dict): break
                    except Exception:
                        continue
                if not isinstance(cfg, dict):
                    return True
                blob = _j.dumps(cfg).lower()
                if key == "nvfp4":
                    has_nvfp4 = "nvfp4" in blob
                    has_fp8   = '"num_bits": 8' in blob or '"num_bits":8' in blob or "float8" in blob
                    has_int4  = "int4" in blob or ('"num_bits": 4' in blob) or ('"num_bits":4' in blob)
                    return has_nvfp4 and not has_fp8
                if key == "fp8":
                    has_fp8  = "float8" in blob or "fp8" in blob
                    has_nvfp4 = "nvfp4" in blob
                    has_bf16_only = "bf16" in blob and not has_fp8 and not has_nvfp4
                    return has_fp8 and not has_nvfp4
                if key == "q4":
                    has_q4   = "awq" in blob or "gptq" in blob or "int4" in blob
                    has_fp8  = "float8" in blob or "fp8" in blob
                    has_nvfp4 = "nvfp4" in blob
                    return has_q4 and not has_fp8 and not has_nvfp4
            except Exception:
                return True
            return True

        # ── 探测源顺序：HF → MS（按 source 偏好调序）──
        hits: list[dict] = []

        if probe_hf:
            try:
                from huggingface_hub import HfApi as _HfApi
                api_c = _HfApi()
                for tail in cands:
                    cid = base + tail
                    try:
                        api_c.model_info(cid, timeout=6)
                        hits.append({"id": cid, "source": "hf"})
                    except TypeError:
                        try:
                            api_c.model_info(cid)
                            hits.append({"id": cid, "source": "hf"})
                        except Exception:
                            pass
                    except Exception:
                        continue
            except Exception:
                pass

        if probe_ms:
            try:
                from modelscope import HubApi as _MSApi
                ms = _MSApi()
                ms_tails = {
                    "fp8":   ["-fp8", "-FP8"],
                    "nvfp4": ["-nvfp4", "-NVFP4"],
                    "q4":    ["-int4", "-AWQ", "-GPTQ", "-q4"],
                }.get(key, cands)
                for tail in ms_tails:
                    cid = base + tail
                    try:
                        info = ms.get_model(cid)
                        if info:
                            hits.append({"id": cid, "source": "modelscope"})
                    except Exception:
                        continue
            except Exception:
                pass

        # ── 优先纯量化仓库：unsloth 提供的大多是纯量化（NVFP4/Q4）──
        # 二次探测：把 unsloth、PrithivMLmods、NV-community 等社区仓库也纳入候选
        community_tails = {
            "nvfp4": ["-NVFP4", "-nvfp4"],
            "fp8":   ["-FP8", "-fp8"],
            "q4":    ["-AWQ", "-GPTQ", "-Q4", "-Int4", "-int4"],
        }.get(key, [])
        community_owners = ["unsloth", "PrithivMLmods", "nv-community", "TheBloke", "tngtech",
                            "Jackrong", "cyankiwi", "lmstudio-community", "Eco-Tech", "merkyor",
                            "prithivMLmods", "selimaktas", "mudler", "lovedheart"]
        if probe_hf or probe_ms:
            try:
                for owner in community_owners:
                    for tail in community_tails:
                        cid = f"{owner}/{base.split('/')[-1]}{tail}"
                        # 简化探测：尝试用 MS 列出（MS 同时索引 HF）
                        if probe_ms:
                            try:
                                from modelscope import HubApi as _MSApi2
                                ms2 = _MSApi2()
                                if ms2.get_model(cid):
                                    hits.append({"id": cid, "source": "modelscope"})
                                    break
                            except Exception:
                                pass
            except Exception:
                pass

        # ── 在所有命中里挑第一个「纯精度」仓库，否则挑第一个命中 ──
        if not hits:
            return {"found": False, "id": None, "key": key, "tried": cands}
        for h in hits:
            if _is_pure_quant_repo(h["id"], key, h["source"]):
                pure = h
                # 也附带 mixed 候选让前端知情
                mixed = [x for x in hits if x["id"] != h["id"]]
                return {
                    "found": True,
                    "id": pure["id"],
                    "source": pure["source"],
                    "pure": True,
                    "alternatives": mixed[:3],
                }
        # 全部都不是纯精度（API 探测不到 / 拿不到 config.json）→ 返回第一个，但加 pure:None
        h = hits[0]
        return {
            "found": True,
            "id": h["id"],
            "source": h["source"],
            "pure": None,
            "alternatives": [x for x in hits[1:] if x["id"] != h["id"]][:3],
        }

    @app.post("/models/download/status", dependencies=auth)
    async def download_status():
        """Progress of in-flight model downloads (SSE-friendly poll).
        Returns {active: [{id, done, total, bytesDone, bytesTotal, status}...]}."""
        return {"active": list(_DOWNLOAD_JOBS.values())}

    @app.post("/models/download", dependencies=auth)
    async def download_model(body: ModelDownloadBody):
        """Start a model download in the background. Returns the job id."""
        import uuid as _uuid
        job_id = f"dl-{_uuid.uuid4().hex[:8]}"
        if body.id in _DOWNLOAD_JOBS or any(j["id"] == body.id for j in _DOWNLOAD_JOBS.values()):
            return {"jobId": job_id, "status": "already-active", "id": body.id}
        _DOWNLOAD_JOBS[body.id] = {
            "id": body.id,
            "source": body.source,
            "done": 0, "total": 0,
            "bytesDone": 0, "bytesTotal": 0,
            "status": "queued", "jobId": job_id,
            "message": "Queued",
        }
        def _download() -> None:
            job = _DOWNLOAD_JOBS[body.id]
            job["status"] = "downloading"
            job["message"] = "正在连接下载源…"
            import threading as _td, time as _ti
            def _stall_watch():
                last, idle = -1, 0
                while True:
                    _ti.sleep(5)
                    if _DOWNLOAD_JOBS.get(body.id) is not job: return
                    if job.get("status") != "downloading": return
                    cur = job.get("bytesDone", 0)
                    idle = 0 if cur != last else idle + 5
                    last = cur
                    if idle >= 90 and job.get("bytesTotal", 0) == 0:
                        job["status"] = "failed"
                        job["message"] = "连接下载源超过 90 秒无响应——请稍后重试或切换镜像"
                        return
            _td.Thread(target=_stall_watch, daemon=True).start()
            try:
                if body.source == "hf":
                    # 弱网保命：连接/元数据 15s 级超时，绝不无限挂起
                    import os as _ot
                    _ot.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "15")
                    _ot.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
                    from huggingface_hub import snapshot_download
                    from tqdm import tqdm, TqdmWarning
                    import warnings
                    warnings.simplefilter("ignore", TqdmWarning)
                    # 真实进度回调：把 tqdm 的字节进度写回 job
                    def _mk_bar(job_ref):
                        class _Bar(tqdm):
                            def update(self, n=1):
                                super().update(n)
                                job_ref["bytesDone"] = int(getattr(self, "n", 0) or 0)
                                job_ref["bytesTotal"] = int(getattr(self, "total", 0) or 0)
                                job_ref["message"] = "下载中 {:.1f}/{:.1f} MB".format(
                                    job_ref["bytesDone"] / 1048576, job_ref["bytesTotal"] / 1048576)
                        return _Bar
                    mirror_sel = str(_load_daemon_settings().get("mirror", "") or "")
                    mdir = _get_model_dir()
                    local_target = os.path.join(mdir, body.id.split("/")[-1]) if mdir else None
                    endpoints = []
                    if mirror_sel:
                        endpoints = [("hf-mirror", "https://hf-mirror.com"), ("huggingface.co", "https://huggingface.co")]
                    else:
                        endpoints = [("huggingface.co", "https://huggingface.co"), ("hf-mirror", "https://hf-mirror.com")]
                    import os as _os
                    last_err = None
                    for ep_name, ep_url in endpoints:
                        job["message"] = "正在从 {} 下载…".format(ep_name)
                        try:
                            _os.environ["HF_ENDPOINT"] = ep_url
                            if local_target:
                                _os.makedirs(local_target, exist_ok=True)
                                target = snapshot_download(body.id, tqdm_class=_mk_bar(job), local_dir=local_target)
                            else:
                                target = snapshot_download(body.id, tqdm_class=_mk_bar(job))
                            job["status"] = "done"
                            job["message"] = "已下载到 {}（{}）".format(target, ep_name)
                            job["path"] = target
                            return
                        except Exception as exc2:
                            last_err = exc2
                            job["message"] = "{} 失败，切换镜像重试…".format(ep_name)
                    job["status"] = "failed"
                    job["message"] = "下载失败: {}".format(str(last_err)[:220])
                elif body.source == "modelscope":
                    from modelscope import snapshot_download as ms_download
                    mdir = _get_model_dir()
                    _tgt = os.path.join(mdir, body.id.split("/")[-1]) if mdir else None
                    if _tgt:
                        # 预取仓库总大小，进度条才有分母
                        try:
                            import httpx as _hx2
                            _rr = _hx2.get(
                                "https://modelscope.cn/api/v1/models/" + body.id + "/repo/files",
                                params={"Recursive": "true"}, timeout=10)
                            _dd = _rr.json()
                            def _sum_files(nodes):
                                s = 0
                                for nd in nodes or []:
                                    if str(nd.get("Type")) == "tree":
                                        s += _sum_files(nd.get("Files") or [])
                                    else:
                                        s += int(nd.get("Size") or 0)
                                return s
                            job["bytesTotal"] = _sum_files((_dd.get("Data") or {}).get("Files"))
                        except Exception:
                            pass
                        import threading as _td2, time as _ti2
                        def _ms_progress():
                            prev = -1
                            while True:
                                _ti2.sleep(3)
                                if _DOWNLOAD_JOBS.get(body.id) is not job: return
                                if job.get("status") not in ("downloading", "queued"): return
                                try:
                                    s = 0
                                    for _rt, _df, _fs in os.walk(_tgt):
                                        for f in _fs: s += os.path.getsize(os.path.join(_rt, f))
                                    if s != prev:
                                        job["bytesDone"] = s
                                        job["message"] = "下载中 {:.1f} MB（ModelScope）".format(s / 1048576)
                                        prev = s
                                except Exception: pass
                        _td2.Thread(target=_ms_progress, daemon=True).start()
                    target = ms_download(body.id, local_dir=_tgt) if _tgt else ms_download(body.id)
                    job["status"] = "done"
                    job["message"] = "已下载到 {}".format(target)
                    job["path"] = target
            except Exception as exc:
                job["status"] = "failed"
                job["message"] = str(exc)[:300]
        threading = __import__("threading")
        t = threading.Thread(target=_download, daemon=True)
        t.start()
        return {"jobId": job_id, "status": "started", "id": body.id}

    _DOWNLOAD_JOBS: dict = {}

    # ---- GPU probe child (lazy-started torch monitor) ----

    _GPU_PROBE: dict = {"proc": None, "last": {}}

    def _ensure_gpu_probe():
        import threading
        import subprocess as _sp
        import sys as _sys
        if _GPU_PROBE["proc"] is not None and _GPU_PROBE["proc"].poll() is None:
            return
        code = (
            "import torch,time,json\n"
            "while True:\n"
            "    try:\n"
            "        f,t=torch.cuda.mem_get_info(0)\n"
            "        n=torch.cuda.get_device_name(0)\n"
            "        print(json.dumps({'free':int(f),'total':int(t),'name':n}),flush=True)\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'err':str(e)[:120]}),flush=True)\n"
            "    time.sleep(2)\n"
        )
        proc = _sp.Popen(
            [_sys.executable, "-u", "-c", code],
            stdout=_sp.PIPE, stderr=_sp.DEVNULL, text=True,
        )
        _GPU_PROBE["proc"] = proc

        def _reader():
            import json as _json
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    _GPU_PROBE["last"] = _json.loads(line)
                except Exception:
                    pass

        threading.Thread(target=_reader, daemon=True).start()

    def _read_panel_asset(name: str) -> str | None:
        import os
        p = os.path.join(os.path.dirname(__file__), name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
        return None

    @app.get("/panel", include_in_schema=False)
    async def control_panel():
        """Serve the FreeToken control panel (static HTML)."""
        from fastapi.responses import HTMLResponse
        html = _read_panel_asset("panel.html")
        if html is not None:
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})
        return HTMLResponse("<h1>Panel not found</h1>", status_code=404)

    @app.get("/panel.js", include_in_schema=False)
    async def control_panel_js():
        """Serve the control panel script."""
        from fastapi.responses import Response
        js = _read_panel_asset("panel.js")
        if js is not None:
            return Response(js, media_type="application/javascript", headers={"Cache-Control": "no-store"})
        return Response("console.error('panel.js missing');", media_type="application/javascript")

    # ---- Same-origin /v1 reverse proxy -----------------------------------------
    # WebView2 blocks cross-origin calls to the engine port (no CORS there), so the
    # panel must reach chat/models through THIS origin. Streams SSE transparently.
    import urllib.request as _urllib_req
    from fastapi.responses import Response as _Resp, StreamingResponse as _Stream

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"], include_in_schema=False)
    async def v1_proxy(path: str, request: Request):
      """Forward a request to the engine's /v1/<path> loopback and return its response.

      Uses urllib + a thread executor instead of httpx. httpx hangs forever on
      the engine's /v1/chat/completions response because the engine streams a
      chunked SSE-style body whose final empty chunk arrives only after the
      response is read; with httpx's connection pool the read block never
      completes. urllib's HTTPResponse handles chunked termination cleanly and
      also works for plain JSON responses and SSE streams.
      """
      try:
        st = manager.status()
        port = st.get("port") or default_serve_port
        if not st.get("running"):
            return _Resp('{"error":{"message":"engine not running"}}',
                         status_code=503, media_type="application/json")
        body = await request.body()
        # Forward the original request body verbatim, but only Content-Type
        # (drop everything else - they belong to the inbound hop and uvicorn
        # recomputes Host / Content-Length / Connection itself).
        fwd_headers = {"Content-Type": request.headers.get("content-type",
                                                            "application/json")}
        # Preserve the caller's Accept header so SSE clients still get SSE.
        accept_hdr = request.headers.get("accept")
        if accept_hdr:
            fwd_headers["Accept"] = accept_hdr
        url = "http://127.0.0.1:%d/v1/%s" % (port, path)
        if request.query_params:
            url += "?" + request.query_params

        def _do_request():
            req = _urllib_req.Request(url, data=body if body else None,
                                      headers=fwd_headers,
                                      method=request.method)
            # The engine's uvicorn may take a while to tokenize + generate;
            # urllib's default is infinite on read, which is what we want for SSE.
            return _urllib_req.urlopen(req, timeout=600)

        try:
            up = await run(proxy_pool, _do_request)
        except Exception as exc:
            return _Resp('{"error":{"message":"engine unreachable: %s"}}' % exc,
                         status_code=502, media_type="application/json")

        # Filter hop-by-hop headers
        hop_by_hop = {"content-length", "transfer-encoding", "connection",
                      "keep-alive", "proxy-authenticate", "proxy-authorization",
                      "te", "trailers", "upgrade"}
        out_headers = {k: v for k, v in up.headers.items()
                       if k.lower() not in hop_by_hop}

        # Detect streaming: if Content-Type is text/event-stream, pipe raw
        # chunks as the response body.
        ct = up.headers.get("content-type", "")
        if "text/event-stream" in ct.lower():
            def _gen():
                try:
                    while True:
                        chunk = up.read(4096)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    try:
                        up.close()
                    except Exception:
                        pass
            return _Stream(_gen(), status_code=up.status, media_type=ct,
                           headers=out_headers)
        # Non-streaming: read everything and return as a single Response.
        try:
            content = up.read()
        finally:
            try:
                up.close()
            except Exception:
                pass
        return _Resp(content, status_code=up.status, media_type=ct,
                     headers=out_headers)
      except Exception as exc:
        return _Resp('{"error":{"message":"proxy failure: %s"}}' % exc,
                     status_code=502, media_type="application/json")


    @app.get("/panel.css", include_in_schema=False)
    async def control_panel_css():
        """Serve the control panel stylesheet."""
        from fastapi.responses import Response
        css = _read_panel_asset("panel.css")
        if css is not None:
            return Response(css, media_type="text/css", headers={"Cache-Control": "no-store"})
        return Response("/* panel.css missing */", media_type="text/css")

    @app.get("/panel-harness.css", include_in_schema=False)
    async def control_panel_harness_css():
        """Serve the DeepSeek-Harness alignment stylesheet."""
        from fastapi.responses import Response
        css = _read_panel_asset("panel-harness.css")
        if css is not None:
            return Response(css, media_type="text/css", headers={"Cache-Control": "no-store"})
        return Response("/* panel-harness.css missing */", media_type="text/css")

    @app.get("/health")
    async def health():
        st = manager.status()
        return {
            "status": "ok",
            "version": DAEMON_VERSION,
            "uptimeS": int(wall_now() - started_wall) if started_wall else 0,
            "engineRunning": bool(st.get("running")),
        }

    # ---- system info (hardware monitors; public like /health) ----

    _sysinfo_cache: dict = {"at": 0.0, "doc": {}}

    @app.get("/sysinfo", include_in_schema=False)
    async def sysinfo():
        """VRAM/RAM totals + utilization for the panel monitors (2s cache)."""
        import time as _time
        now = _time.monotonic()
        if now - _sysinfo_cache["at"] < 2.0 and _sysinfo_cache["doc"]:
            return _sysinfo_cache["doc"]
        doc: dict = {
            "vramUsed": 0, "vramTotal": 0, "gpuUtil": 0, "gpuName": "",
            "memUsed": 0, "memTotal": 0, "cpuUtil": 0,
        }
        # GPU stats: persistent torch probe child (CUDA runtime, no NVML dependency).
        # Falls back to registry for the total when torch is unavailable.
        try:
            _ensure_gpu_probe()
        except Exception:
            pass
        gp = _GPU_PROBE.get("last") or {}
        if gp.get("total"):
            doc["vramTotal"] = int(gp["total"])
            doc["vramUsed"] = max(0, int(gp["total"]) - int(gp.get("free", 0)))
            doc["gpuName"] = gp.get("name") or ""
        else:
            try:
                import winreg as _wr
                k = _wr.OpenKey(
                    _wr.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000",
                )
                raw, _t = _wr.QueryValueEx(k, "HardwareInformation.qwMemorySize")
                _wr.CloseKey(k)
                if isinstance(raw, bytes):
                    raw = int.from_bytes(raw[:8], "little")
                doc["vramTotal"] = int(raw)
            except Exception:
                pass
        # RAM via ctypes GlobalMemoryStatusEx (no psutil dependency)
        try:
            import ctypes as _ct
            class _MEMORYSTATUSEX(_ct.Structure):
                _fields_ = [
                    ("dwLength", _ct.c_ulong), ("dwMemoryLoad", _ct.c_ulong),
                    ("ullTotalPhys", _ct.c_ulonglong), ("ullAvailPhys", _ct.c_ulonglong),
                    ("ullTotalPageFile", _ct.c_ulonglong), ("ullAvailPageFile", _ct.c_ulonglong),
                    ("ullTotalVirtual", _ct.c_ulonglong), ("ullAvailVirtual", _ct.c_ulonglong),
                    ("ullAvailExtendedVirtual", _ct.c_ulonglong),
                ]
            st = _MEMORYSTATUSEX()
            st.dwLength = _ct.sizeof(_MEMORYSTATUSEX)
            if _ct.windll.kernel32.GlobalMemoryStatusEx(_ct.byref(st)):
                doc["memTotal"] = int(st.ullTotalPhys)
                doc["memUsed"] = int(st.ullTotalPhys - st.ullAvailPhys)
                doc["cpuUtil"] = float(st.dwMemoryLoad)
        except Exception:
            pass
        _sysinfo_cache["at"] = now
        _sysinfo_cache["doc"] = doc
        return doc

    # ---- engine lifecycle ----

    @app.post("/engine/start", dependencies=auth)
    async def engine_start(body: StartBody):
        port = resolve_port(body.port)
        try:
            return await run(lifecycle_pool, manager.start, body.model, port, _normalize_engine_args(body.model, body.args))
        except Conflict as exc:
            st = manager.status()
            return JSONResponse(
                status_code=409,
                content={
                    "error": str(exc),
                    "code": "serve_conflict",
                    "currentModel": st.get("model"),
                    "currentPort": st.get("port"),
                },
            )
        except Exception as exc:  # noqa: BLE001 — never propagate a 500-as-crash
            raise HTTPException(status_code=500, detail=f"start failed: {exc}")

    @app.post("/engine/stop", dependencies=auth)
    async def engine_stop(body: StopBody | None = None):
        try:
            return await run(lifecycle_pool, manager.stop, None, bool(body and body.force))
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)

    @app.post("/shutdown", dependencies=auth)
    async def shutdown_daemon(request: Request, body: StopBody | None = None):
        # Tray "Stop daemon" stops everything: stop the engine FIRST so the default detach-on-exit can't
        # leave the ~18GB serve orphaned, THEN bring the daemon down. We reply before uvicorn
        # actually stops (it notices should_exit within ~0.1s) so the client still gets a clean 200.
        try:
            stopped = await run(lifecycle_pool, manager.shutdown, None, bool(body and body.force))
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)
        req = getattr(request.app.state, "request_shutdown", None)
        if req is not None:
            req()
        return {
            "stopping": True,
            "already": stopped.get("already", False),
            "accounting": stopped.get("accounting"),
        }

    @app.post("/engine/switch", dependencies=auth)
    async def engine_switch(body: SwitchBody):
        port = resolve_port(body.port)
        try:
            return await run(
                lifecycle_pool,
                manager.switch,
                body.model,
                port,
                _normalize_engine_args(body.model, body.args),
                body.force,
            )
        except (AccountingPrepareError, AccountingOutboxError) as exc:
            return accounting_error(exc)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"switch failed: {exc}")

    # ---- durable accounting outbox ----

    @app.get("/accounting/pending", dependencies=auth)
    async def accounting_pending():
        try:
            receipts = await run(lifecycle_pool, manager.pending_accounting)
        except AccountingOutboxError as exc:
            return accounting_error(exc)
        return {"receipts": receipts}

    @app.post("/accounting/ack", dependencies=auth)
    async def accounting_ack(body: AccountingAckBody):
        try:
            return await run(lifecycle_pool, manager.ack_accounting, body.receiptId)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except AccountingOutboxError as exc:
            return accounting_error(exc)

    @app.get("/engine/status", dependencies=auth)
    async def engine_status():
        return manager.status()

    @app.get("/engine/metrics", dependencies=auth)
    async def engine_metrics():
        pid = manager.current_pid()
        return await run(proxy_pool, footprint_fn, pid)

    @app.get("/engine/health", dependencies=auth)
    async def engine_health():
        st = manager.status()
        if not st.get("running"):
            return {"reachable": False, "running": False, "daemon": "up", **_engine_summary(st)}
        port = st.get("port") or default_serve_port
        doc = await run(proxy_pool, probe.health, port)
        # The serve's own health fields (status/model/uptimeS/progress) are authoritative for
        # "how is the model doing?"; the daemon only layers on what only it knows, never clobbering
        # the serve's values.
        doc["running"] = True
        doc["daemon"] = "up"
        doc.setdefault("port", st.get("port"))
        doc.setdefault("pid", st.get("pid"))
        doc.setdefault("lastExitCode", st.get("lastExitCode"))
        return doc

    @app.get("/engine/stats", dependencies=auth)
    async def engine_stats():
        st = manager.status()
        if not st.get("running"):
            return {"reachable": False, "running": False}
        port = st.get("port") or default_serve_port
        doc = await run(proxy_pool, probe.stats, port)
        manager.observe_accounting(doc)
        return doc

    @app.get("/engine/logs", dependencies=auth)
    async def engine_logs(request: Request, since: int = 0):
        return _log_stream(request, ring, since)

    @app.get("/engine/logs/snapshot", dependencies=auth)
    async def engine_logs_snapshot(limit: int = 400):
        """JSON 尾部快照——面板日志页轮询用；SSE 流式版仍走 /engine/logs。"""
        with ring._lock:
            recs = list(ring._buf)[-max(10, min(limit, 2000)):]
        return {"lines": [str(r.get("text", "")) for r in recs],
                "kinds": [str(r.get("kind", "line")) for r in recs],
                "seq": [int(r.get("seq", 0)) for r in recs]}

    # ---- checkpoint (phase 3; optional) ----

    if checkpoints is not None:

        @app.post("/checkpoint/start", dependencies=auth)
        async def checkpoint_start(body: CheckpointBody):
            # GPU exclusivity: a convert needs the GPU, so stop any serve first.
            await run(lifecycle_pool, manager.stop)
            try:
                return await run(lifecycle_pool, checkpoints.start, body.id, list(body.args))
            except Conflict as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"checkpoint start failed: {exc}")

        @app.post("/checkpoint/cancel", dependencies=auth)
        async def checkpoint_cancel(body: CancelBody):
            return await run(lifecycle_pool, checkpoints.cancel, body.id)

        @app.get("/checkpoint/status", dependencies=auth)
        async def checkpoint_status():
            return checkpoints.status()

    # ---- hardware bandwidth bench (hardware-adaptive config) ----

    # ---- config schema (desktop panel integration) ----
    @app.get("/engine/options", dependencies=auth)
    async def engine_options():
        """Return the full CLI-options schema for the desktop panel to render dynamically."""
        from freetoken.server.args import _build_option_schema
        try:
            schema = _build_option_schema()
            return {"options": schema}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/engine/estimate", dependencies=auth)
    async def engine_estimate(body: dict):
        """Estimate whether a model can run on this machine with the given options.
        Body fields: model (str), args (list[str]), backend (str).
        Returns a fit report with VRAM/DRAM breakdowns."""
        model = body.get("model", "")
        args = body.get("args", [])
        backend = body.get("backend", "gpu")
        if not model:
            raise HTTPException(status_code=400, detail="model field is required")
        try:
            result = _estimate_model_fit(model, backend, args)
            return {"model": model, "backend": backend, **result}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/bench/run", dependencies=auth)
    async def bench_run(body: BenchBody):
        # GPU exclusivity: the bench allocates transient device memory, so stop any serve first
        # (mirrors /checkpoint/start). Runs `ft bench bw` on the engine HOST (so the profile lands
        # where this daemon's serve reads it) and STREAMS progress back as SSE: `progress` events
        # per measured format, then a terminal `result` (the profile) or `error` event. `body.args`
        # is the raw arg list, so any `ft bench bw` flag (--dtype/--model/--threshold/...) passes
        # through. torch stays out of the daemon (child process), which also frees VRAM on exit.
        await run(lifecycle_pool, manager.stop)

        async def gen():
            env = {**os.environ, "FREETOKEN_BENCH_PROGRESS": "1"}
            argv = [sys.executable, "-m", "freetoken.cli", "bench", "bw", *body.args]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
                )
            except Exception as exc:  # noqa: BLE001
                yield _bench_sse("error", {"message": f"failed to spawn bench: {exc}"})
                return
            tail: collections.deque = collections.deque(maxlen=8)  # last non-progress lines (errors)
            out_path: str | None = None
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                prog = _parse_ftbench(line)
                if prog is not None:
                    yield _bench_sse("progress", prog)
                elif line.startswith("FTBENCH_OUT "):
                    out_path = line[len("FTBENCH_OUT "):]
                elif line:
                    tail.append(line)
            rc = await proc.wait()
            if rc != 0:
                yield _bench_sse("error", {"message": "\n".join(tail) or f"bench exited {rc}"})
                return
            # the file this run wrote (an older engine prints no FTBENCH_OUT: newest file, as before)
            prof = _read_bench_profile(out_path or _bench_profile_path(None))
            if prof is None:
                yield _bench_sse("error", {"message": "bench finished but no profile was written"})
            else:
                yield _bench_sse("result", prof)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/bench/profile", dependencies=auth)
    async def bench_profile():
        def read() -> dict | None:
            return _read_bench_profile(_bench_profile_path(serve_gpu_uuid()))

        def serve_gpu_uuid() -> str | None:
            # the running serve reports the full UUID of its card (/v1/stats gpus); a --gpu given as
            # a UUID prefix would not match the profile file name
            st = manager.status()
            if st.get("running"):
                try:
                    gpus = probe.stats(st.get("port") or default_serve_port).get("gpus") or []
                    if gpus and gpus[0].get("uuid"):
                        return gpus[0]["uuid"]
                except Exception:  # noqa: BLE001 -- the arg below is the fallback
                    pass
            return _serve_gpu_uuid(manager.serve_args())

        return await run(proxy_pool, read)

    @app.get("/engine/ready", dependencies=auth)
    async def engine_ready():
        """Whether the engine is fully loaded and ready to serve requests."""
        st = manager.status()
        if not st.get("running"):
            return {"ready": False, "error": "engine not running"}
        port = st.get("port", 1919)
        try:
            import json as _json
            import urllib.request as _ur
            _req = _ur.Request("http://127.0.0.1:{}/v1/models".format(port))
            with _ur.urlopen(_req, timeout=3.0) as _resp:
                if _resp.status == 200:
                    return {"ready": True, "port": port}
                _body = _json.loads(_resp.read().decode("utf-8", "replace"))
                if isinstance(_body, dict) and _body.get("error") == "model is still loading":
                    return {"ready": False, "error": "model is still loading"}
        except Exception:
            pass
        return {"ready": False, "error": "engine not ready"}

    @app.post("/models/cache/cleanup", dependencies=auth)
    async def models_cache_cleanup():
        """Remove HF cache directories that contain only config files (no weights).
        Returns the list of removed dirs and freed bytes."""
        import pathlib, shutil
        hub = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        if not hub.is_dir():
            return {"removed": [], "freed": 0}
        removed = []
        freed = 0
        for child in hub.iterdir():
            if not child.name.startswith("models--"):
                continue
            snap_dir = child / "snapshots"
            if not snap_dir.is_dir():
                continue
            has_weights = False
            for snap in snap_dir.iterdir():
                for f in snap.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".gguf"):
                        has_weights = True
                        break
                if has_weights:
                    break
            if not has_weights:
                size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
                shutil.rmtree(str(child), ignore_errors=True)
                removed.append(child.name)
                freed += size
        return {"removed": removed, "freed": freed}
    @staticmethod
    def _json_load(path: str) -> dict:
        import json as _j
        with open(path, encoding="utf-8") as _f:
            return _j.load(_f)

    # ---- in-app NVFP4 conversion (MXFP4/raw -> NVFP4) --------------------------
    # Runs checkpoint/quantize.quantize_to_nvfp4 in a daemon thread so the job
    # outlives any single panel request; progress lands in the job dict directly.
    _CONVERT_JOBS: dict = {}

    @app.post("/models/convert-nvfp4", dependencies=auth)
    async def models_convert_nvfp4(body: dict = None):
        """Start an in-app MXFP4/raw -> NVFP4 conversion.
        body: {path, out?}  -- out defaults to a <path>-NVFP4 sibling dir.
        Returns the job id; poll /models/convert-nvfp4/status for progress."""
        import uuid as _uuid, threading as _td
        src = (body or {}).get("path") or ""
        if not src or not os.path.isdir(src):
            raise HTTPException(status_code=400, detail=f"not a model dir: {src!r}")
        if any(j["src"] == src and j["status"] in ("queued", "running")
               for j in _CONVERT_JOBS.values()):
            cur = next(j for j in _CONVERT_JOBS.values()
                       if j["src"] == src and j["status"] in ("queued", "running"))
            return {"error": "conversion already active for this model", "job": cur}
        out = (body or {}).get("out")
        if not out:
            out = src.rstrip("/\\") + "-NVFP4"
        if os.path.isdir(out) and os.listdir(out):
            raise HTTPException(status_code=409, detail=f"output dir not empty: {out!r}")
        job_id = f"cv-{_uuid.uuid4().hex[:8]}"
        job = {"id": job_id, "src": src, "out": out, "status": "queued",
               "phase": "", "done": 0, "total": 0, "message": "排队中",
               "error": None, "startedAt": time.time()}
        _CONVERT_JOBS[job_id] = job

        def _progress(phase: str, done: int, total: int) -> None:
            job["phase"] = phase
            job["done"] = int(done)
            job["total"] = int(total)

        def _write_mtp_sidecar(inter_dir: str, ftw_dir: str) -> int:
            """Gather mtp.* tensors from the stage-1 safetensors into <ftw>/mtp.safetensors.
            The FTW dense pass skips them; the MTP loader reads the sidecar when the
            safetensors index is absent (models/qwen3_5_moe/mtp.py)."""
            import json as _json
            from safetensors import safe_open
            from safetensors.torch import save_file
            idx = _json.load(open(os.path.join(inter_dir, "model.safetensors.index.json"), encoding="utf-8"))
            wm = idx["weight_map"]
            mtp_keys = [k for k in wm if k.startswith("mtp.")]
            if not mtp_keys:
                return 0
            state = {}
            for fname in sorted(set(wm[k] for k in mtp_keys)):
                with safe_open(os.path.join(inter_dir, fname), framework="pt") as f:
                    for k in f.keys():
                        if k.startswith("mtp."):
                            state[k] = f.get_tensor(k)
            save_file(state, os.path.join(ftw_dir, "mtp.safetensors"))
            return len(state)

        def _run() -> None:
            inter = out + "-tmp"  # stage-1 safetensors; removed once the FTW is written
            job["status"] = "running"
            job["message"] = "阶段 1/2: 量化 (MXFP4 -> NVFP4)"
            try:
                import shutil as _sh
                import torch
                from freetoken.checkpoint.quantize import quantize_to_nvfp4
                torch.cuda.init()
                if os.path.isdir(inter):
                    _sh.rmtree(inter, ignore_errors=True)
                r = quantize_to_nvfp4(src, inter, device="cuda", progress=_progress)
                # Release stage-1 buffers before the bank-heavy FTW pass: the PyTorch CPU
                # allocator otherwise keeps ~20 GB of freed decode tensors committed and
                # the expert-host-bank allocation hits WinError 1455 (commit limit).
                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                # stage 2: FTW (engine-native fast-load format, ~20 s loads) in a
                # SUBPROCESS -- a fresh commit space for the ~20 GB expert host banks.
                job["message"] = "阶段 2/2: 打包 FTW (引擎原生格式)"
                _progress("ftw", 0, 0)
                import subprocess as _sp, sys as _sys
                import freetoken as _ft_pkg
                _py_root = os.path.dirname(os.path.dirname(os.path.abspath(_ft_pkg.__file__)))
                env = {**os.environ, "PYTHONPATH": _py_root,
                       "FREETOKEN_CONVERT_PROGRESS": "1"}
                cmd = [_sys.executable, "-m", "freetoken.checkpoint",
                       "--model", inter, "--out", out, "--moe-backend", "offload"]
                proc = _sp.Popen(cmd, env=env, stdout=_sp.PIPE,
                                 stderr=_sp.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
                tail: list = []
                for line in proc.stdout:
                    line = line.rstrip()
                    if line.startswith("FTCONVERT"):
                        parts = line.split()
                        if len(parts) >= 4 and parts[1] == "experts":
                            _progress("ftw-experts", parts[2], parts[3])
                        elif parts[1:2] == ["dense"]:
                            _progress("ftw-dense", 0, 0)
                    elif line:
                        tail.append(line)
                        tail = tail[-20:]
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError("FTW 打包失败: " + chr(10).join(tail[-10:])[:900])
                n_mtp = _write_mtp_sidecar(inter, out)
                _sh.rmtree(inter, ignore_errors=True)  # free the ~20 GB intermediate
                import json as _js
                with open(os.path.join(out, "freetoken_weight.json"), encoding="utf-8") as _f:
                    _idx = _js.load(_f)
                job.update(status="done", message="完成 (FTW, 加载 ~20 s)",
                           stats=r["stats"], shards=len(_idx.get("shards", [])),
                           mtpKeys=n_mtp, finishedAt=time.time())
            except SystemExit as e:
                job.update(status="error", error=str(e), message="失败",
                           finishedAt=time.time())
            except Exception as e:  # noqa: BLE001
                import traceback as _tb
                job.update(status="error", error=str(e),
                           message="失败: " + str(e)[:200],
                           traceback=_tb.format_exc()[-2000:],
                           finishedAt=time.time())

        _td.Thread(target=_run, daemon=True, name=f"ft-convert-{job_id}").start()
        return {"jobId": job_id, "out": out, "status": "queued"}

    @app.post("/models/convert-nvfp4/status", dependencies=auth)
    async def models_convert_nvfp4_status():
        """Progress of in-app conversions: {active: [{id, src, out, status, phase,
        done, total, message, error}...]} (finished jobs stay until the dict resets)."""
        return {"active": list(_CONVERT_JOBS.values())}

    @app.post("/models/local-quants", dependencies=auth)
    async def models_local_quants(body: dict = None):
        """Scan a local model directory for quantization files.
        Returns the detected quantization labels."""
        if not body or not body.get("path"):
            return {"quants": []}
        import pathlib as _pl, json as _json
        p = _pl.Path(body["path"])
        if not p.is_dir():
            return {"quants": []}
        quant_labels = set()
        hf_q = p / "hf_quant_config.json"
        if hf_q.is_file():
            try:
                d = _json.loads(hf_q.read_text())
                if "NVFP4" in str(d): quant_labels.add("NVFP4")
                if "FP8" in str(d): quant_labels.add("FP8")
                if "MXFP4" in str(d): quant_labels.add("MXFP4")
            except: pass
        cfg = p / "config.json"
        if cfg.is_file():
            try:
                d = _json.loads(cfg.read_text())
                qc = d.get("quantization_config", {}) or {}
                if isinstance(qc, dict):
                    if "NVFP4" in qc.get("quant_algo", ""): quant_labels.add("NVFP4")
                    if "FP8" in qc.get("quant_algo", ""): quant_labels.add("FP8")
                    for _k, _v in (qc.get("quantized_layers", {}) or {}).items():
                        if "NVFP4" in str(_v): quant_labels.add("NVFP4")
                        if "FP8" in str(_v): quant_labels.add("FP8")
                        if "MXFP4" in str(_v): quant_labels.add("MXFP4")
            except: pass
        fw = p / "freetoken_weight.json"
        if fw.is_file():
            try:
                d = _json.loads(fw.read_text())
                if "nvfp4" in str(d).lower(): quant_labels.add("NVFP4")
                if "fp8" in str(d).lower(): quant_labels.add("FP8")
            except: pass
        _HAS_KW = {
            "nvfp4": "NVFP4", "mxfp4": "MXFP4", "fp8": "FP8",
            "bf16": "BF16", "fp16": "FP16", "fp32": "FP32",
            "q4": "Q4", "q4_0": "Q4", "q4_k_m": "Q4", "q4_k_s": "Q4",
            "q5": "Q5", "q5_k_m": "Q5", "q5_k_s": "Q5",
            "q6": "Q6", "q6_k": "Q6", "q8": "Q8", "q8_0": "Q8",
            "int4": "INT4", "int8": "INT8", "int5": "INT5",
        }
        for f in p.rglob("*"):
            if not f.is_file(): continue
            name = f.name.lower()
            if not any(name.endswith(s) for s in (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ftw")):
                continue
            for kw, label in _HAS_KW.items():
                if kw in name:
                    quant_labels.add(label)
                    break
            else:
                quant_labels.add("BF16")
        if not quant_labels:
            has_wt = any(f.is_file() and f.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ftw") for f in p.rglob("*"))
            if has_wt: quant_labels.add("BF16")
        _ORDER = ["NVFP4", "FP8", "MXFP4", "Q4", "INT4", "Q5", "Q6", "Q8", "INT8", "BF16", "FP16", "FP32"]
        _RATIO = {
            "NVFP4": 0.5, "MXFP4": 0.5, "FP8": 1.0,
            "Q4": 0.5, "INT4": 0.5, "Q5": 0.65, "INT5": 0.65,
            "Q6": 0.75, "Q8": 1.0, "INT8": 1.0,
            "BF16": 2.0, "FP16": 2.0, "FP32": 4.0,
        }
        result = [{"label": l, "ratio": _RATIO.get(l, 2.0)} for l in _ORDER if l in quant_labels]
        return {"quants": result}


    return app

def _normalize_engine_args(model_path, args):
    """Architecture-aware arg normalization: dense models must NOT receive
    MoE-only flags (e.g. --moe-backend hybrid) — strip them after whitelist."""
    clean = _sanitize_engine_args(args)
    try:
        facts = _get_facts(model_path)
        is_moe = bool(facts.get("num_exp"))
    except Exception:
        return clean
    if is_moe:
        return clean
    out = []
    skip_next = False
    for tok in clean:
        if skip_next:
            skip_next = False
            continue
        if tok == "--moe-backend":
            skip_next = True
            continue
        if tok.startswith("--moe-backend="):
            continue
        out.append(tok)
    return out

def _engine_summary(st: dict) -> dict:
    return {
        "model": st.get("model"),
        "port": st.get("port"),
        "pid": st.get("pid"),
        "uptimeS": st.get("uptimeS", 0),
        "lastExitCode": st.get("lastExitCode"),
    }



def _sse(rec: dict) -> str:
    return f"id: {rec['seq']}\ndata: {json.dumps(rec)}\n\n"



def _sse_gap(dropped: int, from_seq: Any, to_seq: Any) -> str:
    payload = {"kind": "gap", "dropped": dropped, "fromSeq": from_seq, "toSeq": to_seq}
    return f"data: {json.dumps(payload)}\n\n"



def _log_stream(request: Request, ring, since: int) -> StreamingResponse:
    """SSE log stream with replay + live tail. Correctness points:
      * subscribe BEFORE snapshotting the backlog, then dedupe live records by seq → no gap and
        no duplicate across the replay→live boundary;
      * per-subscriber bounded queue, drop-oldest on overflow via ``call_soon_threadsafe`` (the
        mutation runs on the loop thread, so the reader never blocks) and a client-visible gap
        sentinel so a slow client knows it lost lines;
      * ``id:<seq>`` on every frame + ``Last-Event-ID`` honoured for native EventSource resume;
      * a 15 s heartbeat + ``is_disconnected`` check so an idle client's disconnect is detected
        and the subscriber is always removed in ``finally`` (no leak)."""
    loop = asyncio.get_running_loop()
    lei = request.headers.get("last-event-id")
    if lei and lei.isdigit():
        since = int(lei) + 1  # exclusive next-cursor

    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    drop = {"n": 0, "from": None, "to": None}
    # Records with seq < boundary are already covered by the replayed backlog (they landed in the
    # window between subscribe and the snapshot). Skipping them here keeps the gap counters honest
    # — only genuinely-lost LIVE lines feed drop[]. Safe to set after subscribe: the
    # scheduled _put callbacks only run once this handler yields control, by which point boundary
    # is set.
    boundary = {"v": 0}


    def push(rec: dict) -> None:

        def _put() -> None:
            if rec["seq"] < boundary["v"]:
                return  # already delivered via backlog; don't enqueue or count it as dropped
            if q.full():
                try:
                    old = q.get_nowait()
                    drop["n"] += 1
                    if drop["from"] is None:
                        drop["from"] = old["seq"]
                    drop["to"] = old["seq"]
                except asyncio.QueueEmpty:  # pragma: no cover - race-only
                    pass
            q.put_nowait(rec)

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:  # loop is closing during shutdown
            pass

    ring.subscribe(push)
    backlog, cursor = ring.since(since)
    boundary["v"] = cursor

    async def gen():
        try:
            # If the ring evicted records at/after the client's cursor before it (re)connected,
            # announce that lost prefix so the client knows its history is incomplete.
            oldest = backlog[0]["seq"] if backlog else cursor
            if oldest > since:
                yield _sse_gap(oldest - since, since, oldest - 1)
            for rec in backlog:
                yield _sse(rec)
            last_seq = cursor - 1
            while True:
                if await request.is_disconnected():
                    break
                try:
                    rec = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if rec["seq"] <= last_seq:
                    continue  # already delivered in backlog
                if drop["n"]:
                    # Snapshot + reset synchronously BEFORE yielding: during the yield the loop
                    # drains more _put callbacks that may mutate drop[], and those must not be
                    # wiped unreported.
                    n, frm, to = drop["n"], drop["from"], drop["to"]
                    drop["n"], drop["from"], drop["to"] = 0, None, None
                    yield _sse_gap(n, frm, to)
                last_seq = rec["seq"]
                yield _sse(rec)
        finally:
            ring.unsubscribe(push)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
