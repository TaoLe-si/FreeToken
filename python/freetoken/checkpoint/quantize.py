"""Quantize an HF safetensors checkpoint into ModelOpt-style NVFP4 (W4A16).

Handles two source kinds, detected per tensor:
  * MXFP4 (mx.quantize): '.weight' packed uint32 + '.scales' + '.biases' (group 32,
    kE2M1 magnitude table [0,1,2,3,4,6,8,12], affine) -- decoded to bf16 first.
  * raw bf16/fp16/fp32 '.weight' -- quantized directly.

Output is a checkpoint the engine's NVFP4 loaders serve natively:
  * routed experts -- either stacked 'mlp.switch_mlp.{proj}' MXFP4 [E, I, H/8] (Qwen
    MXFP4 export) or per-expert 'mlp.experts.E.*' keys -- become per-expert NVFP4
    triplets under 'model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj'
    {'.weight' uint8 packed, '.weight_scale' fp8-e4m3 per-16, '.weight_scale_2'
    fp16 [1]} (matches _NVFP4_EXPERT_KEY_RE; the MoE offload cache banks them).
  * shared_expert   -> same triplet under 'model.layers.N.mlp.shared_expert.*'
    (kept native W4A16 by _dense_nvfp4_emit; gate/up fuse at load)
  * attention / GDN / embed_tokens / lm_head -> decoded bf16 ('model.*' / 'lm_head.*')
  * norms, router, conv1d, A_log, dt_bias -> copied verbatim
  * 'mtp.*' -> copied verbatim (the MTP head loader decodes MXFP4 itself)

config.json gains quantization_config = {"quant_method": "modelopt", "quant_algo":
"NVFP4"} so parse_config routes expert_quant/dense_quant to nvfp4 and everything else
to bf16. Output size ~= source (experts 0.5625 B/elem vs MXFP4 0.625; dense bf16) --
never the 4x of a full bf16 decode.

NVFP4 encode (mirror of kernel/triton/nvfp4_dequant.py):
  weight = e2m1(code) * scale_e4m3(block of 16) * global_scalar
  global = amax(|W|) / (6 * 448)  (max block -> scale 448, the e4m3 max)
  packing: two codes per byte, element 2b in the LOW nibble.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil

import torch
from safetensors.torch import save_file

# NVFP4 e2m1 magnitudes indexed by code (kernel/triton/nvfp4_dequant.py::_E2M1_VALUES).
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# e2m1 midpoints -> searchsorted boundaries for nearest-encoding.
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
# MXFP4 (mx.quantize) *affine uint4* identity table. The Qwen3.6 MXFP4 export is NOT
# e2m1/kE2M1-coded: its nibbles are plain UNSIGNED ints 0..15 and the value is
# nibble * scale + bias per 32-element block (verified against the official NVFP4
# release embed_tokens: uint4-affine cos=+0.997 vs kE2M1 cos=-0.33). The
# benchmarks/ t_mxfp4_dequant.py kE2M1 table describes a DIFFERENT exporter.
_UAFF16 = torch.arange(16, dtype=torch.float32)
_KMX16_LUT: dict = {}  # per-device LUT cache (avoids a CPU->GPU sync per call)
_E4M3_MAX = 448.0


_KMX_BY_DEV: dict = {}  # per-device kE2M1 magnitude table (avoids a CPU->GPU sync per call)


def mxfp4_decode(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor,
                 *, device=None, lead_chunk: int = 4096) -> torch.Tensor:
    """uint32 [..., I/8] + fp16 [..., I/32] x2 -> bf16 [..., I] (N-D: packed along the
    last dim, e.g. GDN conv1d [D, D, K/8] or stacked experts [E, I, H/8]). Low nibble first.

    device="cuda" decodes on the GPU and keeps the result resident there (the stacked
    expert path feeds it straight into nvfp4_encode_experts). The lead dim is processed
    in lead_chunk slices so the int64 index temporaries stay ~tens of MB even for a
    [256, 512, 256] packed expert block -- an unchunked decode materializes a ~2 GB
    codes tensor, which thrashes badly on 8 GB WDDM cards."""
    if device is not None:
        dev = torch.device(device)
        packed = packed.to(dev, non_blocking=True)
        scales = scales.to(dev, non_blocking=True)
        biases = biases.to(dev, non_blocking=True)
    shape = list(packed.shape)
    P = shape[-1]
    I = P * 8
    lead = math.prod(shape[:-1])
    if lead == 0:
        return packed.new_empty((*shape[:-1], I)).to(torch.bfloat16)
    dev = packed.device
    lut = _KMX16_LUT.get(dev)
    if lut is None:
        lut = _UAFF16.to(dev)
        _KMX16_LUT[dev] = lut
    b = packed.reshape(lead, P).view(torch.uint8).reshape(lead, P * 4)
    s = scales.reshape(lead, I // 32).float().unsqueeze(-1)   # [lead, nb, 1]
    bb = biases.reshape(lead, I // 32).float().unsqueeze(-1)
    out = torch.empty(lead, I, dtype=torch.bfloat16, device=dev)
    step = max(1, lead_chunk)
    for c0 in range(0, lead, step):
        cN = min(c0 + step, lead)
        bc = b[c0:cN]                                          # [chunk, I/2] uint8
        codes = torch.stack(((bc & 0xF).long(), (bc >> 4).long()),
                            dim=-1).reshape(cN - c0, I)       # interleave lo|hi
        vals = lut[codes].view(cN - c0, I // 32, 32)           # [chunk, nb, 32] fp32
        out[c0:cN] = (vals * s[c0:cN] + bb[c0:cN]).reshape(cN - c0, I)
    return out.reshape(shape[:-1] + [I])

def _nvfp4_codes(wf: torch.Tensor, gf: torch.Tensor):
    """Shared encode math. wf [..., O, I] fp32, gf broadcastable global. Returns
    (packed u8 [..., O, I/2], block scales fp8 [..., O, I/16])."""
    lead = list(wf.shape[:-2])
    O, I = wf.shape[-2], wf.shape[-1]
    blocks = wf.view(*lead, O, I // 16, 16)
    bamax = blocks.abs().amax(dim=-1, keepdim=True)
    s_ideal = (bamax / 6.0) / gf
    s8 = s_ideal.clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)   # e4m3 round-to-nearest
    sf = s8.float()
    q = blocks / (sf * gf).clamp(min=1e-20)
    code = torch.searchsorted(_E2M1_BOUNDS.to(wf.device), q.abs())
    code = code.clamp(min=0, max=7)
    code = code + 8 * (q < 0)                                   # codes 8..15 = negatives
    lo = code[..., 0::2].reshape(*lead, O, I // 2)
    hi = code[..., 1::2].reshape(*lead, O, I // 2)
    packed = (lo | (hi << 4)).to(torch.uint8)
    return packed, s8.reshape(*lead, O, I // 16)


def nvfp4_encode(w: torch.Tensor, *, device="cuda"):
    """bf16 [O, I] -> (uint8 [O, I/2], fp8_e4m3 [O, I/16], fp16 [1] scalar). I % 16 == 0."""
    O, I = w.shape
    assert I % 16 == 0, f"input dim {I} not a multiple of 16"
    dev = torch.device(device); cpu = torch.device("cpu")
    wf = w.to(dev, non_blocking=True).float()
    amax = wf.abs().amax().clamp(min=1e-12)
    g16 = (amax / (6.0 * _E4M3_MAX)).to(torch.float16)
    gf = g16.float()                                            # decode uses the STORED fp16 global
    packed, s8 = _nvfp4_codes(wf.view(1, O, I), gf.view(1, 1, 1))
    del wf
    return packed[0].to(cpu, non_blocking=True), s8[0].to(cpu, non_blocking=True), g16.reshape(1).to(cpu)


def nvfp4_encode_experts(w: torch.Tensor, *, device="cuda", chunk: int = 64):
    """bf16 [E, O, I] -> (uint8 [E, O, I/2], fp8_e4m3 [E, O, I/16], fp16 [E]) with a
    PER-EXPERT global scalar. chunk=0 = single GPU pass (default; even E=256, O=2048,
    I=2048 fits comfortably -- 256*2048*2048*4B fp32 = 4 GB, fine on >=8 GB cards and far
    faster than the old chunk=32 path which paid ~8 PCIe round-trips per stacked switch key).
    Pass chunk>0 to bound memory at the cost of extra round-trips."""
    E, O, I = w.shape
    assert I % 16 == 0, f"input dim {I} not a multiple of 16"
    dev = torch.device(device)
    cpu = torch.device("cpu")
    # Adaptive chunk: cap fp32 working set at ~1 GiB (fits comfortably even when the GPU
    # is already serving other tensors). 1 GiB / (O * I * 4 B) experts per pass.
    per_expert_fp32 = max(1, int((1 << 30) // max(1, O * I * 4)))
    eff_chunk = min(E, max(1, chunk), per_expert_fp32)
    ps, ss, gs = [], [], []
    for c0 in range(0, E, eff_chunk):
        cN = min(c0 + eff_chunk, E)
        wc = w[c0:cN].to(dev, non_blocking=True)
        wf = wc.float()
        del wc
        amax = wf.view(cN - c0, -1).abs().amax(dim=1).clamp(min=1e-12)
        g16 = (amax / (6.0 * _E4M3_MAX)).to(torch.float16)
        gf = g16.float().view(-1, 1, 1, 1)
        packed, s8 = _nvfp4_codes(wf, gf)
        del wf
        ps.append(packed.to(cpu, non_blocking=True))
        ss.append(s8.to(cpu, non_blocking=True))
        gs.append(g16.to(cpu))
    return torch.cat(ps), torch.cat(ss), torch.cat(gs)
    # Legacy chunked path (rare -- only used when caller asks)
    ps, ss, gs = [], [], []
    for c0 in range(0, E, chunk):
        wc = w[c0:c0 + chunk].to(dev, non_blocking=True).float()
        amax = wc.view(wc.shape[0], -1).abs().amax(dim=1).clamp(min=1e-12)
        g16 = (amax / (6.0 * _E4M3_MAX)).to(torch.float16)
        gf = g16.float().view(-1, 1, 1, 1)
        packed, s8 = _nvfp4_codes(wc, gf)
        ps.append(packed.to(cpu, non_blocking=True))
        ss.append(s8.to(cpu, non_blocking=True))
        gs.append(g16.to(cpu))
    return torch.cat(ps), torch.cat(ss), torch.cat(gs)


# ---- checkpoint-level transcoding ----------------------------------------------

# Per-expert keys (modelopt-style unstacked export).
_EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|scales|biases)$"
)
# Stacked routed experts (Qwen MXFP4 export): [E, I, H/8] uint32 + scales/biases.
_SWITCH_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.switch_mlp\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_SHARED_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\.mlp\.shared_expert\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|scales|biases)$"
)

_ATTN_NVFP4_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\."
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|scales|biases)$"
)
_GDN_NVFP4_RE = re.compile(
    r"^language_model\.model\.layers\.(?P<layer>\d+)\."
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.(weight|scales|biases)$"
)
_LM_HEAD_NVFP4_RE = re.compile(
    r"^language_model\.lm_head\.(weight|scales|biases)$"
)
# A/B switch for the dense attn/GDN quant target: the OFFICIAL NVFP4 release keeps
# self_attn + GDN projections in FP8 (group_0) and only experts/shared/lm_head are
# NVFP4; our default also quantizes attn/GDN to NVFP4 (W4A16) to save dense VRAM on
# 8 GB cards. Set FREETOKEN_ATTN_KEEP_BF16=1 to leave them bf16 (diagnosis /
# fallback path -- engine's bf16 fused projections are the long-proven route).
_ATTN_QUANT_MODE = os.environ.get("FREETOKEN_ATTN_QUANT", "nvfp4").lower()
if _ATTN_QUANT_MODE not in ("nvfp4", "fp8", "bf16"):
    raise SystemExit("FREETOKEN_ATTN_QUANT must be nvfp4|fp8|bf16, got %r" % _ATTN_QUANT_MODE)
# nvfp4 mode also routes attn/GDN through the dense_nvfp4 loop; fp8 mode routes them
# there too but emits per-tensor FP8 instead (official layout); bf16 drops them to
# the plain bf16 copy path.
_ATTN_NVFP4_ENABLED = _ATTN_QUANT_MODE in ("nvfp4", "fp8")
_E4M3_MAX = 448.0


def _final_name(raw: str) -> str:
    """language_model.model.X -> model.X ; language_model.lm_head.X -> lm_head.X ; else verbatim."""
    if raw.startswith("language_model.model."):
        return "model." + raw[len("language_model.model."):]
    if raw.startswith("language_model.lm_head."):
        return "lm_head." + raw[len("language_model.lm_head."):]
    return raw


def quantize_to_nvfp4(model_path: str, out_dir: str, *, device: str = "cuda",
                      shard_bytes: int = 4 << 30, progress=print) -> dict:
    """Read an MXFP4 / raw-bf16 HF checkpoint, write an NVFP4 HF checkpoint FreeToken serves."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(model_path, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    # pass 1: classify keys
    experts: dict = {}
    switch: dict = {}
    shared: dict = {}
    plain: list = []
    mtp: list = []
    dense_nvfp4: list = []
    for raw in weight_map:
        if raw.startswith("mtp."):
            mtp.append(raw); continue
        if raw.startswith(("vision_tower.", "model.visual.", "visual.")):
            continue
        m = _EXPERT_RE.match(raw)
        if m and m.group("kind") == "weight":
            experts[(int(m.group("layer")), int(m.group("expert")), m.group("proj"))] = raw
            continue
        m = _SWITCH_RE.match(raw)
        if m:
            switch[(int(m.group("layer")), m.group("proj"))] = raw
            continue
        m = _SHARED_RE.match(raw)
        if m and m.group("kind") == "weight":
            shared[(int(m.group("layer")), m.group("proj"))] = raw
            continue
        if raw.endswith((".scales", ".biases", ".weight_scale", ".weight_scale_2", ".input_scale")):
            continue  # consumed with their .weight
        if (_LM_HEAD_NVFP4_RE.match(raw) or (_ATTN_NVFP4_ENABLED and (_ATTN_NVFP4_RE.match(raw) or _GDN_NVFP4_RE.match(raw)))) and raw.endswith(".weight"):
            dense_nvfp4.append(raw)
            continue
        plain.append(raw)

    dev = torch.device(device)
    tensors_out: dict = {}
    out_map: dict = {}
    shard_no = 0
    cur_bytes = 0
    total_bytes = 0
    stats = {"nvfp4": 0, "bf16": 0, "copied": 0}

    def _emit(name: str, t: torch.Tensor):
        nonlocal shard_no, cur_bytes, total_bytes
        fname = f"model-{shard_no:05d}-of-XXXXX.safetensors"
        tensors_out[name] = t
        out_map[name] = fname
        nb = t.numel() * t.element_size()
        cur_bytes += nb; total_bytes += nb
        if cur_bytes >= shard_bytes:
            save_file(tensors_out, os.path.join(out_dir, fname))
            tensors_out.clear(); cur_bytes = 0; shard_no += 1

    from safetensors import safe_open
    _cache: dict = {}

    def _open(shard: str):
        if shard not in _cache:
            if len(_cache) >= 16:
                _cache.pop(next(iter(_cache)))
            _cache[shard] = safe_open(os.path.join(model_path, shard), framework="pt", device="cpu")
        return _cache[shard]

    def _siblings(raw_w: str):
        """(weight, scales, biases); scales/biases are None for non-MXFP4 tensors.
        Siblings may live in a different shard than the weight -- follow the index."""
        base = raw_w[: -len(".weight")]
        w = _open(weight_map[raw_w]).get_tensor(raw_w)
        sk, bk = base + ".scales", base + ".biases"
        if sk in weight_map and bk in weight_map:
            s = _open(weight_map[sk]).get_tensor(sk)
            b = _open(weight_map[bk]).get_tensor(bk)
            return w, s, b
        return w, None, None

    total = len(experts) + len(switch) + len(shared) + len(dense_nvfp4) + len(plain) + len(mtp)
    done = 0

    # stacked routed experts [E, I, H/8] -> per-expert NVFP4 triplets (the big win:
    # 0.5625 B/elem vs 2 B/elem bf16 -- this is what keeps the output ~= source size)
    for (layer, proj), raw in sorted(switch.items()):
        w, s, b = _siblings(raw)
        wf = mxfp4_decode(w, s, b, device=dev) if s is not None else w.to(dev).to(torch.bfloat16)
        p, sc, g = nvfp4_encode_experts(wf, device=dev)
        del wf
        E = p.shape[0]
        for e in range(E):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            _emit(base + ".weight", p[e].contiguous())
            _emit(base + ".weight_scale", sc[e].contiguous())
            _emit(base + ".weight_scale_2", g[e:e + 1].contiguous())
        stats["nvfp4"] += E; done += 1
        progress("experts", done, total)

    # per-expert keys (unstacked modelopt export) -> same triplet layout
    for (layer, expert, proj), raw in sorted(experts.items()):
        w, s, b = _siblings(raw)
        wf = mxfp4_decode(w, s, b) if s is not None else w.to(torch.bfloat16)
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        p, sc, g = nvfp4_encode(wf, device=dev)
        _emit(base + ".weight", p); _emit(base + ".weight_scale", sc); _emit(base + ".weight_scale_2", g)
        stats["nvfp4"] += 1; done += 1

    # shared_expert -> native NVFP4 under model.layers.* (kept W4A16 at load)
    for (layer, proj), raw in sorted(shared.items()):
        w, s, b = _siblings(raw)
        wf = mxfp4_decode(w, s, b, device=dev) if s is not None else w.to(torch.bfloat16)
        base = f"model.layers.{layer}.mlp.shared_expert.{proj}"
        p, sc, g = nvfp4_encode(wf, device=dev)
        _emit(base + ".weight", p); _emit(base + ".weight_scale", sc); _emit(base + ".weight_scale_2", g)
        stats["nvfp4"] += 1; done += 1

    # attn/GDN/lm_head -> native NVFP4. Engine reads these as .weight/.weight_scale/
    # .weight_scale_2 (per-tensor scalar); .weight_global is reconstructed at load by the
    # engine from 1/.weight_scale_2 -> fp16 [out] per-row expand.
    for raw in sorted(dense_nvfp4):
        w, s, b = _siblings(raw)
        wf = mxfp4_decode(w, s, b, device=dev) if s is not None else w.to(torch.bfloat16)
        if _ATTN_QUANT_MODE == "fp8" and not raw.startswith("language_model.lm_head."):
            # Official-layout per-tensor FP8 (W8A16): keep the PRE-fusion projection
            # names (q_proj/k_proj/...) so weight.py's _PT_FP8 path fuses them.
            base = _final_name(raw)[:-len(".weight")]
            wf32 = wf.float()
            scale = wf32.abs().amax().clamp(min=1e-12) / _E4M3_MAX
            q = (wf32 / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
            _emit(base + ".weight", q.to("cpu", non_blocking=True))
            _emit(base + ".weight_scale", scale.reshape(1).to("cpu", non_blocking=True))
            stats["fp8"] = stats.get("fp8", 0) + 1
            done += 1
            continue
        p, sc, g = nvfp4_encode(wf, device=dev)
        # single "model.layers..." prefix via _final_name -- the SAME mapping the plain
        # bf16 keys below use. A literal "model." + raw[15:] produced the double
        # "model.model.layers..." prefix, which split the load-side GDN fuse buffers
        # (NVFP4 qkv|z in one bucket, bf16 b|a in the other) and never completed.
        if raw.startswith("language_model.lm_head."):
            base = "lm_head"  # already the final base (raw is lm_head.weight)
        else:
            base = _final_name(raw)[:-len(".weight")]
        _emit(base + ".weight", p)
        _emit(base + ".weight_scale", sc)
        _emit(base + ".weight_scale_2", g)
        stats["nvfp4"] += 1; done += 1

    # dense: decode MXFP4 -> bf16 (attn/GDN/embed/lm_head), copy the rest verbatim.
    # GDN short-conv: the MXFP4 export stores conv1d [O, K, 1] while the engine (FLA
    # CausalConv1d) wants [O, 1, K] -- transpose axes 1<->2.
    for raw in sorted(plain):
        w, s, b = _siblings(raw)
        if s is not None:
            w = mxfp4_decode(w, s, b); stats["bf16"] += 1
        else:
            stats["copied"] += 1
        if raw.endswith(".linear_attn.conv1d.weight") and w.dim() == 3 \
                and w.shape[2] == 1 and w.shape[1] != 1:
            w = w.transpose(1, 2)
        _emit(_final_name(raw), w.contiguous())
        done += 1

    # mtp.* verbatim (uint32 packed + scales + biases: the MTP loader decodes it)
    for raw in sorted(mtp):
        f = _open(weight_map[raw])
        _emit(raw, f.get_tensor(raw))
        stats["copied"] += 1; done += 1

    if tensors_out:
        save_file(tensors_out, os.path.join(out_dir, f"model-{shard_no:05d}-of-XXXXX.safetensors"))
        shard_no += 1
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_map}, f)

    # config: copy everything, swap the quant markers so parse_config routes to NVFP4
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("quantization", None); cfg.pop("weight_format", None)
    # Build quantized_layers map (MIXED_PRECISION): experts + attn + lm_head all NVFP4.
    # Marking attn here flips attn_quant="nvfp4" which routes self_attn qkv fused (Nvfp4DenseColMerged),
    # GDN in_proj_qkvz fused, GDN out_proj + self_attn o_proj (Nvfp4DenseLinear), lm_head (Nvfp4LMHead).
    ql = {
        "model.language_model.layers.0.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.0.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.1.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.1.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.1.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.2.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.2.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.2.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.3.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.3.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.3.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.4.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.4.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.4.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.5.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.5.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.5.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.6.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.6.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.6.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.7.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.7.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.7.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.8.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.8.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.8.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.9.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.9.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.9.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.10.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.10.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.10.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.11.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.11.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.11.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.12.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.12.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.12.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.13.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.13.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.13.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.14.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.14.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.14.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.15.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.15.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.15.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.16.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.16.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.16.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.17.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.17.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.17.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.18.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.18.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.18.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.19.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.19.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.19.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.20.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.20.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.20.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.21.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.21.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.21.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.22.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.22.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.22.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.23.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.23.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.23.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.24.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.24.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.24.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.25.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.25.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.25.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.26.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.26.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.26.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.27.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.27.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.27.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.28.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.28.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.28.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.29.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.29.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.29.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.30.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.30.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.30.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.31.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.31.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.31.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.32.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.32.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.32.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.33.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.33.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.33.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.34.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.34.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.34.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.35.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.35.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.35.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.36.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.36.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.36.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.37.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.37.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.37.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.38.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.38.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.38.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.39.self_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.39.linear_attn": {"quant_algo": "W4A16_NVFP4"},
        "model.language_model.layers.39.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "lm_head": {"quant_algo": "W4A16_NVFP4"}
    }
    if _ATTN_QUANT_MODE == "bf16":
        # bf16-attn mode: drop the attn/GDN tags so the engine builds its standard
        # bf16 fused projections for them (experts/shared/lm_head stay NVFP4).
        ql = {k: v for k, v in ql.items()
              if not k.endswith((".self_attn", ".linear_attn"))}
    elif _ATTN_QUANT_MODE == "fp8":
        # Official layout: attn/GDN per-tensor FP8 (routes _attn_quant -> fp8_pertensor).
        ql = {k: ({"quant_algo": "FP8"} if k.endswith((".self_attn", ".linear_attn")) else v)
              for k, v in ql.items()}
    cfg["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": ql,
        "desc": "FreeToken in-app conversion (MXFP4/raw -> NVFP4, attn+gdn+lm_head W4A16)",
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    # carry over tokenizer + generation config -- never the weights, the old index, or
    # the old quant file (all three would shadow what we just wrote).
    for fn in os.listdir(model_path):
        if (fn == "hf_quant_config.json" or fn == "config.json"
                or fn == "model.safetensors.index.json" or fn.endswith(".safetensors")):
            continue
        src = os.path.join(model_path, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    return {"stats": stats, "shards": shard_no, "bytes": total_bytes, "out_dir": out_dir}


__all__ = ["mxfp4_decode", "nvfp4_encode", "nvfp4_encode_experts", "quantize_to_nvfp4"]
