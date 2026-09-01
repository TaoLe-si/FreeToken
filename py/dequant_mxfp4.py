"""Convert MLX-style MXFP4 (u32 packed + f16 scales + f16 biases, group 32) HF checkpoint
to a dequantized bf16 HF directory that FreeToken's existing loaders/converters understand.
Splits fused switch_mlp experts into per-expert keys; extracts mtp.* into a separate file."""
import json, os, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = r"E:\models\Qwen3.6-35B-A3B-MXFP4-MTP"
OUT = r"E:\models\Qwen3.6-35B-A3B-NVFP4-MTP-debf"
MTP_OUT = r"E:\models\Qwen3.6-35B-A3B-NVFP4-MTP-debf\mtp_bf16.safetensors"
PROG = r"E:\FreeToken\py\deq_progress.json"

def report(phase, done, total):
    with open(PROG, "w") as f:
        json.dump({"phase": phase, "done": done, "total": total, "ts": time.time()}, f)

def dequant(w_u32, scales, biases):
    # MLX affine MXFP4: last dim packs 8 nibbles per u32; group of 32 shares scale+bias.
    lead = list(w_u32.shape[:-1])
    b = w_u32.view(torch.uint8)                       # [..., D*4]
    lo = (b & 0x0F)
    hi = (b >> 4)
    inter = torch.stack((lo, hi), dim=-1)             # [..., D*4, 2]
    nib = inter.reshape(*lead, -1)                    # [..., L]
    L = nib.shape[-1]
    G = 32
    q = nib.to(torch.float32) - 8.0
    q = q.reshape(*lead, L // G, G)
    s = scales.to(torch.float32).unsqueeze(-1)        # [..., L//G, 1]
    bi = biases.to(torch.float32).unsqueeze(-1)
    w = s * q + bi
    return w.reshape(*lead, L).to(torch.bfloat16)

def map_name(n):
    if n.startswith("language_model."):
        return n[len("language_model."):]
    return n

os.makedirs(OUT, exist_ok=True)
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
keys = list(wm.keys())
report("start", 0, len(keys))

# group by shard for IO locality
by_shard = {}
for k in keys:
    by_shard.setdefault(wm[k], []).append(k)

out_index = {"metadata": {"format": "pt"}, "weight_map": {}}

_handles = {}
def gt(name):
    sh = wm[name]
    h = _handles.get(sh)
    if h is None:
        h = safe_open(os.path.join(SRC, sh), framework="pt", device="cpu")
        _handles[sh] = h
    return h.get_tensor(name)
bucket, bucket_bytes, shard_no = {}, 0, 0
SHARD_LIMIT = 4 << 30

def flush_bucket():
    global bucket, bucket_bytes, shard_no
    if not bucket:
        return
    shard_no += 1
    name = "model-%05d-of-XXXXX.safetensors" % shard_no
    save_file(bucket, os.path.join(OUT, name), metadata={"format": "pt"})
    for k in bucket:
        out_index["weight_map"][k] = name
    bucket, bucket_bytes = {}, 0

mtp_tensors = {}
n_done = 0
t0 = time.time()
for shard in sorted(by_shard):
    with safe_open(os.path.join(SRC, shard), framework="pt", device="cpu") as _unused_f:
        for k in by_shard[shard]:
            if k.endswith(".scales") or k.endswith(".biases"):
                continue
            base = k
            # [fix] scales/biases are SIBLINGS of .weight (X.scales), not children
            stem = base[:-len(".weight")] if base.endswith(".weight") else base
            is_q = (stem + ".scales") in wm
            dst = map_name(base)
            if is_q:
                t = dequant(gt(base), gt(stem + ".scales"),
                            gt(stem + ".biases"))
            else:
                t = gt(base)
                if t.dtype == torch.float16:
                    t = t.to(torch.bfloat16)
                # ft-conv1d-fix: MLX stores conv1d as [out, k, in]; FreeToken expects [out, in, k]
                if dst.endswith("linear_attn.conv1d.weight"):
                    t = t.transpose(-1, -2).contiguous()
            if base.startswith("mtp."):
                # keep mtp in its own namespace file (dequanted bf16)
                mtp_tensors[base] = t.to(torch.bfloat16).contiguous()
            elif base.startswith("vision_tower."):
                pass  # text-only serving drops vision weights
            else:
                # split fused switch_mlp experts -> per-expert keys
                if ".switch_mlp." in dst:
                    pre, proj = dst.split(".switch_mlp.")
                    E = t.shape[0]
                    for e in range(E):
                        ek = "%s.experts.%d.%s" % (pre.replace(".mlp", ".mlp"), e, proj)
                        bucket[ek] = t[e].contiguous()
                        bucket_bytes += t[e].numel() * 2
                else:
                    bucket[dst] = t.contiguous()
                    bucket_bytes += t.numel() * 2
            if bucket_bytes >= SHARD_LIMIT:
                flush_bucket()
            n_done += 1
            if n_done % 200 == 0:
                report("convert", n_done, len(keys))
flush_bucket()
report("save_mtp", n_done, len(keys))
save_file(mtp_tensors, MTP_OUT, metadata={"format": "pt"})

# finalize index with real shard count
total = shard_no
fixed = {}
for k, v in out_index["weight_map"].items():
    fixed[k] = v.replace("of-XXXXX", "of-%05d" % total)
json.dump({"metadata": {"format": "pt"}, "weight_map": fixed},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)

# rename shards to match total count
import glob as _g
for pth in _g.glob(os.path.join(OUT, "model-*-of-XXXXX.safetensors")):
    n = pth.split("-of-")[0][-5:]
    os.rename(pth, pth.replace("-of-XXXXX", "-of-%05d" % total))

# copy config/tokenizer essentials
import shutil
for fn in ("config.json","generation_config.json","tokenizer.json","tokenizer_config.json",
           "vocab.json","merges.txt","chat_template.jinja","preprocessor_config.json"):
    srcp = os.path.join(SRC, fn)
    if os.path.isfile(srcp):
        shutil.copyfile(srcp, os.path.join(OUT, fn))

report("done", n_done, len(keys))
print("CONVERT DONE", n_done, "tensors,", shard_no, "shards,", round(time.time()-t0,1), "s")
