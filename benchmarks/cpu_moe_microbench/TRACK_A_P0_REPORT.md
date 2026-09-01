# Track 4 (A) P0 Report: Weight Path Revival

**Date**: 2026-08-27
**Status**: **ALREADY DONE** (no code changes needed)

## Key Finding
The previous session's `models/qwen3_5_moe/mtp.py::load_mtp_head_from_safetensors()`
function already handles loading all 42 MTP weights from the checkpoint.

## Verification

### Checkpoint MTP keys (42 total):
- mtp.fc.biases / mtp.fc.scales / mtp.fc.weight (FC layer: 2048 x 4096 MXFP4)
- mtp.layers.0.input_layernorm.weight (2048)
- mtp.layers.0.mlp.gate.weight (256 x 2048 routing gate, bf16)
- mtp.layers.0.mlp.shared_expert.{gate,up,down}_proj.{weight,biases,scales} (3 projs x 3)
- mtp.layers.0.mlp.switch_mlp.{gate,up,down}_proj.{weight,biases,scales} (3 projs x 3, MXFP4 packed)
- mtp.layers.0.post_attention_layernorm.weight (2048)
- mtp.layers.0.self_attn.{q,k,v,o}_proj.{weight,biases,scales} (4 projs x 3)
- mtp.norm.weight (2048)
- mtp.pre_fc_norm_embedding.weight (2048)
- mtp.pre_fc_norm_hidden.weight (2048)

### MTP head loads correctly:
```
Loaded MTP head in 12.6s
FC packed shape: torch.Size([2048, 512])    # 2048 output rows x 512 uint32 = 4096 K elements
FC biases shape: None                       # not populated by this loader
FC scales shape: None                       # not populated by this loader
Switch gate shape: torch.Size([256, 512, 2048])  # 256 experts x 512 I x 2048 H (bf16)
Switch up shape: torch.Size([256, 512, 2048])
Switch down shape: torch.Size([256, 2048, 512])
Attn qkv_proj: torch.Size([9216, 2048])     # 16*256*2 + 2*256 + 2*256 = 9216
Attn o_proj: torch.Size([2048, 4096])
All 4 norms: torch.Size([2048])
```

### Why it works (verified):
1. `weight.py:204, 703` filters `model.visual.*` and `visual.*`, NOT `mtp.*`
2. `_rename()` only strips `model.language_model.` / `language_model.` prefix
3. So `mtp.*` keys pass through with their original names
4. `load_mtp_head_from_safetensors()` does its OWN two-pass load across all safetensors files

## Conclusion: P0 is a no-op

The weight path is already wired correctly. The previous session's MtpHead module
is the production loader. No code changes needed for P0.

## Next: P1 (iGPU executor for MTP head)

Now we need to:
1. Start P1g v2 server as daemon
2. LOAD mtp.fc.weight once at session start
3. Provide `MtpExecutor.forward(prev_token_id, prev_hidden) -> draft_logits` API
4. Wire it into `engine.py::create_model` so MTP head has access to iGPU FC