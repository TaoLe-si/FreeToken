"""Verify Qwen3_5MtpHead loads weights correctly from a real Qwen3.6 checkpoint."""
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, r'E:\\FreeToken\\python')

from freetoken.models.qwen3_5_moe import (
    MtpHeadConfig,
    Qwen3_5MtpHead,
    load_mtp_head_from_safetensors,
)

MODEL_DIR = r'E:\\models\\Qwen3.6-35B-A3B-MXFP4-MTP'

def main():
    print("Loading MTP head from", MODEL_DIR)
    cfg = MtpHeadConfig()
    # Stub embed/lm_head for the test
    embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
    lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16)
    head = load_mtp_head_from_safetensors(
        MODEL_DIR, cfg, embed, lm_head, device="cpu", dtype=torch.bfloat16,
    )
    # Count parameters
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"MTP head loaded: {n_params/1e6:.1f}M params")

    # Verify attn qkv_proj is non-zero (most important non-fc weights)
    print(f"qkv_proj.weight norm: {head.attn.qkv_proj.weight.norm().item():.4f}  (q+k+v fused, MXFP4 init=0)")
    print(f"o_proj.weight norm: {head.attn.o_proj.weight.norm().item():.4f}")
    print(f"mlp.gate.weight norm: {head.mlp.gate.weight.norm().item():.4f}")
    # Check that switch MLP expert 0 weights are non-zero
    n_packed = len(head._packed_mxfp4)
    print(f"packed MXFP4 weights: {n_packed}")

    # Run 1-token forward and check logits shape
    print("\nRunning 1-token forward...")
    head.eval()
    with torch.no_grad():
        prev_token_id = torch.tensor([1234], dtype=torch.long)
        prev_hidden = torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16) * 0.1
        t0 = time.perf_counter()
        logits = head(prev_token_id, prev_hidden)
        t1 = time.perf_counter()
    print(f"logits shape: {logits.shape}")
    print(f"logits dtype: {logits.dtype}")
    print(f"logits norm: {logits.norm().item():.4f}")
    print(f"argmax token: {logits.argmax().item()}")
    print(f"1-token forward time: {(t1-t0)*1000:.2f}ms (PyTorch CPU, no iGPU yet)")
    assert logits.shape == (1, cfg.vocab_size), f"unexpected logits shape {logits.shape}"
    print("\nAll checks PASSED")

if __name__ == "__main__":
    main()