import sys, torch
sys.path.insert(0, "E:/FreeToken/python")
from freetoken.models.qwen3_5_moe.mtp import MtpHeadConfig, load_mtp_head_from_safetensors

cfg = MtpHeadConfig()
embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=torch.bfloat16)
lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16)
head = load_mtp_head_from_safetensors(
    "E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4", cfg, embed, lm_head,
    igpu_fc=None, device="cpu", dtype=torch.bfloat16)
print("head loaded:", head is not None)
print("fc path:", type(head.igpu_fc).__name__)
# 抽查解码数值: fc 权重应非零且有限
w = head.igpu_fc.w
print("fc w:", tuple(w.shape), "finite:", torch.isfinite(w).all().item(), "std:", w.std().item())
# 单步前向 (随机输入) —— 形状/NaN 检查
head.eval()
with torch.inference_mode():
    logits, state = head.forward_with_state(
        torch.tensor([100]), torch.randn(1, cfg.hidden_size, dtype=torch.bfloat16), position=0)
    print("logits:", tuple(logits.shape), "finite:", torch.isfinite(logits).all().item())
    print("state:", tuple(state.shape))
    # 连续 draft 3 步
    ids = head.draft_ids = []
    hid = state
    tok = torch.tensor([int(logits[0].argmax())])
    for i in range(3):
        logits, hid = head.forward_with_state(tok, hid, position=i + 1)
        tok = torch.tensor([int(logits[0].argmax())])
        ids.append(int(tok))
    print("draft ids (随机权重下, 仅验证不NaN):", ids)
