"""
关键测试: dGPU CUDA kernel 直接读 WDDM pinned memory (zero-copy).
这是 'dGPU WDDM 跑 decode' 方案的核心.

对比:
- dGPU kernel 读 VRAM (基线, 256 GB/s)
- dGPU kernel 读 WDDM pinned (zero-copy, 走 PCIe, ~6 GB/s 但页命中后可能 80 GB/s)
- dGPU kernel 读 WDDM 但 batch prefetch

注意: PyTorch 不支持 cuda @ cpu (zero-copy matmul).
真实场景要么:
  1. CUDA UVA + 自定义 kernel (复杂)
  2. cudaMemcpy 整段到 VRAM 再算 (本测试)
  3. PyTorch 默认: tensor.cpu() @ tensor.cuda() 会自动 sync (慢)
"""
import os, sys, time
import torch

print(f"[dGPU] {torch.cuda.get_device_name(0)}\n")

H = 2048
TOPK = 8

def time_ms(fn, n=20, warmup=3):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

# ============================================================
# [A] 64 KB WDDM bank (decode 实际尺寸)
# ============================================================
print("="*60)
print("[A] 64KB bank (单 layer gate_up 部分, fp32 2048x32)")
print("="*60)

hidden_vram = torch.randn(1, H, device='cuda', dtype=torch.float32)
weights_vram = torch.randn(H, 32, device='cuda', dtype=torch.float32)
weights_pinned = torch.randn(H, 32, pin_memory=True, dtype=torch.float32)

# warm up VRAM
for _ in range(5): _ = hidden_vram @ weights_vram
torch.cuda.synchronize()

# warm up WDDM pinned
weights_pinned.to('cuda', non_blocking=True)
torch.cuda.synchronize()

ms = time_ms(lambda: hidden_vram @ weights_vram)
print(f"  A.1 VRAM:    {ms:.4f} ms")

# WDDM pinned: 必须先 H2D 再算 (PyTorch 限制)
def wddm_64kb_h2d_matmul():
    v = weights_pinned.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    return hidden_vram @ v
ms = time_ms(wddm_64kb_h2d_matmul)
print(f"  A.2 WDDM+H2D: {ms:.4f} ms (H2D 64KB + matmul)")

# ============================================================
# [B] 40 个 256KB bank 模拟 decode
# ============================================================
print("\n" + "="*60)
print("[B] 40 layer MoE, each reads 1MB bank (256K floats)")
print("="*60)

N = 40
banks_vram = [torch.randn(H, 128, device='cuda', dtype=torch.float32) for _ in range(N)]
banks_wddm = [torch.randn(H, 128, pin_memory=True, dtype=torch.float32) for _ in range(N)]

# warm up WDDM banks
for b in banks_wddm:
    b.to('cuda', non_blocking=True)
torch.cuda.synchronize()

def decode_40_vram():
    for b in banks_vram: _ = hidden_vram @ b

def decode_40_wddm():
    for b in banks_wddm:
        v = b.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        _ = hidden_vram @ v

ms = time_ms(decode_40_vram)
print(f"  B.1 40 layers VRAM:    {ms:.3f} ms => {1000/ms:.0f} tok/s")

ms = time_ms(decode_40_wddm)
print(f"  B.2 40 layers WDDM+H2D: {ms:.3f} ms => {1000/ms:.0f} tok/s")

# ============================================================
# [C] 真实 MoE 权重 (256 MB bank)
# ============================================================
print("\n" + "="*60)
print("[C] 真实 256MB bank (每层 256MB gate_up + 128MB down)")
print("="*60)
print("  注: 这种规模 PyTorch WDDM pinned alloc 可能失败或 page fault 多")

try:
    big_vram = torch.randn(256*1024*1024//4, device='cuda', dtype=torch.float32)
    big_wddm = torch.randn(256*1024*1024//4, pin_memory=True, dtype=torch.float32)
    print(f"  alloc big_wddm (256MB pinned): OK")
    
    # warm up
    _ = hidden_vram @ big_vram.view(H, 32768)
    big_wddm.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    _ = hidden_vram @ big_vram.view(H, 32768)
    
    def read_big_vram():
        return hidden_vram @ big_vram.view(H, 32768)
    
    def read_big_wddm_h2d():
        v = big_wddm.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        return hidden_vram @ v.view(H, 32768)
    
    ms = time_ms(read_big_vram, n=5)
    print(f"  C.1 read 256MB VRAM:   {ms:.3f} ms")
    
    ms = time_ms(read_big_wddm_h2d, n=5)
    print(f"  C.2 read 256MB WDDM+H2D: {ms:.3f} ms")
    
    # 5 个 sequential
    print("\n  C.3 5 个 256MB bank sequential:")
    bigs_vram = [torch.randn(256*1024*1024//4, device='cuda', dtype=torch.float32) for _ in range(5)]
    bigs_wddm = [torch.randn(256*1024*1024//4, pin_memory=True, dtype=torch.float32) for _ in range(5)]
    for b in bigs_wddm: b.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    
    def read_5_vram():
        for b in bigs_vram: _ = hidden_vram @ b.view(H, 32768)
    def read_5_wddm():
        for b in bigs_wddm:
            v = b.to('cuda', non_blocking=True)
            torch.cuda.synchronize()
            _ = hidden_vram @ v.view(H, 32768)
    
    ms = time_ms(read_5_vram, n=3)
    print(f"    VRAM:   {ms:.3f} ms ({ms/5:.1f} ms/layer)")
    ms = time_ms(read_5_wddm, n=3)
    print(f"    WDDM:   {ms:.3f} ms ({ms/5:.1f} ms/layer)")
except Exception as e:
    print(f"  C section failed: {e}")

# ============================================================
# [D] 关键: 模拟 MoE 真正访问模式 (8 个 expert 中选 8)
# ============================================================
print("\n" + "="*60)
print("[D] Real MoE pattern: per token, pick 8 experts from 256")
print("="*60)
print("  简化: 每层权重是 [256 experts, H, H/8] (gate_up 简化)")
print("  decode 路径: hidden @ expert_weights[topk_ids]")

E = 256
EXPERT_OUT = 256  # expert 输出维度简化
expert_vram = torch.randn(E, H, EXPERT_OUT, device='cuda', dtype=torch.float32)
expert_wddm = torch.randn(E, H, EXPERT_OUT, pin_memory=True, dtype=torch.float32)

# warm up
expert_wddm.to('cuda', non_blocking=True)
torch.cuda.synchronize()

topk_ids = torch.randint(0, E, (TOPK,), device='cuda', dtype=torch.int64)
topk_w = torch.ones(TOPK, device='cuda', dtype=torch.float32) / TOPK

def moe_8_experts_vram():
    """8 experts matmul + sum"""
    selected = expert_vram[topk_ids]  # [8, H, EXPERT_OUT]
    out = (hidden_vram.unsqueeze(0) @ selected).squeeze(0)  # [8, EXPERT_OUT]
    return (out * topk_w.unsqueeze(-1)).sum(0)

def moe_8_experts_wddm_h2d():
    """WDDM -> VRAM -> 8 experts"""
    v = expert_wddm.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    selected = v[topk_ids]
    out = (hidden_vram.unsqueeze(0) @ selected).squeeze(0)
    return (out * topk_w.unsqueeze(-1)).sum(0)

ms = time_ms(moe_8_experts_vram, n=20)
print(f"  D.1 1 layer MoE 8-experts VRAM: {ms:.3f} ms")

ms = time_ms(moe_8_experts_wddm_h2d, n=20)
print(f"  D.2 1 layer MoE 8-experts WDDM: {ms:.3f} ms (H2D 1GB)")

# 40 layers MoE
print("\n  D.3 40 layer MoE 8-experts:")
expert_layers_vram = [torch.randn(E, H, EXPERT_OUT, device='cuda', dtype=torch.float32) for _ in range(N)]
expert_layers_wddm = [torch.randn(E, H, EXPERT_OUT, pin_memory=True, dtype=torch.float32) for _ in range(N)]
for e in expert_layers_wddm: e.to('cuda', non_blocking=True)
torch.cuda.synchronize()

def moe_40_vram():
    for ex in expert_layers_vram:
        s = ex[topk_ids]
        _ = ((hidden_vram.unsqueeze(0) @ s).squeeze(0) * topk_w.unsqueeze(-1)).sum(0)

def moe_40_wddm():
    for ex in expert_layers_wddm:
        v = ex.to('cuda', non_blocking=True)
        torch.cuda.synchronize()
        s = v[topk_ids]
        _ = ((hidden_vram.unsqueeze(0) @ s).squeeze(0) * topk_w.unsqueeze(-1)).sum(0)

ms = time_ms(moe_40_vram, n=5)
print(f"    VRAM: {ms:.3f} ms ({ms/40:.1f} ms/layer) => {1000/ms:.0f} tok/s")

ms = time_ms(moe_40_wddm, n=5)
print(f"    WDDM: {ms:.3f} ms ({ms/40:.1f} ms/layer) => {1000/ms:.0f} tok/s")

print("\n" + "="*60)
print("总结:")
print("="*60)
print("  VRAM = 256 GB/s, kernel 几乎瞬时")
print("  WDDM pinned -> VRAM 受 PCIe 6 GB/s 限制")
print("  但 'MoE decode = random access WDDM' 如果命中 page cache 极快")
print("  如果纯走 PCIe, 256MB H2D 一层 = 256MB/6GB = 42ms, 不现实")
print("  所以: 关键看 [B] vs [D] 实测差距")
