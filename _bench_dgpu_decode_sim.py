"""
模拟真实 decode 场景:
- 模型权重放 dGPU WDDM shared pool (14 GB 触发 page fault)
- 每步 decode: 读 40 个 16 KB 权重 + 写回结果
- 测总延迟

对比:
- 旧路径 (iGPU GTT + PCIe): 49ms compute + 4ms PCIe ≈ 53ms
- 新路径 (dGPU WDDM shared): 只测 WDDM access 延迟
"""
import os, sys, time
import torch

print(f"[dGPU] {torch.cuda.get_device_name(0)}")

# 模拟 40 个 bank, 总 17 GB
# gate_up_packed 256 MB × 40 = 10.2 GB
# 其他权重 ~ 7 GB
# 我们只放 10.2 GB, 模拟门控和上投影 (占用最大的)
N_BANKS = 40
BANK_MB = 256  # gate_up_packed
TOTAL_BANK_MB = N_BANKS * BANK_MB
print(f"[SETUP] Allocating {N_BANKS}×{BANK_MB}MB = {TOTAL_BANK_MB/1024:.1f} GB in dGPU WDDM shared pool...")

t0 = time.perf_counter()
banks = torch.empty(N_BANKS * BANK_MB * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
torch.cuda.synchronize()
print(f"  alloc: {time.perf_counter()-t0:.2f}s")
print(f"  is_pinned: {banks.is_pinned()}")

# 触发 page fault
print("[PAGE-FAULT] Touching all banks (warm up)...")
t0 = time.perf_counter()
banks.zero_()
torch.cuda.synchronize()
print(f"  warm zero: {time.perf_counter()-t0:.2f}s")

# === Decode 模拟 ===
# 每步: hidden (8KB) + topk_ids (32B) + topk_weights (32B) 写
# 每层: kernel 读 256MB gate_up + 128MB down + scales + output 8KB
# 简化: 每步读 16KB 随机 bank, 写 8KB 输出
print("\n[DECODE] Simulating 5 decode steps, each step touches 40 banks × 16KB:")
def decode_step():
    # 模拟 40 层: 每层从不同 bank 读 16KB (代表 mat-mul gate up 部分)
    s = 0
    for layer in range(40):
        off = layer * BANK_MB * 1024 * 1024 + (layer * 1024) % (BANK_MB * 1024 * 1024 - 16*1024)
        chunk = banks[off:off+16*1024]
        s += chunk.sum().item()
    return s

times = []
for trial in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s = decode_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)
    print(f"  step {trial+1}: {dt:.2f} ms (sum={s})")
print(f"  avg: {sum(times)/len(times):.2f} ms => {1000/(sum(times)/len(times)):.1f} tok/s (WDDM-access-only)")

# === 加 PCIe H2D 输入 ===
print("\n[DECODE-FULL] Full decode step with PCIe H2D input:")
x_pinned = torch.empty(8*1024, dtype=torch.float32, pin_memory=True)
out_pinned = torch.empty(8*1024, dtype=torch.float32, pin_memory=True)
x_vram = torch.zeros(8*1024, device='cuda')

def full_decode_step():
    # 1. H2D input
    x_vram.copy_(x_pinned, non_blocking=True)
    torch.cuda.synchronize()
    # 2. Read 40 banks
    s = 0
    for layer in range(40):
        off = layer * BANK_MB * 1024 * 1024 + (layer * 1024) % (BANK_MB * 1024 * 1024 - 16*1024)
        s += banks[off:off+16*1024].sum().item()
    # 3. D2H output
    out_pinned.copy_(x_vram, non_blocking=True)
    torch.cuda.synchronize()
    return s

times = []
for trial in range(5):
    t0 = time.perf_counter()
    s = full_decode_step()
    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)
    print(f"  full step {trial+1}: {dt:.2f} ms")
print(f"  avg: {sum(times)/len(times):.2f} ms => {1000/(sum(times)/len(times)):.1f} tok/s (with PCIe)")

print("\n" + "="*60)
print("对比总结 (理论上限)")
print("="*60)
print(f"  iGPU GTT + PCIe (Form-2): 49ms compute + 4ms PCIe ≈ 53ms (~19 tok/s)")
print(f"  dGPU WDDM shared: 见上面实测数字")
print(f"  dGPU VRAM only (model partially in VRAM): N/A (8GB 不够 17.5GB 模型)")
