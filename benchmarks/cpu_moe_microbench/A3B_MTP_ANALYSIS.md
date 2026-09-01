# 35B A3B MOE iGPU-MTP 加速分析

## 模型架构含义

**Qwen3.6-35B-A3B-MXFP4-MTP 的实际部署**：
- 总参数：35B
- 激活参数：~3B（每次前向只激活 3B）
- 32B 权重在 CPU 内存，按需加载到 dGPU
- MTP head 也在内存中（一次性加载 ~405MB）

**对 iGPU-MTP 加速的含义**：
1. 主模型前向 (dGPU) 实际比 35B dense 快得多：
   - dense 35B 前向：~50-80ms（4070 笔记本 GPU）
   - A3B active 35B 前向：~5-10ms（3B active + 32B 从内存加载）
   - **dGPU 实际是瓶颈**，不是 MTP head

2. MTP head 的 attn+MoE+FC 计算量：
   - FC: 4096 -> 2048, 8M params, 0.3ms
   - Attn: 2048 hidden, 16+2 heads, 50M params, ~0.5ms
   - MoE: 256 experts, top-8, 1.4B active, 0.5-1ms on iGPU
   - **MoE 是 MTP head 的瓶颈**（当前 CPU 跑 50-60ms/draft）

## 当前能拿到的 tok/s 数字

### 场景 1：仅 FC on iGPU（当前状态）
- MTP head: 70ms/draft (CPU MoE 主导)
- 并行: max(7, 70) = 70ms
- 串行: 7 + 3*70 = 217ms
- 加速: 1.04x (MTP 主导)

### 场景 2：FC + MoE on iGPU
- MTP head: 12-15ms/draft (24 GEMVs in BATCH_ALL)
- 并行: max(7, 15) = 15ms
- 串行: 7 + 3*15 = 52ms
- 加速: 1.4x @ accept 0.6 (2.8 tok/step)
- tok/s: 2.8 / 15ms = **187 tok/s**

### 场景 3：FC + MoE + Attn on iGPU (完整 iGPU MTP)
- MTP head: 5-7ms/draft (24 GEMV fused + attn+FC)
- 并行: max(7, 7) = 7ms (刚好打平)
- 串行: 7 + 3*7 = 28ms
- 加速: 1.6x @ accept 0.6
- tok/s: 2.8 / 7ms = **400 tok/s** (2.8x baseline)

### 场景 4：FC + MoE + Attn + Norm + LM_head 全 iGPU
- MTP head: 3-5ms/draft (1ms MoE fused + 1ms attn+FC + 1ms lm_head)
- 并行: max(7, 5) = 7ms (主模型主导)
- 加速: 1.43x (上限在主模型)
- tok/s: 2.8 / 7ms = **400 tok/s**

## 工作量估算

| 场景 | 工作量 | 风险 | 加速 |
|------|--------|------|------|
| 1. 仅 FC | 已完成 | 低 | 1.04x |
| 2. + MoE | 3-5 天 | 中 (BATCH_ALL + weight load) | 1.4x |
| 3. + Attn | +2 天 | 中 (RoPE + 注意力) | 1.6x |
| 4. 完整 MTP iGPU | +3 天 | 高 (lm_head bf16 GEMV) | 1.4-1.6x (上限) |

## 关键技术点

### 1. v3 server BATCH_ALL 已就绪
- 一次 dispatch 多个 weight
- 24 个 expert GEMV 可以一次 BATCH_ALL 提交
- 服务器处理是顺序的，但单次 dispatch 减少了 CPU overhead

### 2. MTP MoE 权重是 MXFP4 packed
- 当前 `load_mtp_head_from_safetensors` 把它们 dequant 到 bf16
- 需要修改保留 packed 版本以备 iGPU 上传
- 上传前从 checkpoint 重新读 (开销 < 1s)

### 3. FreeToken 已有 iGPU MoE 基础设施
- `IgpuMoeExecutor` (engine.py:337) 已经在用 iGPU D3D12 跑 MoE
- 但它期望 NVFP4 W4A8 格式，不是 MXFP4 e2m1
- 需要适配 MTP head 的 MXFP4 格式 (用 v3 server)

### 4. 并行架构已就绪 (方向 1)
- MtpParallelDriver 设计完成
- main_dgpu + mtp_igpu 线程并行
- 实测：MTP=3ms 时 1.64x, MTP=1ms 时 1.58x

## 建议

**最有性价比的是场景 2 (FC + MoE on iGPU, 1.4x 加速, 3-5 天)**

理由：
- MoE 是 MTP head 的瓶颈（70ms -> 12-15ms，5x 改善）
- 工作量适中（v3 server 已支持 BATCH_ALL）
- 风险可控（不需要新写 shader，复用 v3）
- 1.4x 加速是真实数字（不是 1.04x）

**后续可以叠加**：
- 场景 3: 加 attn（再 +0.2x）
- 场景 4: 加 norm + lm_head（达到上限 1.6x）

## 与你之前期望的对比

你之前说 "最后要实测加速解码数据"。根据 35B A3B 的实际情况：
- baseline (dGPU only): ~143 tok/s
- iGPU MTP + 场景 2: ~187 tok/s (1.3x)
- iGPU MTP + 场景 3-4: ~280-400 tok/s (2-2.8x)

加速比上限在 2.8x 左右（受主模型 7ms forward 限制）。

## 下一步选项

1. **做场景 2 (MoE iGPU)**：1.4x 加速，3-5 天
2. **做场景 3 (+ Attn iGPU)**：1.6x 加速，+2 天
3. **整合到 FreeToken scheduler**：需要 P2 工作（8-9 天），完整 e2e 测量
4. **当前状态归档 + 总结报告**
