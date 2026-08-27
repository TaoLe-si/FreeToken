# P0 完成报告 (2026-08-27 16:50)

## 根本原因 (root cause)
原 t_mxfp4_gemv_v3_server.exe 加载了 `t_mxfp4_gemv_sk.dxil`（MXFP4 e8m0 byte 格式 shader）
但 Qwen3.6-35B-A3B-MXFP4-MTP 模型的 MXFP4 权重实际是 **NVFP4 格式**（fp16 scale + per-block bias），
需要 `t_nvfp4_gemv_sk.dxil`。

同时 v3 server 的 dispatch 路径 slot 2/3 错位（rB/rA 装的数据与 shader 期望的 t2/bias, t3/act 不匹配）。

## 修复
1. **dxil 文件**: t_mxfp4_gemv_sk.dxil → t_nvfp4_gemv_sk.dxil (6984 bytes)
2. **dispatch 路径** (line 248-256) 改为:
   - slot 0 = rW (packed) → t0
   - slot 1 = rS (scl)    → t1
   - slot 2 = rA (bias)   → t2 (per-block bias NVFP4)
   - slot 3 = rB (act)    → t3 (act float)
   - slot 4 = rG (gbl)    → t4
   - slot 5 = rR (rowBias)→ t5 (新增)
   - slot 6 = rOut        → u0
   - slot 7 = rCb         → b0
3. **rA 资源大小**: M*4 → M*ns*4 (per-block bias)
4. **act 转 float**: int32 → float (server 端转换)
5. **barrier 数组**: 6 → 7 (加 rR barrier)
6. **协议**: szB 现在是 M*ns*4 (per-block bias, NVFP4 格式)

## 验证
- T1: zero = 0 ✓
- T2: bias=5, scl=0, zero input = 0 ✓ (公式 (0+5)*0=0)
- T3: packed=1 scale=1 act=1 bias=0 = 32 ✓
- T4: packed=1 scale=1 act=1 bias=5 = 37 ✓
- T5: scl=0, bias=5, zero input = 0 ✓ (公式 (0+5)*0=0)
- T6: M=4 K=32 = [32,32,32,32] ✓
- T7: M=4 K=32 + bias=1 = [33,33,33,33] ✓

## 重要文件
- t_mxfp4_gemv_v3_server.cpp (修复后, 360 行)
- t_mxfp4_gemv_v3_server.exe (280576 bytes, 编译通过)
- t_mxfp4_gemv_sk.dxil = t_nvfp4_gemv_sk.dxil (copy)
- t_p0_diag3.py (更新为 NVFP4 协议)
- t_p0_simple.py (更新为 NVFP4 协议)

## 下一步
- A 场景: MTP head FC + MoE on iGPU
- B 场景: +Attn on iGPU
- C 场景: FreeToken scheduler integration + e2e tok/s
- D 场景: archive + final report
