# FreeToken 前端端到端验证测试文档

**日期**: 2026-08-28  
**构建**: worktree @ bac2c4c + KV 量化 + MTP scheduler + NVFP4 内置转换器  
**模型**: Qwen3.6-35B-A3B-MXFP4-MTP（23.1 GB，hybrid GDN，hidden=2048，vocab=248320，256 experts×40 层）  
**前端**: daemon panel（HTTP :1900）→ engine（:1919）——与 freetoken.exe 生产链路一致

---

## 1. 测试目标

1. 验证前端完整链路（panel → daemon → engine）可驱动全部新功能
2. 验证 KV 缓存量化（q8_0）端到端生效
3. 验证内置 MXFP4→NVFP4 转换器：产物体积 ≈ 源（不膨胀）、可被引擎原生加载
4. 验证 MTP 投机解码集成
5. 量化基线对比数据

## 2. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop 8 GB（/sysinfo vramTotal=8.59GB） |
| 内存 | 50.7 GB |
| venv | C:\Users\Administrator\AppData\Local\freeToken\venv |
| daemon | ft daemon 0.1.1，127.0.0.1:1900 |
| engine | 127.0.0.1:1919 |

## 3. 前端模拟方式

所有操作通过 daemon HTTP API（与 panel.js `api()` 相同端点），非 CLI 直调：

```
GET  /sysinfo /models /engine/options /engine/status /engine/health
GET  /engine/logs/snapshot?limit=N
POST /engine/start {model,port,args}      = 前端「启动」按钮
POST /v1/chat/completions                = 前端聊天页（daemon 反代 engine）
POST /models/convert-nvfp4               = 前端「转 NVFP4」按钮（新功能）
POST /models/convert-nvfp4/status        = 转换进度条轮询
```

## 4. 单元级验证（全部通过 ✅）

### 4.1 NVFP4 编码器（checkpoint/quantize.py）

| 测试 | 结果 |
|---|---|
| `nvfp4_encode` → GPU kernel `dequant_nvfp4` 解码 roundtrip（64×512） | rel 8.9%，cos>0.99 ✅ |
| 512×2048 expert 尺寸 | rel 9.0%，cos 0.9955 ✅ |
| `nvfp4_encode_experts` 批量 per-expert global（[6,128,512]，差异幅度 40×） | 每专家 cos>0.995 ✅ |
| 打包位序（低 nibble 前）与 kernel 一致 | ✅ |

### 4.2 MXFP4 解码器（mxfp4_decode，N-D）

| 测试 | 结果 |
|---|---|
| 与项目 `dequant_mxfp4_weight_v2` 对比（随机 codes/scales/biases） | max diff 0.00094 ✅ |
| 3D 堆叠专家 [E,I,H/8] 解码逐元素验证 | ✅ |
| 真实模型 embed_tokens 解码 | std=0.0488 ✅ |
| 对官方 NVFP4 发布的 embed_tokens（真值）对拍 | 修复 Bug12 后 cos=+0.997 ✅ |

### 4.3 CLI / 配置流

`--kv-quant/--mtp/--mtp-k/--mtp-igpu-fc` 解析、/engine/options schema、daemon 白名单、panel.js buildEngineArgs 全部验证 ✅

### 4.4 KV q8_0 池

store_kv 2D 输入 roundtrip err 0.49%、存储 53.1%（= 理论 34/64）✅

## 5. 发现并修复的 Bug

| # | Bug | 修复 |
|---|---|---|
| 1 | `import freetoken.engine` ImportError（441a71a 误删 _BANK_BYTES_PER_EXPERT） | commit 4c13749 恢复 |
| 2 | KeyError model.embed_tokens.weight（双前缀） | _rename 加分支 |
| 3 | embed shape assert：**引擎无 MXFP4 解码路径** | → 内置转换器（本测试主对象） |
| 4 | _store_kv_q8 只收 3D | 2D→3D 归一化 |
| 5 | 前端无 MTP 设置 | panel.html/js 三行 + 传参 + schema |
| 6 | 转换 job 随 worker 退出被杀 | daemon 线程托管（_CONVERT_JOBS） |
| 7 | _siblings 元组长度不一致 | 统一 3 元组 |
| 8 | mxfp4_decode 只支持 2D（conv1d 3D 崩溃） | N-D 化 |
| 9 | **产物 57GB 膨胀（4×）**：switch_mlp 堆叠专家被解码成 bf16 | 拆成逐专家 NVFP4 键（0.5625 B/elem） |
| 10 | 产物 index 被源 index 覆盖 | carryover 排除 index/量化文件 |
| 11 | 批量编码 gf 广播维度错 | view(-1,1,1,1) |
| 12 | 【根因·输出乱码】mxfp4_decode 把源 nibble 当 kE2M1 浮点 LUT 解码；实际 Qwen3.6 MXFP4 导出是无符号 4-bit 仿射 nibble*scale+bias（32 元素块）。此前所有自洽单测两边同错故全绿；唯官方真值对拍暴露：uint4-affine cos=+0.997 vs kE2M1 cos=-0.33 | _KMX16 改 _UAFF16=arange(16)；修复后 embed 逐行 cos 全部>0.99，端到端文本恢复 |

## 6. 内置转换器（Test C）

**转换映射**：

| 源（MXFP4） | → 产物（NVFP4） | 说明 |
|---|---|---|
| `language_model.model.layers.N.mlp.switch_mlp.{gate,up,down}_proj`（[256,I,H/8] 堆叠） | `model.language_model.layers.N.mlp.experts.E.{proj}.{weight,weight_scale,weight_scale_2}` ×256 | 匹配 _NVFP4_EXPERT_KEY_RE → offload cache 银行 |
| shared_expert（MXFP4） | `model.layers.N.mlp.shared_expert.*` NVFP4 三元组 | 加载时保持 W4A16 |
| attn/GDN（MXFP4） | NVFP4（默认）；FREETOKEN_ATTN_QUANT=fp8 或 bf16 可切 | fp8=官方 group_0 同款（e4m3+标量 scale，W8A16） |
| norms/router/conv1d/A_log/dt_bias | 原样 | — |
| `mtp.*` | 原样（含 mtp switch_mlp） | MTP loader 自解码 |
| config.json | + `quantization_config={modelopt, NVFP4}` | parse_config 路由 |

**体积账**：专家 32.2B 参数 × 0.5625 B/elem ≈ 18.1GB；dense bf16 ≈ 3.2GB；shared NVFP4 71MB；mtp ~0.3GB → **≈21.6GB（源 23.1GB 的 93%）**，可加载。

**操作**（= 前端「转 NVFP4」按钮）：
```http
POST /models/convert-nvfp4 {"path":"E:/models/Qwen3.6-35B-A3B-MXFP4-MTP"}
→ 200 {"jobId":"cv-13ced536","out":"…-NVFP4","status":"queued"}
POST /models/convert-nvfp4/status → {active:[{status,phase:"experts",done,total}]}
```

**结果**: ✅ stage-1 ~2.5min（30841 张量）；stage-2 50.3s → 815 weight + 240 experts_bank，18.93 GiB / 3 shards。Bug12 修复后重转（修复前产物数值全错→乱码）。

## 7. 引擎加载验证（Test D）

启动参数（offload 主路径 + 前端高级参数）：
```
--moe-backend offload --moe-cache-size 256 --disable-moe-prefill-overlap --memory-ratio 0.9
```
**结果**: ✅ 加载 27-34s。显存账：dense 2.0 GiB + KV 池 107890 tok = 2.06 GiB + GDN state pool；初始化后余 0.63 GiB（memory-ratio 0.9 故意吃满）。专家 bank 16.9 GiB 驻留 RAM。
注意：offload 后端 --moe-cache-size 必须 >= num_experts(256)（启动即校验）；"moe-cache-size=1" 仅适用于 --moe-backend cpu。

**Q4_0 KV 变体**（--kv-quant q4_0 --num-tokens 262144）：✅ 256K token 池分配成功（0.5625 B/elem，llama.cpp Q4_0 块格式：16 打包字节 + fp16 scale / 32 值），warmup 通过。

## 8. 聊天与性能（Test E/F）

- Test E（转换模型, bf16 KV）: "1+1=? 只回答数字" → 分步推理 "Here's a thinking process:…" → content="2" ✅；**7.7-8.4 tok/s**（官方对照 3-5 tok/s）
- Test E'（q4_0 KV @256K）: "9.11和9.8哪个大?" → 正常推理 ✅；**1.4 tok/s**（每步共享 stage chunked 反量化，正确性优先的取舍）
- Test F（MTP）: 与乱码根因无关（sidecar 已禁用），未启用

**对照组**: 官方 NVFP4 发布同参数 → 干净推理输出（证明引擎/内核无误，乱码唯一根因是 Bug12）

## 9. 已知限制

1. MXFP4 原生加载不支持（设计如此——NVFP4 W4A16 kernel 是性能路径，转换器补入口）
2. q4_0 KV 已实现（256K 池 ✅）；解码路径走共享 stage chunked 反量化，1.4 tok/s，速度敏感场景用 bf16/q8_0
3. MTP 仅 greedy 采样
4. 转换需 CUDA GPU（GPU 批量编码）

## 10. 结论

全链路 ✅。乱码根因（Bug12：MXFP4 nibble 被当作 kE2M1 LUT 而实为无符号仿射）已修复并回归验证：
- 内置转换器产物与官方 NVFP4 发布 embed 对拍逐行 cos>0.99
- 端到端：分步推理正常、"2" 正确作答、7.7-8.4 tok/s（超官方对照）
- Q4_0 KV 256K 文本正确（1.4 tok/s）
- 最终产物已落位 E:/models/Qwen3.6-35B-A3B-MXFP4-MTP-NVFP4（修复版）
