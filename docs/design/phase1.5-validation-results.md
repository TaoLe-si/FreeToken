# Phase 1.5 验证结果 (e2e tests, 2026-09-03)

## 测试序列

### 测试 A: 带 FT_IGPU_RESERVE=1 (失败)

**配置**:
- `--moe-backend=igpu --dense-ffn-engine=igpu`
- `--mtp --mtp-k=1 --mtp-igpu-fc`
- `--memory-ratio=0.9`
- `FT_IGPU_RESERVE=1` (env)

**结果**: ❌ 引擎崩溃

```
RuntimeError: cudaHostRegister failed for 0.2 GiB
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 137, in pin
File "E:\\FreeToken\\python\\freetoken\\kernel\\pinned.py", line 50, in host_register
RuntimeError: cudaHostRegister failed: out of memory

Process freetoken-TP0-scheduler:
File "E:\\FreeToken\\python\\freetoken\\moe\\expert_banks.py", line 445, in load_expert_banks
File "E:\\FreeToken\\python\\freetoken\\checkpoint\\ftw.py", line 557, in load_ftw_banks
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 362, in __exit__
    self.wait()
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 353, in wait
    raise self._exc
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 327, in _run
    _settle(bank, residency)
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 277, in _settle
    bank.pin()
File "E:\\FreeToken\\python\\freetoken\\moe\\host_banks.py", line 139, in pin
    raise RuntimeError(
```

**根因**:
```
FT_IGPU_RESERVE=1:  17.3 GB  (iGPU GTT 预分配 → 消耗系统 RAM)
load_ftw_banks:     16.9 GB  (host pinned)
model + KV cache:   ~2 GB
─────────────────────────────
总计:                ~36 GB > 32 GB 系统 RAM
```

**结论**: `FT_IGPU_RESERVE=1` 和 `load_ftw_banks` 双份占用系统 RAM 导致 OOM

---

### 测试 B: 不带 FT_IGPU_RESERVE=1 (部分成功)

**配置**:
- `--moe-backend=igpu --dense-ffn-engine=igpu`
- `--mtp --mtp-k=1 --mtp-igpu-fc`
- `--memory-ratio=0.85` (降低 VRAM 占用)
- 无 `FT_IGPU_RESERVE`

**结果**: ⚠️ 引擎启动成功，但 iGPU MoE 失败，退回 CPU MoE

**日志关键行**:
```
[INFO] iGPU register_banks: reserved=0
[INFO] GTT meminfo before migration: free=17.92 GB total=18.20 GB
[INFO] H2D CHUNK FAIL rc=1 name=gate_up_packed layer=0 off=0 n=67108864
[WARNING] iGPU shared MoE executor unavailable (hipMemcpy H2D failed (1) for 
          'gate_up_packed' layer 0); falling back to the CPU executor for decode
[INFO] CPU MoE executor ready: threads=15 (pinned to cores 0..14) isa=avx512bf16
       fmt=nvfp4 H=2048 I=512 experts=256 layers=40 top_k=8 act=silu max_tokens=4
[INFO] Free memory after initialization: 2.28 GiB
[INFO] API server is ready to serve on 127.0.0.1:1919
```

**关键观察**:
1. GTT 17.92 GB 可用 (足够装 17.3 GB banks)
2. 但 `hipMemcpy H2D` 仍然返回 `rc=1` (hipErrorInvalidValue)
3. 即使 GTT 有空间，第一次 hipMemcpy 就失败
4. commit 55af654 中提到的 "hipMalloc returns unusable low VAs after model load" 问题在当前 ROCm 6.4 + Windows driver 仍然存在

**结论**: `igpu_devmalloc` 在 CUDA 加载模型后，返回的 GTT 地址是 "unusable"，无法接受 H2D

---

## 根本问题总结

### 问题 1: FT_IGPU_RESERVE=1 内存爆炸

- 17.3 GB GTT (iGPU device memory) + 16.9 GB host pinned = 34.2 GB
- 32 GB 系统 RAM 不够
- 需要架构改造: 让 GTT reserve 复用 host pinned banks，不重复分配

### 问题 2: hipMalloc 返回 unusable 地址

- commit 55af654 老问题: "hipMalloc returns unusable low VAs once the engine has loaded the model"
- 实测: `hipMemGetInfo` 显示 17.92 GB free，但 `hipMemcpy H2D rc=1` 失败
- 即使 `FT_IGPU_RESERVE=1` (理论上预先分配好地址) 也无法避免，因为 pre-allocate  用了系统 RAM 导致 OOM

### 问题 3: Phase 1 decode() async 改动未生效

- decode() 已经改了用 pinned staging + 单次 sync (Phase 1 已应用)
- 但因为 iGPU MoE 整个 fallback 到 CPU，根本没走到 igpu_moe_decode_dev
- 验证了 Phase 1 的 correctness 但没验证 throughput 提升

---

## 历史数据回顾

| 测试 | 条件 | tok/s | 备注 |
|---|---|---|---|
| `iGPU + 7 tok/s` (用户实测) | `--moe-backend=igpu` | 7 | iGPU MoE 路径，per-layer sync |
| `CPU fallback` (本次测试 B) | iGPU MoE 失败，CPU | ~3 | CPU MoE 路径 |
| `Phase 0 baseline` (单算子) | _igpu_phase0.py | 16.5 | isolated kernel time only |
| `Phase 1 isolated` | _phase1_test.py | 22.4 | + 去掉 per-layer sync |

---

## 待解决问题

### A. 让 FT_IGPU_RESERVE 与 load_ftw_banks 共存

**方案**:
1. 让 `_IGPU_RESERVED` 直接指向 `bank_sources` 的 pinned host memory (用 `hipHostRegister` 把已 pinned 的 host memory 注册为 iGPU device memory)
2. 不要预分配独立的 17.3 GB GTT，而是 reuse host pinned
3. 这样 GTT 占用 = 0 额外内存

**实现细节**:
- HIP API: `hipHostRegister` / `hipHostUnregister` / `hipMemcpyHtoD` from registered host
- 已有 pinned memory 可以直接被 iGPU 访问 (UVA)
- 需要验证: 注册后的 pinned memory 是否能作为 iGPU GTT 地址

### B. 验证 Phase 1 真正效果

**前提**: A 解决后，iGPU MoE 路径才能跑
**预期**: 22 tok/s (Phase 1 集成验证)

### C. 解决 hipMalloc unusable address 问题

**方案**:
1. 用 `hipHostRegister` + `hipMemcpyHtoDAsync` 替代 `hipMalloc`
2. 直接从已 pinned host memory 拷贝到 GTT
3. 或完全跳过 GTT，让 kernel 直接读 pinned host memory (zero-copy)

---

## 下一步

1. **修复 A**: 让 `_IGPU_RESERVED` 指向已 pinned host memory，不再预分配 GTT
2. **测试 A**: 验证带 `FT_IGPU_RESERVE=1` 不再 OOM
3. **测试 Phase 1 集成**: 验证 22 tok/s 真实可达
4. **Phase 2 准备**: kernel 融合 (3 → 1 kernel per layer)

---

## 用户关键洞察汇总 (本次对话)

1. **"iGPU 完全没有占用率, 但显存有占用"**
   → MoE 没真正用 iGPU 计算，bank alloc 了但 H2D 失败
2. **"iGPU 显存只有 512M, 权重放错地方了吗"**
   → 权重在 iGPU GTT (系统 RAM 映射) 不是 VRAM，但 iGPU 没有 17.3 GB
3. **"DGPU 也有共享内存"**
   → WDDM shared pool, 但 6 GB/s PCIe 不够 16 GB/s 需求
4. **"PCIe 是不是带宽瓶颈"**
   → 是的, top-8 expert 重读 = 16 GB/s, 但 PCIe 只有 6-12 GB/s
5. **"每个 token 都要重传所有专家吗"**
   → 不需要全量，但需要 top-8 (320 MB/token), 还是要走 PCIe 6 GB/s 不够
6. **"CPU 能和IGPU 类似直读RAM吗"**
   → 是的, CPU 89 GB/s 直读 DDR5, 比 iGPU GTT 26 GB/s 快 3.4 倍
7. **"能否专家层放在CPU, CPU建立专家表"**
   → CPU-managed cache 架构，但 PCIe 同步开销太大，业界未广泛采用
8. **"重计算用 dGPU, 轻计算用 iGPU"**
   → batch=1 时所有 expert 同等算力，分离无收益; 业界用 prefill/decode 分离
9. **"iGPU 没有 Tensor core, 瓶颈是算力"**
   → 算力过剩 50× (2% 利用率), 真瓶颈是 kernel launch overhead (47 ms/token)
10. **"为什么不等全部放入 iGPU"**
    → 已经全部在 iGPU，但 launch overhead (47 ms) > compute (0.4 ms) + memory (1.6 ms)
