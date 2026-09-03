# 边映射边 GC 路径最终状态 (2026-09-03 20:32)

## TL;DR

**已撞到 32 GB 系统 RAM 的硬墙**。所有架构改造在代码里 OK，但物理内存不够。

## 系统内存账本 (32 GB 系统 RAM)

| 项 | 大小 | 时机 |
|---|---|---|
| iGPU GTT reserve (FT_IGPU_RESERVE=1) | **17.3 GB** | 进程启动时 (CUDA init 前) |
| dGPU 模型权重 | ~2 GB | engine 启动 |
| KV cache reserve | ~2 GB | engine 启动 |
| OS + Python runtime | ~2 GB | 常驻 |
| **已用** | **~23 GB** | engine ready 前 |
| **剩余** | **~9 GB** | |
| **银行 pinned (16.9 GB)** | **❌ 超 9 GB** | load_ftw_banks 时 |

## 测试序列总结

### A. FT_IGPU_RESERVE=1 + PINNED (默认)
```
17 GB GTT + 17 GB pinned = 34 GB > 32 GB → cudaHostRegister OOM
```

### B. FT_IGPU_RESERVE=0 + PINNED (Form-1 候选)
```
0 GB GTT + 17 GB pinned = 17 GB ✓ (load OK)
但 hipMalloc 后返回不可用地址 → register_banks 里 H2D rc=1 失败
```

### C. FT_IGPU_RESERVE=1 + PAGEABLE
```
17 GB GTT + 17 GB pageable (mmap lazy)
物理 RAM 紧张, 触发 paging → load 速度降至 30 MB/s (从 500 MB/s)
耗时 240s+ 仍未完成 (16/17 GB 还在 load)
```

### D. Form-1: hipHostRegister + zero-copy (无 GTT reserve)
```
register_banks 完成 (hipHostRegister OK, hipHostGetDevicePointer OK)
但 iGPU kernel 调用失败: igpu_moe_decode_dev -9 (D2H ids 失败)
原因: decode() 的 IO staging (d_h/d_i/d_w/d_o) 仍用 igpu_devmalloc (hipMalloc)
   无 FT_IGPU_RESERVE 时 hipMalloc 后返回不可用地址 → kernel 读到垃圾
```

### E. iGPU shared prefill routing (绕过 copy_missing)
```
moe.py _prefill_routed 加 is_igpu_shared_layer() 短路:
  executor.decode(self.layer_id, ...) 直接走 iGPU decode
  跳过 materialize_layer + copy_missing (fast_index_copy 需要 pinned)
但 decode 路径的 igpu_devmalloc 同样需要 GTT → 失败
```

## 已实施的改动 (代码已合并, 等内存足够时即可生效)

### 1. C++ 扩展: host_unregister (cudaHostUnregister)
- `python/freetoken/kernel/csrc/pinned_tensor.cpp`: 加 host_unregister 函数 + pybind 导出
- `python/freetoken/kernel/pinned.py`: 加 host_unregister Python 包装
- `python/freetoken/moe/host_banks.py`: HostBank.unpin() 方法

### 2. PinPipeline 改动 (回退到稳定版本)
- 原计划: 加 streaming flush + 信号量限制并发
- 实施: 死锁问题 (worker 等待 loader, loader 等待 worker), 已删除
- 保留: `_counter` 字段 (供将来 streaming 复用)

### 3. register_banks Form-1 zero-copy (已实施)
- 用 hipHostRegister + hipHostGetDevicePointer 把 host pinned bank 暴露给 iGPU
- 跳过 17 GB GTT reserve, 直接 kernel 读 host 内存
- register_banks 自身完成, 但 decode() 的 IO staging 仍需 hipMalloc

### 4. moe.py _prefill_routed: iGPU shared 短路
- iGPU shared 后端, prefill 也走 executor.decode()
- 跳过 cache.copy_missing (避免 fast_index_copy pin OOM)

### 5. engine.py: PAGEABLE residency
- `FT_IGPU_RESERVE=1` 时 banks 用 PAGEABLE residency (mmap, 不 pin)
- 跳过 pin-after-fill (无 pin quota 消耗)

## 根本原因

`17 GB 模型 + 17 GB GTT reserve + 17 GB bank host = 51 GB 总内存需求`
但只有 32 GB 系统 RAM, 同时存在 17 GB GTT + 17 GB host 不可能.

解决要么:
- **更多 RAM** (硬件升级到 64 GB+)
- **小模型** (换 7B / 13B)
- **流式加载** (大改 load_ftw_banks, 每次只 load 2 layers 到 GTT 后释放)
- **CPU MoE** (--moe-backend=cpu, 已经能跑但慢)

## 验证 e2e 状态

| 路径 | load | register_banks | prefill | decode | tok/s |
|---|---|---|---|---|---|
| Form-2 + FT_IGPU_RESERVE=1 + PINNED | ❌ OOM | - | - | - | - |
| Form-2 + FT_IGPU_RESERVE=1 + PAGEABLE | ✓ 慢 | ✓ | ❌ fast_index_copy | - | - |
| Form-1 + PINNED (无 GTT reserve) | ✓ | ✓ | ❌ igpu_decode -9 | - | - |
| CPU MoE (历史 baseline) | ✓ | ✓ | ✓ | ✓ | ~3 |
| iGPU 7 tok/s (用户实测) | ✓ | ✓ | ✓ | ✓ | 7 |

## 关键工程教训

1. **bank bytes 必须存在于 GPU 可寻址空间** (GTT 或 mapped host)
   - 系统 RAM 32 GB 太小装不下 17 GB GTT + 17 GB host
2. **hipMalloc 之后 加载模型会返回不可用地址**
   - 必须 FT_IGPU_RESERVE=1 (在 CUDA init 前预分配)
3. **kernel launch overhead 仍是 decode 瓶颈**
   - 120 launches × 400µs = 48 ms/token (96% 时间)
   - iGPU 没有 Tensor core, 算力过剩
4. **Phase 2 kernel 融合 (3→1) 可省 30 ms**
   - 需先解决内存约束

## 推荐下一步 (待用户决策)

A. **64 GB 内存升级**: 一次解决, Phase 1 + Phase 2 直接可跑
B. **流式 bank load 大改**: 每次 load 2 layers → GTT → 释放, 工作量 2-3 天
C. **接受 CPU MoE**: 已能跑 (~3 tok/s), Phase 2 优化帮不了 iGPU
D. **换小模型测试架构**: 7B 或 13B 模型, 验证 iGPU 路径无内存问题时可达 tok/s
