# FreeToken C++ 胶水层下沉设计 (MTP 关键路径)

> 目标: 把 FreeToken 桌面端 Python 胶水层拖累 MTP 性能的关键路径下沉到 C++, 通过 pybind11 直接链接到现有 Python 调用方, 消除 GIL / dict-lookup / 每帧 numpy 往返带来的 ~10-15ms/round Python overhead.

---

## 1. 背景与现状

| 路径 | 当前实现 | Python overhead 占比 | 每 round 调用次数 (K=2 verify + K draft) |
|---|---|---|---|
| `python/freetoken/kernel/igpu_fc.py` | `subprocess.Popen` + 同步 `read_exact` + `numpy` 往返 | ~3-4ms / call (含 GIL + 2x memcpy) | verify: 1 (per-layer fwd ×24); draft: 1/step × K steps |
| `python/freetoken/models/qwen3_5_moe/gdn.py::forward` | 24 层 × 多 ops × `torch.Tensor` 字典访问 | ~50us / launch × ~96 launches = ~5ms | verify: 1/层 × 24 = 24; draft: 0 |
| `python/freetoken/models/qwen3_5_moe/mtp.py::MtpHead.forward_with_state` | 6 个 torch op 链 + 1 iGPU call | ~3-4ms / call | draft: K 次 / round |
| `python/freetoken/engine/mtp_driver.py::MtpDriver.draft` | K 次循环调 `forward_with_state` + `.item()` GIL sync | ~0.5ms / step (含 `.item()` 同步) | 1 调 / round (但循环内含 K 个 `forward_with_state`) |

合计 K=2 draft round 大致: 24 × GDN launch ~5ms + K × MtpHead ~7ms + iGPU FC ~7ms + driver loop overhead ~1ms ≈ **20-25ms Python overhead/round**, 与设计文档描述一致.

---

## 2. 范围界定

下沉优先级 (P0 → P2):

### P0 - 必做 (直接瓶颈)

1. **`IgpuService`** —— 把 `IgpuFcClient` (一次性 STATELESS 路径) 和 `IgpuFcSticky` (持久 weight 路径) 合并为一个 C++ 类. 两类的 IPC 协议在 `benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server.cpp` 已稳定 (FC_LOAD / FC_CALL / STATELESS).
2. **`MtpHead`** —— 把 `Qwen3_5MtpHead.forward_with_state` 单步 fused forward (RMSNorm × 2, cat, fc, residual, RMSNorm × 2, attn, MoE, RMSNorm, lm_head logits) 整个下沉为一个 C++ 类方法, 直接吃 torch::Tensor 输入, 内部驱动 iGPU + dGPU ops, 返回 torch::Tensor.

### P1 - 推荐 (增益大, 风险中等)

3. **`GdnDispatcher`** —— `Qwen3_5GatedDeltaNet.forward` 的 24 层循环主路径 (decode / prefill / mtp_verify 三条 branch + 5 个子 op) 整体下沉. 每个 layer 一个 `LayerState` (in_proj, conv1d.weight, A_log, dt_bias, norm.weight, out_proj), 24 个 layer 共用同一个 dispatcher 实例.

### P2 - 可选 (锦上添花)

4. **`MtpDriver::draft` loop** —— Python 端只调一次 C++ `MtpDriver.draft_k_steps(uid, prev_token_id, prev_hidden, position, K)`, 内部在 C++ 端完成 K 次 `forward_with_state` (避免 `.item()` GIL sync).

**不纳入**:
- 自定义 triton / fla kernel 移植 (那是另一条 workstream, 见 `fla` 集成). 我们只下沉**胶水层**, 计算 kernel 继续调 `freetoken.kernel.fla` / `torch::matmul`.
- `extend_context` 的 batched fc (一次性 per-request, 频率低, 不值).
- `MtpHeadAttention` / `MtpHeadMoe` 子模块的进一步算子融合 (留给 Phase 2 fused kernel 设计).

---

## 3. 公共 API 设计

### 3.1 pybind11 模块

- 模块名: **`_freetoken_igpu`** (放在 `python/freetoken/kernel/_freetoken_igpu/`)
- 与现有 `_pinned_tensor` / `_cpu_moe` 并列, 由同一 `setup.py` 注册构建.
- 链接 `libtorch_cuda` (从已安装的 torch 包, `TORCH_LIB_DIR`), 复用现有 csrc 的链接配置 (`-lcudart`).
- **不**额外引入 pybind11 头: `torch::extension.h` 已自带 pybind11.

### 3.2 C++ 类签名 (header-only view)

```cpp
// python/freetoken/kernel/csrc/glue/igpu_service.h
#include <torch/extension.h>
#include <cstdint>

namespace ft::glue {

class IgpuService {
public:
    // 构造: 启动 iGPU D3D12 server subprocess. server_path 必填 (Windows only).
    // max_M / max_K / max_ns 仅用于启动时 sanity check, 不限制协议.
    IgpuService(std::string server_path, int max_M, int max_K, int max_ns);
    ~IgpuService();

    IgpuService(const IgpuService&) = delete;
    IgpuService& operator=(const IgpuService&) = delete;

    // 一次性 STATELESS 路径 (替代 IgpuFcClient.forward).
    // packed: (M, K//8) uint32 CPU pinned
    // act:   (K,)      int32   CPU pinned
    // scales:(M, K//32) float32 CPU pinned
    // biases:(M, K//32) float32 CPU pinned
    // 返回:   (M,)     float32 torch::Tensor (CPU, 同步返回; 内部已对齐 iGPU 输出)
    torch::Tensor forward_stateless(
        torch::Tensor packed, torch::Tensor act,
        torch::Tensor scales, torch::Tensor biases);

    // 持久 sticky 路径 (替代 IgpuFcSticky.__init__ + __call__ + update_weight + close).
    // 在构造完成后异步上传 weights (FC_LOAD), 后续 call 仅传 act.
    // act: (K,) float32 CPU pinned -> (M,) float32 CPU pinned.
    torch::Tensor fc_call(torch::Tensor act);

    // 热更新 weight (替代 update_weight).
    void update_weight(torch::Tensor packed, torch::Tensor scales, torch::Tensor biases);

    // 关闭 + kill subprocess.
    void close();

    // server 启动超时 (默认 15s, 替代原 _open 中的 sleep + log 轮询).
    void set_ready_timeout(int seconds);

    // 返回最近 N 行 server stderr (替代 get_log), 用于 C++ 端聚合上报 Python.
    std::vector<std::string> get_log(int last_n);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- GdnDispatcher: per-instance 持有 24 个 layer 的权重 ----
//
class GdnDispatcher {
public:
    struct LayerSpec {
        int64_t hidden_size;
        int64_t num_k_heads, num_v_heads;
        int64_t head_k_dim, head_v_dim;
        int64_t conv_kernel_size;
        double rms_norm_eps;
        bool split_proj;        // fp8 / nvfp4 时 true
        bool nvfp4_attn;        // out_proj 用 NVFP4
        bool block_fp8;         // in_proj 用 block fp8
        bool pertensor_fp8;     // in_proj 用 per-tensor fp8
        std::string expert_quant; // "none"/"fp8_block"/...
        std::string attn_quant;   // "none"/"nvfp4"/...
    };

    // layers: 24 个 layer 的 spec + 权重 tensor.
    // pool:   LinearStatePool 的 dGPU 张量句柄 (recurrent_states / conv_states)
    //         — 直接传入 torch::Tensor, 不在 C++ 端管理生命周期.
    GdnDispatcher(std::vector<LayerSpec> specs,
                  torch::Tensor pool_recurrent,   // [L, S, K, V] fp32
                  torch::Tensor pool_conv,        // [L, S, conv_dim, K-1] (dtype 任意)
                  std::vector<torch::Tensor> in_proj_qkvz_weights,   // 每层 1 个
                  std::vector<torch::Tensor> in_proj_ba_weights,      // split_proj 时有, 否则空
                  std::vector<torch::Tensor> in_proj_weights,         // fused 时有, 否则空
                  std::vector<torch::Tensor> conv1d_weights,
                  std::vector<torch::Tensor> a_log,                  // [num_v_heads] fp32
                  std::vector<torch::Tensor> dt_bias,                // [num_v_heads] fp32
                  std::vector<torch::Tensor> norm_weights,           // [head_v_dim]
                  std::vector<torch::Tensor> out_proj_weights);

    // decode forward: 1 call / token / layer / round.
    // 输入: hidden_states [B, H]; fla_meta { cu_seqlens, cache_indices, has_initial_state, track_dst, track_h_row, track_conv_src }
    // 返回: [B, H] bf16.
    torch::Tensor forward_decode(
        torch::Tensor hidden_states,
        torch::Tensor cu_seqlens,            // [B+1] int64
        torch::Tensor cache_indices,         // [B]   int32
        c10::optional<torch::Tensor> has_initial_state);

    // prefill forward: chunk + 一次性 recurrent + conv1d varlen.
    // 输入: hidden_states [total, H]; fla_meta { cu_seqlens, cache_indices, has_initial_state, fresh_state_indices, track_* }
    // 返回: [total, H] bf16.
    torch::Tensor forward_prefill(
        torch::Tensor hidden_states,
        torch::Tensor cu_seqlens,
        torch::Tensor cache_indices,
        torch::Tensor has_initial_state,
        c10::optional<torch::Tensor> fresh_state_indices,
        c10::optional<torch::Tensor> track_dst,
        c10::optional<torch::Tensor> track_h_row,
        c10::optional<torch::Tensor> track_conv_src);

    // mtp_verify forward (varlen over K+1 tokens, per-step snap).
    torch::Tensor forward_mtp_verify(
        torch::Tensor hidden_states,         // [K+1, H]
        torch::Tensor cu_seqlens,            // [0, K+1]
        torch::Tensor cache_indices,         // [1]
        torch::Tensor has_initial_state,
        c10::optional<torch::Tensor> snap_slots);  // [K+1] int64

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- MtpHead: 单步 fused forward ----
//
class MtpHead {
public:
    struct Cfg {
        int64_t hidden_size, vocab_size;
        int64_t num_experts, num_experts_per_tok;
        int64_t moe_intermediate, shared_expert_intermediate;
        int64_t head_dim, num_qo_heads, num_kv_heads;
        double partial_rotary_factor, rms_norm_eps, rope_base;
        bool norm_topk_prob;
    };

    MtpHead(Cfg cfg,
            std::shared_ptr<IgpuService> igpu_fc,    // 注入
            torch::Tensor pre_fc_norm_emb_weight,
            torch::Tensor pre_fc_norm_hid_weight,
            torch::Tensor input_layernorm_weight,
            torch::Tensor post_attention_layernorm_weight,
            torch::Tensor mtp_norm_weight,
            // attn
            torch::Tensor qkv_proj_weight, torch::Tensor q_norm_weight, torch::Tensor k_norm_weight, torch::Tensor o_proj_weight,
            // moe
            torch::Tensor gate_weight,
            torch::Tensor switch_gate, torch::Tensor switch_up, torch::Tensor switch_down,
            torch::Tensor shared_gate, torch::Tensor shared_up, torch::Tensor shared_down,
            torch::Tensor shared_expert_gate_weight,
            // embeddings (来自主模型)
            torch::Tensor embed_table,             // [V, H]
            std::shared_ptr<void> lm_head);        // 任意可调用对象 (Python-side LM head)

    // 单步 fused forward.
    // prev_token_id: int64 scalar
    // prev_hidden:   [1, H] bf16/fp16/fp32 (caller's device)
    // position:      int (rope position)
    // 返回: pair(logits [1, V] bf16, state [1, H] bf16).
    std::tuple<torch::Tensor, torch::Tensor> forward_with_state(
        int64_t prev_token_id,
        torch::Tensor prev_hidden,
        int64_t position);

    // 给已提交的 committed tokens (verify 后) 追加 KV.
    void extend_context(torch::Tensor tokens, torch::Tensor hiddens, int64_t start_pos);

    // Draft cache rollback / reset.
    void truncate_kv(int64_t n);
    void reset_draft_cache();
    int64_t kv_len();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ft::glue

// ---- pybind11 模块入口 ----
//
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<ft::glue::IgpuService, std::shared_ptr<ft::glue::IgpuService>>(m, "IgpuService")
        .def(py::init<std::string, int, int, int>(), py::arg("server_path"), py::arg("max_M")=8, py::arg("max_K")=4096, py::arg("max_ns")=128)
        .def("forward_stateless", &ft::glue::IgpuService::forward_stateless)
        .def("fc_call", &ft::glue::IgpuService::fc_call)
        .def("update_weight", &ft::glue::IgpuService::update_weight)
        .def("close", &ft::glue::IgpuService::close)
        .def("set_ready_timeout", &ft::glue::IgpuService::set_ready_timeout)
        .def("get_log", &ft::glue::IgpuService::get_log);

    py::class_<ft::glue::GdnDispatcher>(m, "GdnDispatcher")
        .def(py::init<...>())
        .def("forward_decode", &ft::glue::GdnDispatcher::forward_decode)
        .def("forward_prefill", &ft::glue::GdnDispatcher::forward_prefill)
        .def("forward_mtp_verify", &ft::glue::GdnDispatcher::forward_mtp_verify);

    py::class_<ft::glue::MtpHead>(m, "MtpHead")
        .def(py::init<...>())
        .def("forward_with_state", &ft::glue::MtpHead::forward_with_state)
        .def("extend_context", &ft::glue::MtpHead::extend_context)
        .def("truncate_kv", &ft::glue::MtpHead::truncate_kv)
        .def("reset_draft_cache", &ft::glue::MtpHead::reset_draft_cache)
        .def("kv_len", &ft::glue::MtpHead::kv_len);
}
```

### 3.3 Python 侧包装 (瘦包装, 不留逻辑)

```python
# python/freetoken/kernel/_igpu_glue.py
import os
import torch
from . import _freetoken_igpu as _C   # pybind11 module

class IgpuServicePy:
    """Thin Python wrapper over _freetoken_igpu.IgpuService.
    Kept for back-compat: existing callers (mtp.py / mtp_driver.py) see the same
    surface as the old IgpuFcSticky. The torch() bridge that maps cuda/cpu in ->
    same-device out is still done in Python (it owns the .unsqueeze(0) shape
    convention MtpHead.forward_with_state expects)."""
    def __init__(self, packed, K, scales=None, biases=None, server_path=None):
        self._impl = _C.IgpuService(server_path or _default_server_path(), ...)
        self._impl.update_weight(packed.cpu(), scales.cpu() if scales is not None else None,
                                  biases.cpu() if biases is not None else None)
        self.K = K; self.M = packed.shape[0]
    def __call__(self, act_flat): ...
    def batch(self, x): ...
    def update_weight(self, ...): ...
    def close(self): ...
    def torch(self): return _TorchFcBridge(self)
```

### 3.4 内存管理策略

- **dGPU 端张量 (recurrent state, pool, embed, lm_head)**: 全部用 `torch::Tensor` 句柄传入, C++ 端**不持有**所有权, 不调用 `.reset()` / `~Tensor()`. 通过 `at::Tensor` 引用计数自动管理. Python 端 update pool 时重新调 init.
- **CPU pinned 张量 (igpu IPC 临时 buf)**: 复用 `freetoken.kernel._pinned_tensor` 暴露的 `alloc_pinned_tensor`. 在 `IgpuService::Impl` 内一次性预分配最大尺寸 (M×K 字节) 的 pinned buffer, 每次 `forward_stateless` / `fc_call` 时通过 `cudaMemcpyAsync` 从 input tensor 拷到 pinned buf. 不每次分配/释放 pinned 内存.
- **iGPU 输出 (M floats)**: iGPU server 写 readback heap, C++ 端直接 map + memcpy 到返回的 CPU `torch::Tensor` (一次性 `.empty({M}).pin_memory()`).
- **避免**: `std::vector<uint8_t>` 临时 buf 拷贝, `numpy` ↔ `torch::Tensor` 双向转换.

### 3.5 CUDA stream sync

iGPU server 是独立进程, 不共享 dGPU stream. 同步分两步:

1. **dGPU → host**: caller 的 torch::Tensor 是 dGPU 的, `cudaMemcpyAsync(d2h)` 在 dGPU **default stream** 上发起, C++ 端 `cudaStreamSynchronize(default_stream)` 等写完再写 iGPU subprocess pipe. 这是一处必要的 host sync, 但**仅一次** (替代原 Python 端 `.cpu()` 同步).
2. **iGPU subprocess IPC**: pipe write (atomic, blocking) + read_exact (blocking). 这部分 sync 是固有的, 不可消, 但 C++ 端通过 `poll` + non-blocking io + 自旋等待 syscall (`read_exact`) 比 Python `subprocess` + GIL 调度延迟更低.

`MtpHead.forward_with_state` 内: dGPU compute → `.cpu()` 到 pinned buf → iGPU server call → iGPU output → `.to(dGPU_device, non_blocking=True)` → 后续 dGPU compute (RMSNorm / attn / MoE). **保留一处 d2h + h2d** (不可避, MTP fc 必须吃 fp32 CPU 输入, 这是协议要求). 整条 fused forward 内的所有 dGPU op 保持在 default stream, C++ 端不再插入额外 sync.

### 3.6 错误处理

- iGPU subprocess 退出 / 协议错误 → C++ 抛 `std::runtime_error(msg + server_log_tail)`, pybind11 自动转 Python `RuntimeError`.
- torch::Tensor 形状 / dtype 不匹配 → `TORCH_CHECK` 抛 `c10::Error` → pybind11 转 `RuntimeError`.
- GdnDispatcher 中 fla kernel 调用失败 → 透传 `c10::Error`, 附 layer_id + kernel 名.
- LM head 在 Python 端 (因为 LM head 是共享, 由主 engine 管), 通过 `std::function` callback 注入; C++ 端捕获 Python exception 并以 `std::runtime_error` 形式抛出 (pybind11 9.x 自动转, 11+ 需显式 `e py::cast`).

---

## 4. 编译系统

### 4.1 推荐方案: 复用现有 `setup.py` (而不是新开 pybind11_add_module + CMake)

**理由**:
1. 现有 `_pinned_tensor` / `_cpu_moe` 已用 `torch.utils.cpp_extension.BuildExtension`, 链路稳定.
2. `pybind11_add_module` 是 CMake-side, 需要新开 `CMakeLists.txt` + 把 wheel 构建链路 (`setup.py` → `pip install`) 双轨, 维护成本高.
3. 现有 `setup.py` 已校验 nvcc ↔ torch CUDA 版本对齐 (`_toolchain.py::check_nvcc_matches_torch`), 这是硬约束.
4. libtorch C++ API 在 `torch::extension.h` 里就够了, 不需要额外 CMake.

### 4.2 集成方式

```python
# setup.py 增量修改
from torch.utils.cpp_extension import CppExtension

setup(
    ext_modules=[
        ...  # 现有 _pinned_tensor / _cpu_moe 不动
        CppExtension(
            name="freetoken.kernel._freetoken_igpu",
            sources=[
                "python/freetoken/kernel/csrc/glue/igpu_service.cpp",
                "python/freetoken/kernel/csrc/glue/gdn_dispatcher.cpp",
                "python/freetoken/kernel/csrc/glue/mtp_head.cpp",
                "python/freetoken/kernel/csrc/glue/pybind_module.cpp",
            ],
            include_dirs=cuda_include_dirs + [
                str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "glue"),
            ],
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=["-O3", "-std=c++17", "-pthread"],
            extra_ldflags=["-lstdc++fs"],   # for std::filesystem
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
```

### 4.3 跨平台要点

- **Windows (MSVC + CUDA 13)**: 默认 MSVC ABI, `_read` / `_write` 走 `_setmode(_O_BINARY)`. iGPU server 进程用 `CreateProcessW` (替代 Python 的 `subprocess.Popen`), 通过 `STARTUPINFO` 重定向 stdin/stdout pipe. stderr 用 `CreateFileW` + `ReadFile` 异步读取 (event-driven, 不开 background thread).
- **Linux (gcc + CUDA)**: iGPU server 路径在 Linux 上**不存在** (D3D12 是 Windows only). Linux build 必须把 `IgpuService` 编译为 stub: 构造时返回"unavailable", `forward_stateless` / `fc_call` 抛 `std::runtime_error("iGPU service is Windows-only")`. Pybind11 在 Linux 上 module 可正常导入, 但运行时会失败. 这与 `benchmarks/cpu_moe_microbench/t_mxfp4_gemv_v3_server.cpp` 的 Windows-only 性质一致.
- **macOS**: 同 Linux, 不支持 iGPU, 视为 Linux-like.

### 4.4 编译时间优化

- **ccache**: `setup.py` 启动时检测 `CCACHE_DIR`, 注入 `extra_compile_args=["-fprofile-arcs", "-ftest-coverage"]` 仅 debug build. release 用 `ccache nvcc` wrapper (环境变量驱动, 无需改 setup.py).
- **增量编译**: ninja + ccache 默认就行. 关键: 把 `GdnDispatcher` 拆为 `gdn_dispatcher.cpp` + `gdn_layer_state.cpp` (template header only), 改 layer spec 时只重编 dispatcher 主文件.
- **预编译头**: `torch/extension.h` + `pybind11/pybind11.h` 用 `-include` 注入, 减少 nvcc 重 parse 时间.
- **unity build (jumbo)**: 4 个 csrc 文件可以用 `BuildExtension` 的 `with_options(jit_compile=False)` 手动 unity, 把 pybind_module.cpp + 各 impl.cpp 合一. 权衡 debug 单步可读性.

---

## 5. 测试策略

### 5.1 数值对齐 (vs Python 版)

新增 `tests/test_glue_cpp.py`:

- **`test_igpu_stateless_equivalence`**: 随机 packed/scales/biases (M=8, K=2048/4096), 同一 act, 调 `IgpuFcClient.forward` (Python) vs `IgpuService.forward_stateless` (C++), 断言 `np.allclose(rtol=1e-4, atol=1e-5)`.
- **`test_igpu_sticky_equivalence`**: 同上, 但走 `IgpuFcSticky` vs `IgpuService.fc_call`. 注意 sticky path 在 Linux 不跑 (server unavailable), Windows-only.
- **`test_mtp_head_equivalence`**: 加载相同 mtp head 权重到 Python 版 (`Qwen3_5MtpHead`) 和 C++ 版 (`MtpHead`), 给相同 prev_token_id + prev_hidden, 比 logits (`cos ≥ 0.9999`) + state (`max abs diff ≤ 1e-3 bf16`).
- **`test_gdn_dispatcher_equivalence`**: decode 路径, 1-2 个 token, 比 `Qwen3_5GatedDeltaNet.forward` vs `GdnDispatcher.forward_decode`.

### 5.2 性能 benchmark

新增 `benchmarks/test_glue_overhead.py`:

- **`test_draft_round_overhead`**: 测 K=2 draft round 的 wall time:
  - baseline: 全 Python (现状)
  - P0: 仅 `IgpuService` + `MtpHead` 下沉
  - P1: + `GdnDispatcher` 下沉
  - P2: + `MtpDriver.draft` loop 下沉
  预期每步 1-2ms 递减, K=2 round 从 ~50ms → ~25ms.
- **`test_i_gpu_throughput`**: 单纯 fc call 吞吐: 1000 次 fc_call, 比 Python wrapper vs C++ wrapper 的 QPS.
- **`test_gdn_launch_overhead`**: 24 层 decode, 比 Python 链 vs C++ dispatcher 的 launch overhead (用 `torch.cuda.Event` 测纯 GPU time + host wall time 差).

### 5.3 集成测试

- **`test_end_to_end_mtp_draft`**: 加载真实 checkpoint (gated by `pytest.mark.needs_weights`), 跑 5 round `MtpDriver.draft`, 比 accept rate / 数值 vs Python baseline. 不允许任何退化.
- **`test_concurrent_drivers`**: 两个 driver 实例 (各持有独立 IgpuService) 同进程跑, 验证 `IgpuService` 不是 singleton, 各管各的 subprocess + fd.
- **`test_subprocess_crash_recovery`**: 模拟 server 进程 crash (kill -9), 验证后续 `fc_call` 抛清晰 `RuntimeError`, 不 zombie fd.

---

## 6. 风险点

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | iGPU subprocess stdin/stdout 阻塞导致 GIL 抢占 | 高 | C++ 端用 OS 原生 pipe (`CreatePipe` / `pipe2`), 不走 Python `subprocess`, 阻塞发生在 C++ 线程, Python 不感知. |
| R2 | dGPU `.cpu()` / iGPU `.to(dGPU, non_blocking=True)` 隐式 sync 增多 | 高 | 复用现有 `_pinned_tensor` 已暴露的 pinned buf, 把 `.cpu()` 换成 `cudaMemcpyAsync(d2h, pinned_buf)`. 验证: `torch.cuda.current_stream().synchronize()` 在 MtpHead 内只发生 1 次 (fc call 后). |
| R3 | LM head 由 Python 侧管理, C++ 端回调引入 Python GIL | 中 | 用 `py::function` 包裹, 在 C++ 端先 `py::gil_scoped_release` 再调 (lm_head 是 dGPU compute, 不需要 GIL), 调完 `py::gil_scoped_acquire`. 验证 GIL 持有时长 < 5us per call. |
| R4 | 24 层 GDN weights 占内存大 (~3 GB), 一次性 `init` 传 24×9 = 216 个 torch::Tensor 慢 | 中 | `GdnDispatcher.__init__` 只持有 `torch::Tensor` 句柄 (引用计数), 不拷 data. Python 端 lazy 在 `__init__` 之前 `cudaMallocAsync` pool, C++ 端 `pool_recurrent` / `pool_conv` 一次性传引用. |
| R5 | C++ 端 per-call 创建临时 torch::Tensor 触发 dispatcher overhead | 中 | 复用 `at::empty` 的 `MemoryFormat::Contiguous` + 预分配 staging buf, 避免每次走 c10 allocator. |
| R6 | 跨平台: Linux 没有 iGPU server, 必须 graceful fallback | 中 | `IgpuService` 构造时 `if constexpr (!platform_supports_igpu)` 抛 `std::runtime_error("iGPU not available on this platform")`. Python 包装 try/except, 回落 `TorchNvfp4Fc`. |
| R7 | libtorch 版本不匹配导致 ABI 错 | 低 | 复用 `_check_toolchain()` 现有机制. build isolation 用 `torch>=2.11,<2.12` (与 pyproject.toml 一致). |
| R8 | pybind11 + torch::Tensor 的 GIL 管理 (Python 11+ 需显式) | 低 | 集中在 `pybind_module.cpp` 入口处显式 `gil_scoped_acquire/release`, 不在 hot path 内部争抢. |
| R9 | iGPU server 端 stderr 日志暴增, C++ ring buffer 满 | 低 | `get_log(last_n)` 改为 `std::deque<std::string>` + size cap (1024 行), 满了 pop front. |
| R10 | MTP verify varlen GDN path 涉及 snap copy, C++ 端要管 pool slot 写冲突 | 中 | 复用 Python 现有 `LinearStatePool.copy_from` + `conv_states[li][dst].copy_(...)` 的语义, 在 C++ 端用 `at::Tensor::index_copy_` 一行实现. |

---

## 7. 工作量估算与优先级

| Task | 工作量 (人天) | 依赖 | 优先级 |
|---|---|---|---|
| pybind11 骨架 + `IgpuService` (stateless + sticky) | 3 | 无 | **P0** |
| `MtpHead` fused forward (吃 torch::Tensor 输入, 内调 IgpuService + dGPU ops) | 4 | P0.1 (IgpuService) | **P0** |
| `GdnDispatcher` (24 层 + 3 paths: decode / prefill / mtp_verify) | 4 | 无 (但测试需 mtp head) | **P1** |
| `MtpDriver.draft` loop 下沉 (`MtpDriver.draft_k_steps`) | 1 | P0.2 (MtpHead) | **P2** |
| 数值对齐测试 (4 个 equivalence test) | 2 | P0.1, P0.2, P1 | **P0** (与 P0/P1 并行) |
| 性能 benchmark (3 个 benchmark) | 1 | 全部 | **P1** |
| 集成测试 (E2E draft + crash recovery) | 1 | 全部 | **P1** |
| 集成到现有代码路径 (替换 IgpuFcSticky / MtpHead.forward_with_state 调用方) | 2 | 全部 | **P0** (收尾) |
| Linux stub fallback + Windows-only CI 验证 | 0.5 | 无 | **P0** (前置) |
| **总计** | **18.5** | | |

注: 原 prompt 给的是 12-14 天, 我们这里把 GDN dispatcher 拆细 (3 paths × 8 layers worth of code), 多估了 4-5 天. 如果允许 "GDN dispatcher 只做 decode 路径 (P1a), prefill/varlen 留 Phase 2", 可压缩到 14 天.

---

## 8. 与并行其他工作的依赖关系

| 平行工作 | 接口契约 | 我们需要的 |
|---|---|---|
| **iGPU server (`t_mxfp4_gemv_v3_server.exe`) 维护** | FC_LOAD / FC_CALL / STATELESS / QUIT 协议 (现状, 稳定) | 不变. server 端不动. |
| **fla kernel 升级** (`freetoken.kernel.fla`) | `chunk_gated_delta_rule` / `fused_sigmoid_gating_delta_rule_update` API | 不变. GdnDispatcher 调同一 Python binding (走 pybind + Python 回调, 或直接调 ATen custom op). |
| **LM head 量化** (NVFP4 LM head) | `Nvfp4LMHead.forward(state)` API | MtpHead 内 lm_head callback 需要适配新签名. 若 LM head 改为 ATen custom op, MtpHead 直接调 C++, 不需 Python callback. |
| **MoE dense FFN 走 iGPU** (`--dense-ffn-engine igpu`) | iGPU service 多 weight 接口 (`IgpuMultiClient.call_all`) | 不在本次范围. 但**预留**: `IgpuService` 类可加 `call_multi(weights: list[...], acts: list[...])` 成员供后续 MoE dense FFN 下沉复用. |
| **Checkpoint loader 重构** (`load_mtp_head_from_safetensors`) | 输出 `_packed_mxfp4` dict | 不变. MtpHead 构造继续吃这个 dict 的 tensors. |
| **MTP verify rollback** (`_mtp_process_verify`) | `MtpHead.truncate_kv(n)` API | 不变. 已在 MtpHead 设计里. |
| **Windows / Linux CI 矩阵** | 现有 CUDA 13 + torch 2.11 + sglang-kernel 0.4.5 | C++ build 与 pyproject.toml 共用一套, 自动覆盖. iGPU 测试仅 Windows runner. |

**冲突点**: 若 "fla kernel 升级" 工作想把 `gdn_decode_fla` 改为 ATen custom op (跳过 Python `freetoken.kernel.fla` import), GdnDispatcher 可以直接调 ATen op, 进一步去掉一次 Python 跳转. 这是一个**加分项**, 但不在本设计硬约束内.

**无依赖**: 本设计与以下工作正交:
- 模型转换 (`checkpoint/quantize.py`)
- 前端 API server (`freetoken/api`)
- Scheduler / KV cache (`freetoken.engine.scheduler`)

---

## 9. 里程碑

| Milestone | 交付物 | 可测指标 |
|---|---|---|
| M1 (Day 3) | `IgpuService` 单测通过 (stateless + sticky), Python 包装 + fallback 完整 | iGPU QPS 比 Python wrapper +30% |
| M2 (Day 7) | `MtpHead` 单测通过 (数值对齐) | MTP head 单步 wall time 从 ~3.5ms → ~1.0ms |
| M3 (Day 11) | `GdnDispatcher` decode 路径单测通过 | GDN 24 层 launch wall time 从 ~5ms → ~2ms |
| M4 (Day 14) | 集成到 `MtpDriver`, E2E MTP draft 不退化 | K=2 round wall time 从 ~50ms → ~25ms |
| M5 (Day 17) | 性能 benchmark + 文档化 | 收尾 + 报告 |

---

## 10. 不在范围内 (明确)

- fla / triton kernel 自身重写为 C++ (那是 kernel 团队, 不是胶水)
- MTP verify varlen GDN 的 snap copy 优化 (现 C++ 端用 `index_copy_`, 已等价)
- LM head 量化 (另一 workstream)
- 多 iGPU server (single server 假设, 与现状一致)
- CUDA graph capture (留 Phase 2, 触发条件: 全 fused forward 不再有 CPU sync)

---

## 11. 后续可扩展 (Phase 2 候选)

1. **CUDA graph capture** —— `MtpHead.forward_with_state` 全 fused 后可尝试 capture 为 CUDA graph, 进一步去 Python GIL 抢占.
2. **多 iGPU server 池** —— 大 batch draft 时, 多实例 `IgpuService` 各自管一个 subprocess, round-robin 调度.
3. **Async pipelining** —— dGPU compute (RMSNorm / cat) 与 iGPU fc call 重叠 (iGPU 是独立进程, 可与 dGPU 并行, 但需双 buffer 管理 cat input 的所有权).
4. **NUMA-aware pinning** —— Linux 上 (即使无 iGPU) 可用同一类 API 把 CPU bf16 compute MoE 路径包成 C++ `GluonExecutor`, 复用 pybind 模板.

---

*作者: FreeToken Perf / 状态: 设计评审稿 v1 / 待评审*
