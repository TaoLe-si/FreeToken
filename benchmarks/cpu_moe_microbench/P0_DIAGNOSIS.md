# P0 Bug 完整诊断报告

## 状态
- **分支**: feature/igpu-mtp-mxfp4
- **v3 server.exe**: 2026-08-27 编译 (279KB)
- **shader (t_mxfp4_gemv_sk.hlsl)**: 声明 StructuredBuffer<uint> packed/scl, int act
- **shader (.dxil, 7408 bytes)**: HLSL 编译后的 bytecode

## P0 现象
所有 STATELESS 调用返回 outv[0] = 0，不论输入。

## 根因
dxil 把 `scl` 和 `act` 编译为 **float element**（不是 uint/int）：

dxil 反汇编显示:
```
Resource bind info for scl: float $Element; Size: 4
Resource bind info for act: float $Element; Size: 4
Resource bind info for packed: uint $Element; Size: 4
Resource bind info for bias: float $Element; Size: 4
Resource bind info for gbl: float $Element; Size: 4
```

shader 主循环中的 IR 指令:
```
%36 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(... T1/scl ...)  ; 读 float
%43 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(... T2/act ...)  ; 读 float
%45 = fmul fast float %44, %42  ; 用 float 算
%52 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(... T2/act ...)  ; 读 float
%61 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(... T2/act ...)  ; 读 float
```

但 v3 server 写入的是:
- rB (act): 写入 int32 raw bytes (e.g., act=1 → 0x00000001)
- rS (scl): 写入 uint32 raw bytes (e.g., scl=127 → 0x0000007F)

当 dxc 编译时把 act 解释为 float:
- act=1 (int32, 0x00000001) 作为 float 读 = 1.4e-45 (subnormal, 几乎为 0)
- scl=127 (uint32, 0x0000007F) 作为 float 读 = 1.55e-43 (subnormal, 几乎为 0)

所以 `fmul weight * act * scale` 全部是 0。

## 修复方案

### 方案 A (推荐): 修改 v3 server, 写入 float
在 v3 server.cpp 的 STATELESS 处理中:
- 把 act (int32) 转换为 float 写入 rB
- 把 scl (uint8 编码 e8m0) 解码为 float 写入 rS

### 方案 B: 修改 shader
- 把 `StructuredBuffer<int> act` 改成 `StructuredBuffer<float> act`
- 把 `StructuredBuffer<uint> scl` 改成 `StructuredBuffer<float> scl`
- 修改使用方式

## 关键文件
- `E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_v3_server.cpp` (354 行)
- `E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_sk.hlsl` (91 行)
- `E:\FreeToken\benchmarks\cpu_moe_microbench\t_mxfp4_gemv_sk.dxil` (7408 bytes, 已编译)

## 工具
- DXC: `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe`
- MSVC: `C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe`
- 构建脚本: `E:\FreeToken\benchmarks\cpu_moe_microbench\build_v3_server.bat`
- 构建 shader: `E:\FreeToken\benchmarks\cpu_moe_microbench\build_mxfp4_gemv.bat`
- 测试脚本: `E:\FreeToken\benchmarks\cpu_moe_microbench\t_p0_diag3.py`

## 测试用例 (应通过)
| Test | packed | scl | act | bias | expected outv[0] |
|------|--------|-----|-----|------|------------------|
| T1 | 0x0 | 0 | 0 | 0 | 0 |
| T2 | 0x0 | 0 | 0 | 5.0 | 5.0 (证明 bias 路径) |
| T3 | 0x11111111 | 0x7F | 1 | 0 | 32.0 (基本 GEMV) |
| T4 | 0x11111111 | 0x7F | 1 | 5.0 | 37.0 (GEMV + bias) |
| T5 | M=4 K=32 同T3 | | | | [32,32,32,32] |
| T6 | M=4 K=32 + bias=[1,2,3,4] | | | | [33,34,35,36] |
| T7 | M=1 K=4096 真实 MTP fc 权重 | | | | 应与 PyTorch ref bit-exact (rel err < 1e-3) |
