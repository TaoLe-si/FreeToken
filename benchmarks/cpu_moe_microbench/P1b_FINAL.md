# [P1b] 最终失败根因 — 已定位多个问题

[P1a] 通过（合成数据 + MXFP4 GEMV，max rel diff 3.7e-4）。

[P1b] 在真实模型权重 + bf16 per-block scale/bias 下失败。**多个独立 bug**：

## Bug 1：DXC fast-math 重排 bias 加法（最严重）

源码：
```hlsl
float wsum = Σ(W[k] * act[k], k=0..31);  // 32-element sum
acc += (wsum + bbias) * scale;            // bias 加一次
```

DXC fast-math 重排为：
```
Σ(W[k] * act[k] + bbias, k=0..31) * scale   // bias 加 32 次
```

结果：bias 被放大了32倍（for K=4096: 128 micro-blocks × 32 = 4096 次）。

Mathematically equal 但在 fp arithmetic 下不等价（reordering sums changes rounding）。
**workaround**：`precise` 限定符部分有效。

## Bug 2：`numthreads(32,1,1)` 与 reduce 循环不匹配（之前已修）

`sh[256]` 是为256线程设计的 reduce（`for (s=128; s>0; s>>=1)`）。
改成 numthreads=32 后，`sh[32..255]` 未初始化被读入，导致 garbage。
**已修**：改回 `numthreads(256,1,1)`。

## Bug 3：1row 文件 nbPerRow 错误（之前已修）

`M=1 K=32` 时 `nbPerRow = K/8 = 4`，但测试文件写成 `1`，导致 shader 用错的索引。

## Bug 4：DXC register 重排（最隐蔽）

当 `scl` 和 `bias` 被 debug override 跳过时，DXC 删除它们。
**但同时把后续 register 重新编号**：
- shader: `packed : t0, scl : t1, bias : t2, act : t3, gbl : t4, rowBias : t5`
- DXC 优化后：`packed : t0, act : t1, gbl : t2, rowBias : t3, outv : u0`
- cpp 端还按原 binding 发送 → 数据错位

**这是 debug override 测试时多次产生乱码输出的根因。**

## Bug 5：cpp 端 stale compile（我自己的错）

多次修改 cpp/shader 后，build 失败（如 `(ComPtr<...>&)` 语法错误），
但我没注意到编译失败，跑的是旧 exe。

## 最终状态

`P1b` kernel 在最简设置下（`scale=1, bias=0` + 真实权重）输出 garbage，
不是 -27（sum of W for row0）。经过5+轮的修复（包括 register 重排、bias 重排、
stale build、1row nbPerRow 错误），仍无法得到与 PyTorch reference 数值对齐的结果。

放弃项目。

详细时间线在 P1a_STATUS.md 和 P1b_STATUS.md。
