# iGPU 共享内存池 MoE decode 执行器 — 架构设计（v1）

## 决策背景
- 显存池容量不允许 decode 整层驻留（40 层 x 256 专家 = 10240 实例 vs 868 槽）。
- CPU executor 判定不通过（0.44 t/s vs 基线 4.11；RAM 带宽墙）。
- 用户拍板：专家 bank 驻留 CPU/iGPU 共享内存池（APU unified memory），
  由 iGPU 直接读取 —— 消除 16.9GiB 装不下 dGPU 显存的问题，且省去每步 PCIe
  拷贝（slot cache 的存在意义消失）。

## 模型核算（Qwen3.6-35B-A3B MXFP4-NVFP4）
- H=2048, I=512, top_k=8；每专家 gate_up(2x512x2048)+down(2048x512) = 3.1M 参数
- nvfp4 打包约 1.59MB/专家/层；40 层 x 256 = 16.3GB —— 必须驻共享内存
- 每 decode token 的 iGPU 读取量：40 层 x 8 路由 x 1.59MB 约 508MB/token

## 性能模型（780M 级 iGPU，共享 DDR5 实测读带宽按 60-100GB/s）
- 纯带宽下限：508MB / 60-100GBps 约 5-8.5ms/token 即 120-190 t/s
- dispatch 数量是第二个瓶颈：v3 server 单次 dispatch 0.2-0.5ms。
  - 现状逐专家：40 层 x 8 x 2 = 640 次 → 不可用（numpy 路径的病根）
  - 每层融合一次（8 专家 gate_up+down 一把）：80 次 → 24-40ms/token 约 25-40 t/s
  - 整步一次 dispatch（目标）：吞吐逼近带宽上限（100+ t/s），
    叠加 MTP 命中（d1 79%）→ 有效 t/s 目标不低于 2x 当前基线。
- 对比基线 17 t/s（用户机）：判据 = iGPU 整步方案必须 > 25 t/s 才算兑现。

## 三大件
### 1. 共享内存池 bank 驻留（host 侧）
- 复用 OffloadMoeCache 的 host banks（已 pinned、CPU 可读布局），通过 C++
  IgpuService 注册为 iGPU 可见 host 指针（HIP: hipHostRegister；
  D3D12: cross-adapter shared heap / open-existing）。
- 零拷贝：不做每步 H2D；LOAD 时把 bank 指针交给 server。
- 池按 (layer, expert) 连续分桶，路由 id 直接索引，无 slot/LRU ——
  从根上消灭 copy_missing（Bug A 消失，decode replay 变安全）。

### 2. 整步批量 shader（iGPU 侧）
- 输入：路由表 [40x8] int32（pinned）+ 每层激活 [2048] f32 + bank 指针。
- 层间串行依赖（层 n+1 输入依赖层 n 输出）→ 实际粒度是每层一次 dispatch，
  40 次/步：40 x 0.3-0.45ms = 12-18ms/token 约 55-85 t/s（仍达标）；
  gate_up+down 融合成单 kernel 可再省一半次数。
- 输出：每层 MoE 输出写回 pinned 环形缓冲，供主模型后续子层读。

### 3. CUDA-graph 桥（engine 侧，仿 CpuMoeExecutor flag-sync）
- 图内节点：D2H(路由+激活→pinned) → memop(done=0) → host node(向 server
  非阻塞提交) → memop(ready) → H2D(结果)。
- server 与 CUDA 进程解耦（沿用子进程模式），host node 仅非阻塞 write(pipe)；
  完成标志写共享 ready 页 —— 与 CpuMoeExecutor 的 done/ready 槽位同构，已被
  证明可捕获。
- pinned IO 多缓冲（每层 x 2 槽乒乓）：修复 CPU 路径复现的 overlap 踩踏。

## 分阶段落地
- P0（判定性，1 天）：目标机上先跑 igpu_bw.py 拿共享内存读带宽实数；再以
  手写路由表让 iGPU server 批量读共享 bank 跑一层 8 专家 GEMV，对拍 dGPU
  参考。门槛：推算整步 > 25 t/s 才继续，否则停。
- P1（server）：BATCH_LAYER 协议（一次 payload = 一层全部路由 + 激活），
  gate_up+down 融合 shader；共享 bank 注册接口。
- P2（engine）：IgpuSharedMoeExecutor（对齐 CpuMoeExecutor 接口）+ 图桥 +
  多缓冲；decode_target=igpu-shared 门控下重开 decode replay 与 verify graph。
- P3（验收）：planets/france 逐字等价 + 200-token 基准 > 25 t/s + state-hash
  位级对比。

## 风险
- 共享内存读带宽实测值是全案前提（开发机无 AMD iGPU，需目标机先测）。
- 层间 40 次 dispatch 可能被 launch/调度开销支配 → P0 同时测空 dispatch
  往返，决定是否把 GDN 子层也并入 server。
- HIP vs D3D12：现栈为 D3D12/DXIL；HIP server 已有雏形
  （t_mxfp4_gemv_v3_hip_server），共享内存映射最直接，建议 P0 用 HIP 先证带宽。
