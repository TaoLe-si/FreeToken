# P1g+ to MTP Integration Execution Plan (4 Tracks)

**End goal**: Validate iGPU-assisted dGPU MTP decoding tok/s speedup with measured data
**Current state** (from session history + my probe):
- OK P1g sticky server (`t_mxfp4_gemv_v2_server.exe`) verified bit-exact
- OK Phase 0: `models/qwen3_5_moe/mtp.py` exists (MtpHead + loader)
- OK Phase 1: `engine/mtp_driver.py` + `kernel/igpu_fc.py` exist (MtpDriver)
- OK `t_mtp_head_driver.py` e2e driver exists
- PENDING Phase 2: scheduler integration (the main work)
- PENDING Real MXFP4 e8m0 scales path (kernel uses 0.01 magic)
- PENDING M>1 path validation
- PENDING Performance comparison numbers (dGPU-only vs iGPU-MTP)

## Track 1: Plan B - Multi-weight sticky cache validation
**Goal**: Verify 8-16 different-shape weights can LOAD/CALL simultaneously without interference

Sub-tasks:
1. Write `t_test_v2_multi_weights.py` simulating MTP head's full LOAD
   - fc: M=1, K=4096
   - attn q/k/v/o: M=2048/256/256/2048, K=2048/2048/2048/4096
   - 256 switch experts: M=512, K=2048 each (or sample)
2. For each weight, CALL with different act, compare to PyTorch ref
3. Verify rW/rAct/rGbl resources do not interfere in LOAD/CALL sequences
4. Edge case: M change (1->8) triggers realloc
5. **Key insight**: If Phase 0 only uses fc on iGPU (attn+MoE on PyTorch),
   we only need 1 weight. Reduce B to: LOAD-rewrite-same-name cache consistency

Acceptance:
- All weights match PyTorch ref (4-digit precision)
- LOAD-rewrite same name gives identical output to first LOAD
- 100 LOAD/CALL cycles no memory leak (check server stderr)

Effort: 0.5-1 day

## Track 2: Plan E - M>1 (MoE top-8 experts) validation
**Goal**: Verify P1g server output for M=8 is sum of 8x M=1 outputs (linearity)

Sub-tasks:
1. Write `t_test_v2_m8.py`:
   - LOAD fc M=8 K=4096
   - CALL with same act for all 8 rows
   - Compare to PyTorch ref: sum of 8 single-row GEMVs
2. Verify resource realloc on M change: rW/rAct/rGbl/rOut/rRb
3. **Known risk** (P1d_STATUS): M>1 realloc has NaN bug
   - Diagnose: rOut state transition, rOutS variable, CopyResource sizes
4. If bug found, write minimal M-realloc unit test in P1g server

Acceptance:
- M=8 LOAD+CALL output = sum of 8x M=1 outputs (fp32 bit-exact or 1e-4)
- 100 M=8 CALLs no NaN/Inf
- M change (1->8->1) stable

Effort: 0.5-1 day

## Track 3: Plan C - Real MXFP4 e8m0 scales support
**Goal**: Remove the kernel's hardcoded 0.01f magic, use real e8m0 scales

Pre-investigation (MUST do first):
1. Check P1g server currently uploads zeros to rB (slot 2 = scales)
2. Empirical: 4.6925 (a=0.05) / 602 (sum_nibbles) = 0.00780
   - kernel formula: `(float)wsum * 0.01f * (float)sb`, with sb=128 -> 1.28
   - Real e8m0 scale sb=128 = 2^(128-127) = 2.0
   - 0.01 vs exp2(sb-127)/100 = systematic 100x error in current outputs
3. **All current outputs have 100x systematic error** vs true MXFP4 math

Sub-tasks (after B+E):
1. Modify HLSL source `d3d12_gemv_sk.hlsl`:
   - Change `(float)wsum * 0.01f * (float)sb` -> `(float)wsum * exp2((float)sb - 127.0f)`
2. Recompile to .dxbc
3. Modify P1g LOAD command: read real e8m0 scales from checkpoint, upload to rB
4. Validate: output should be 4.6925 / 1.28 = 3.66, PyTorch ref < 0.5% error

Risks:
- Shader change may break current stable (if wrong) baseline
- Current .dxbc source unclear (d3d12_gemv_sk.hlsl vs t_mxfp4_gemv_sk.hlsl)
- Need to verify d3d12_gemv_sk.hlsl compile matches current .dxbc before changing

Acceptance:
- Same fc_w + act=0.05 + real scales -> output 3.66 (vs current 4.69 with 100x error)
- PyTorch ref error < 0.5%

Effort: 1-2 days

## Track 4: Plan A - Integrate into FreeToken scheduler (the main course)
**Goal**: iGPU-MTP in real inference, measure tok/s speedup

Pre-dep: B + E complete (C optional, not blocking integration path)

Sub-tasks (per e82ea6b1 supplementary report, 8-9 person-days):

### P0: weight path revival (1 day)
- Delete the mtp filter in `weight.py:204, 703` (or yield mtp keys to state_dict)
- Verify `load_mtp_head_from_safetensors` still works

### P1: iGPU executor for MTP head (1.5 days)
- New file `engine/mtp_executor.py` (modeled after `_init_igpu_executor` at engine.py:868):
  - Start P1g v2 server as daemon
  - LOAD mtp.fc.weight once
  - Provide `forward(hidden_states, prev_token_id) -> draft_logits`
- Modify `engine.py`:
  - `create_model` (engine.py:425) instantiate MtpExecutor
  - `forward_batch` (engine.py:1268) two-phase: main model forward -> last hidden -> MTP executor

### P2: scheduler integration (3-4 days)
- Modify `core.py`: `Req` add `draft_ids: list[int]`, `verified_len: int`
- Modify `scheduler/cache.py`: add `free_partial(req, to_seq_len)` (0.5d, ref scheduler/cache.py:316-320)
- Modify `scheduler/scheduler.py`:
  - `_make_positions` (scheduler.py:882) support K+1 token
  - `_process_last_data` K-token loop
  - `DecodeManager.schedule_next_batch` push K tokens per step
- Modify `engine/graph.py`: GraphRunner decode graph K-dim capture (1 day)

### P3: e2e benchmark (1 day)
- CLI flags: `--mtp-k N --mtp-head-device igpu`
- Test scenarios:
  - baseline: dGPU-only, MTP off, measure tok/s
  - dGPU+MTP-drafter-iGPU: MTP-K=3, measure tok/s
  - Compute speedup
- Acceptance:
  - single batch tok/s speedup >= 1.3x (when MTP accept rate > 0.5)
  - batch=8 + MTP-K=3: dGPU does 1/(1+K*accept_rate) fewer forwards

Key risks:
- KV partial rollback: cache_req_to_len already exists, extend it
- CUDA graph recapture: cuda_graph_bs + new K dim
- iGPU <-> dGPU sync: MTP head eats main model last hidden (D2D copy)
- MtpDriver existing rollback uses cache_req_to_len, scheduler needs to invoke it

Effort: 8-9 days (matches e82ea6b1 estimate)

## Recommended execution order

```
Day 0-0.5:  Track 1 (B) + Track 2 (E) [parallel]
Day 0.5-1:  Track 1 (B) done + Track 2 (E) done
Day 1-2:    Track 3 (C) shader recompile + precision verify
Day 2-3:    Track 4 P0 (weight path) + P1 (iGPU executor wrap)
Day 4-6:    Track 4 P2 (scheduler/cache integration)
Day 7-8:    Track 4 P3 (graph recapture + e2e benchmark)
Day 9:      Final report + speedup data
```

Final deliverables:
- `P1g_STATUS.md` -> `P2_STATUS.md` full integration report
- iGPU-MTP vs dGPU-only tok/s comparison table
- Key files: MtpExecutor, scheduler.diff, benchmark script