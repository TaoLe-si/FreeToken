"""Host-side dense SwiGLU MLP executors for B-group layers.

Nvfp4HostFfnExecutor: NVFP4 packed host weights -> per-output-channel INT8
cache (half the bf16 footprint; bf16 for all layers exhausted 47 GB commit).
CpuDenseFfnExecutor: FP8-per-tensor tail layers -> bf16 pinned cache.
"""

from __future__ import annotations

import torch


def _dequant_fp8_bf16(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 per-tensor/per-channel weight -> bf16 (mirrors loader-side math)."""
    w = w.detach().to("cpu")
    s = s.detach().to("cpu")
    wf = w.to(torch.float32)
    sf = s.to(torch.float32)
    if sf.dim() == 0:
        out = wf * sf
    elif sf.dim() == 1:
        s2 = sf.reshape(-1, 1) if sf.shape[0] == wf.shape[0] else sf.reshape(1, -1)
        out = wf * s2
    else:
        out = wf * sf
    return out.to(torch.bfloat16)


class CpuDenseFfnExecutor:
    """Dense SwiGLU MLP on pinned host memory (fused gate|up column-merged input)."""

    def __init__(self, *, gate_up_w, gate_up_s, down_w, down_s, output_sizes):
        if not output_sizes:
            raise ValueError("CpuDenseFfnExecutor requires merged output_sizes")
        self.output_sizes = [int(v) for v in output_sizes]
        gu = _dequant_fp8_bf16(gate_up_w.to("cpu"), gate_up_s.to("cpu"))
        dn = _dequant_fp8_bf16(down_w.to("cpu"), down_s.to("cpu"))
        self.gate_up_w = gu.t().contiguous()
        self.down_w = dn.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        xc = x.detach().to("cpu")
        gu = xc @ self.gate_up_w
        g, u = gu.split(self.output_sizes, dim=-1)
        y = torch.nn.functional.silu(g) * u
        out = y @ self.down_w
        return out.to(dev)


_NVFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class Nvfp4HostFfnExecutor:
    """Dense SwiGLU MLP on CPU from host-resident NVFP4 weights.

    Dequantizes ONCE at construction into a per-output-channel INT8 cache
    (fp32 row scales): half the bf16 steady-state footprint. Forward upcasts
    int8 to bf16 in bounded K-chunks and accumulates in fp32."""

    def __init__(self, *, gate_up_w, gate_up_scale, gate_up_global,
                 down_w, down_scale, down_global):
        lut = _NVFP4_LUT.to("cpu")

        def deq_i8(w_u8, sc_fp8, gl_f16):
            w = w_u8.detach().to("cpu")
            s = sc_fp8.detach().to("cpu").float()
            kfull = w.shape[1] * 2
            if s.dim() == 2 and s.shape[0] == kfull // 16 and s.shape[1] == w.shape[0]:
                s = s.t()
            sf = s.repeat_interleave(16, dim=1) if s.dim() == 2 else s.unsqueeze(1)
            g = gl_f16.detach().to("cpu").float().reshape(-1)
            if g.shape[0] != w.shape[0]:
                g = torch.ones(w.shape[0])
            N = w.shape[1] * 2
            qi = torch.empty(w.shape[0], N, dtype=torch.int8)
            qs = None
            chunk = max(1024, (192 * 1024 * 1024) // max(1, N * 4))
            for r0 in range(0, w.shape[0], chunk):
                r1 = min(w.shape[0], r0 + chunk)
                wc = w[r0:r1]
                lo = lut[(wc & 0xF).long()]
                hi = lut[(wc >> 4).long()]
                Wc = torch.stack((lo, hi), dim=-1).reshape(r1 - r0, -1)
                B = Wc.float() * sf[r0:r1] * g[r0:r1].unsqueeze(1)
                # Per-OUTPUT-CHANNEL (row) scales: rows of the packed weight are
                # out_features, so scales stay attached to GEMM output columns.
                amax = B.abs().amax(dim=1).clamp_min(1e-8)
                if qs is None:
                    qs = amax / 127.0
                else:
                    qs = torch.cat((qs, amax / 127.0))
                q = torch.round(B / qs[-(r1 - r0):].unsqueeze(1)).clamp_(-127, 127)
                qi[r0:r1] = q.to(torch.int8)
            return qi, qs  # [out, in], [out]

        gu_q, gu_s = deq_i8(gate_up_w, gate_up_scale, gate_up_global)
        dn_q, dn_s = deq_i8(down_w, down_scale, down_global)
        # deq_i8 returns [out, in]; transpose ONCE here to GEMM layout [in, out].
        self.gate_up_w = gu_q.t().contiguous()   # [K, 2I] int8
        self.gate_up_s = gu_s                    # [2I] fp32 (output channels)
        self.down_w = dn_q.t().contiguous()      # [I, H] int8
        self.down_s = dn_s                       # [H] fp32
        self.I = self.gate_up_w.shape[1] // 2

    def _mm_i8(self, x, Wi, s, chunk=2048):
        M, K = x.shape
        N = Wi.shape[1]
        out = torch.empty(M, N, dtype=torch.float32)
        xf = x.float()
        for k0 in range(0, K, chunk):
            k1 = min(K, k0 + chunk)
            Wc = (Wi[k0:k1].float() * s.unsqueeze(0)).to(torch.bfloat16).float()
            out += xf[:, k0:k1] @ Wc
        return out.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        xc = x.detach().to("cpu")
        g = self._mm_i8(xc, self.gate_up_w, self.gate_up_s)
        act = g[:, self.I:] * (g[:, :self.I] * torch.sigmoid(g[:, :self.I]))
        out = self._mm_i8(act, self.down_w, self.down_s)
        return out.to(dev)
