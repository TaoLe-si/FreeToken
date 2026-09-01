// MTP head MoE server (Phase 2.3, 2026-08-29).
// Reference: t_mxfp4_gemv_v3_server.cpp (D3D12 device/queue/cmdlist 模板).
//
// 协议:
//   MOE_LOAD <E=256> <I=512> <H=2048> <bytes>\n  + body  ->  OK\n
//   MOE_FORWARD\n + hidden_f32[2048]                ->  out_f32[2048]
//   QUIT\n                                          ->  shutdown
//
// Sticky: expert weights 一次性上传, 常驻 iGPU VRAM (~1.3 GB total).
// 
// 实施 P0 骨架:
//   - D3D12 device + queue + 4 个 command list (route / expert_8x / shared / combine)
//   - 4 个 compute shader 编译 (fxc / dxc 编译 HLSL -> DXIL 写 .dxil 文件, 启动时读)
//   - 4 个 DXIL 通过 MOE_LOAD 上传 weights
//   - MOE_FORWARD 1 dispatch per kernel
//   - readback 输出到 stdout
// 
// TODO (后续 PR): 数值对齐 vs PyTorch MtpHeadMoe; 性能 benchmark.

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <io.h>
#include <fcntl.h>
#include <string>
#include <cerrno>

using Microsoft::WRL::ComPtr;

static bool readN(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t got = 0;
    while (got < n) {
        int r = _read(fd, p + got, (unsigned int)(n - got));
        if (r <= 0) { if (r == 0) return false; if (errno == EINTR) continue; return false; }
        got += (size_t)r;
    }
    return true;
}
static bool readLine(int fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
        int r = _read(fd, &c, 1);
        if (r <= 0) return false;
        if (c == '\n') break;
        out += c;
    }
    return true;
}
static void writeAll(int fd, const void* buf, size_t n) {
    _write(fd, buf, (unsigned int)n);
    _flushall();
}

static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f;
    return d;
}

struct MoeState {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> uploadAlloc, dispatchAlloc;
    ComPtr<ID3D12GraphicsCommandList> uploadList, dispatchList;
    ComPtr<ID3D12Fence> fence;
    HANDLE fenceEvent = nullptr;
    UINT64 fenceVal = 0;

    // Shaders
    ComPtr<ID3D12PipelineState> psoRoute, psoExpert, psoShared, psoCombine;
    ComPtr<ID3D12RootSignature> rootSig;

    // Sticky weight buffers
    ComPtr<ID3D12Resource> bExpertGate, bExpertUp, bExpertDown;
    ComPtr<ID3D12Resource> bSharedGate, bSharedUp, bSharedDown, bSharedGw;
    ComPtr<ID3D12Resource> bRouterW;
    UINT64 capExpert = 0;  // bytes per expert buffer

    // Per-call scratch
    ComPtr<ID3D12Resource> bHiddenIn, bTop8Idx, bTop8W, bExpertOut, bSharedOut, bOut;

    UINT32 E = 256, I = 512, H = 2048;
    bool sticky_loaded = false;
};

static bool loadDxilPso(MoeState& s, const char* dxil_path, ComPtr<ID3D12PipelineState>& pso) {
    std::ifstream fi(dxil_path, std::ios::binary);
    if (!fi) { fprintf(stderr, "missing %s\n", dxil_path); return false; }
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { fprintf(stderr, "empty %s\n", dxil_path); return false; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = s.rootSig.Get();
    psd.CS = { dxil.data(), dxil.size() };
    if (FAILED(s.device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso)))) {
        fprintf(stderr, "CreateComputePipelineState failed for %s\n", dxil_path);
        return false;
    }
    return true;
}

static bool submit(MoeState& s, ComPtr<ID3D12GraphicsCommandList> lst, const char* tag) {
    if (FAILED(lst->Close())) { fprintf(stderr, "[%s] Close failed\n", tag); return false; }
    ID3D12CommandList* ls[] = { lst.Get() };
    s.queue->ExecuteCommandLists(1, ls);
    s.fenceVal++;
    s.queue->Signal(s.fence.Get(), s.fenceVal);
    ResetEvent(s.fenceEvent);
    s.fence->SetEventOnCompletion(s.fenceVal, s.fenceEvent);
    if (WaitForSingleObject(s.fenceEvent, 5000) != WAIT_OBJECT_0) {
        fprintf(stderr, "[%s] fence wait timeout\n", tag);
        return false;
    }
    lst->Reset(s.uploadAlloc.Get(), nullptr);  // reset for next use (caller-agnostic)
    return true;
}

int main() {
    fprintf(stderr, "t_mtp_moe_server starting...\n");
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    MoeState s;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&s.device));
    if (FAILED(hr)) { fprintf(stderr, "device create failed\n"); return 1; }
    fprintf(stderr, "device ok\n");

    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    s.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&s.queue));
    s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.uploadAlloc));
    s.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s.dispatchAlloc));
    s.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s.fence));
    s.fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&s.uploadList));
    s.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s.uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&s.dispatchList));

    // Root signature: t0..t4 + u0..u1 + b0 cbuffer. Matches HLSL.
    D3D12_DESCRIPTOR_RANGE ranges[8] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 5; ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors = 2; ranges[1].BaseShaderRegister = 0;
    // We'll register SRV/UAV ranges as needed per kernel; for the skeleton, a single
    // 5-SRV + 2-UAV range works for all 4 kernels (extra slots are just unused).
    D3D12_ROOT_PARAMETER rp[3] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[0].DescriptorTable = { 1, &ranges[0] };
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable = { 1, &ranges[1] };
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[2].Descriptor = { 0, 0 };  // b0
    rp[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 3; rd.pParameters = rp;
    rd.NumStaticSamplers = 0; rd.pStaticSamplers = nullptr;
    rd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3D10Blob> sigBlob, errBlob;
    if (FAILED(D3D12SerializeRootSignature(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob))) {
        fprintf(stderr, "D3D12SerializeRootSignature failed\n");
        if (errBlob) fprintf(stderr, "%s\n", (const char*)errBlob->GetBufferPointer());
        return 1;
    }
    if (FAILED(s.device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
                                            IID_PPV_ARGS(&s.rootSig)))) {
        fprintf(stderr, "CreateRootSignature failed\n");
        return 1;
    }

    // Load DXIL PSOs from sibling files. (HLSL must be pre-compiled with
    // dxc/fxc -- the build script does that. We just load the .dxil here.)
    if (!loadDxilPso(s, "t_mtp_moe_route.dxil", s.psoRoute)) return 1;
    fprintf(stderr, "pso route ok\n");
    if (!loadDxilPso(s, "t_mtp_moe_expert.dxil", s.psoExpert)) return 1;
    fprintf(stderr, "pso expert ok\n");
    if (!loadDxilPso(s, "t_mtp_moe_shared.dxil", s.psoShared)) return 1;
    fprintf(stderr, "pso shared ok\n");
    if (!loadDxilPso(s, "t_mtp_moe_combine.dxil", s.psoCombine)) return 1;
    fprintf(stderr, "pso combine ok\n");

    fprintf(stderr, "t_mtp_moe_server ready\n");

    // ---- command loop ----
    while (true) {
        std::string line;
        if (!readLine(0, line)) break;
        if (line.empty()) continue;
        std::vector<std::string> t;
        size_t pos = 0;
        while (pos < line.size()) {
            while (pos < line.size() && line[pos] == ' ') pos++;
            if (pos >= line.size()) break;
            size_t s2 = pos;
            while (pos < line.size() && line[pos] != ' ') pos++;
            t.push_back(line.substr(s2, pos - s2));
        }
        if (t.empty()) continue;
        std::string cmd = t[0];
        try {
            if (cmd == "QUIT") { fprintf(stderr, "QUIT\n"); break; }
            else if (cmd == "MOE_LOAD") {
                if (t.size() < 4) { fprintf(stderr, "MOE_LOAD: bad args\n"); continue; }
                UINT32 E = std::stoul(t[1]), I = std::stoul(t[2]), H = std::stoul(t[3]);
                UINT64 bodySize = (UINT64)E * I * H * 2 * 3  // gate + up (bf16, E*I*H*2 each)
                                + (UINT64)E * H * I * 2    // down
                                + (UINT64)I * H * 2 * 3    // shared gate + up + gw (I*H*2 each, I bf16 shared_gw)
                                + (UINT64)H * I * 2        // shared_down
                                + (UINT64)E * H * 2;       // router_w
                std::vector<uint8_t> body(bodySize);
                if (!readN(0, body.data(), bodySize)) { fprintf(stderr, "MOE_LOAD: read fail\n"); continue; }
                // Upload -- TODO: actually upload to GPU resources.
                // For P0 skeleton, just acknowledge.
                s.sticky_loaded = true;
                writeAll(1, "OK\n", 3);
                fprintf(stderr, "MOE_LOAD E=%u I=%u H=%u body=%llu bytes\n", E, I, H, (unsigned long long)bodySize);
            }
            else if (cmd == "MOE_FORWARD") {
                if (!s.sticky_loaded) { fprintf(stderr, "MOE_FORWARD before MOE_LOAD\n"); continue; }
                // Read hidden_f32[2048]
                std::vector<float> hidden(s.H);
                if (!readN(0, hidden.data(), s.H * 4)) { fprintf(stderr, "MOE_FORWARD: read fail\n"); continue; }
                // P0 stub: return zeros. Real impl: 4 dispatches, readback out.
                std::vector<float> out(s.H, 0.0f);
                writeAll(1, out.data(), s.H * 4);
                fprintf(stderr, "MOE_FORWARD done (P0 stub output)\n");
            }
            else {
                fprintf(stderr, "unknown cmd: %s\n", cmd.c_str());
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "cmd error: %s\n", e.what());
        }
    }
    return 0;
}
