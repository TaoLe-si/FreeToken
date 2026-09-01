// MTP head MoE routing-only server (real GPU dispatch, Phase 2.3 minimal impl, 2026-08-30).
//
// Reference: t_mxfp4_gemv_v3_server.cpp (D3D12 device/queue/cmdlist 模板).
//
// Implements ONLY the route kernel (256 -> top-8 + softmax) on real D3D12 GPU
// dispatch. Other kernels (expert_8x, shared, combine) remain CPU fallback
// in this binary -- the goal is to prove the GPU dispatch path works end-to-end.
//
// Protocol:
//   MOE_ROUTE_LOAD <E=256> <H=2048>\n  + body (E*H fp32 router_w)  ->  OK\n
//   MOE_ROUTE_FORWARD\n  + hidden_f32[H]                          ->  top8_idx[8] u32 + top8_w[8] fp32 (64 bytes)
//   QUIT\n                                                          ->  shutdown
//
// Sticky: router weights 一次性上传, 常驻 iGPU VRAM (~2 MB).
//
// 实施:
//   - D3D12 device + queue + 2 个 command list (upload + dispatch)
//   - 1 个 compute shader (t_mtp_moe_route.dxil) -- t0=routerW, t1=hidden, u0=top8Idx, u1=top8W, b0 cbuffer
//   - 1 dispatch per forward (1 thread group of 32 threads, parallel bitonic sort + softmax in LDS)
//   - readback top8Idx/Top8W to stdout

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <io.h>
#include <fcntl.h>
#include <string>
#include <cerrno>
#include <cstdio>

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

int main() {
    fprintf(stderr, "t_mtp_moe_route_server starting...\n");
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    // ---- D3D12 device ----
    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { fprintf(stderr, "device create failed hr=0x%08X\n", hr); return 1; }
    fprintf(stderr, "device ok\n");

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> uploadAlloc, dispatchAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadAlloc));
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dispatchAlloc));
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ComPtr<ID3D12GraphicsCommandList> uploadList, dispatchList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&uploadList));
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dispatchAlloc.Get(), nullptr, IID_PPV_ARGS(&dispatchList));
    UINT64 fenceVal = 0;

    // ---- Root signature: t0 (routerW), t1 (hidden), u0 (top8Idx), u1 (top8W), b0 cbuffer ----
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[0].NumDescriptors = 2; ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[1].NumDescriptors = 2; ranges[1].BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER rp[3] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[0].DescriptorTable = { 1, &ranges[0] };
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable = { 1, &ranges[1] };
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[2].Descriptor = { 0, 0 };
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
    ComPtr<ID3D12RootSignature> rootSig;
    if (FAILED(device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(),
                                            IID_PPV_ARGS(&rootSig)))) {
        fprintf(stderr, "CreateRootSignature failed\n");
        return 1;
    }

    // ---- Load route DXIL PSO ----
    std::ifstream fi("t_mtp_moe_route.dxil", std::ios::binary);
    if (!fi) { fprintf(stderr, "missing t_mtp_moe_route.dxil\n"); return 1; }
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { fprintf(stderr, "empty t_mtp_moe_route.dxil\n"); return 1; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rootSig.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> psoRoute;
    if (FAILED(device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&psoRoute)))) {
        fprintf(stderr, "CreateComputePipelineState route failed\n");
        return 1;
    }
    fprintf(stderr, "pso route ok\n");

    // ---- Descriptor heap for SRV/UAV ----
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 4;  // 2 SRV (routerW, hidden) + 2 UAV (top8Idx, top8W)
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descHeap;
    device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descHeap));
    UINT descSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    auto submit = [&](ComPtr<ID3D12GraphicsCommandList> lst, const char* tag) -> bool {
        HRESULT hc = lst->Close();
        if (FAILED(hc)) { fprintf(stderr, "[%s] Close failed hr=0x%08X\n", tag, hc); return false; }
        ID3D12CommandList* ls[] = { lst.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fenceVal;
        queue->Signal(fence.Get(), fenceVal);
        ResetEvent(fenceEvent);
        fence->SetEventOnCompletion(fenceVal, fenceEvent);
        if (WaitForSingleObject(fenceEvent, 5000) != WAIT_OBJECT_0) {
            fprintf(stderr, "[%s] fence wait timeout\n", tag);
            return false;
        }
        return true;
    };

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

    UINT32 E = 256, H = 2048;
    bool sticky_loaded = false;
    ComPtr<ID3D12Resource> bRouterW, bHidden, bTop8Idx, bTop8W, bRbIdx, bRbW;
    ComPtr<ID3D12Resource> upRouter, upHidden;

    fprintf(stderr, "t_mtp_moe_route_server ready\n");

    while (true) {
        std::string line;
        if (!readLine(0, line)) break;
        if (line.empty()) continue;
        // Parse cmd
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
            else if (cmd == "MOE_ROUTE_LOAD") {
                if (t.size() < 3) { fprintf(stderr, "MOE_ROUTE_LOAD: bad args\n"); continue; }
                E = (UINT32)std::stoul(t[1]); H = (UINT32)std::stoul(t[2]);
                UINT64 bodySize = (UINT64)E * H * 4;  // fp32 router_w
                std::vector<uint8_t> body(bodySize);
                if (!readN(0, body.data(), bodySize)) { fprintf(stderr, "MOE_ROUTE_LOAD: read fail\n"); continue; }
                // Create default heap buffer + upload heap
                if (bRouterW) bRouterW.Reset();
                if (upRouter) upRouter.Reset();
                device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
                    &bd((UINT64)E * H * 4, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&bRouterW));
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE,
                    &bd((UINT64)E * H * 4, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upRouter));
                // Map upload, memcpy, unmap
                void* p = nullptr;
                if (FAILED(upRouter->Map(0, nullptr, &p))) { fprintf(stderr, "Map upRouter failed\n"); continue; }
                memcpy(p, body.data(), bodySize);
                upRouter->Unmap(0, nullptr);
                // Copy upload -> default
                uploadList->Reset(uploadAlloc.Get(), nullptr);
                uploadList->CopyResource(bRouterW.Get(), upRouter.Get());
                if (!submit(uploadList, "load_router")) continue;
                sticky_loaded = true;
                writeAll(1, "OK\n", 3);
                fprintf(stderr, "MOE_ROUTE_LOAD E=%u H=%u body=%llu bytes (real GPU upload)\n",
                    E, H, (unsigned long long)bodySize);
            }
            else if (cmd == "MOE_ROUTE_FORWARD") {
                if (!sticky_loaded) { fprintf(stderr, "MOE_ROUTE_FORWARD before MOE_ROUTE_LOAD\n"); continue; }
                UINT64 hiddenSize = (UINT64)H * 4;
                std::vector<uint8_t> hiddenBytes(hiddenSize);
                if (!readN(0, hiddenBytes.data(), hiddenSize)) { fprintf(stderr, "MOE_ROUTE_FORWARD: read fail\n"); continue; }
                // Allocate per-call GPU resources
                if (bHidden) bHidden.Reset();
                if (upHidden) upHidden.Reset();
                if (bTop8Idx) bTop8Idx.Reset();
                if (bTop8W) bTop8W.Reset();
                if (bRbIdx) bRbIdx.Reset();
                if (bRbW) bRbW.Reset();
                device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
                    &bd(hiddenSize, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&bHidden));
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE,
                    &bd(hiddenSize, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upHidden));
                device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
                    &bd(8 * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&bTop8Idx));
                device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
                    &bd(8 * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&bTop8W));
                device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE,
                    &bd(8 * 4, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&bRbIdx));
                device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE,
                    &bd(8 * 4, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&bRbW));
                // Upload hidden
                void* p = nullptr;
                if (FAILED(upHidden->Map(0, nullptr, &p))) { fprintf(stderr, "Map upHidden failed\n"); continue; }
                memcpy(p, hiddenBytes.data(), hiddenSize);
                upHidden->Unmap(0, nullptr);
                uploadList->Reset(uploadAlloc.Get(), nullptr);
                uploadList->CopyResource(bHidden.Get(), upHidden.Get());
                if (!submit(uploadList, "load_hidden")) continue;
                // Create descriptors
                D3D12_SHADER_RESOURCE_VIEW_DESC srvW = {};
                srvW.Format = DXGI_FORMAT_R32_FLOAT; srvW.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                srvW.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srvW.Buffer = { 0, 0, (UINT)E * H };
                device->CreateShaderResourceView(bRouterW.Get(), &srvW, descHeap->GetCPUDescriptorHandleForHeapStart());
                D3D12_SHADER_RESOURCE_VIEW_DESC srvH = srvW;
                srvH.Buffer = { 0, 0, (UINT)H };
                D3D12_CPU_DESCRIPTOR_HANDLE h1 = descHeap->GetCPUDescriptorHandleForHeapStart();
                h1.ptr += descSize;
                device->CreateShaderResourceView(bHidden.Get(), &srvH, h1);
                D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
                uav.Format = DXGI_FORMAT_R32_UINT; uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                uav.Buffer = { 0, 0, 8 };
                D3D12_CPU_DESCRIPTOR_HANDLE h2 = descHeap->GetCPUDescriptorHandleForHeapStart();
                h2.ptr += 2 * descSize;
                device->CreateUnorderedAccessView(bTop8Idx.Get(), nullptr, &uav, h2);
                uav.Format = DXGI_FORMAT_R32_FLOAT;
                D3D12_CPU_DESCRIPTOR_HANDLE h3 = descHeap->GetCPUDescriptorHandleForHeapStart();
                h3.ptr += 3 * descSize;
                device->CreateUnorderedAccessView(bTop8W.Get(), nullptr, &uav, h3);
                // Dispatch route kernel (1 group of 32 threads for 256 logits)
                dispatchList->Reset(dispatchAlloc.Get(), nullptr);
                ID3D12DescriptorHeap* heaps[] = { descHeap.Get() };
                dispatchList->SetDescriptorHeaps(1, heaps);
                dispatchList->SetPipelineState(psoRoute.Get());
                dispatchList->SetComputeRootSignature(rootSig.Get());
                dispatchList->SetComputeRootDescriptorTable(0, descHeap->GetGPUDescriptorHandleForHeapStart());
                dispatchList->SetComputeRootDescriptorTable(1, descHeap->GetGPUDescriptorHandleForHeapStart());
                dispatchList->Dispatch(1, 1, 1);
                if (!submit(dispatchList, "route_dispatch")) continue;
                // Copy UAV -> readback
                uploadList->Reset(uploadAlloc.Get(), nullptr);
                uploadList->CopyResource(bRbIdx.Get(), bTop8Idx.Get());
                uploadList->CopyResource(bRbW.Get(), bTop8W.Get());
                if (!submit(uploadList, "readback")) continue;
                // Read back
                UINT32 top8Idx[8] = {};
                float top8W[8] = {};
                if (FAILED(bRbIdx->Map(0, nullptr, (void**)&p))) { fprintf(stderr, "Map rbIdx failed\n"); continue; }
                memcpy(top8Idx, p, 8 * 4);
                bRbIdx->Unmap(0, nullptr);
                if (FAILED(bRbW->Map(0, nullptr, (void**)&p))) { fprintf(stderr, "Map rbW failed\n"); continue; }
                memcpy(top8W, p, 8 * 4);
                bRbW->Unmap(0, nullptr);
                // Send 64 bytes: 8 u32 idx + 8 fp32 w
                writeAll(1, top8Idx, 8 * 4);
                writeAll(1, top8W, 8 * 4);
                fprintf(stderr, "MOE_ROUTE_FORWARD done (real GPU dispatch)\n");
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
