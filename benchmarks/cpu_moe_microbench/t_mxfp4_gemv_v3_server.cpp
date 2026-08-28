// [P1h] iGPU MXFP4 GEMV server with TRUE MXFP4 e8m0 scale bindings + per-row bias.
//
// Outputs B*M floats. Bias is per-row (M floats per item).
// Multi-GEMV shader used (all in one Dispatch).

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <unordered_map>
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
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}

struct Mxfp4Weight {
    UINT64 M = 0;
    UINT32 K = 0;
    UINT32 nb = 0;
    UINT32 ns = 0;
    ComPtr<ID3D12Resource> rW, rS, rO, rB, rA, rG, rR, rOut, rRb, rCb;
    D3D12_RESOURCE_STATES rWSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rSSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rASt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rBSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rGSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rRSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> uploadLoad;
    UINT64 uploadLoadCap = 0;
    ComPtr<ID3D12Resource> uploadCall;
    UINT64 uploadCallCap = 0;
    bool loaded = false;
};

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    fprintf(stderr, "ready fd0=%d fd1=%d fd2=%d\n", _fileno(stdin), _fileno(stdout), _fileno(stderr)); fflush(stderr);

    ComPtr<ID3D12Debug> dbg;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); }
    ComPtr<ID3D12InfoQueue> iq;
    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    HRESULT hr_dev = D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr_dev)) { fprintf(stderr, "device create failed hr=0x%08X\n", hr_dev); return 1; }
    if (SUCCEEDED(device.As(&iq))) {}
    fprintf(stderr, "device ok\n"); fflush(stderr);

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> uploadAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadAlloc));
    ComPtr<ID3D12GraphicsCommandList> uploadList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&uploadList));
    uploadList->Close();
    ComPtr<ID3D12CommandAllocator> dispatchAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dispatchAlloc));
    ComPtr<ID3D12GraphicsCommandList> dispatchList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dispatchAlloc.Get(), nullptr, IID_PPV_ARGS(&dispatchList));
    dispatchList->Close();
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;

    std::ifstream fi("t_nvfp4_gemv_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { fprintf(stderr, "missing t_mxfp4_gemv_sk.dxil\n"); return 1; }
    D3D12_ROOT_PARAMETER rp[8] = {};
    for (int i = 0; i < 6; ++i) { rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; rp[i].Descriptor.ShaderRegister = (UINT)i; rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; }
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[6].Descriptor.ShaderRegister = 0; rp[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[7].Descriptor.ShaderRegister = 0; rp[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = { 8, rp, 0, nullptr };
    ComPtr<ID3DBlob> sig, errb;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &errb);
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    fprintf(stderr, "pso ok\n"); fflush(stderr);

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

    auto submit = [&](ComPtr<ID3D12GraphicsCommandList> lst, const char* tag) -> bool {
        HRESULT hc = lst->Close();
        if (FAILED(hc)) { fprintf(stderr, "submit[%s] Close failed hr=0x%08X\n", tag, hc); fflush(stderr); return false; }
        ID3D12CommandList* ls[] = { lst.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fv;
        HRESULT hs = queue->Signal(fence.Get(), fv);
        if (FAILED(hs)) { fprintf(stderr, "submit[%s] Signal failed hr=0x%08X\n", tag, hs); fflush(stderr); return false; }
        ResetEvent(ev); fence->SetEventOnCompletion(fv, ev);
        DWORD wr = WaitForSingleObject(ev, 30000);
        if (wr != WAIT_OBJECT_0) { fprintf(stderr, "submit[%s] Wait res=%lu fv=%llu\n", tag, wr, fv); fflush(stderr); return false; }
        return true;
    };

    auto createWeight = [&](UINT64 M, UINT32 K) -> Mxfp4Weight {
        Mxfp4Weight w;
        w.M = M; w.K = K; w.nb = K / 8; w.ns = K / 32;
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.nb * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rW));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rS));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rO));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rA));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rG));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rR));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.rOut));
        device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRb));
        device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.rCb));
        return w;
    };

    std::unordered_map<std::string, Mxfp4Weight> weights;
    fprintf(stderr, "mxfp4-v3 server ready\n"); fflush(stderr);

    while (true) {
        std::string line;
        if (!readLine(0, line)) { fprintf(stderr, "readline fail\n"); break; }
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

        if (cmd == "QUIT") { fprintf(stderr, "QUIT\n"); break; }
        else if (cmd == "STATELESS") {
            if (t.size() < 7) { fprintf(stderr, "STATELESS: bad args\n"); continue; }
            UINT32 M = std::stoul(t[1]);
            UINT32 K = std::stoul(t[2]);
            UINT32 szP = std::stoul(t[3]);
            UINT32 szS = std::stoul(t[4]);
            UINT32 szA = std::stoul(t[5]);
            UINT32 szB = std::stoul(t[6]);
            if ((K & 31) != 0) { fprintf(stderr, "STATELESS: K not mult of 32\n"); continue; }
            UINT32 nb = K / 8, ns = K / 32;
            UINT32 szOut = M * 4;
            UINT32 szO = M * ns * 4;
            UINT32 szG = M * 4;
            std::vector<uint8_t> packed(szP), scales(szS), act(szA), bias(szB);
            if (!readN(0, packed.data(), szP)) { fprintf(stderr, "STATELESS packed read fail\n"); continue; }
            if (!readN(0, scales.data(), szS)) { fprintf(stderr, "STATELESS scales read fail\n"); continue; }
            if (!readN(0, act.data(), szA)) { fprintf(stderr, "STATELESS act read fail\n"); continue; }
            if (!readN(0, bias.data(), szB)) { fprintf(stderr, "STATELESS bias read fail\n"); continue; }
            // Force recreation of all resources for each STATELESS request
            weights.erase("__stateless__");
            Mxfp4Weight& w = weights["__stateless__"] = createWeight(M, K);
            UINT64 total = (UINT64)szP + szS + szO + szA + szB + szG + szG;
            if (w.uploadLoadCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.uploadLoad));
                w.uploadLoadCap = total;
            }
            UINT64 offP = 0, offS = szP, offA = offS + szS, offB = offA + szA, offG = offB + szB, offR = offG + szG;
            void* m = nullptr; w.uploadLoad->Map(0, nullptr, &m);
            std::memcpy((char*)m + offP, packed.data(), szP);
            std::memcpy((char*)m + offS, scales.data(), szS);
            std::memset((char*)m + offS + szS, 0, szO);
            // Bit-cast act: int32 (same bit pattern as float32) -> float (memcpy, NOT (float)int)
            {
                float* aDst = (float*)((char*)m + offA);
                int32_t* aSrc = (int32_t*)act.data();
                UINT32 nAct = szA / 4;
                for (UINT32 i = 0; i < nAct; i++) std::memcpy(&aDst[i], &aSrc[i], 4);
            }
            std::memcpy((char*)m + offB, bias.data(), szB);
            { float* g = (float*)((char*)m + offG); for (UINT32 i = 0; i < M; i++) g[i] = 1.0f;
              float* rb = (float*)((char*)m + offR); for (UINT32 i = 0; i < M; i++) rb[i] = 0.0f; }
            w.uploadLoad->Unmap(0, nullptr);
            uploadList->Reset(uploadAlloc.Get(), nullptr);
            uploadList->CopyBufferRegion(w.rW.Get(), 0, w.uploadLoad.Get(), offP, szP);
            uploadList->CopyBufferRegion(w.rS.Get(), 0, w.uploadLoad.Get(), offS, szS);
            uploadList->CopyBufferRegion(w.rB.Get(), 0, w.uploadLoad.Get(), offA, szA);
            uploadList->CopyBufferRegion(w.rA.Get(), 0, w.uploadLoad.Get(), offB, szB);
            uploadList->CopyBufferRegion(w.rG.Get(), 0, w.uploadLoad.Get(), offG, szG);
            uploadList->CopyBufferRegion(w.rR.Get(), 0, w.uploadLoad.Get(), offR, szG);
            { D3D12_RESOURCE_BARRIER bs[7] = {}; int nbb = 0;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rW.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rS.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rB.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rA.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rG.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rR.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
              uploadList->ResourceBarrier(nbb, bs); }
            if (!submit(uploadList, "STATELESS-up")) continue;
            // Ensure upload completes before dispatch
            queue->Signal(fence.Get(), ++fv);
            fence->SetEventOnCompletion(fv, ev);
            WaitForSingleObject(ev, 5000);
                        w.rWSt = w.rSSt = w.rBSt = w.rASt = w.rGSt = w.rRSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            void* cm = nullptr; w.rCb->Map(0, nullptr, &cm);
            struct { uint32_t K, nbPerRow, nsPerRow, pad; } cbv = { K, ns, ns, 0 };  // nbPerRow=块数(K/32), 非uint数!
            std::memcpy(cm, &cbv, sizeof(cbv));
            w.rCb->Unmap(0, nullptr);
            // Recreate PSO for each STATELESS request (fix NVFP4 NaN)
            ComPtr<ID3D12PipelineState> psoStateless;
            device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&psoStateless));
            dispatchList->SetPipelineState(psoStateless.Get());
            dispatchList->Reset(dispatchAlloc.Get(), psoStateless.Get());
                        dispatchList->SetComputeRootSignature(rs.Get());
            dispatchList->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());  // t0: packed
            dispatchList->SetComputeRootShaderResourceView(1, w.rS->GetGPUVirtualAddress());  // t1: scl (NVFP4 fp16)
            dispatchList->SetComputeRootShaderResourceView(2, w.rA->GetGPUVirtualAddress());  // t2: per-row bias (NVFP4, M*4)
            dispatchList->SetComputeRootShaderResourceView(3, w.rB->GetGPUVirtualAddress());  // t3: act (float K)
            dispatchList->SetComputeRootShaderResourceView(4, w.rG->GetGPUVirtualAddress());  // t4: gbl (per-row)
            dispatchList->SetComputeRootShaderResourceView(5, w.rR->GetGPUVirtualAddress());  // t5: rowBias (per-row)
            dispatchList->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());  // u0: outv
            dispatchList->SetComputeRootConstantBufferView(7, w.rCb->GetGPUVirtualAddress());  // b0: cbuffer
            if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                dispatchList->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }

            
            fprintf(stderr, "  DEBUG: M=%u K=%u nb=%u ns=%u\n", (unsigned)M, (unsigned)K, (unsigned)nb, (unsigned)ns);
            fprintf(stderr, "  DEBUG: rW ptr=%llx size=%llu\n", (unsigned long long)w.rW->GetGPUVirtualAddress(), (unsigned long long)w.rW->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rS ptr=%llx size=%llu\n", (unsigned long long)w.rS->GetGPUVirtualAddress(), (unsigned long long)w.rS->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rA ptr=%llx size=%llu\n", (unsigned long long)w.rA->GetGPUVirtualAddress(), (unsigned long long)w.rA->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rB ptr=%llx size=%llu\n", (unsigned long long)w.rB->GetGPUVirtualAddress(), (unsigned long long)w.rB->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rG ptr=%llx size=%llu\n", (unsigned long long)w.rG->GetGPUVirtualAddress(), (unsigned long long)w.rG->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rR ptr=%llx size=%llu\n", (unsigned long long)w.rR->GetGPUVirtualAddress(), (unsigned long long)w.rR->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rOut ptr=%llx size=%llu\n", (unsigned long long)w.rOut->GetGPUVirtualAddress(), (unsigned long long)w.rOut->GetDesc().Width);
            fprintf(stderr, "  DEBUG: rCb ptr=%llx size=%llu\n", (unsigned long long)w.rCb->GetGPUVirtualAddress(), (unsigned long long)w.rCb->GetDesc().Width);
            // Read back first few values from upload buffer
            void* dbg_m = nullptr; w.uploadLoad->Map(0, nullptr, &dbg_m);
            float* dbg_f = (float*)((char*)dbg_m + offP);
            fprintf(stderr, "  DEBUG: upload[0..3]=0x%08X 0x%08X 0x%08X 0x%08X\n", 
                    ((uint32_t*)dbg_m)[0], ((uint32_t*)dbg_m)[1], ((uint32_t*)dbg_m)[2], ((uint32_t*)dbg_m)[3]);
            fprintf(stderr, "  DEBUG: upload[offS..offS+3]=0x%08X 0x%08X 0x%08X 0x%08X\n", 
                    ((uint32_t*)((char*)dbg_m + offS))[0], ((uint32_t*)((char*)dbg_m + offS))[1], 
                    ((uint32_t*)((char*)dbg_m + offS))[2], ((uint32_t*)((char*)dbg_m + offS))[3]);
            fprintf(stderr, "  DEBUG: upload[offA..offA+3]=%f %f %f %f\n", 
                    ((float*)((char*)dbg_m + offA))[0], ((float*)((char*)dbg_m + offA))[1],
                    ((float*)((char*)dbg_m + offA))[2], ((float*)((char*)dbg_m + offA))[3]);
            fprintf(stderr, "  DEBUG: upload[offB..offB+3]=%f %f %f %f\n", 
                    ((float*)((char*)dbg_m + offB))[0], ((float*)((char*)dbg_m + offB))[1],
                    ((float*)((char*)dbg_m + offB))[2], ((float*)((char*)dbg_m + offB))[3]);
            w.uploadLoad->Unmap(0, nullptr);
            
            dispatchList->Dispatch((UINT)M, 1, 1);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
              dispatchList->ResourceBarrier(1, &b);
              w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE; }
            dispatchList->CopyResource(w.rRb.Get(), w.rOut.Get());
            // Read back rOut
            void* om = nullptr; w.rRb->Map(0, nullptr, &om);
            fprintf(stderr, "  DEBUG: outv raw bytes = %02X %02X %02X %02X\n",
                    ((uint8_t*)om)[0], ((uint8_t*)om)[1], ((uint8_t*)om)[2], ((uint8_t*)om)[3]);
            w.rRb->Unmap(0, nullptr);
            
            if (!submit(dispatchList, "STATELESS-dispatch")) continue;
            w.rRb->Map(0, nullptr, &om);
            fprintf(stderr, "  DEBUG: rRb bytes after submit = %02X %02X %02X %02X\n",
                    ((uint8_t*)om)[0], ((uint8_t*)om)[1], ((uint8_t*)om)[2], ((uint8_t*)om)[3]);
            writeAll(1, &szOut, 4);
            writeAll(1, om, szOut);
            w.rRb->Unmap(0, nullptr);
        } else if (cmd == "MULTI_GEMV") {
            // MULTI_GEMV B K szPPerItem szSPerItem szAPerItem szBPerItem gblPerItem
            // Body: B items concatenated (each: packed + scales + act + bias + gbl)
            if (t.size() < 8) { fprintf(stderr, "MULTI_GEMV: bad args\n"); continue; }
            UINT32 B = std::stoul(t[1]);
            UINT32 K = std::stoul(t[2]);
            UINT32 szPPer = std::stoul(t[3]);
            UINT32 szSPer = std::stoul(t[4]);
            UINT32 szAPer = std::stoul(t[5]);
            UINT32 szBPer = std::stoul(t[6]);
            UINT32 gblPer = std::stoul(t[7]);
            if ((K & 31) != 0) { fprintf(stderr, "MULTI_GEMV: K not mult of 32\n"); continue; }
            UINT32 nb = K / 8, ns = K / 32;   // nb = packed uints/row; ns = 32-elem blocks/row
            UINT32 szOut = B * 4;
            UINT64 perItem = (UINT64)szPPer + szSPer + szAPer + szBPer + gblPer;
            UINT64 bodyBytes = (UINT64)B * perItem;
            UINT64 totalBytes = bodyBytes + (UINT64)B * 4;  // 尾部追加 rowBias 零填充
            std::vector<uint8_t> body(totalBytes, 0);
            if (!readN(0, body.data(), bodyBytes)) { fprintf(stderr, "MULTI_GEMV body read fail\n"); continue; }
            // rowBias 区 (body 之后) 保持全零
            Mxfp4Weight w = createWeight(B, K);
            weights["__multi__"] = w;
            Mxfp4Weight& mw = weights["__multi__"];
            if (mw.uploadLoadCap < totalBytes) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(totalBytes, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&mw.uploadLoad));
                mw.uploadLoadCap = totalBytes;
            }
            { void* m = nullptr; mw.uploadLoad->Map(0, nullptr, &m);
              std::memcpy(m, body.data(), (size_t)totalBytes);
              mw.uploadLoad->Unmap(0, nullptr); }
            // 上传: 每项数据按行主序展开到 GPU buffer
            uploadList->Reset(uploadAlloc.Get(), nullptr);
            for (UINT32 i = 0; i < B; i++) {
                UINT64 off = (UINT64)i * perItem;
                uploadList->CopyBufferRegion(mw.rW.Get(), (UINT64)i * nb * 4, mw.uploadLoad.Get(), off, szPPer);                                          // packed
                uploadList->CopyBufferRegion(mw.rS.Get(), (UINT64)i * ns * 4, mw.uploadLoad.Get(), off + szPPer, szSPer);                                  // scales
                uploadList->CopyBufferRegion(mw.rB.Get(), (UINT64)i * K * 4, mw.uploadLoad.Get(), off + szPPer + szSPer, szAPer);                          // act
                uploadList->CopyBufferRegion(mw.rA.Get(), (UINT64)i * ns * 4, mw.uploadLoad.Get(), off + szPPer + szSPer + szAPer, szBPer);                // bias(每块)
                uploadList->CopyBufferRegion(mw.rG.Get(), (UINT64)i * 4,     mw.uploadLoad.Get(), off + szPPer + szSPer + szAPer + szBPer, gblPer);       // gbl
                uploadList->CopyBufferRegion(mw.rR.Get(), (UINT64)i * 4,     mw.uploadLoad.Get(), bodyBytes + (UINT64)i * 4, 4);                          // rowBias=0
            }
            { D3D12_RESOURCE_BARRIER bs[6] = {}; int nbb = 0;
              auto addSRV = [&](ID3D12Resource* r) {
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = r;
                bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb; };
              addSRV(mw.rW.Get()); addSRV(mw.rS.Get()); addSRV(mw.rB.Get());
              addSRV(mw.rA.Get()); addSRV(mw.rG.Get()); addSRV(mw.rR.Get());
              uploadList->ResourceBarrier(nbb, bs); }
            if (!submit(uploadList, "MULTI_GEMV-up")) continue;
            mw.rWSt = mw.rSSt = mw.rBSt = mw.rASt = mw.rGSt = mw.rRSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            void* cm = nullptr; mw.rCb->Map(0, nullptr, &cm);
            struct { uint32_t K, nbPerRow, nsPerRow, pad; } cbv = { K, ns, ns, 0 };  // nbPerRow=块数(K/32)
            std::memcpy(cm, &cbv, sizeof(cbv));
            mw.rCb->Unmap(0, nullptr);
            // 绑定 (与 STATELESS 相同的 8 槽布局)
            dispatchList->Reset(dispatchAlloc.Get(), pso.Get());
            dispatchList->SetComputeRootSignature(rs.Get());
            dispatchList->SetComputeRootShaderResourceView(0, mw.rW->GetGPUVirtualAddress());  // t0 packed
            dispatchList->SetComputeRootShaderResourceView(1, mw.rS->GetGPUVirtualAddress());  // t1 scl
            dispatchList->SetComputeRootShaderResourceView(2, mw.rA->GetGPUVirtualAddress());  // t2 bias(每块)
            dispatchList->SetComputeRootShaderResourceView(3, mw.rB->GetGPUVirtualAddress());  // t3 act
            dispatchList->SetComputeRootShaderResourceView(4, mw.rG->GetGPUVirtualAddress());  // t4 gbl
            dispatchList->SetComputeRootShaderResourceView(5, mw.rR->GetGPUVirtualAddress());  // t5 rowBias
            dispatchList->SetComputeRootUnorderedAccessView(6, mw.rOut->GetGPUVirtualAddress()); // u0 outv
            dispatchList->SetComputeRootConstantBufferView(7, mw.rCb->GetGPUVirtualAddress());  // b0 cbuffer
            if (mw.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = mw.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = mw.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                dispatchList->ResourceBarrier(1, &b);
                mw.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
            dispatchList->Dispatch(B, 1, 1);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = mw.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
              dispatchList->ResourceBarrier(1, &b);
              mw.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE; }
            dispatchList->CopyResource(mw.rRb.Get(), mw.rOut.Get());
            if (!submit(dispatchList, "MULTI_GEMV-dispatch")) continue;
            void* om = nullptr; mw.rRb->Map(0, nullptr, &om);
            writeAll(1, &szOut, 4);
            writeAll(1, om, szOut);
            mw.rRb->Unmap(0, nullptr);
            fprintf(stderr, "  MULTI_GEMV B=%u done\n", B); fflush(stderr);
                } else {
            fprintf(stderr, "unknown cmd: %s\\n", cmd.c_str());
        }
    }
    return 0;
}
