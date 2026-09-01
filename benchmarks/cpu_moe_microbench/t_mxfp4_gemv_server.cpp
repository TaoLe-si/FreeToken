// [P1c] Generic MXFP4 GEMV server (robust rewrite)
// Protocol (stdin binary): hdr [M,K,szPacked,szScales,szBiases] (5xuint32 LE)
//   then packed bytes, scales bytes, biases bytes, act (K*4 float)
// Response (stdout binary): len (uint32) + M*4 bytes float out
// Full sync after every submit; explicit per-resource state tracking.
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
static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    fprintf(stderr, "ready fd0=%d fd1=%d fd2=%d\n", _fileno(stdin), _fileno(stdout), _fileno(stderr)); fflush(stderr);

    ComPtr<ID3D12Debug> dbg;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); }
    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    HRESULT hr_dev = D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr_dev)) { fprintf(stderr, "device create failed hr=0x%08X\n", hr_dev); return 1; }
    fprintf(stderr, "device ok\n"); fflush(stderr);
    ComPtr<ID3D12InfoQueue> iq;
    if (SUCCEEDED(device.As(&iq))) { }

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

    std::ifstream fi("t_mxfp4_gemv_sk.dxil", std::ios::binary);
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
    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rGbl, rRowB, rOut, rRb, rCb;
    D3D12_RESOURCE_STATES rWState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rSState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rBState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rActState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rGblState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rRowBState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOutState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> uploadBuf;
    UINT64 uploadCap = 0;
    UINT64 curM = 0, curK = 0;

    auto submit = [&](ComPtr<ID3D12GraphicsCommandList> lst, const char* tag) -> bool {
        HRESULT hc = lst->Close();
        if (FAILED(hc)) {
            fprintf(stderr, "submit[%s] Close failed hr=0x%08X\n", tag, hc); fflush(stderr);
            if (iq) {
                UINT64 n = iq->GetNumStoredMessages();
                fprintf(stderr, "  === %llu stored validation messages ===\n", (unsigned long long)n); fflush(stderr);
                for (UINT64 i = 0; i < n; i++) {
                    SIZE_T sz = 0; iq->GetMessage(i, nullptr, &sz);
                    std::vector<uint8_t> mbuf(sz);
                    D3D12_MESSAGE* m = (D3D12_MESSAGE*)mbuf.data();
                    iq->GetMessage(i, m, &sz);
                    fprintf(stderr, "  [%llu] %s\n", (unsigned long long)i, m->pDescription ? m->pDescription : "(no desc)"); fflush(stderr);
                }
                iq->ClearStoredMessages();
            }
            return false;
        }
        ID3D12CommandList* ls[] = { lst.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fv;
        HRESULT hs = queue->Signal(fence.Get(), fv);
        if (FAILED(hs)) { fprintf(stderr, "submit[%s] Signal failed hr=0x%08X\n", tag, hs); fflush(stderr); return false; }
        ResetEvent(ev); fence->SetEventOnCompletion(fv, ev);
        DWORD wr = WaitForSingleObject(ev, 30000);
        if (wr != WAIT_OBJECT_0) { fprintf(stderr, "submit[%s] Wait res=%lu fv=%llu\n", tag, (unsigned long)wr, (unsigned long long)fv); fflush(stderr); return false; }
        return true;
    };

    while (true) {
        uint32_t hdr[6];
        if (!readN(0, hdr, 24)) { fprintf(stderr, "header read failed\n"); break; }
        uint32_t M = hdr[0], K = hdr[1];
        uint32_t szPacked = hdr[2], szAct = hdr[3], szScales = hdr[4], szBiases = hdr[5];
        if ((K & 31) != 0) { fprintf(stderr, "K not mult of 32: %u\n", (unsigned)K); break; }
        uint32_t nb = K / 8, ns = K / 32;
        if (szAct != (uint32_t)K*4) { fprintf(stderr, "szAct mismatch\n"); break; }
        std::vector<uint8_t> packed(szPacked), act(szAct), scales(szScales), biases(szBiases);
        if (!readN(0, packed.data(), szPacked)) { fprintf(stderr, "packed read failed\n"); break; }
        if (!readN(0, act.data(), szAct)) { fprintf(stderr, "act read failed\n"); break; }
        if (!readN(0, scales.data(), szScales)) { fprintf(stderr, "scales read failed\n"); break; }
        if (!readN(0, biases.data(), szBiases)) { fprintf(stderr, "biases read failed\n"); break; }

        auto t0 = std::chrono::high_resolution_clock::now();
        // (re)alloc stage resources if shape changed
        if (M != curM || K != curK) { fprintf(stderr, "REALLOC curM=%u M=%u curK=%llu K=%u\n", (unsigned)curM, (unsigned)M, (unsigned long long)curK, (unsigned)K); fflush(stderr);
            rW.Reset(); rS.Reset(); rB.Reset(); rAct.Reset(); rGbl.Reset(); rRowB.Reset(); rOut.Reset(); rRb.Reset(); rCb.Reset();
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * nb * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rW));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rS));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rB));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rAct));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rGbl));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRowB));
            device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&rOut));
            device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRb));
            device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&rCb));
            rWState = D3D12_RESOURCE_STATE_COPY_DEST;
            rSState = D3D12_RESOURCE_STATE_COPY_DEST;
            rBState = D3D12_RESOURCE_STATE_COPY_DEST;
            rActState = D3D12_RESOURCE_STATE_COPY_DEST;
            rGblState = D3D12_RESOURCE_STATE_COPY_DEST;
            rRowBState = D3D12_RESOURCE_STATE_COPY_DEST;
            rOutState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            curM = M; curK = K;
        }

        // ---- batch upload: one list, all copies + transitions ----
        UINT64 szG = (UINT64)M * 4;
        UINT64 offP = 0, offA = szPacked, offS = szPacked + szAct, offB = szPacked + szAct + szScales;
        UINT64 offG = offB + szBiases, offR = offG + szG;
        UINT64 total = offR + szG;
        if (uploadCap < total) {
            uploadBuf.Reset();
            device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
            uploadCap = total;
            fprintf(stderr, "uploadBuf grew to %llu\n", (unsigned long long)total); fflush(stderr);
        }
                void* um = nullptr; uploadBuf->Map(0, nullptr, &um);
        std::memcpy((char*)um + offP, packed.data(), szPacked);
        std::memcpy((char*)um + offA, act.data(), szAct);
        std::memcpy((char*)um + offS, scales.data(), szScales);
        std::memcpy((char*)um + offB, biases.data(), szBiases);
        {
            float* g = (float*)((char*)um + offG); for (UINT32 i = 0; i < M; i++) g[i] = 1.0f;
            float* rb = (float*)((char*)um + offR); for (UINT32 i = 0; i < M; i++) rb[i] = 0.0f;
        }
        uploadBuf->Unmap(0, nullptr);
uploadList->Reset(uploadAlloc.Get(), nullptr);
        // transition NPSR->COPY_DEST where needed (skip if already COPY_DEST to avoid no-op barriers)
        {
            D3D12_RESOURCE_BARRIER bs[4]; int nb = 0;
#define ADDTRANS(res, st) do { if (st == D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) { bs[nb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nb].Transition.pResource = res.Get(); bs[nb].Transition.Subresource = 0; bs[nb].Transition.StateBefore = st; bs[nb].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST; ++nb; st = D3D12_RESOURCE_STATE_COPY_DEST; } } while(0)
            ADDTRANS(rW, rWState); ADDTRANS(rS, rSState); ADDTRANS(rB, rBState); ADDTRANS(rAct, rActState); ADDTRANS(rGbl, rGblState); ADDTRANS(rRowB, rRowBState);
#undef ADDTRANS
            if (nb > 0) uploadList->ResourceBarrier(nb, bs);
        }
        uploadList->CopyBufferRegion(rW.Get(), 0, uploadBuf.Get(), offP, szPacked);
        uploadList->CopyBufferRegion(rS.Get(), 0, uploadBuf.Get(), offA, szAct);  // T1 = act
        uploadList->CopyBufferRegion(rB.Get(), 0, uploadBuf.Get(), offS, szScales);  // T2 unused
        uploadList->CopyBufferRegion(rAct.Get(), 0, uploadBuf.Get(), offA, szAct);  // rAct=act bytes (T3=HLSL t3=act)
        uploadList->CopyBufferRegion(rGbl.Get(), 0, uploadBuf.Get(), offG, szG);
        uploadList->CopyBufferRegion(rRowB.Get(), 0, uploadBuf.Get(), offR, szG);
        {
            D3D12_RESOURCE_BARRIER bs[6] = {};
            bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[0].Transition.pResource = rW.Get(); bs[0].Transition.Subresource = 0; bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bs[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[1].Transition.pResource = rS.Get(); bs[1].Transition.Subresource = 0; bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bs[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[2].Transition.pResource = rB.Get(); bs[2].Transition.Subresource = 0; bs[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[2].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bs[3].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[3].Transition.pResource = rAct.Get(); bs[3].Transition.Subresource = 0; bs[3].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[3].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bs[4].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[4].Transition.pResource = rGbl.Get(); bs[4].Transition.Subresource = 0; bs[4].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[4].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bs[5].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[5].Transition.pResource = rRowB.Get(); bs[5].Transition.Subresource = 0; bs[5].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[5].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            uploadList->ResourceBarrier(6, bs);
        }
        rWState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        rSState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        rBState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        rActState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        rGblState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        rRowBState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if (!submit(uploadList, "upload")) break;

        // ---- dispatch ----
        void* cm = nullptr; rCb->Map(0, nullptr, &cm);
        struct { uint32_t K, nbPerRow, nsPerRow, pad; } cbv = { K, nb, ns, 0 };
        std::memcpy(cm, &cbv, sizeof(cbv));
        rCb->Unmap(0, nullptr);
        HRESULT hda = dispatchAlloc->Reset();
        if (FAILED(hda)) { fprintf(stderr, "dispatchAlloc reset hr=0x%08X\n", hda); break; }
        dispatchList->Reset(dispatchAlloc.Get(), pso.Get());
        dispatchList->SetComputeRootSignature(rs.Get());
        dispatchList->SetComputeRootShaderResourceView(0, rW->GetGPUVirtualAddress());
        dispatchList->SetComputeRootShaderResourceView(1, rS->GetGPUVirtualAddress());
        dispatchList->SetComputeRootShaderResourceView(2, rB->GetGPUVirtualAddress());
        dispatchList->SetComputeRootShaderResourceView(3, rAct->GetGPUVirtualAddress());
        dispatchList->SetComputeRootShaderResourceView(4, rGbl->GetGPUVirtualAddress());
        dispatchList->SetComputeRootShaderResourceView(5, rRowB->GetGPUVirtualAddress());
        dispatchList->SetComputeRootUnorderedAccessView(6, rOut->GetGPUVirtualAddress());
        dispatchList->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
        fprintf(stderr, "  before dispatch M=%u rOutState=%d\n", (unsigned)M, (int)rOutState); fflush(stderr);
        if (rOutState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = rOut.Get(); b.Transition.Subresource = 0;
            b.Transition.StateBefore = rOutState; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            dispatchList->ResourceBarrier(1, &b);
            rOutState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        dispatchList->Dispatch((UINT)M, 1, 1);
        {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = rOut.Get(); b.Transition.Subresource = 0;
            b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            dispatchList->ResourceBarrier(1, &b);
            rOutState = D3D12_RESOURCE_STATE_COPY_SOURCE;
        }
        dispatchList->CopyResource(rRb.Get(), rOut.Get());
        if (!submit(dispatchList, "dispatch")) break;

        void* om = nullptr; rRb->Map(0, nullptr, &om);
        uint32_t szOut = M * 4;
        fprintf(stderr, "  readback:"); fflush(stderr);
        for (uint32_t di = 0; di < szOut && di < 32; di++) fprintf(stderr, " %02x", (unsigned)((unsigned char*)om)[di]);
        fprintf(stderr, "\n"); fflush(stderr);
        for (uint32_t di = 0; di < szOut/4 && di < 8; di++) fprintf(stderr, "  v[%u]=%.3f", di, (double)((float*)om)[di]);
        fprintf(stderr, "\n"); fflush(stderr);
        _write(1, &szOut, 4);
        _write(1, om, szOut);
        rRb->Unmap(0, nullptr);
        _flushall();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "Dispatch M=%u K=%u: %.3fms\n", (unsigned)M, (unsigned)K, ms); fflush(stderr);
    }
    return 0;
}
