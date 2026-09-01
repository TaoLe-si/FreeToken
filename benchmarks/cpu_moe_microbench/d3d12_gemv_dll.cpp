#include <d3d12.h>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <stdio.h>
#include <cstring>
#include <cstdint>
#include <new>
using namespace Microsoft::WRL;

// ---- D3D12 helper
static D3D12_RESOURCE_DESC make_buf(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = sz; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1; d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f;
    return d;
}

// ---- internal state
struct IgpuHandle {
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> upload, uav, readback;
    ComPtr<ID3D12Fence> fence;
    HANDLE ev;
    UINT64 fenceVal = 0;
    UINT inc = 0;
    UINT64 curTotal = 0;
    UINT curM = 0, curK = 0;
    bool ready = false;
    // Weight-persistence: the Python executor passes the SAME stable host
    // buffers every call for weights/scales, so re-upload only activations.
    const void* last_pk = nullptr;
    const void* last_sc = nullptr;
    unsigned char* pmap = nullptr;
    float* omap = nullptr;
    char errmsg[256];
};

// ---- find AMD adapter
static int _find_amd_adapter(ComPtr<IDXGIAdapter1>& out) {
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return -1;
    for (UINT i = 0; ; ++i) {
        ComPtr<IDXGIAdapter1> adp;
        if (factory->EnumAdapters1(i, &adp) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d; adp->GetDesc1(&d);
        if (d.VendorId == 0x1002) { out = adp; return (int)i; }
    }
    return -1;
}

// ---- exported API

extern "C" {

__declspec(dllexport) void* igpu_create() {
    IgpuHandle* h = new (std::nothrow) IgpuHandle();
    if (!h) return nullptr;
    h->ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!h->ev) { delete h; return nullptr; }
    // AMD adapter
    ComPtr<IDXGIAdapter1> adp;
    if (_find_amd_adapter(adp) < 0) {
        // fallback to default
        ComPtr<IDXGIFactory1> f;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f)))) { delete h; return nullptr; }
        if (FAILED(f->EnumAdapters1(0, &adp))) { delete h; return nullptr; }
    }
    if (FAILED(D3D12CreateDevice(adp.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&h->dev)))) {
        snprintf(h->errmsg, sizeof(h->errmsg), "D3D12CreateDevice failed"); delete h; return nullptr;
    }
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
    if (FAILED(h->dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&h->queue)))) {
        snprintf(h->errmsg, sizeof(h->errmsg), "CreateCommandQueue failed"); delete h; return nullptr;
    }
    // load dxil from embedded or external? We'll use a file path or embed.
    // For now, path: d3d12_gemv_sk.dxil in CWD.
    FILE* f = fopen("d3d12_gemv_sk.dxil", "rb");
    if (!f) { snprintf(h->errmsg, sizeof(h->errmsg), "dxil open failed"); delete h; return nullptr; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* dxil = new char[sz];
    fread(dxil, 1, sz, f); fclose(f);
    // root signature
    D3D12_DESCRIPTOR_RANGE ranges[6] = {};
    for (int i = 0; i < 6; ++i) {
        ranges[i].RangeType = (i < 5) ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[i].NumDescriptors = 1; ranges[i].BaseShaderRegister = (i < 5) ? (UINT)i : 0;
        ranges[i].RegisterSpace = 0; ranges[i].OffsetInDescriptorsFromTableStart = i;
    }
    D3D12_ROOT_PARAMETER rp[2] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[0].DescriptorTable.NumDescriptorRanges = 6; rp[0].DescriptorTable.pDescriptorRanges = ranges;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[1].Constants.Num32BitValues = 4; rp[1].Constants.ShaderRegister = 0; rp[1].Constants.RegisterSpace = 0;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2; rsd.pParameters = rp; rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> rsb, errb;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsb, &errb)) ||
        FAILED(h->dev->CreateRootSignature(0, rsb->GetBufferPointer(), rsb->GetBufferSize(), IID_PPV_ARGS(&h->rs)))) {
        snprintf(h->errmsg, sizeof(h->errmsg), "root signature failed"); delete[] dxil; delete h; return nullptr;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = h->rs.Get(); psd.CS.pShaderBytecode = dxil; psd.CS.BytecodeLength = sz;
    if (FAILED(h->dev->CreateComputePipelineState(&psd, IID_PPV_ARGS(&h->pso)))) {
        snprintf(h->errmsg, sizeof(h->errmsg), "PSO failed"); delete[] dxil; delete h; return nullptr;
    }
    delete[] dxil;
    if (FAILED(h->dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&h->alloc))) ||
        FAILED(h->dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&h->fence))) ||
        FAILED(h->dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, h->alloc.Get(), nullptr, IID_PPV_ARGS(&h->cl)))) {
        snprintf(h->errmsg, sizeof(h->errmsg), "alloc/fence/cl failed"); delete h; return nullptr;
    }
    h->cl->Close();
    h->inc = h->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    h->ready = true;
    return (void*)h;
}

__declspec(dllexport) int igpu_gemv(void* handle, int M, int K,
    const unsigned char* packed, const unsigned int* scl,
    const int* act, const float* asb, const float* gbl, float* out) {
    if (!handle || !packed || !scl || !act || !asb || !gbl || !out) return -1;
    IgpuHandle* h = (IgpuHandle*)handle;
    if (!h->ready) return -2;
    if (M <= 0 || K <= 0 || (K & 15)) return -3;
    int NB = K / 16;
    UINT64 off_pk = 0, off_scl = (UINT64)M * NB * 8, off_act = off_scl + (UINT64)M * NB * 4,
           off_asb = off_act + (UINT64)NB * 16 * 4, off_gbl = off_asb + NB * 4,
           total = off_gbl + (UINT64)M * 4;
    D3D12_HEAP_PROPERTIES hup = { D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hrb = { D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hdef = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    if (total > h->curTotal || (UINT)M != h->curM || (UINT)K != h->curK) {
        h->upload.Reset(); h->uav.Reset(); h->readback.Reset(); h->heap.Reset();
        if (FAILED(h->dev->CreateCommittedResource(&hup, D3D12_HEAP_FLAG_NONE, &make_buf(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&h->upload)))) return -4;
        if (FAILED(h->dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &make_buf((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&h->readback)))) return -4;
        if (FAILED(h->dev->CreateCommittedResource(&hdef, D3D12_HEAP_FLAG_NONE, &make_buf((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&h->uav)))) return -4;
        D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 6, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
        h->dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&h->heap));
        D3D12_CPU_DESCRIPTOR_HANDLE ch = h->heap->GetCPUDescriptorHandleForHeapStart();
        auto mk_srv = [&](UINT64 off, UINT num, UINT stride) {
            D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
            s.Format = DXGI_FORMAT_UNKNOWN; s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.Buffer.FirstElement = (UINT)(off / stride); s.Buffer.NumElements = num; s.Buffer.StructureByteStride = stride;
            h->dev->CreateShaderResourceView(h->upload.Get(), &s, ch); ch.ptr += h->inc;
        };
        mk_srv(off_pk, M * NB, 8);
        mk_srv(off_scl, M * NB, 4);
        mk_srv(off_act, NB * 16, 4);
        mk_srv(off_asb, NB, 4);
        mk_srv(off_gbl, M, 4);
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; u.Buffer.NumElements = M; u.Buffer.StructureByteStride = 4;
        h->dev->CreateUnorderedAccessView(h->uav.Get(), nullptr, &u, ch);
        h->curTotal = total; h->curM = M; h->curK = K;
    }
    // upload data -- skip the ~133 MB weight/scale copy when the caller passed
    // the same stable buffers as last time (per-token saving: entire decode).
    void* pmap = nullptr;
    h->upload->Map(0, nullptr, &pmap);
    const bool w_stable = (packed == h->last_pk) && (scl == h->last_sc)
                          && (total == h->curTotal) && (M == h->curM) && (K == h->curK);
    if (!w_stable) {
        memcpy(pmap, packed, off_scl);
        memcpy((char*)pmap + off_scl, scl, off_act - off_scl);
        h->last_pk = packed; h->last_sc = scl;
    }
    memcpy((char*)pmap + off_act, act, off_asb - off_act);
    memcpy((char*)pmap + off_asb, asb, off_gbl - off_asb);
    memcpy((char*)pmap + off_gbl, gbl, total - off_gbl);
    h->upload->Unmap(0, nullptr);
    // record command list
    h->alloc->Reset();
    h->cl->Reset(h->alloc.Get(), h->pso.Get());
    h->cl->SetComputeRootSignature(h->rs.Get());
    ID3D12DescriptorHeap* heaps[] = { h->heap.Get() };
    h->cl->SetDescriptorHeaps(1, heaps);
    h->cl->SetComputeRootDescriptorTable(0, h->heap->GetGPUDescriptorHandleForHeapStart());
    UINT32 pconst[4] = { (UINT32)K, (UINT32)NB, 0, 0 };
    float gs = 0.25f;
    memcpy(&pconst[2], &gs, 4);
    h->cl->SetComputeRoot32BitConstants(1, 4, pconst, 0);
    h->cl->Dispatch((UINT)M, 1, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = h->uav.Get(); bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    h->cl->ResourceBarrier(1, &bar);
    h->cl->CopyResource(h->readback.Get(), h->uav.Get());
    h->cl->Close();
    // execute
    ID3D12CommandList* lists[] = { h->cl.Get() };
    h->queue->ExecuteCommandLists(1, lists);
    h->queue->Signal(h->fence.Get(), ++h->fenceVal);
    h->fence->SetEventOnCompletion(h->fenceVal, h->ev);
    WaitForSingleObject(h->ev, 60000);
    // readback
    float* omap = nullptr;
    h->readback->Map(0, nullptr, (void**)&omap);
    memcpy(out, omap, (size_t)M * 4);
    h->readback->Unmap(0, nullptr);
    return 0;
}

__declspec(dllexport) void igpu_destroy(void* handle) {
    if (!handle) return;
    IgpuHandle* h = (IgpuHandle*)handle;
    h->ready = false;
    // wait for pending work
    h->queue->Signal(h->fence.Get(), ++h->fenceVal);
    h->fence->SetEventOnCompletion(h->fenceVal, h->ev);
    WaitForSingleObject(h->ev, 5000);
    if (h->ev) CloseHandle(h->ev);
    delete h;
}

__declspec(dllexport) const char* igpu_errmsg(void* handle) {
    if (!handle) return "null handle";
    return ((IgpuHandle*)handle)->errmsg;
}

} // extern "C"
