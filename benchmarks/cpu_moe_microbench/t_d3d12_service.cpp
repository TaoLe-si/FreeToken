#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <stdio.h>
#include <vector>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <io.h>
using namespace Microsoft::WRL;
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

static D3D12_RESOURCE_DESC make_buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1; d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = flags;
    return d;
}

static bool rd_exact(void* buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread((char*)buf + got, 1, n - got, stdin);
        if (r == 0) return false;
        got += r;
    }
    return true;
}

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    fprintf(stderr, "[igpu] init\n");
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    int use_idx = -1;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.VendorId == 0x1002) { use_idx = (int)i; break; }
    }
    if (use_idx < 0) { factory->EnumAdapters1(0, &adapter); } else { factory->EnumAdapters1((UINT)use_idx, &adapter); }
    DXGI_ADAPTER_DESC1 ad; adapter->GetDesc1(&ad);
    fprintf(stderr, "[igpu] adapter: %ls\n", ad.Description);

    ComPtr<ID3D12Device> dev;
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { fprintf(stderr, "[igpu] dev fail %08X\n", hr); return 1; }
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
    ComPtr<ID3D12CommandQueue> queue;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    D3D12_HEAP_PROPERTIES hup = { D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hrb = { D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hdef = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };

    ComPtr<ID3D12Resource> src, outbuf, uav;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> rs;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12Fence> fence;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    float g_const = 0.25f;
    UINT inc = 0;
    UINT64 cur_total = 0;
    unsigned char* pmap = nullptr;
    float* omap = nullptr;
    uint32_t cur_M = 0, cur_K = 0;

    // 读 dxil (从 exe 同一目录)
    wchar_t exe_path[MAX_PATH];
    GetModuleFileNameW(nullptr, exe_path, MAX_PATH);
    wchar_t* p = wcsrchr(exe_path, L'\\');
    if (p) { p[1] = 0; }
    wcscat_s(exe_path, L"d3d12_gemv_sk.dxil");
    FILE* f = _wfopen(exe_path, L"rb");
    if (!f) { fprintf(stderr, "[igpu] dxil open fail: %ls\n", exe_path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> dxil(sz); fread(dxil.data(), 1, sz, f); fclose(f);

    // 根签名（6 描述符 + 4 root constants）
    D3D12_DESCRIPTOR_RANGE ranges[6] = {};
    for (int i = 0; i < 6; ++i) {
        ranges[i].RangeType = (i < 5) ? D3D12_DESCRIPTOR_RANGE_TYPE_SRV : D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[i].NumDescriptors = 1; ranges[i].BaseShaderRegister = (i < 5) ? (UINT)i : 0; ranges[i].RegisterSpace = 0;
        ranges[i].OffsetInDescriptorsFromTableStart = i;
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
    ComPtr<ID3DBlob> rsblob, errblob;
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsblob, &errblob);
    if (FAILED(hr)) { fprintf(stderr, "[igpu] rs serialize fail\n"); return 1; }
    hr = dev->CreateRootSignature(0, rsblob->GetBufferPointer(), rsblob->GetBufferSize(), IID_PPV_ARGS(&rs));
    if (FAILED(hr)) { fprintf(stderr, "[igpu] rs fail\n"); return 1; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get(); psd.CS.pShaderBytecode = dxil.data(); psd.CS.BytecodeLength = dxil.size();
    hr = dev->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) { fprintf(stderr, "[igpu] pso fail %08X\n", hr); return 1; }
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    fprintf(stderr, "[igpu] ready\n");
    fflush(stderr);

    while (true) {
        uint32_t hdr[2];
        if (!rd_exact(hdr, 8)) break;
        uint32_t M = hdr[0], K = hdr[1];
        uint32_t NB = K / 16;
        if (M == 0 || K == 0 || (K & 15) != 0) { fprintf(stderr, "[igpu] bad hdr M=%u K=%u\n", M, K); return 2; }
        UINT64 off_pk = 0, off_scl = (UINT64)M * NB * 8, off_act = off_scl + (UINT64)M * NB * 4,
                 off_asb = off_act + (UINT64)NB * 16 * 4, off_gbl = off_asb + NB * 4,
                 total = off_gbl + (UINT64)M * 4;
        if (total > cur_total || M != cur_M || K != cur_K) {
            // 重建资源
            src.Reset(); outbuf.Reset(); uav.Reset(); heap.Reset(); cl.Reset();
            HRESULT hr1 = dev->CreateCommittedResource(&hup, D3D12_HEAP_FLAG_NONE, &make_buffer_desc(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&src)); if (FAILED(hr1)) { fprintf(stderr, "[igpu] src fail %08X total=%llu\n", hr1, total); return 3; }
            HRESULT hr2 = dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outbuf)); if (FAILED(hr2)) { fprintf(stderr, "[igpu] outbuf fail %08X\n", hr2); return 3; }
            HRESULT hr3 = dev->CreateCommittedResource(&hdef, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav)); if (FAILED(hr3)) { fprintf(stderr, "[igpu] uav fail %08X\n", hr3); return 3; }
            D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 6, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
            dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
            D3D12_CPU_DESCRIPTOR_HANDLE ch = heap->GetCPUDescriptorHandleForHeapStart();
            auto mk_srv = [&](UINT64 off, UINT num, UINT stride) {
                D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
                s.Format = DXGI_FORMAT_UNKNOWN; s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                s.Buffer.FirstElement = off / stride; s.Buffer.NumElements = num; s.Buffer.StructureByteStride = stride;
                dev->CreateShaderResourceView(src.Get(), &s, ch); ch.ptr += inc;
            };
            mk_srv(off_pk, M * NB, 8);
            mk_srv(off_scl, M * NB, 4);
            mk_srv(off_act, NB * 16, 4);
            mk_srv(off_asb, NB, 4);
            mk_srv(off_gbl, M, 4);
            D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
            u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; u.Buffer.NumElements = M; u.Buffer.StructureByteStride = 4;
            dev->CreateUnorderedAccessView(uav.Get(), nullptr, &u, ch);
            dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
            cl->SetPipelineState(pso.Get());
            cl->SetComputeRootSignature(rs.Get());
            ID3D12DescriptorHeap* heaps[] = { heap.Get() };
            cl->SetDescriptorHeaps(1, heaps);
            cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
            uint32_t pconst[4] = { K, NB, 0, 0 };
            memcpy(&pconst[2], &g_const, 4);
            cl->SetComputeRoot32BitConstants(1, 4, pconst, 0);
            cur_total = total; cur_M = M; cur_K = K;
        }
        // 读数据
        std::vector<unsigned char> buf(total);
        if (!rd_exact(buf.data(), total)) { fprintf(stderr, "[igpu] short payload\n"); return 4; }
        src->Map(0, nullptr, (void**)&pmap);
        memcpy(pmap, buf.data(), total);
        src->Unmap(0, nullptr);
        // 重录命令列表（M/K 依赖 dispatch 数）
        alloc->Reset();
        cl->Reset(alloc.Get(), pso.Get());
        cl->SetComputeRootSignature(rs.Get());
        ID3D12DescriptorHeap* heaps[] = { heap.Get() };
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        uint32_t pconst[4] = { K, NB, 0, 0 };
        memcpy(&pconst[2], &g_const, 4);
        cl->SetComputeRoot32BitConstants(1, 4, pconst, 0);
        cl->Dispatch(M, 1, 1);
        D3D12_RESOURCE_BARRIER bar = {};
        bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition.pResource = uav.Get(); bar.Transition.Subresource = 0;
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        cl->ResourceBarrier(1, &bar);
        cl->CopyResource(outbuf.Get(), uav.Get());
        cl->Close();
        ID3D12CommandList* lists[] = { cl.Get() };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
        outbuf->Map(0, nullptr, (void**)&omap);
        fwrite(omap, 4, M, stdout);
        fflush(stdout);
        outbuf->Unmap(0, nullptr);
    }
    fprintf(stderr, "[igpu] bye\n");
    return 0;
}
