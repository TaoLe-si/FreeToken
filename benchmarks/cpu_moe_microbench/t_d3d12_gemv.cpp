#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
using namespace Microsoft::WRL;
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

static const int kE2M1x2[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

static D3D12_RESOURCE_DESC make_buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1; d.SampleDesc.Quality = 0;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = flags;
    return d;
}

#define TRACE(...) do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)
int main(int argc, char** argv) {
    const UINT M = argc > 1 ? (UINT)atoi(argv[1]) : 4096;
    const UINT K = argc > 2 ? (UINT)atoi(argv[2]) : 4096;
    const UINT NB = K / 16;
    const int iters = argc > 3 ? atoi(argv[3]) : 20;
    printf("GEMV M=%u K=%u NB=%u\n", M, K, NB);

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
    printf("using: %ls\n", ad.Description);

    ComPtr<ID3D12Debug> dbg;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); printf("debug layer on\n"); }
    ComPtr<ID3D12Device> dev;
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { printf("dev fail %08X\n", hr); return 1; }
    TRACE("device ok\n");
    ComPtr<ID3D12InfoQueue> iq;
    dev.As(&iq);
    auto dump_msgs = [&]() {
        if (!iq) return;
        UINT64 n = iq->GetNumStoredMessages();
        for (UINT64 i = 0; i < n && i < 25; ++i) {
            SIZE_T len = 0;
            iq->GetMessage(i, nullptr, &len);
            std::vector<BYTE> buf(len + 16);
            D3D12_MESSAGE* m = (D3D12_MESSAGE*)buf.data();
            if (SUCCEEDED(iq->GetMessage(i, m, &len))) printf("  [DBG] %s\n", m->pDescription);
        }
        iq->ClearStoredMessages();
    };
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
    ComPtr<ID3D12CommandQueue> queue;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    // ---- 数据（真实 W4A8 布局）----
    std::vector<unsigned char> packed((size_t)M * NB * 8);
    std::vector<uint32_t> scl((size_t)M * NB);
    std::vector<int32_t> act((size_t)NB * 16);
    std::vector<float> asb(NB);
    srand(42);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = (unsigned char)(rand() & 0xFF);
    for (size_t i = 0; i < scl.size(); ++i) scl[i] = (uint32_t)(rand() & 0x7F);
    for (size_t i = 0; i < act.size(); ++i) act[i] = rand() % 255 - 127;  // -127..127
    for (size_t i = 0; i < NB; ++i) asb[i] = 0.01f + 0.05f * (float)(rand() % 100) / 100.0f;
    const float g = 0.25f;
    std::vector<float> gbl(M);
    for (size_t i = 0; i < M; ++i) gbl[i] = 0.5f + 0.5f * (float)(rand() % 100) / 100.0f;  // 0.5..1.0 每行

    // CPU 参考（同 HLSL 逻辑：wsum*0.01*scale + asb）
    std::vector<float> ref(M);
    for (UINT r = 0; r < M; ++r) {
        float acc = 0.0f;
        for (UINT b = 0; b < NB; ++b) {
            const unsigned char* pk = packed.data() + (size_t)r * NB * 8 + (size_t)b * 8;
            int wsum = 0;
            for (int j = 0; j < 8; ++j) {
                unsigned char byte = pk[j];
                wsum += kE2M1x2[byte & 0xF] * act[(size_t)b * 16 + j] + kE2M1x2[byte >> 4] * act[(size_t)b * 16 + 8 + j];
            }
            acc += (float)wsum * 0.01f * (float)(scl[(size_t)r * NB + b] & 0xFF) + asb[b];
        }
        ref[r] = acc * g * gbl[r];
    }

    // ---- 资源（upload 直读 = B 组共享内存场景）----
    const UINT64 off_pk = 0, off_scl = (UINT64)M * NB * 8, off_act = off_scl + (UINT64)M * NB * 4,
                 off_asb = off_act + (UINT64)NB * 16 * 4, off_gbl = off_asb + NB * 4,
                 total = off_gbl + (UINT64)M * 4;
    D3D12_HEAP_PROPERTIES hup = { D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hrb = { D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    ComPtr<ID3D12Resource> src, outbuf;
    dev->CreateCommittedResource(&hup, D3D12_HEAP_FLAG_NONE, &make_buffer_desc(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&src));
    dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outbuf));
    unsigned char* pm = nullptr;
    src->Map(0, nullptr, (void**)&pm);
    memcpy(pm + off_pk, packed.data(), packed.size());
    memcpy(pm + off_scl, scl.data(), scl.size() * 4);
    memcpy(pm + off_act, act.data(), act.size() * 4);
    memcpy(pm + off_asb, asb.data(), asb.size() * 4);
    memcpy(pm + off_gbl, gbl.data(), gbl.size() * 4);
    src->Unmap(0, nullptr);
    // 输出 UAV buffer（默认堆）
    D3D12_HEAP_PROPERTIES hdef = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    ComPtr<ID3D12Resource> uav;
    dev->CreateCommittedResource(&hdef, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav));

    // ---- 描述符堆（4 SRV + 1 UAV）----
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 6, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    ComPtr<ID3D12DescriptorHeap> heap;
    dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE ch = heap->GetCPUDescriptorHandleForHeapStart();
    auto mk_srv = [&](UINT64 off, UINT num, UINT stride) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Format = DXGI_FORMAT_UNKNOWN; s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Buffer.FirstElement = off / stride; s.Buffer.NumElements = num; s.Buffer.StructureByteStride = stride;
        dev->CreateShaderResourceView(src.Get(), &s, ch); ch.ptr += inc;
    };
    mk_srv(off_pk, M * NB, 8);              // uint2（8B/元素）
    mk_srv(off_scl, M * NB, 4);              // uint
    mk_srv(off_act, NB * 16, 4);             // int
    mk_srv(off_asb, NB, 4);                  // float
    mk_srv(off_gbl, M, 4);                   // float（每行 global scale）
    TRACE("SRVs created\n");
    D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
    u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; u.Buffer.NumElements = M; u.Buffer.StructureByteStride = 4;
    dev->CreateUnorderedAccessView(uav.Get(), nullptr, &u, ch);
    TRACE("UAV created\n");

    // ---- 根签名（表 + 3 root constants）----
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
    rp[1].Constants.Num32BitValues = 3; rp[1].Constants.ShaderRegister = 0; rp[1].Constants.RegisterSpace = 0;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2; rsd.pParameters = rp; rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> rsblob, errblob;
    TRACE("serializing RS...\n");
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsblob, &errblob);
    TRACE("serialized: %08X blob=%p\n", hr & 0xFFFFFFFF, rsblob.Get());
    if (FAILED(hr)) { printf("rs serialize fail %08X\n", hr); return 1; }
    ComPtr<ID3D12RootSignature> rs;
    TRACE("creating RS...\n");
    hr = dev->CreateRootSignature(0, rsblob->GetBufferPointer(), rsblob->GetBufferSize(), IID_PPV_ARGS(&rs));
    TRACE("RS created: %08X\n", hr & 0xFFFFFFFF);
    if (FAILED(hr)) { printf("rs create fail %08X\n", hr); dump_msgs(); return 1; }

    // ---- PSO ----
    const char* dxilname = argc > 4 ? argv[4] : "d3d12_gemv.dxil";
    TRACE("loading dxil %s...\n", dxilname);
    FILE* f = fopen(dxilname, "rb");
    if (!f) { printf("dxil open fail\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> dxil(sz); fread(dxil.data(), 1, sz, f); fclose(f);
    TRACE("dxil size=%ld\n", sz);
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get(); psd.CS.pShaderBytecode = dxil.data(); psd.CS.BytecodeLength = dxil.size();
    ComPtr<ID3D12PipelineState> pso;
    TRACE("creating PSO...\n");
    hr = dev->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    TRACE("PSO: %08X\n", hr & 0xFFFFFFFF);
    if (FAILED(hr)) { printf("pso fail %08X\n", hr); dump_msgs(); return 1; }

    // ---- timestamp query ----
    D3D12_QUERY_HEAP_DESC qhd = { D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0 };
    ComPtr<ID3D12QueryHeap> thq;
    hr = dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&thq));
    if (FAILED(hr)) { printf("query heap fail %08X\n", hr); return 1; }
    ComPtr<ID3D12Resource> tsbuf;
    hr = dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &make_buffer_desc(16, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tsbuf));
    if (FAILED(hr)) { printf("tsbuf fail %08X\n", hr); return 1; }
    UINT64 ts_freq = 0;
    queue->GetTimestampFrequency(&ts_freq);
    TRACE("ts freq=%llu\n", ts_freq);

    // ---- 命令列表 ----
    ComPtr<ID3D12CommandAllocator> alloc;
    TRACE("creating allocator...\n");
    hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    TRACE("allocator: %08X\n", hr & 0xFFFFFFFF);
    ComPtr<ID3D12GraphicsCommandList> cl;
    TRACE("creating cmdlist...\n");
    hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
    TRACE("cmdlist: %08X\n", hr & 0xFFFFFFFF);
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(rs.Get());
    ID3D12DescriptorHeap* heaps[] = { heap.Get() };
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    uint32_t pconst[4] = { K, NB, 0, 0 };
    memcpy(&pconst[2], &g, 4);
    cl->SetComputeRoot32BitConstants(1, 4, pconst, 0);
    cl->EndQuery(thq.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    {
        const char* mode = argc > 5 ? argv[5] : "gemv";
        UINT64 nelem = (mode[0] == 'c') ? (UINT64)M * NB : (mode[0] == 's' ? (UINT64)M : (UINT64)M);
        UINT64 ngroups = (mode[0] == 's') ? M : (nelem + 255) / 256;   // split-K: 每组一行
        cl->Dispatch((UINT)ngroups, 1, 1);
    }
    cl->EndQuery(thq.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = uav.Get(); bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cl->ResourceBarrier(1, &bar);
    cl->CopyResource(outbuf.Get(), uav.Get());
    TRACE("closing...\n");
    hr = cl->Close();
    TRACE("close: %08X\n", hr & 0xFFFFFFFF);
    if (FAILED(hr)) dump_msgs();

    ComPtr<ID3D12Fence> fence;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto run_once = [&]() {
        ID3D12CommandList* lists[] = { cl.Get() };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
    };
    TRACE("executing...\n");
    run_once();
    TRACE("executed\n");
    float* op = nullptr;
    TRACE("mapping readback...\n");
    outbuf->Map(0, nullptr, (void**)&op);
    TRACE("mapped\n");
    long bad = 0; double maxerr = 0;
    for (UINT r = 0; r < M; ++r) {
        double e = fabs((double)op[r] - ref[r]);
        double rel = 1e-3 * (fabs(ref[r]) + 1.0);
        if (e > rel) { bad++; if (e > maxerr) maxerr = e; }
    }
    TRACE("GEMV out[0..3] = %.4f %.4f %.4f %.4f (ref %.4f %.4f %.4f %.4f)\n",
           op[0], op[1], op[2], op[3], ref[0], ref[1], ref[2], ref[3]);
    TRACE("correct: bad=%ld/%u maxerr=%.3f %s\n", bad, M, maxerr, bad == 0 ? "OK" : "FAIL");
    uint64_t* ts = nullptr;
    tsbuf->Map(0, nullptr, (void**)&ts);
    fprintf(stderr, "TSRAW: %llu %llu freq=%llu\n", ts[0], ts[1], ts_freq); fflush(stderr);
    double best = 1e18;
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        run_once();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt > 0 && dt < best) best = dt;
    }
    fprintf(stderr, "TSRAW2: %llu %llu\n", ts[0], ts[1]); fflush(stderr);
    outbuf->Unmap(0, nullptr);
    tsbuf->Unmap(0, nullptr);
    double wbytes = (double)M * NB * 8 + (double)M * NB * 4 + (double)NB * 16 * 4 + (double)NB * 4;
    TRACE("GEMV D3D12 GPU: %.3f ms -> %.1f GB/s (weights+act read) -> %.1f G MAC/s\n",
           best * 1000, wbytes / best / 1e9, (double)M * K / best / 1e9);
    TRACE("DONE\n");
    return 0;
}
