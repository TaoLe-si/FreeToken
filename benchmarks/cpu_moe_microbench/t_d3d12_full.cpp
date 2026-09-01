#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <vector>
#include <chrono>
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

int main() {
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { printf("factory fail %08X\n", hr); return 1; }
    ComPtr<IDXGIAdapter1> adapter;
    int use_idx = -1;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        printf("adapter[%u]: %ls vid=%04X\n", i, d.Description, d.VendorId);
        if (d.VendorId == 0x1002) { use_idx = (int)i; break; }
    }
    if (use_idx < 0) { use_idx = 0; factory->EnumAdapters1(0, &adapter); }
    else { factory->EnumAdapters1((UINT)use_idx, &adapter); }
    DXGI_ADAPTER_DESC1 ad; adapter->GetDesc1(&ad);
    printf("using: %ls\n", ad.Description);

    ComPtr<ID3D12Device> dev;
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { printf("dev fail %08X\n", hr); return 1; }

    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_COMPUTE, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0 };
    ComPtr<ID3D12CommandQueue> queue;
    hr = dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) { printf("queue fail %08X\n", hr); return 1; }

    const UINT N = 1 << 20;
    D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    D3D12_HEAP_PROPERTIES hrb = { D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0 };
    ComPtr<ID3D12Resource> uav, rbuf;
    hr = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)N * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav));
    if (FAILED(hr)) { printf("uav fail %08X\n", hr); return 1; }
    hr = dev->CreateCommittedResource(&hrb, D3D12_HEAP_FLAG_NONE, &make_buffer_desc((UINT64)N * 4, D3D12_RESOURCE_FLAG_NONE),
                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rbuf));
    if (FAILED(hr)) { printf("rb fail %08X\n", hr); return 1; }

    // 描述符堆
    D3D12_DESCRIPTOR_HEAP_DESC hd = { D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0 };
    ComPtr<ID3D12DescriptorHeap> heap;
    dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavd = {};
    uavd.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavd.Buffer.FirstElement = 0; uavd.Buffer.NumElements = N;
    uavd.Buffer.StructureByteStride = 4; uavd.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    dev->CreateUnorderedAccessView(uav.Get(), nullptr, &uavd, heap->GetCPUDescriptorHandleForHeapStart());

    // 根签名（1 个 UAV 描述符表）
    D3D12_DESCRIPTOR_RANGE range = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0 };
    D3D12_ROOT_PARAMETER rp = {};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 1; rp.DescriptorTable.pDescriptorRanges = &range;
    rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 1; rsd.pParameters = &rp; rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> rsblob, errblob;
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &rsblob, &errblob);
    if (FAILED(hr)) { printf("rs serialize fail %08X\n", hr); return 1; }
    ComPtr<ID3D12RootSignature> rs;
    hr = dev->CreateRootSignature(0, rsblob->GetBufferPointer(), rsblob->GetBufferSize(), IID_PPV_ARGS(&rs));
    if (FAILED(hr)) { printf("rs fail %08X\n", hr); return 1; }

    // PSO（加载 DXIL）
    FILE* f = fopen("t_d3d12.dxil", "rb");
    if (!f) { printf("dxil open fail\n"); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> dxil(sz); fread(dxil.data(), 1, sz, f); fclose(f);
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS.pShaderBytecode = dxil.data(); psd.CS.BytecodeLength = dxil.size();
    ComPtr<ID3D12PipelineState> pso;
    hr = dev->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) { printf("pso fail %08X\n", hr); return 1; }

    ComPtr<ID3D12CommandAllocator> alloc;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->SetPipelineState(pso.Get());
    cl->SetComputeRootSignature(rs.Get());
    ID3D12DescriptorHeap* heaps[] = { heap.Get() };
    cl->SetDescriptorHeaps(1, heaps);
    cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    cl->Dispatch(N / 256, 1, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = uav.Get(); bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    cl->ResourceBarrier(1, &bar);
    cl->CopyResource(rbuf.Get(), uav.Get());
    cl->Close();

    ComPtr<ID3D12Fence> fence;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    auto run_once = [&]() {
        ID3D12CommandList* lists[] = { cl.Get() };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), 1);
        fence->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, 30000);
    };
    run_once();  // 正确性
    float* p = nullptr;
    rbuf->Map(0, nullptr, (void**)&p);
    printf("D3D12 outv[0..3] = %.3f %.3f %.3f %.3f\n", p[0], p[1], p[2], p[3]);
    printf("expected: %.3f %.3f %.3f %.3f\n", 0.5f, 1.5001f, 2.5002f, 3.5003f);
    bool ok = fabsf(p[0] - 0.5f) < 1e-3f && p[3] > 3.0f;
    printf("RESULT: %s\n", ok ? "WRITE SURVIVES (no DCE)" : "WRITE ELIMINATED");
    // 计时
    double best = 1e18;
    for (int i = 0; i < 20; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        run_once();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt < best) best = dt;
    }
    rbuf->Unmap(0, nullptr);
    printf("dispatch+wait: %.3f ms -> %.1f GB/s writes\n", best * 1000, (double)N * 4 / best / 1e9);
    return 0;
}
