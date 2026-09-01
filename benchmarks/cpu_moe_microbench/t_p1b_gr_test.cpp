#define NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
using Microsoft::WRL::ComPtr;

static D3D12_RESOURCE_DESC make_buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

int main() {
    uint32_t M = 128;
    std::cout << "P1B test 2 starting" << std::endl;

    ComPtr<IDXGIFactory1> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    factory->EnumAdapters1(0, &adapter);

    ComPtr<ID3D12Device> device;
    D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));

    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    std::ifstream f("t_p1b_gr_test_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::cout << "DXIL: " << dxil.size() << std::endl;

    D3D12_ROOT_PARAMETER rp[1] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[0].Descriptor.RegisterSpace = 0;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 1; rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    std::cout << "PSO OK" << std::endl;

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    ComPtr<ID3D12Resource> uav;
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
        &make_buffer_desc(M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav));
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };
    ComPtr<ID3D12Resource> rb;
    device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE,
        &make_buffer_desc(M * 4, D3D12_RESOURCE_FLAG_NONE),
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb));

    list->Close();
    alloc->Reset();
    list->Reset(alloc.Get(), pso.Get());
    list->SetComputeRootSignature(rs.Get());
    list->SetComputeRootUnorderedAccessView(0, uav->GetGPUVirtualAddress());
    list->Dispatch(M, 1, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = uav.Get();
    bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &bar);
    list->CopyResource(rb.Get(), uav.Get());
    list->Close();
    {
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> fence;
        device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        queue->Signal(fence.Get(), 1);
        fence->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, 30000);
    }
    void* om = nullptr;
    rb->Map(0, nullptr, &om);
    std::ofstream fo("t_p1b_gr_test_output.bin", std::ios::binary);
    fo.write((const char*)om, M * 4);
    fo.close();
    rb->Unmap(0, nullptr);
    std::cout << "Wrote t_p1b_gr_test_output.bin" << std::endl;
    return 0;
}
