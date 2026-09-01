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
    std::cout << "Hello D3D12 start" << std::endl;
    ComPtr<IDXGIFactory1> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    factory->EnumAdapters1(0, &adapter);

    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { std::cerr << "D3D12CreateDevice failed" << std::endl; return 1; }
    std::cout << "Device OK" << std::endl;

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));

    ComPtr<ID3D12GraphicsCommandList> list;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    std::cout << "CreateCommandList: " << std::hex << hr << std::endl;

    // Load DXIL
    std::ifstream f("t_hello_d3d12.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::cout << "DXIL: " << dxil.size() << " bytes, magic=" << std::string((char*)dxil.data(), 4) << std::endl;

    // Root sig: 1 root UAV
    D3D12_ROOT_PARAMETER rp[1] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[0].Descriptor.RegisterSpace = 0;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 1; rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    std::cout << "SerializeRootSig: " << std::hex << hr << std::endl;

    ComPtr<ID3D12RootSignature> rootSig;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    std::cout << "CreateRootSig: " << std::hex << hr << std::endl;

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rootSig.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    hr = device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    std::cout << "CreatePSO: " << std::hex << hr << std::endl;
    if (FAILED(hr)) return 1;

    // Resources
    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    ComPtr<ID3D12Resource> uav;
    hr = device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE,
        &make_buffer_desc(4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&uav));
    std::cout << "CreateUAV: " << std::hex << hr << std::endl;

    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };
    ComPtr<ID3D12Resource> rb;
    hr = device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE,
        &make_buffer_desc(4, D3D12_RESOURCE_FLAG_NONE),
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb));
    std::cout << "CreateReadback: " << std::hex << hr << std::endl;

    // IMPORTANT: must Close the open list before Reset. The list was opened by CreateCommandList.
    hr = list->Close();
    std::cout << "Close: " << std::hex << hr << std::endl;

    // Now Reset + record
    hr = alloc->Reset();
    std::cout << "AllocReset: " << std::hex << hr << std::endl;
    hr = list->Reset(alloc.Get(), pso.Get());
    std::cout << "ListReset: " << std::hex << hr << std::endl;

    list->SetComputeRootSignature(rootSig.Get());
    list->SetComputeRootUnorderedAccessView(0, uav->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = uav.Get();
    bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &bar);
    list->CopyResource(rb.Get(), uav.Get());
    hr = list->Close();
    std::cout << "Close2: " << std::hex << hr << std::endl;

    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    {
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        queue->Signal(fence.Get(), 1);
        fence->SetEventOnCompletion(1, ev);
        DWORD w = WaitForSingleObject(ev, 30000);
        std::cout << "Wait: " << w << std::endl;
    }

    void* m = nullptr;
    rb->Map(0, nullptr, &m);
    uint32_t v = 0;
    std::memcpy(&v, m, 4);
    rb->Unmap(0, nullptr);
    std::cout << "Result = " << v << " (expected 42)" << std::endl;
    return v == 42 ? 0 : 1;
}
