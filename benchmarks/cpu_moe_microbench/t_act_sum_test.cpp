#define NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
using Microsoft::WRL::ComPtr;
static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}
int main() {
    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a; f->EnumAdapters1(0, &a);
    ComPtr<ID3D12Device> d; D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d));
    ComPtr<ID3D12CommandQueue> q;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q));
    ComPtr<ID3D12CommandAllocator> al; d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al));
    ComPtr<ID3D12GraphicsCommandList> l; d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al.Get(), nullptr, IID_PPV_ARGS(&l));

    std::ifstream fi("t_test_act_sum.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    std::cout << "dxil " << dxil.size() << std::endl;

    D3D12_ROOT_PARAMETER rp[2] = {};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[0].Descriptor.ShaderRegister = 2;
    rp[0].Descriptor.RegisterSpace = 0;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[1].Descriptor.ShaderRegister = 0;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 2; rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    ComPtr<ID3D12RootSignature> rs;
    d->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    d->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    std::cout << "pso ok" << std::endl;

    // act buffer: 32 int (=128 bytes). Each int = 1.
    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    ComPtr<ID3D12Resource> actBuf;
    d->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(128, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&actBuf));
    ComPtr<ID3D12Resource> actUp;
    d->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(128, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&actUp));
    void* m = nullptr;
    actUp->Map(0, nullptr, &m);
    int* mi = (int*)m;
    for (int i = 0; i < 32; i++) mi[i] = 1;
    actUp->Unmap(0, nullptr);

    ComPtr<ID3D12Resource> outBuf;
    d->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&outBuf));
    ComPtr<ID3D12Resource> outRb;
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };
    d->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&outRb));

    l->Close();
    al->Reset();
    l->Reset(al.Get(), pso.Get());
    l->CopyResource(actBuf.Get(), actUp.Get());
    l->Close();
    {
        ID3D12CommandList* ls[] = { l.Get() };
        q->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f2; d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        q->Signal(f2.Get(), 1);
        f2->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, 30000);
    }
    // Barrier: actBuf COPY_SOURCE -> NON_PIXEL_SHADER_RESOURCE
    al->Reset();
    l->Reset(al.Get(), pso.Get());
    D3D12_RESOURCE_BARRIER bb = {};
    bb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bb.Transition.pResource = actBuf.Get();
    bb.Transition.Subresource = 0;
    bb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    bb.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    l->ResourceBarrier(1, &bb);
    l->SetComputeRootSignature(rs.Get());
    l->SetComputeRootShaderResourceView(0, actBuf->GetGPUVirtualAddress());
    l->SetComputeRootUnorderedAccessView(1, outBuf->GetGPUVirtualAddress());
    l->Dispatch(1, 1, 1);
    D3D12_RESOURCE_BARRIER bar = {};
    bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    bar.Transition.pResource = outBuf.Get();
    bar.Transition.Subresource = 0;
    bar.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    bar.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    l->ResourceBarrier(1, &bar);
    l->CopyResource(outRb.Get(), outBuf.Get());
    l->Close();
    {
        ID3D12CommandList* ls[] = { l.Get() };
        q->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f2; d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        q->Signal(f2.Get(), 2);
        f2->SetEventOnCompletion(2, ev);
        WaitForSingleObject(ev, 30000);
    }
    // Verify actBuf by readback
    ComPtr<ID3D12Resource> actRb;
    d->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(32, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&actRb));
    al->Reset();
    l->Reset(al.Get(), nullptr);
    l->CopyResource(actRb.Get(), actBuf.Get());
    l->Close();
    {
        ID3D12CommandList* ls[] = { l.Get() };
        q->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f2; d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        q->Signal(f2.Get(), 3);
        f2->SetEventOnCompletion(3, ev);
        WaitForSingleObject(ev, 30000);
    }
    void* arm = nullptr;
    actRb->Map(0, nullptr, &arm);
    std::cout << "actBuf bytes: ";
    for (int i = 0; i < 32; i++) std::cout << (int)((unsigned char*)arm)[i] << " ";
    std::cout << std::endl;
    actRb->Unmap(0, nullptr);

    void* om = nullptr;
    outRb->Map(0, nullptr, &om);
    float v = 0;
    std::memcpy(&v, om, 4);
    outRb->Unmap(0, nullptr);
    std::cout << "Result = " << v << " (expected 32)" << std::endl;
    return 0;
}
