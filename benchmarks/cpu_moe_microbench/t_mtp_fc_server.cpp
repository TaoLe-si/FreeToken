#define NOMINMAX
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <io.h>
#include <cstring>
#include <chrono>
#include <cstdio>

using Microsoft::WRL::ComPtr;
static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}

int main(int argc, char** argv) {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);

    if (argc < 2) { std::cerr << "usage: mtp_fc_server weights.bin [M]\n"; return 1; }
    uint32_t M = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 1;
    uint32_t K = 4096, nbPerRow = 512, nsPerRow = 128;

    std::ifstream fi(argv[1], std::ios::binary);
    if (!fi.good()) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
    uint32_t fm, fk, fnb, fns;
    fi.read((char*)&fm, 4); fi.read((char*)&fk, 4);
    fi.read((char*)&fnb, 4); fi.read((char*)&fns, 4);
    if (fm == M && fk == K && fnb == nbPerRow && fns == nsPerRow) {
        M = fm; K = fk; nbPerRow = fnb; nsPerRow = fns;
    }
    std::vector<uint32_t> fcW(M * nbPerRow);
    std::vector<float> fcB(M * nsPerRow), fcS(M * nsPerRow);
    fi.read((char*)fcW.data(), M * nbPerRow * 4);
    fi.read((char*)fcB.data(), M * nsPerRow * 4);
    fi.read((char*)fcS.data(), M * nsPerRow * 4);
    std::cerr << "Loaded weights M=" << M << " K=" << K << "\n";

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

    std::ifstream fi2("t_mtp_fc_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi2)), std::istreambuf_iterator<char>());
    D3D12_ROOT_PARAMETER rp[8] = {};
    for (int i = 0; i < 6; ++i) {
        rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[i].Descriptor.ShaderRegister = (UINT)i;
    }
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[6].Descriptor.ShaderRegister = 0;
    rp[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[7].Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 8; rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));
    std::cerr << "PSO ready\n";

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };
    UINT64 szW = (UINT64)M * nbPerRow * 4;
    UINT64 szS = (UINT64)M * nsPerRow * 4;
    UINT64 szAct = (UINT64)K * 4;
    UINT64 szOut = (UINT64)M * 4;
    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rOut, rRb, rCb;
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szW, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rW));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szS, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rS));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szS, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rB));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szAct, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rAct));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szOut, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&rOut));
    device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(szOut, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRb));
    device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&rCb));

    auto mkUp = [&](UINT64 sz, ComPtr<ID3D12Resource>& r) {
        device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(sz, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r));
    };
    auto upload = [&](ComPtr<ID3D12Resource>& dst, const void* src, UINT64 bytes) {
        ComPtr<ID3D12Resource> u; mkUp(bytes, u);
        void* m = nullptr; u->Map(0, nullptr, &m);
        std::memcpy(m, src, (size_t)bytes); u->Unmap(0, nullptr);
        list->Reset(alloc.Get(), nullptr);
        list->CopyResource(dst.Get(), u.Get());
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f2; device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        queue->Signal(f2.Get(), 1); f2->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, 30000);
    };
    upload(rW, fcW.data(), szW);
    upload(rS, fcS.data(), szS);
    upload(rB, fcB.data(), szS);
    void* cm = nullptr; rCb->Map(0, nullptr, &cm);
    struct { uint32_t K, nbPerRow, nsPerRow, pad; } cb = { K, nbPerRow, nsPerRow, 0 };
    std::memcpy(cm, &cb, sizeof(cb));
    rCb->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 1;

    std::cerr << "Ready. Reading act from stdin, writing outv to stdout.\n";
    std::cout.flush();
    std::cerr.flush();

    while (true) {
        uint32_t len = 0;
        std::cin.read((char*)&len, 4);
        if (!std::cin || std::cin.eof()) break;
        if (len != szAct) { std::cerr << "wrong act size " << len << " expected " << szAct << "\n"; break; }
        std::vector<float> act(K);
        if (!std::cin.read((char*)act.data(), szAct)) break;
        std::cerr << "act[:5]=" << act[0] << " " << act[1] << " " << act[2] << " " << act[3] << " " << act[4] << " sum=";
        double s = 0; for (auto v : act) s += v;
        std::cerr << s << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        std::cerr << "uploading act: szAct=" << szAct << " act.size()=" << act.size() << "\n";
        // For iter >= 1, rAct is in NON_PIXEL_SHADER_RESOURCE. Need barrier back to COPY_DEST.
        {
            D3D12_RESOURCE_BARRIER bpre = {};
            bpre.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            bpre.Transition.pResource = rAct.Get();
            bpre.Transition.Subresource = 0;
            bpre.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            bpre.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            list->Reset(alloc.Get(), nullptr);
            list->ResourceBarrier(1, &bpre);
            list->Close();
            ID3D12CommandList* ls_pre[] = { list.Get() };
            queue->ExecuteCommandLists(1, ls_pre);
            queue->Signal(fence.Get(), fv);
            fence->SetEventOnCompletion(fv, ev);
            WaitForSingleObject(ev, 30000);
            fv++;
        }
        upload(rAct, act.data(), szAct);
        // Barrier: rAct COPY_DEST -> NON_PIXEL_SHADER_RESOURCE
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER bb = {};
        bb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bb.Transition.pResource = rAct.Get();
        bb.Transition.Subresource = 0;
        bb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        bb.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1, &bb);
        list->Close();
        ID3D12CommandList* ls0[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls0);
        queue->Signal(fence.Get(), fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
        fv++;
        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(rs.Get());
        list->SetComputeRootShaderResourceView(0, rW->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1, rS->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, rB->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(3, rAct->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(6, rOut->GetGPUVirtualAddress());
        list->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
        list->Dispatch(M, 1, 1);
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        queue->Signal(fence.Get(), fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
        fv++;
        list->Reset(alloc.Get(), nullptr);
        list->CopyResource(rRb.Get(), rOut.Get());
        list->Close();
        queue->ExecuteCommandLists(1, ls);
        queue->Signal(fence.Get(), fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
        fv++;
        void* om = nullptr; rRb->Map(0, nullptr, &om);
        float val0 = ((float*)om)[0];
        std::cerr << "rb val0=" << val0 << "\n";
        std::cout.write((const char*)&szOut, 4);
        std::cout.write((const char*)om, szOut);
        std::cout.flush();
        rRb->Unmap(0, nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "Dispatch: " << ms << "ms\n";
    }
    return 0;
}
