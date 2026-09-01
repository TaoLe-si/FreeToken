#define NOMINMAX
#include <random>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

using Microsoft::WRL::ComPtr;

static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}
static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t)(p * (v.size() - 1))];
}

int main(int argc, char** argv) {
    int N = 100;
    if (argc >= 2) N = atoi(argv[1]);

    std::cout << "NVFP4 GEMV test" << std::endl;

    uint32_t M = 2048, K = 4096;
    uint32_t nbPerRow = K / 8u, nsPerRow = K / 32u;

    std::vector<uint32_t> fcW;
    std::vector<float> fcB, fcS, act;

    bool use_external = false;
    {
        std::ifstream fs2("t_p1b_m128.bin", std::ios::binary);
        if (!fs2.good()) fs2.open("t_mtp_fc_1row.bin", std::ios::binary);
        if (fs2.good()) {
            uint32_t m2, k2, nbp2, nsp2;
            fs2.read((char*)&m2, 4); fs2.read((char*)&k2, 4);
            fs2.read((char*)&nbp2, 4); fs2.read((char*)&nsp2, 4);
            M = m2; K = k2; nbPerRow = nbp2; nsPerRow = nsp2;
            fcW.resize(M * nbPerRow); fs2.read((char*)fcW.data(), M * nbPerRow * 4);
            fcB.resize(M * nsPerRow); fs2.read((char*)fcB.data(), M * nsPerRow * 4);
            fcS.resize(M * nsPerRow); fs2.read((char*)fcS.data(), M * nsPerRow * 4);
            act.resize(K); fs2.read((char*)act.data(), K * 4);
            use_external = true;
            std::cout << "Loaded 1row M=" << M << " K=" << K << std::endl;
        }
    }
    if (!use_external) {
        std::ifstream fs2("t_mtp_fc_with_act.bin", std::ios::binary);
        if (fs2.good()) {
            uint32_t m2, k2, nbp2, nsp2;
            fs2.read((char*)&m2, 4); fs2.read((char*)&k2, 4);
            fs2.read((char*)&nbp2, 4); fs2.read((char*)&nsp2, 4);
            if (m2 == M && k2 == K) {
                fcW.resize(M * nbPerRow); fs2.read((char*)fcW.data(), M * nbPerRow * 4);
                fcB.resize(M * nsPerRow); fs2.read((char*)fcB.data(), M * nsPerRow * 4);
                fcS.resize(M * nsPerRow); fs2.read((char*)fcS.data(), M * nsPerRow * 4);
                act.resize(K); fs2.read((char*)act.data(), K * 4);
                use_external = true;
                std::cout << "Loaded external M=" << M << " K=" << K << std::endl;
            }
        }
    }
    if (!use_external) {
        std::mt19937 rng(42);
        std::normal_distribution<float> n01(0.0f, 1.0f);
        act.resize(K);
        for (auto& v : act) v = n01(rng);
    }
    std::cout << "M=" << M << " K=" << K << std::endl;

    // init D3D12 device
    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    ComPtr<ID3D12CommandQueue> queue2;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue2));
    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    std::ifstream fi2("t_nvfp4_gemv_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi2)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { std::cerr << "missing t_nvfp4_gemv_sk.dxil" << std::endl; return 1; }

    D3D12_ROOT_PARAMETER rp[8] = {};
    for (int i = 0; i < 6; ++i) {
        rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[i].Descriptor.ShaderRegister = (UINT)i;
        rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[6].Descriptor.ShaderRegister = 0;
    rp[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[7].Descriptor.ShaderRegister = 0;
    rp[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 8; rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, err;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    ComPtr<ID3D12RootSignature> rsig;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rsig));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rsig.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

    auto mkDef = [&](UINT64 sz, D3D12_RESOURCE_STATES init, D3D12_RESOURCE_FLAGS flags, ComPtr<ID3D12Resource>& r) {
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(sz, flags), init, nullptr, IID_PPV_ARGS(&r));
    };
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
        queue2->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f2; device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f2));
        HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        queue2->Signal(f2.Get(), 1); f2->SetEventOnCompletion(1, ev);
        WaitForSingleObject(ev, 30000);
    };

    UINT64 szW = (UINT64)M * nbPerRow * 4;
    UINT64 szBS = (UINT64)M * nsPerRow * 4;
    UINT64 szAct = (UINT64)K * 4;
    UINT64 szRow = (UINT64)M * 4;
    UINT64 szOut = (UINT64)M * 4;

    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rGbl, rRowB, rOut, rRb, rCb;
    mkDef(szW, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rW);
    mkDef(szBS, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rS);
    mkDef(szBS, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rB);
    mkDef(szAct, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rAct);
    mkDef(szRow, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rGbl);
    mkDef(szRow, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, rRowB);
    mkDef(szOut, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, rOut);
    device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(szOut, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRb));
    mkUp(256, rCb);

    if (fcW.empty()) fcW.resize(M * nbPerRow);
    if (fcB.empty()) fcB.resize(M * nsPerRow);
    if (fcS.empty()) fcS.resize(M * nsPerRow, 1.0f);
    upload(rW, fcW.data(), szW);
    upload(rS, fcS.data(), szBS);
    upload(rB, fcB.data(), szBS);
    upload(rAct, act.data(), szAct);
    std::vector<float> ones(M, 1.0f), zeros(M, 0.0f);
    upload(rGbl, ones.data(), szRow);
    upload(rRowB, zeros.data(), szRow);

    void* cm = nullptr; rCb->Map(0, nullptr, &cm);
    struct { uint32_t K, nbPerRow, nsPerRow, pad; } cb = { K, nbPerRow, nsPerRow, 0 };
    std::memcpy(cm, &cb, sizeof(cb));
    rCb->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;

    auto dispatch = [&]() {
        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(rsig.Get());
        list->SetComputeRootShaderResourceView(0, rW->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1, rS->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, rB->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(3, rAct->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(4, rGbl->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(5, rRowB->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(6, rOut->GetGPUVirtualAddress());
        list->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
        list->Dispatch(M, 1, 1);
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue2->ExecuteCommandLists(1, ls);
        queue2->Signal(fence.Get(), ++fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
    };

    // Zero-init outv via copy from a zero upload buffer
    {
        std::vector<float> zerosOut(szOut / 4, 0.0f);
        list->Reset(alloc.Get(), nullptr);
        // (no-op - skip zero init)
        list->Close();
        ID3D12CommandList* ls0[] = { list.Get() };
        queue2->ExecuteCommandLists(1, ls0);
        ComPtr<ID3D12Fence> f0; device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f0));
        HANDLE ev0 = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        queue2->Signal(f0.Get(), 999); f0->SetEventOnCompletion(999, ev0);
        WaitForSingleObject(ev0, 30000);
        // Now dispatch many times to clear UAV initial state
        for (int i = 0; i < 10; ++i) dispatch();
    }
    std::vector<double> times;
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dispatch();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double sum = 0; for (double t : times) sum += t;
    double avg = sum / times.size();
    double p50 = percentile(times, 0.50);
    std::cout << "RESULT M=" << M << " K=" << K << " avg_ms=" << avg << " p50_ms=" << p50 << std::endl;

    list->Reset(alloc.Get(), nullptr);
    list->CopyResource(rRb.Get(), rOut.Get());
    list->Close();
    ID3D12CommandList* ls[] = { list.Get() };
    queue2->ExecuteCommandLists(1, ls);
    queue2->Signal(fence.Get(), ++fv);
    fence->SetEventOnCompletion(fv, ev);
    WaitForSingleObject(ev, 30000);
    void* om = nullptr; rRb->Map(0, nullptr, &om);
    std::ofstream fo("t_mtp_fc_output.bin", std::ios::binary);
    fo.write((const char*)om, (std::streamsize)szOut);
    fo.close();
    rRb->Unmap(0, nullptr);
    std::cout << "Wrote t_mtp_fc_output.bin" << std::endl;
    return 0;
}
