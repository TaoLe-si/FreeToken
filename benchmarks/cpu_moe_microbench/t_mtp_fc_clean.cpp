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

int main(int argc, char** argv) {
    uint32_t M = 2048, K = 4096;
    uint32_t nbPerRow = K / 8u, nsPerRow = K / 32u;
    int N = 100;
    if (argc >= 2) M = (uint32_t)atoi(argv[1]);
    if (argc >= 3) K = (uint32_t)atoi(argv[2]);
    if (argc >= 4) N = atoi(argv[3]);
    std::cout << "MTP FC test M=" << M << " K=" << K << " N=" << N << std::endl;

    // Load inputs
    std::ifstream fi("t_mtp_fc_with_act.bin", std::ios::binary);
    if (!fi.good()) { std::cerr << "missing t_mtp_fc_with_act.bin" << std::endl; return 1; }
    uint32_t fm, fk, fnb, fns;
    fi.read((char*)&fm, 4); fi.read((char*)&fk, 4);
    fi.read((char*)&fnb, 4); fi.read((char*)&fns, 4);
    if (fm == M && fk == K) {
        nbPerRow = fnb; nsPerRow = fns;
        std::cout << "Loaded from file: nbPerRow=" << nbPerRow << " nsPerRow=" << nsPerRow << std::endl;
    }
    fi.close();
    
    // For now, generate random act on host (will load from file later)
    std::mt19937 rng(42);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> act(K);
    for (auto& v : act) v = n01(rng);

    // D3D12 setup
    ComPtr<IDXGIFactory1> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    factory->EnumAdapters1(0, &adapter);

    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::cerr << "D3D12CreateDevice failed" << std::endl; return 1;
    }

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

    std::ifstream fi2("t_nvfp4_gemv_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi2)), std::istreambuf_iterator<char>());
    std::cout << "DXIL " << dxil.size() << std::endl;

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
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) {
        std::cerr << "RootSig failed" << std::endl; return 1;
    }
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));

    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso)))) {
        std::cerr << "PSO failed" << std::endl; return 1;
    }
    std::cout << "PSO OK" << std::endl;

    // Resources (sizes)
    UINT64 szW = (UINT64)M * nbPerRow * 4;
    UINT64 szS = (UINT64)M * nsPerRow * 4;
    UINT64 szAct = (UINT64)K * 4;
    UINT64 szRow = (UINT64)M * 4;
    UINT64 szOut = (UINT64)M * 4;

    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rGbl, rRowB, rOut, rRb, rCb;
    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

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

    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szW, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rW));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szS, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rS));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szS, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rB));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szAct, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rAct));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szRow, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rGbl));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szRow, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRowB));
    device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(szOut, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&rOut));
    device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(szOut, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rRb));
    mkUp(256, rCb);

    // Load REAL weights + biases + scales + act from t_mtp_fc_with_act.bin
    std::vector<uint32_t> fcW(M * nbPerRow);
    std::vector<float> fcB(M * nsPerRow), fcS(M * nsPerRow);
    std::vector<float> fcGbl(M, 1.0f), fcRowB(M, 0.0f), zerosF2(M, 0.0f);
    {
        std::ifstream fi2("t_mtp_fc_with_act.bin", std::ios::binary);
        std::cout << "fi2.good=" << fi2.good() << std::endl;
        if (fi2.good()) {
            uint32_t fm, fk, fnb, fns;
            fi2.read((char*)&fm, 4); fi2.read((char*)&fk, 4);
            fi2.read((char*)&fnb, 4); fi2.read((char*)&fns, 4);
            std::cout << "file hdr: M=" << fm << " K=" << fk << " nbPerRow=" << fnb << " nsPerRow=" << fns << std::endl;
            std::cout << "cpp hdr: M=" << M << " K=" << K << " nbPerRow=" << nbPerRow << " nsPerRow=" << nsPerRow << std::endl;
            if (fm == M && fk == K && fnb == nbPerRow && fns == nsPerRow) {
                fi2.read((char*)fcW.data(), M * nbPerRow * 4);
                fi2.read((char*)fcB.data(), M * nsPerRow * 4);
                fi2.read((char*)fcS.data(), M * nsPerRow * 4);
                fi2.read((char*)act.data(), K * 4);
                std::cout << "Loaded real weights + biases + scales + act from file" << std::endl;
                std::cout << "fcW[:5]=" << fcW[0] << " " << fcW[1] << " fcB[0]=" << fcB[0] << " fcS[0]=" << fcS[0] << " act[0]=" << act[0] << std::endl;
            } else {
                std::cerr << "File dimensions mismatch" << std::endl; return 1;
            }
        } else {
            std::cerr << "Missing t_mtp_fc_with_act.bin" << std::endl; return 1;
        }
    }
    upload(rW, fcW.data(), szW);
    upload(rS, fcS.data(), szS);
    upload(rB, fcB.data(), szS);
    upload(rAct, act.data(), szAct);
    upload(rGbl, fcGbl.data(), szRow);
    upload(rRowB, zerosF2.data(), szRow);

    void* cm = nullptr; rCb->Map(0, nullptr, &cm);
    struct { uint32_t K, nbPerRow, nsPerRow, pad; } cb = { K, K / 32u, nsPerRow, 0 };  // nbPerRow=块数(K/32), 非uint数(K/8)!
    std::memcpy(cm, &cb, sizeof(cb));
    rCb->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    for (int warm = 0; warm < 10; ++warm) {
        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(rs.Get());
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
        queue->ExecuteCommandLists(1, ls);
        UINT64 fv = warm + 2;
        queue->Signal(fence.Get(), fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
    }

    std::vector<double> times;
    times.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(rs.Get());
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
        queue->ExecuteCommandLists(1, ls);
        UINT64 fv = 12 + i;
        queue->Signal(fence.Get(), fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    double sum = 0; for (double t : times) sum += t;
    double avg = sum / times.size();
    std::sort(times.begin(), times.end());
    double p50 = times[times.size() / 2];
    std::cout << "RESULT M=" << M << " K=" << K << " avg_ms=" << avg << " p50_ms=" << p50 << std::endl;

    // Copy to readback
    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
    list->CopyResource(rRb.Get(), rOut.Get());
    list->Close();
    ID3D12CommandList* ls[] = { list.Get() };
    queue->ExecuteCommandLists(1, ls);
    UINT64 fv = 200;
    queue->Signal(fence.Get(), fv);
    fence->SetEventOnCompletion(fv, ev);
    WaitForSingleObject(ev, 30000);

    void* om = nullptr; rRb->Map(0, nullptr, &om);
    std::ofstream fo("t_mtp_fc_clean_output.bin", std::ios::binary);
    fo.write((const char*)om, (std::streamsize)szOut);
    fo.close();
    rRb->Unmap(0, nullptr);
    std::cout << "Wrote t_mtp_fc_clean_output.bin" << std::endl;
    return 0;
}
