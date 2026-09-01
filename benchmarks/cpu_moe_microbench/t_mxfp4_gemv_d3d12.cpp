#define NOMINMAX
// MXFP4 GEMV host benchmark for AMD 780M (D3D12).
// Build deps: d3d12.lib dxgi.lib
// Usage: t_mxfp4_gemv_d3d12.exe [M=2048] [K=4096] [N=1000]

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cstdint>

using Microsoft::WRL::ComPtr;

static std::vector<uint8_t> readFile(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t i = (size_t)(p * (v.size() - 1));
    return v[i];
}

int main(int argc, char** argv) {
    uint32_t M = 2048, K = 4096;
    int N = 1000;
    if (argc >= 2) M = (uint32_t)atoi(argv[1]);
    if (argc >= 3) K = (uint32_t)atoi(argv[2]);
    if (argc >= 4) N = atoi(argv[3]);
    if ((K % 32) != 0) { std::cerr << "K must be multiple of 32" << std::endl; return 1; }
    uint32_t nbPerRow = K / 8u;    // uint32 packed per row
    uint32_t nsPerRow = K / 32u;   // uint32 scl per row
    std::cout << "MXFP4 GEMV D3D12 test: M=" << M << " K=" << K << " N=" << N << std::endl;
    std::cout << "starting..." << std::endl;

    // ---- Adapter: pick AMD (0x1002) only, else default ----
    ComPtr<IDXGIFactory4> factory;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<IDXGIAdapter1> amdAdapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (d.VendorId == 0x1002) { amdAdapter = adapter; break; }
    }
    if (amdAdapter) {
        adapter = amdAdapter;
        std::cout << "Using AMD adapter (vendor 0x1002)" << std::endl;
    } else {
        factory->EnumAdapters1(0, &adapter);
        std::cout << "Using default adapter" << std::endl;
    }

    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        std::cerr << "D3D12CreateDevice failed" << std::endl; return 1;
    }

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));

    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    list->Close();

    // ---- Root sig: 5 SRV root + 1 UAV root + 1 CBV root ----
    // Use ROOT_DESCRIPTOR (not table) for direct SetComputeRootShaderResourceView / UAV / CBV.
    D3D12_ROOT_PARAMETER rp[7] = {};
    for (int i = 0; i < 5; ++i) {
        rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[i].Descriptor.ShaderRegister = (UINT)i;
        rp[i].Descriptor.RegisterSpace = 0;
        rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[5].Descriptor.ShaderRegister = 0;
    rp[5].Descriptor.RegisterSpace = 0;
    rp[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[6].Descriptor.ShaderRegister = 0;
    rp[6].Descriptor.RegisterSpace = 0;
    rp[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd = {};
    rsd.NumParameters = 7;
    rsd.pParameters = rp;
    rsd.NumStaticSamplers = 0;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sigBlob, errBlob;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob))) {
        if (errBlob) std::cerr << "RootSig err: " << (const char*)errBlob->GetBufferPointer() << std::endl;
        std::cerr << "RootSig serialize failed" << std::endl; return 1;
    }
    ComPtr<ID3D12RootSignature> rootSig;
    if (FAILED(device->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)))) {
        std::cerr << "CreateRootSignature failed" << std::endl; return 1;
    }
    std::cout << "RootSig OK" << std::endl;

    // ---- Pipeline state from DXIL ----
    auto dxil = readFile("t_mxfp4_gemv_sk.dxil");
    if (dxil.empty()) { std::cerr << "DXIL missing" << std::endl; return 1; }
    ComPtr<ID3D12PipelineState> pso;
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rootSig.Get();
    psd.CS = { dxil.data(), dxil.size() };
    if (FAILED(device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso)))) {
        std::cerr << "PSO create failed" << std::endl; return 1;
    }
    std::cout << "PSO OK" << std::endl;

    // ---- Resources ----
    auto mkBuf = [&](UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES init, ComPtr<ID3D12Resource>& r, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES hp = { heap };
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1; rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = flags;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, init, nullptr, IID_PPV_ARGS(&r));
    };

    auto uploadTo = [&](ComPtr<ID3D12Resource>& dst, const void* src, UINT64 bytes) {
        D3D12_HEAP_PROPERTIES hp = { D3D12_HEAP_TYPE_UPLOAD };
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
        rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> u;
        device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&u));
        void* m = nullptr;
        u->Map(0, nullptr, &m);
        std::memcpy(m, src, (size_t)bytes);
        u->Unmap(0, nullptr);
        list->Reset(alloc.Get(), pso.Get());
        list->CopyResource(dst.Get(), u.Get());
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        ComPtr<ID3D12Fence> f; device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
        queue->Signal(f.Get(), 1);
        while (f->GetCompletedValue() < 1) Sleep(1);
    };

    // Generate random inputs (mt19937 seed=42, matching Python reference)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> u8(0, 255);
    std::uniform_int_distribution<int> s8(-128, 127);
    std::uniform_real_distribution<float> fb(-0.5f, 0.5f);
    std::uniform_real_distribution<float> fg(0.5f, 2.0f);
    std::normal_distribution<float> nscl(127.0f, 5.0f);

    std::vector<uint32_t> packedData((size_t)M * nbPerRow);
    for (auto& v : packedData) v = (uint32_t)(u8(rng) | (u8(rng) << 8) | (u8(rng) << 16) | (u8(rng) << 24));
    std::vector<uint32_t> sclData((size_t)M * nsPerRow);
    for (auto& v : sclData) {
        int b0 = std::max(100, std::min(154, (int)nscl(rng)));
        int b1 = std::max(100, std::min(154, (int)nscl(rng)));
        int b2 = std::max(100, std::min(154, (int)nscl(rng)));
        int b3 = std::max(100, std::min(154, (int)nscl(rng)));
        v = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
    }
    std::vector<int8_t> actData(K);
    for (auto& v : actData) v = (int8_t)s8(rng);
    std::vector<float> biasData(M), gblData(M);
    for (auto& v : biasData) v = fb(rng);
    for (auto& v : gblData) v = fg(rng);

    UINT64 packedBytes = (UINT64)M * nbPerRow * 4;
    UINT64 sclBytes    = (UINT64)M * nsPerRow * 4;
    UINT64 actBytes    = (UINT64)K;
    UINT64 biasBytes   = (UINT64)M * 4;
    UINT64 gblBytes    = (UINT64)M * 4;
    UINT64 outBytes    = (UINT64)M * 4;

    ComPtr<ID3D12Resource> resPacked, resScl, resAct, resBias, resGbl, resOut, resReadback, resCB;
    mkBuf(packedBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, resPacked);
    mkBuf(sclBytes,    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, resScl);
    mkBuf(actBytes,    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, resAct);
    mkBuf(biasBytes,   D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, resBias);
    mkBuf(gblBytes,    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, resGbl);
    mkBuf(outBytes,    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, resOut, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    mkBuf(outBytes,    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, resReadback);
    mkBuf(256,         D3D12_HEAP_TYPE_UPLOAD,   D3D12_RESOURCE_STATE_GENERIC_READ, resCB);

    // Optional: load single-row input for debug
    bool use_single = false;
    {
        std::ifstream fi("t_mxfp4_single_inputs.bin", std::ios::binary);
        if (fi.good()) {
            char magic[4]; fi.read(magic, 4);
            if (std::string(magic, 4) == "SGL1") {
                use_single = true;
                uint32_t dims[2]; fi.read((char*)dims, 8);
                M = dims[0]; K = dims[1];
                nbPerRow = K / 8u; nsPerRow = K / 32u;
                packedData.resize(M * nbPerRow); fi.read((char*)packedData.data(), M * nbPerRow * 4);
                sclData.resize(M * nsPerRow); fi.read((char*)sclData.data(), M * nsPerRow * 4);
                actData.resize(K); fi.read((char*)actData.data(), K);
                biasData.resize(M); fi.read((char*)biasData.data(), M * 4);
                gblData.resize(M); fi.read((char*)gblData.data(), M * 4);
                std::cout << "Loaded single-row inputs M=" << M << " K=" << K << std::endl;
            }
        }
    }
    // Dump inputs for Python reference comparison (only for the random case)
    if (!use_single) {
        std::ofstream fi("t_mxfp4_inputs.bin", std::ios::binary);
        fi.write("PK10", 4);
        uint32_t dims[2] = { M, K };
        fi.write((const char*)dims, 8);
        fi.write((const char*)packedData.data(), (std::streamsize)packedBytes);
        fi.write((const char*)sclData.data(), (std::streamsize)sclBytes);
        fi.write((const char*)actData.data(), (std::streamsize)actBytes);
        fi.write((const char*)biasData.data(), (std::streamsize)biasBytes);
        fi.write((const char*)gblData.data(), (std::streamsize)gblBytes);
    }
    uploadTo(resPacked, packedData.data(), packedBytes);
    uploadTo(resScl,    sclData.data(),    sclBytes);
    uploadTo(resAct,    actData.data(),    actBytes);
    uploadTo(resBias,   biasData.data(),   biasBytes);
    uploadTo(resGbl,    gblData.data(),    gblBytes);

    void* cm = nullptr;
    resCB->Map(0, nullptr, &cm);
    struct { uint32_t K, nbPerRow, nsPerRow, pad; } cb = { K, nbPerRow, nsPerRow, 0 };
    std::memcpy(cm, &cb, sizeof(cb));
    resCB->Unmap(0, nullptr);

    // ---- Benchmark ----
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;

    auto dispatch = [&]() {
        alloc->Reset();
        list->Reset(alloc.Get(), pso.Get());
        list->SetComputeRootSignature(rootSig.Get());
        list->SetComputeRootShaderResourceView(0, resPacked->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(1, resScl->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, resAct->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(3, resBias->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(4, resGbl->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(5, resOut->GetGPUVirtualAddress());
        list->SetComputeRootConstantBufferView(6, resCB->GetGPUVirtualAddress());
        list->Dispatch(M, 1, 1);
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        queue->Signal(fence.Get(), ++fv);
        fence->SetEventOnCompletion(fv, ev);
        WaitForSingleObject(ev, 30000);
    };

    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) dispatch();
    std::cout << "Measuring..." << std::endl;

    std::vector<double> times;
    times.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dispatch();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double sum = 0; for (double t : times) sum += t;
    double avg = sum / times.size();
    double p50 = percentile(times, 0.50);
    double p99 = percentile(times, 0.99);
    double tflops = 2.0 * (double)M * (double)K / 1e9 / (p50 / 1000.0);
    std::cout << "RESULT M=" << M << " K=" << K
              << " avg_ms=" << avg
              << " p50_ms=" << p50
              << " p99_ms=" << p99
              << " p50_TFLOPs=" << tflops << std::endl;

    // ---- Read back output to file ----
    list->Reset(alloc.Get(), nullptr);
    list->CopyResource(resReadback.Get(), resOut.Get());
    list->Close();
    ID3D12CommandList* ls2[] = { list.Get() };
    queue->ExecuteCommandLists(1, ls2);
    queue->Signal(fence.Get(), ++fv);
    fence->SetEventOnCompletion(fv, ev);
    WaitForSingleObject(ev, 30000);

    void* mp = nullptr;
    resReadback->Map(0, nullptr, &mp);
    std::ofstream f("t_mxfp4_gemv_output.bin", std::ios::binary);
    f.write((const char*)mp, (std::streamsize)outBytes);
    f.close();
    resReadback->Unmap(0, nullptr);
    std::cout << "Wrote t_mxfp4_gemv_output.bin (" << outBytes << " bytes)" << std::endl;
    return 0;
}
