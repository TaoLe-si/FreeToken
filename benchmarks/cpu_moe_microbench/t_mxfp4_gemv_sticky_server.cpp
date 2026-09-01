// [P1f] iGPU MXFP4 GEMV server with sticky-weight support
//
// Protocol (text commands, line-based, server reads from stdin):
//   LOAD <name> <M> <K> <packed_bytes>\n
//     -> stores weight in named slot. Server replies "OK <name> <bytes_received>\n"
//   CALL <name> <K> <act_bytes> <scales_bytes> <biases_bytes>\n
//     -> dispatches using pre-loaded weight. Server replies <4 byte len><M floats>
//   STATELESS <M> <K> <szP> <szA> <szS> <szB> <packed> <act> <scales> <biases>\n
//     -> original stateless protocol. Server replies <4 byte len><M floats>
//   QUIT\n
//     -> server exits cleanly
//
// Each "weight" is identified by a short name (e.g. "fc", "q", "k", "v", "o",
// "gate", "up", "down"). The server holds up to 16 named weights.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <io.h>
#include <fcntl.h>
#include <string>
using Microsoft::WRL::ComPtr;

static bool readN(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t got = 0;
    while (got < n) {
        int r = _read(fd, p + got, (unsigned int)(n - got));
        if (r <= 0) { if (r == 0) return false; if (errno == EINTR) continue; return false; }
        got += r;
    }
    return true;
}
static bool readLine(int fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
        int r = _read(fd, &c, 1);
        if (r <= 0) return false;
        if (c == '\n') break;
        out += c;
    }
    return true;
}
static void writeAll(int fd, const void* buf, size_t n) {
    _write(fd, buf, (unsigned int)n);
    _flushall();
}

static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}

// Per-named-weight resources
struct StickyWeight {
    UINT64 M = 0;            // number of output rows
    UINT32 K = 0;            // input dim
    UINT32 nb = 0;           // K / 8
    UINT32 ns = 0;           // K / 32
    ComPtr<ID3D12Resource> rW;     // packed [M*nb*4]
    ComPtr<ID3D12Resource> rS;     // scales [M*ns*4] (placeholder)
    ComPtr<ID3D12Resource> rB;     // biases [M*ns*4] (placeholder)
    ComPtr<ID3D12Resource> rAct;   // activation [K*4] (input)
    ComPtr<ID3D12Resource> rGbl;   // per-row global scale [M*4]
    ComPtr<ID3D12Resource> rRowB;  // per-row bias [M*4]
    ComPtr<ID3D12Resource> rOut;   // output [M*4] UAV
    ComPtr<ID3D12Resource> rRb;    // readback [M*4]
    ComPtr<ID3D12Resource> uploadBuf;  // upload heap buffer (reused)
    UINT64 uploadCap = 0;
    D3D12_RESOURCE_STATES rWState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rSState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rBState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rActState = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOutState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
};

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    fprintf(stderr, "ready fd0=%d fd1=%d fd2=%d\n", _fileno(stdin), _fileno(stdout), _fileno(stderr));

    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    HRESULT hr = D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) { fprintf(stderr, "device failed\n"); return 1; }

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    list->Close();
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;

    std::ifstream fi("t_mxfp4_gemv_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { fprintf(stderr, "missing t_mxfp4_gemv_sk.dxil\n"); return 1; }
    D3D12_ROOT_PARAMETER rp[8] = {};
    for (int i = 0; i < 6; ++i) { rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; rp[i].Descriptor.ShaderRegister = (UINT)i; rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; }
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[6].Descriptor.ShaderRegister = 0; rp[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[7].Descriptor.ShaderRegister = 0; rp[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = { 8, rp, 0, nullptr };
    ComPtr<ID3DBlob> sig, errb;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &errb);
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rs));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psd = {};
    psd.pRootSignature = rs.Get();
    psd.CS = { dxil.data(), dxil.size() };
    ComPtr<ID3D12PipelineState> pso;
    device->CreateComputePipelineState(&psd, IID_PPV_ARGS(&pso));

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

    auto submit = [&]() {
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fv;
        queue->Signal(fence.Get(), fv);
        ResetEvent(ev); fence->SetEventOnCompletion(fv, ev);
        return WaitForSingleObject(ev, 30000) == WAIT_OBJECT_0;
    };
    auto waitBar = [&](ComPtr<ID3D12Resource> res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after, D3D12_RESOURCE_STATES& st) {
        if (st == after) return;
        alloc->Reset(); list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = res.Get(); b.Transition.Subresource = 0;
        b.Transition.StateBefore = before; b.Transition.StateAfter = after;
        list->ResourceBarrier(1, &b);
        submit();
        st = after;
    };
    auto copyTo = [&](ComPtr<ID3D12Resource> dst, ComPtr<ID3D12Resource> src, UINT64 bytes, D3D12_RESOURCE_STATES& st) {
        alloc->Reset(); list->Reset(alloc.Get(), nullptr);
        list->CopyResource(dst.Get(), src.Get());
        submit();
    };

    std::unordered_map<std::string, StickyWeight> weights;

    auto ensureWeight = [&](const std::string& name, UINT64 M, UINT32 K) -> StickyWeight* {
        auto it = weights.find(name);
        if (it != weights.end() && it->second.M == M && it->second.K == K) return &it->second;
        StickyWeight w;
        w.M = M; w.K = K;
        w.nb = K / 8; w.ns = K / 32;
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.nb * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rW));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rAct));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rS));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rGbl));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRowB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.rOut));
        device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRb));
        weights[name] = std::move(w);
        return &weights[name];
    };

    // Initial reply: server ready
    std::cerr << "sticky server ready\n";
    fflush(stderr);

    while (true) {
        std::string line;
        if (!readLine(0, line)) break;
        if (line.empty()) continue;
        // Tokenize
        std::vector<std::string> toks;
        size_t pos = 0;
        while (pos < line.size()) {
            while (pos < line.size() && line[pos] == ' ') pos++;
            if (pos >= line.size()) break;
            size_t start = pos;
            while (pos < line.size() && line[pos] != ' ') pos++;
            toks.push_back(line.substr(start, pos - start));
        }
        if (toks.empty()) continue;
        std::string cmd = toks[0];

        if (cmd == "QUIT") {
            std::cerr << "QUIT\n"; break;
        } else if (cmd == "LOAD") {
            // LOAD <name> <M> <K> <packed_size>
            if (toks.size() < 4) { fprintf(stderr, "LOAD: bad args\n"); continue; }
            std::string name = toks[1];
            UINT64 M = std::stoull(toks[2]);
            UINT32 K = std::stoull(toks[3]);
            // Read packed_size from line tail (or from header)
            UINT64 packed_size;
            if (toks.size() >= 5) packed_size = std::stoull(toks[4]);
            else packed_size = (UINT64)M * (K / 8) * 4;
            // Read exactly packed_size bytes
            std::vector<uint8_t> packed(packed_size);
            if (!readN(0, packed.data(), packed_size)) { fprintf(stderr, "LOAD: read failed\n"); continue; }
            StickyWeight* w = ensureWeight(name, M, K);
            // Upload packed via upload heap
            UINT64 total = packed_size + (UINT64)M * 4 + (UINT64)M * 4;
            if (w->uploadCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w->uploadBuf));
                w->uploadCap = total;
            }
            void* m = nullptr; w->uploadBuf->Map(0, nullptr, &m);
            std::memcpy(m, packed.data(), packed_size);
            // gbl=1, rowB=0
            float* g = (float*)((char*)m + packed_size);
            float* rb = (float*)((char*)m + packed_size + (UINT64)M * 4);
            for (UINT32 i = 0; i < M; i++) { g[i] = 1.0f; rb[i] = 0.0f; }
            w->uploadBuf->Unmap(0, nullptr);
            // Transition rW: COPY_DEST -> COPY_DEST (no-op, it's already in COPY_DEST after create)
            // Copy
            submit();
            void* om = nullptr; w.rRb->Map(0, nullptr, &om);
            uint32_t szOut = (uint32_t)w.M * 4;
            _write(1, &szOut, 4);
            _write(1, om, szOut);
            w.rRb->Unmap(0, nullptr);
            _flushall();
            std::cerr << "CALLED " << name << " M=" << w.M << " K=" << w.K << " bytes=" << total << "\n";
        } else if (cmd == "STATELESS") {
            // Original stateless protocol (kept for backward compat)
            if (toks.size() < 5) { fprintf(stderr, "STATELESS: bad args\n"); continue; }
            UINT32 M = std::stoul(toks[1]);
            UINT32 K = std::stoul(toks[2]);
            UINT32 szP = std::stoul(toks[3]);
            UINT32 szA = std::stoul(toks[4]);
            UINT32 szS = std::stoul(toks[5].size() ? toks[5] : "0");
            UINT32 szB = toks.size() > 6 ? std::stoul(toks[6]) : 0;
            UINT32 nb = K / 8; UINT32 ns = K / 32;
            UINT32 szOut = M * 4;
            std::vector<uint8_t> packed(szP), act(szA), scales(szS), biases(szB);
            if (!readN(0, packed.data(), szP)) { fprintf(stderr, "STATELESS: packed read failed\n"); continue; }
            if (!readN(0, act.data(), szA)) { fprintf(stderr, "STATELESS: act read failed\n"); continue; }
            if (szS > 0 && !readN(0, scales.data(), szS)) { fprintf(stderr, "STATELESS: scales read failed\n"); continue; }
            if (szB > 0 && !readN(0, biases.data(), szB)) { fprintf(stderr, "STATELESS: biases read failed\n"); continue; }
            StickyWeight* w = ensureWeight("__stateless__", M, K);
            UINT64 total = szP + szA + szS + szB + (UINT64)M * 8;
            if (w->uploadCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w->uploadBuf));
                w->uploadCap = total;
            }
            void* m = nullptr; w->uploadBuf->Map(0, nullptr, &m);
            std::memcpy((char*)m, packed.data(), szP);
            std::memcpy((char*)m + szP, act.data(), szA);
            std::memcpy((char*)m + szP + szA, scales.data(), szS);
            std::memcpy((char*)m + szP + szA + szS, biases.data(), szB);
            float* g = (float*)((char*)m + szP + szA + szS + szB);
            float* rb = g + M;
            for (UINT32 i = 0; i < M; i++) { g[i] = 1.0f; rb[i] = 0.0f; }
            w->uploadBuf->Unmap(0, nullptr);
            submit();
            void* om = nullptr; w->rRb->Map(0, nullptr, &om);
            _write(1, &szOut, 4);
            _write(1, om, szOut);
            w->rRb->Unmap(0, nullptr);
            _flushall();
        } else {
            fprintf(stderr, "unknown cmd: %s\n", cmd.c_str());
        }
    }
    return 0;
}
