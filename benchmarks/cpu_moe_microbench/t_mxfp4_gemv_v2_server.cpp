// [P1g] iGPU MXFP4 GEMV server with sticky-weight cache
//
// Clean rewrite based on the verified P1d stateless server (t_mxfp4_gemv_server.cpp).
// Same root signature: 6 SRV (slot 0-5) + 1 UAV (slot 6) + 1 CBV (slot 7), all dense.
// Same per-resource state tracking + barrier logic + submit pattern.
//
// Protocol (text line + binary body):
//   LOAD <name> <M> <K> <packed_size>\n<packed_bytes>
//     -> store weight in named slot. Resources are created on first LOAD for that name
//        and the packed weight + per-row gbl=1.0 / rowB=0.0 are uploaded once.
//        Server replies "OK <name> <M> <K>\n"
//   CALL <name> <act_size> <scales_size> <biases_size>\n<act_bytes><scales_bytes><biases_bytes>
//     -> dispatches using the pre-loaded weight. Only the activation is re-uploaded.
//        Server replies <4-byte uint32 len><M*4 bytes float32>
//   STATELESS <M> <K> <szP> <szA> <szS> <szB>\n<packed><act><scales><biases>
//     -> same as P1d stateless protocol. Server replies <4-byte uint32 len><M*4 bytes>
//   QUIT\n
//     -> server exits cleanly
//
// Weight cache: up to 16 named weights, keyed by short name (e.g. "fc", "q", "k", ...).
// After LOAD, the resources rW/rS/rB/rGbl/rRowB are in NON_PIXEL_SHADER_RESOURCE state.
// CALL only touches rAct (COPY_DEST -> upload -> NPSR) and re-dispatches.

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
#include <cerrno>
using Microsoft::WRL::ComPtr;

static bool readN(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    size_t got = 0;
    while (got < n) {
        int r = _read(fd, p + got, (unsigned int)(n - got));
        if (r <= 0) { if (r == 0) return false; if (errno == EINTR) continue; return false; }
        got += (size_t)r;
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

// Per-named-weight resources. Sizes:
//   rW     : packed [M * nb * 4]      (nb = K/8)
//   rS     : scales [K * 4]            (T1 alias: holds act, like P1d)
//   rB     : biases [M * ns * 4]      (ns = K/32)
//   rAct   : activation [K * 4]
//   rGbl   : per-row global scale [M * 4]   (set to 1.0 at LOAD)
//   rRowB  : per-row bias [M * 4]           (set to 0.0 at LOAD)
//   rOut   : output [M * 4] (UAV)
//   rRb    : readback [M * 4]
// Note: rS holds K*4 (not M*ns*4) because the shader reads slot 1 (rS) as act in the P1d protocol.
// Scales and biases are uploaded only on STATELESS (or future LOAD-extension), but the
// kernel does not actually use them for the MTP fc call (verified by P1d with szS=szB=0).
struct StickyWeight {
    UINT64 M = 0;
    UINT32 K = 0;
    UINT32 nb = 0;
    UINT32 ns = 0;
    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rGbl, rRowB, rOut, rRb, rCb;
    D3D12_RESOURCE_STATES rWSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rSSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rBSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rGblSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rRowBSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    // Upload heap for the LOAD command (one-shot, holds packed+gbl+rowB).
    ComPtr<ID3D12Resource> uploadLoad;
    UINT64 uploadLoadCap = 0;
    // Upload heap for CALL command (reused across calls, holds act+scales+biases).
    ComPtr<ID3D12Resource> uploadCall;
    UINT64 uploadCallCap = 0;
    bool loaded = false;
};

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    fprintf(stderr, "ready fd0=%d fd1=%d fd2=%d\n", _fileno(stdin), _fileno(stdout), _fileno(stderr)); fflush(stderr);

    // D3D12 debug layer
    ComPtr<ID3D12Debug> dbg;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) { dbg->EnableDebugLayer(); }
    ComPtr<ID3D12InfoQueue> iq;
    // Find AMD adapter (Radeon 780M)
    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    HRESULT hr_dev = D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr_dev)) { fprintf(stderr, "device create failed hr=0x%08X\n", hr_dev); return 1; }
    if (SUCCEEDED(device.As(&iq))) {}
    fprintf(stderr, "device ok\n"); fflush(stderr);

    // Two command queues / allocators / lists (P1d pattern)
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> uploadAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&uploadAlloc));
    ComPtr<ID3D12GraphicsCommandList> uploadList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, uploadAlloc.Get(), nullptr, IID_PPV_ARGS(&uploadList));
    uploadList->Close();
    ComPtr<ID3D12CommandAllocator> dispatchAlloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dispatchAlloc));
    ComPtr<ID3D12GraphicsCommandList> dispatchList;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dispatchAlloc.Get(), nullptr, IID_PPV_ARGS(&dispatchList));
    dispatchList->Close();
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;

    // Load shader
    std::ifstream fi("t_mxfp4_gemv_sk.dxil", std::ios::binary);
    std::vector<uint8_t> dxil((std::istreambuf_iterator<char>(fi)), std::istreambuf_iterator<char>());
    if (dxil.empty()) { fprintf(stderr, "missing t_mxfp4_gemv_sk.dxil\n"); return 1; }
    // Dense 8-slot root sig: 6 SRV (slot 0-5), 1 UAV (slot 6), 1 CBV (slot 7)
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
    fprintf(stderr, "pso ok\n"); fflush(stderr);

    D3D12_HEAP_PROPERTIES hpDef = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_HEAP_PROPERTIES hpUp = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES hpRb = { D3D12_HEAP_TYPE_READBACK };

    auto submit = [&](ComPtr<ID3D12GraphicsCommandList> lst, const char* tag) -> bool {
        HRESULT hc = lst->Close();
        if (FAILED(hc)) {
            fprintf(stderr, "submit[%s] Close failed hr=0x%08X\n", tag, hc); fflush(stderr);
            if (iq) {
                UINT64 n = iq->GetNumStoredMessages();
                for (UINT64 i = 0; i < n; i++) {
                    SIZE_T sz = 0; iq->GetMessage(i, nullptr, &sz);
                    std::vector<uint8_t> mbuf(sz);
                    D3D12_MESSAGE* m = (D3D12_MESSAGE*)mbuf.data();
                    iq->GetMessage(i, m, &sz);
                    fprintf(stderr, "  [%llu] %s\n", (unsigned long long)i, m->pDescription ? m->pDescription : "(no desc)");
                }
                iq->ClearStoredMessages();
            }
            return false;
        }
        ID3D12CommandList* ls[] = { lst.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fv;
        HRESULT hs = queue->Signal(fence.Get(), fv);
        if (FAILED(hs)) { fprintf(stderr, "submit[%s] Signal failed hr=0x%08X\n", tag, hs); fflush(stderr); return false; }
        ResetEvent(ev); fence->SetEventOnCompletion(fv, ev);
        DWORD wr = WaitForSingleObject(ev, 30000);
        if (wr != WAIT_OBJECT_0) { fprintf(stderr, "submit[%s] Wait res=%lu fv=%llu\n", tag, (unsigned long)wr, (unsigned long long)fv); fflush(stderr); return false; }
        return true;
    };

    // Create resources for a newly-loaded weight.
    auto createWeight = [&](UINT64 M, UINT32 K) -> StickyWeight {
        StickyWeight w;
        w.M = M; w.K = K; w.nb = K / 8; w.ns = K / 32;
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.nb * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rW));
        // rS holds K*4 (T1 alias for act, P1d layout)
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rS));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * w.ns * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rAct));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rGbl));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRowB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.rOut));
        device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRb));
        device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.rCb));
        return w;
    };

    std::unordered_map<std::string, StickyWeight> weights;
    std::cerr << "sticky-v2 server ready\n"; fflush(stderr);

    while (true) {
        std::string line;
        if (!readLine(0, line)) { fprintf(stderr, "readline fail\n"); break; }
        if (line.empty()) continue;
        // Tokenize first 5 tokens
        std::vector<std::string> t;
        size_t pos = 0;
        while (pos < line.size()) {
            while (pos < line.size() && line[pos] == ' ') pos++;
            if (pos >= line.size()) break;
            size_t s2 = pos;
            while (pos < line.size() && line[pos] != ' ') pos++;
            t.push_back(line.substr(s2, pos - s2));
        }
        if (t.empty()) continue;
        std::string cmd = t[0];

        if (cmd == "QUIT") {
            std::cerr << "QUIT\n";
            break;
        } else if (cmd == "LOAD") {
            // LOAD <name> <M> <K> <packed_size>
            if (t.size() < 5) { fprintf(stderr, "LOAD: bad args (need 4)\n"); continue; }
            std::string name = t[1];
            UINT64 M = std::stoull(t[2]);
            UINT32 K = std::stoull(t[3]);
            UINT64 packed_size = std::stoull(t[4]);
            // Erase any existing weight with this name (shape may differ)
            weights.erase(name);
            StickyWeight& w = weights[name] = createWeight(M, K);
            // Allocate persistent upload buffer
            UINT64 total = packed_size + (UINT64)M * 4 + (UINT64)M * 4;
            if (w.uploadLoadCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.uploadLoad));
                w.uploadLoadCap = total;
            }
            // Read packed bytes
            std::vector<uint8_t> packed(packed_size);
            if (!readN(0, packed.data(), packed_size)) { fprintf(stderr, "LOAD: packed read failed\n"); weights.erase(name); continue; }
            // Map upload buffer, fill packed + gbl + rowB
            void* m = nullptr; w.uploadLoad->Map(0, nullptr, &m);
            std::memcpy(m, packed.data(), packed_size);
            float* g = (float*)((char*)m + packed_size);
            float* rb = g + M;
            for (UINT32 i = 0; i < M; i++) { g[i] = 1.0f; rb[i] = 0.0f; }
            w.uploadLoad->Unmap(0, nullptr);
            // Batched upload + transition: packed->rW, gbl->rGbl, rowB->rRowB, then COPY_DEST->NPSR
            uploadList->Reset(uploadAlloc.Get(), nullptr);
            uploadList->CopyBufferRegion(w.rW.Get(), 0, w.uploadLoad.Get(), 0, packed_size);
            uploadList->CopyBufferRegion(w.rGbl.Get(), 0, w.uploadLoad.Get(), packed_size, (UINT64)M * 4);
            uploadList->CopyBufferRegion(w.rRowB.Get(), 0, w.uploadLoad.Get(), packed_size + (UINT64)M * 4, (UINT64)M * 4);
            {
                D3D12_RESOURCE_BARRIER bs[3] = {};
                bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[0].Transition.pResource = w.rW.Get(); bs[0].Transition.Subresource = 0; bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                bs[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[1].Transition.pResource = w.rGbl.Get(); bs[1].Transition.Subresource = 0; bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                bs[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[2].Transition.pResource = w.rRowB.Get(); bs[2].Transition.Subresource = 0; bs[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[2].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                uploadList->ResourceBarrier(3, bs);
            }
            if (!submit(uploadList, "LOAD")) { fprintf(stderr, "LOAD submit failed\n"); weights.erase(name); continue; }
            w.rWSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rGblSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rRowBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.loaded = true;
            // Reply
            std::string reply = "OK " + name + " " + std::to_string(M) + " " + std::to_string(K) + "\n";
            writeAll(1, reply.data(), reply.size());
            std::cerr << "LOADED " << name << " M=" << M << " K=" << K << " packed=" << packed_size << "\n"; fflush(stderr);
        } else if (cmd == "CALL") {
            // CALL <name> <szA> <szS> <szB>
            if (t.size() < 4) { fprintf(stderr, "CALL: bad args\n"); continue; }
            std::string name = t[1];
            UINT32 szA = std::stoul(t[2]);
            UINT32 szS = std::stoul(t[3]);
            UINT32 szB = std::stoul(t[4]);
            auto it = weights.find(name);
            if (it == weights.end() || !it->second.loaded) { fprintf(stderr, "CALL: no weight %s\n", name.c_str()); continue; }
            StickyWeight& w = it->second;
            UINT64 total = (UINT64)szA + szS + szB;
            if (total == 0) { fprintf(stderr, "CALL: zero bytes\n"); continue; }
            std::vector<uint8_t> body(total);
            if (!readN(0, body.data(), total)) { fprintf(stderr, "CALL: body read fail\n"); continue; }
            // Allocate / reuse upload buffer
            if (w.uploadCallCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.uploadCall));
                w.uploadCallCap = total;
            }
            void* m = nullptr; w.uploadCall->Map(0, nullptr, &m);
            std::memcpy(m, body.data(), total);
            w.uploadCall->Unmap(0, nullptr);
            auto t0 = std::chrono::high_resolution_clock::now();
            // Upload act + (optional) scales + biases to rS (slot 1, T1) and rB (slot 2) and rAct (slot 3).
            // Following P1d layout:
            //   - rS (slot 1) holds the act bytes (T1 = act), so we copy szA bytes there.
            //   - rB (slot 2) holds the scales (unused by shader, but we fill it for completeness).
            //   - rAct (slot 3) holds the act bytes (T3 = act).
            uploadList->Reset(uploadAlloc.Get(), nullptr);
            // rS COPY_DEST -> ensure
            if (w.rSSt != D3D12_RESOURCE_STATE_COPY_DEST) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rS.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rSSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                uploadList->ResourceBarrier(1, &b);
            }
            uploadList->CopyBufferRegion(w.rS.Get(), 0, w.uploadCall.Get(), 0, szA);
            // rB: scales
            if (szS > 0) uploadList->CopyBufferRegion(w.rB.Get(), 0, w.uploadCall.Get(), szA, szS);
            // rAct: act
            uploadList->CopyBufferRegion(w.rAct.Get(), 0, w.uploadCall.Get(), 0, szA);
            // Transition rS, rB, rAct COPY_DEST -> NPSR
            {
                D3D12_RESOURCE_BARRIER bs[3] = {};
                int nbb = 0;
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rS.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                if (szS > 0) {
                    bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rB.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                }
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rAct.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                uploadList->ResourceBarrier(nbb, bs);
            }
            if (!submit(uploadList, "CALL-up")) { fprintf(stderr, "CALL upload submit fail\n"); continue; }
            w.rSSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rActSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // cbuffer (must upload to upload heap)
            void* cm = nullptr; w.rCb->Map(0, nullptr, &cm);
            struct { uint32_t K, nbPerRow, nsPerRow, pad; } cbv = { w.K, w.nb, w.ns, 0 };
            std::memcpy(cm, &cbv, sizeof(cbv));
            w.rCb->Unmap(0, nullptr);
            // Dispatch
            dispatchList->Reset(dispatchAlloc.Get(), pso.Get());
            dispatchList->SetComputeRootSignature(rs.Get());
            dispatchList->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(1, w.rS->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(2, w.rB->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(3, w.rAct->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress());
            dispatchList->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());
            dispatchList->SetComputeRootConstantBufferView(7, w.rCb->GetGPUVirtualAddress());
            if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                dispatchList->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
            dispatchList->Dispatch((UINT)w.M, 1, 1);
            {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                dispatchList->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
            dispatchList->CopyResource(w.rRb.Get(), w.rOut.Get());
            if (!submit(dispatchList, "CALL-dispatch")) { fprintf(stderr, "CALL dispatch submit fail\n"); continue; }
            void* om = nullptr; w.rRb->Map(0, nullptr, &om);
            uint32_t szOut = (uint32_t)w.M * 4;
            float v0 = szOut >= 4 ? ((float*)om)[0] : 0.0f;
            fprintf(stderr, "  CALL %s M=%llu K=%u v[0]=%.4f\n", name.c_str(), (unsigned long long)w.M, w.K, (double)v0); fflush(stderr);
            writeAll(1, &szOut, 4);
            writeAll(1, om, szOut);
            w.rRb->Unmap(0, nullptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            fprintf(stderr, "  CALL %s: %.3fms\n", name.c_str(), ms); fflush(stderr);
        } else if (cmd == "STATELESS") {
            // STATELESS <M> <K> <szP> <szA> <szS> <szB>
            if (t.size() < 7) { fprintf(stderr, "STATELESS: bad args\n"); continue; }
            UINT32 M = std::stoul(t[1]);
            UINT32 K = std::stoul(t[2]);
            UINT32 szP = std::stoul(t[3]);
            UINT32 szA = std::stoul(t[4]);
            UINT32 szS = std::stoul(t[5]);
            UINT32 szB = std::stoul(t[6]);
            if ((K & 31) != 0) { fprintf(stderr, "STATELESS: K not mult of 32\n"); continue; }
            UINT32 nb = K / 8, ns = K / 32;
            UINT32 szOut = M * 4;
            std::vector<uint8_t> packed(szP), act(szA), scales(szS), biases(szB);
            if (!readN(0, packed.data(), szP)) { fprintf(stderr, "STATELESS packed read fail\n"); continue; }
            if (!readN(0, act.data(), szA)) { fprintf(stderr, "STATELESS act read fail\n"); continue; }
            if (szS > 0 && !readN(0, scales.data(), szS)) { fprintf(stderr, "STATELESS scales read fail\n"); continue; }
            if (szB > 0 && !readN(0, biases.data(), szB)) { fprintf(stderr, "STATELESS biases read fail\n"); continue; }
            // Use a one-shot temporary StickyWeight
            StickyWeight& w = weights["__stateless__"] = createWeight(M, K);
            UINT64 total = (UINT64)szP + szA + szS + szB + (UINT64)M * 4 + (UINT64)M * 4;
            if (w.uploadLoadCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.uploadLoad));
                w.uploadLoadCap = total;
            }
            UINT64 offP = 0, offA = szP, offS = szP + szA, offB = szP + szA + szS, offG = offB + szB, offR = offG + (UINT64)M * 4;
            void* m = nullptr; w.uploadLoad->Map(0, nullptr, &m);
            std::memcpy((char*)m + offP, packed.data(), szP);
            std::memcpy((char*)m + offA, act.data(), szA);
            std::memcpy((char*)m + offS, scales.data(), szS);
            std::memcpy((char*)m + offB, biases.data(), szB);
            {
                float* g = (float*)((char*)m + offG); for (UINT32 i = 0; i < M; i++) g[i] = 1.0f;
                float* rb = (float*)((char*)m + offR); for (UINT32 i = 0; i < M; i++) rb[i] = 0.0f;
            }
            w.uploadLoad->Unmap(0, nullptr);
            // Single batched list: copy packed->rW, act->rS (T1), scales->rB, act->rAct (T3), gbl, rowB; transition to NPSR; then dispatch.
            uploadList->Reset(uploadAlloc.Get(), nullptr);
            uploadList->CopyBufferRegion(w.rW.Get(), 0, w.uploadLoad.Get(), offP, szP);
            uploadList->CopyBufferRegion(w.rS.Get(), 0, w.uploadLoad.Get(), offA, szA);
            if (szS > 0) uploadList->CopyBufferRegion(w.rB.Get(), 0, w.uploadLoad.Get(), offS, szS);
            uploadList->CopyBufferRegion(w.rAct.Get(), 0, w.uploadLoad.Get(), offA, szA);
            uploadList->CopyBufferRegion(w.rGbl.Get(), 0, w.uploadLoad.Get(), offG, (UINT64)M * 4);
            uploadList->CopyBufferRegion(w.rRowB.Get(), 0, w.uploadLoad.Get(), offR, (UINT64)M * 4);
            {
                D3D12_RESOURCE_BARRIER bs[6] = {};
                int nbb = 0;
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rW.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rS.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                if (szS > 0) { bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rB.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb; }
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rAct.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rGbl.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                bs[nbb].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[nbb].Transition.pResource = w.rRowB.Get(); bs[nbb].Transition.Subresource = 0; bs[nbb].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[nbb].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; ++nbb;
                uploadList->ResourceBarrier(nbb, bs);
            }
            if (!submit(uploadList, "STATELESS-up")) continue;
            w.rWSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rSSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rActSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rGblSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rRowBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // cbuffer
            void* cm = nullptr; w.rCb->Map(0, nullptr, &cm);
            struct { uint32_t K, nbPerRow, nsPerRow, pad; } cbv = { K, nb, ns, 0 };
            std::memcpy(cm, &cbv, sizeof(cbv));
            w.rCb->Unmap(0, nullptr);
            // Dispatch
            dispatchList->Reset(dispatchAlloc.Get(), pso.Get());
            dispatchList->SetComputeRootSignature(rs.Get());
            dispatchList->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(1, w.rS->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(2, w.rB->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(3, w.rAct->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress());
            dispatchList->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress());
            dispatchList->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());
            dispatchList->SetComputeRootConstantBufferView(7, w.rCb->GetGPUVirtualAddress());
            if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                dispatchList->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
            dispatchList->Dispatch(M, 1, 1);
            {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                dispatchList->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE;
            }
            dispatchList->CopyResource(w.rRb.Get(), w.rOut.Get());
            if (!submit(dispatchList, "STATELESS-dispatch")) continue;
            void* om = nullptr; w.rRb->Map(0, nullptr, &om);
            float v0 = szOut >= 4 ? ((float*)om)[0] : 0.0f;
            fprintf(stderr, "  STATELESS M=%u K=%u v[0]=%.4f\n", M, K, (double)v0); fflush(stderr);
            writeAll(1, &szOut, 4);
            writeAll(1, om, szOut);
            w.rRb->Unmap(0, nullptr);
        } else {
            fprintf(stderr, "unknown cmd: %s\n", cmd.c_str());
        }
    }
    return 0;
}
