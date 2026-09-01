// [P1f] iGPU MXFP4 server with multiple pre-loaded weights (simpler than sticky)
// Protocol (text commands):
//   LOAD <name> <M> <K> <packed_size>\n<packed_bytes>
//     -> store weight, reply "OK <name>\n"
//   CALL <name>\n<act_bytes><scales_bytes><biases_bytes>
//     -> dispatch, reply <4-byte len><M floats>
//   QUIT\n
//
// Simpler: separate command + payload reads.
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
#include <chrono>
#include <map>
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

static D3D12_RESOURCE_DESC bd(UINT64 sz, D3D12_RESOURCE_FLAGS f) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz;
    d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = f; return d;
}

struct W {
    UINT64 M; UINT32 K; UINT32 nb; UINT32 ns;
    ComPtr<ID3D12Resource> rW, rS, rB, rAct, rGbl, rRowB, rOut, rRb, up;
    UINT64 upCap = 0;
    D3D12_RESOURCE_STATES rWSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    D3D12_RESOURCE_STATES rSSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rBSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rGblSt = D3D12_RESOURCE_STATE_COPY_DEST;
    D3D12_RESOURCE_STATES rRowBSt = D3D12_RESOURCE_STATE_COPY_DEST;
    bool loaded = false;
};

int main() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    fprintf(stderr, "multi server ready\n");

    ComPtr<IDXGIFactory1> f; CreateDXGIFactory1(IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x1002) break;
    }
    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return 1;

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
    if (dxil.empty()) return 1;
    // Shader uses t0, t3, t4, t5. Need slots for these. Use dense array of 8 entries
    // but only bind slots 0, 3, 4, 5 (others stay empty).
    D3D12_ROOT_PARAMETER rp[8] = {};
    for (int i = 0; i < 8; ++i) { rp[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV; rp[i].Descriptor.ShaderRegister = (UINT)i; rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL; }
    rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[6].Descriptor.ShaderRegister = 0; rp[6].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV; rp[7].Descriptor.ShaderRegister = 0; rp[7].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd = { 8, rp, 0, nullptr };
    ComPtr<ID3DBlob> sig;
    D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, nullptr);
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

    auto createW = [&](UINT64 M, UINT32 K) {
        W w;
        w.M = M; w.K = K; w.nb = K/8; w.ns = K/32;
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*w.nb*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rW));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd((UINT64)K*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rAct));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*w.ns*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rS));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*w.ns*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rGbl));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRowB));
        device->CreateCommittedResource(&hpDef, D3D12_HEAP_FLAG_NONE, &bd(M*4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&w.rOut));
        device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd(M*4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&w.rRb));
        return w;
    };

    auto submit = [&]() {
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        queue->ExecuteCommandLists(1, ls);
        ++fv;
        queue->Signal(fence.Get(), fv);
        ResetEvent(ev); fence->SetEventOnCompletion(fv, ev);
        HRESULT hr = queue->Signal(fence.Get(), fv);
        bool ok = (SUCCEEDED(hr) && WaitForSingleObject(ev, 30000) == WAIT_OBJECT_0);
        if (!ok) { fprintf(stderr, "submit failed hr=0x%08X fence=%llu\\n", (unsigned)hr, (unsigned long long)fv); fflush(stderr); }
        return ok;
    };

    std::map<std::string, W> ws;

    while (true) {
        std::string line;
        fprintf(stderr, "  waiting for line...\n"); fflush(stderr);
        if (!readLine(0, line)) break;
        fprintf(stderr, "  got line: '%s'\n", line.c_str()); fflush(stderr);
        if (line.empty()) continue;
        // Parse first token
        size_t sp = line.find(' ');
        std::string cmd = (sp == std::string::npos) ? line : line.substr(0, sp);
        std::string rest = (sp == std::string::npos) ? "" : line.substr(sp + 1);

        if (cmd == "QUIT") { std::cerr << "QUIT\n"; break; }
        if (cmd == "LOAD") {
            // LOAD <name> <M> <K> -> read packed_size after M, K (or compute)
            // format: LOAD <name> <M> <K>
            std::vector<std::string> t;
            size_t p = 0;
            while (p < rest.size()) {
                while (p < rest.size() && rest[p] == ' ') p++;
                size_t s2 = p;
                while (p < rest.size() && rest[p] != ' ') p++;
                t.push_back(rest.substr(s2, p - s2));
            }
            if (t.size() < 3) { fprintf(stderr, "LOAD bad\n"); continue; }
            std::string name = t[0];
            UINT64 M = std::stoull(t[1]);
            UINT32 K = std::stoull(t[2]);
            UINT64 packed_size = (UINT64)M * (K/8) * 4;
            // Allocate and read
            auto it = ws.find(name);
            if (it != ws.end()) ws.erase(it);
            ws[name] = createW(M, K);
            W& w = ws[name];
            // read packed bytes
            std::vector<uint8_t> packed(packed_size);
            if (!readN(0, packed.data(), packed_size)) { fprintf(stderr, "LOAD read fail\n"); continue; }
            // Upload packed + gbl + rowB (use persistent mapped upload heap)
            UINT64 total = packed_size + M*4 + M*4;
            if (w.upCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.up));
                w.upCap = total;
            }
            void* m = nullptr; w.up->Map(0, nullptr, &m);
            std::memcpy(m, packed.data(), packed_size);
            float* g = (float*)((char*)m + packed_size);
            float* rb = g + M;
            fprintf(stderr, "  LOAD upbuf: g ptr offset=%llu bytes\n", (unsigned long long)((char*)g - (char*)m)); fflush(stderr);
            for (UINT32 i = 0; i < M; i++) { g[i] = 1.0f; rb[i] = 0.0f; }
            fprintf(stderr, "  LOAD upbuf: g[0] after write=%f\n", g[0]); fflush(stderr);
            w.up->Unmap(0, nullptr);

            // Single batched command list: copy packed/gbl/rowB to default heap, then barrier to NPSR
            fprintf(stderr, "  LOAD state before copy: rW=%d rGbl=%d rRowB=%d\n", (int)w.rWSt, (int)w.rGblSt, (int)w.rRowBSt); fflush(stderr);
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            list->CopyBufferRegion(w.rW.Get(), 0, w.up.Get(), 0, packed_size);
            list->CopyBufferRegion(w.rGbl.Get(), 0, w.up.Get(), packed_size, (UINT64)M * 4);
            list->CopyBufferRegion(w.rRowB.Get(), 0, w.up.Get(), packed_size + (UINT64)M * 4, (UINT64)M * 4);
            // Barrier: rW/rGbl/rRowB COPY_DEST -> NPSR
            {
                D3D12_RESOURCE_BARRIER bs[3] = {};
                bs[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[0].Transition.pResource = w.rW.Get(); bs[0].Transition.Subresource = 0; bs[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; bs[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                bs[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[1].Transition.pResource = w.rGbl.Get(); bs[1].Transition.Subresource = 0; bs[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; bs[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                bs[2].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; bs[2].Transition.pResource = w.rRowB.Get(); bs[2].Transition.Subresource = 0; bs[2].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; bs[2].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                list->ResourceBarrier(3, bs);
            }
            submit();
            w.rWSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rGblSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.rRowBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // VERIFY: readback rGbl/ rRowB
            {
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rGbl.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; list->ResourceBarrier(1, &b); }
                if (!submit()) { fprintf(stderr, "  LOAD submit failed!\\n"); fflush(stderr); }
                ComPtr<ID3D12Resource> rb;
                device->CreateCommittedResource(&hpRb, D3D12_HEAP_FLAG_NONE, &bd((UINT64)M * 4, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb));
                // Transition rGbl to COPY_SOURCE for readback
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rGbl.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rGblSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE; list->ResourceBarrier(1, &b); }
                submit();
                list->CopyResource(rb.Get(), w.rGbl.Get());
                submit();
                void* g = nullptr; rb->Map(0, nullptr, &g);
                fprintf(stderr, "  LOAD verify rGbl[0]=%f\n", ((float*)g)[0]); fflush(stderr);
                rb->Unmap(0, nullptr);
                // Transition back to NPSR
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rGbl.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
                submit();
            }
            w.rGblSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            w.loaded = true;
            std::string reply = "OK " + name + " " + std::to_string(M) + " " + std::to_string(K) + "\n";
            _write(1, reply.data(), (unsigned int)reply.size());
            _flushall();
            std::cerr << "LOADED " << name << " M=" << M << " K=" << K << "\n";
        } else if (cmd == "CALL") {
            // CALL <name> -> reply is binary: 4-byte len + M*4 floats
            std::string name = rest;
            while (name.size() && name[0] == ' ') name = name.substr(1);
            while (name.size() && name.back() == ' ') name.pop_back();
            auto it = ws.find(name);
            if (it == ws.end()) { fprintf(stderr, "CALL: no weight %s\n", name.c_str()); continue; }
            W& w = it->second;
            UINT32 szA = w.K * 4, szS = w.M * w.ns * 4, szB = w.M * w.ns * 4;
            UINT64 total = szA + szS + szB;
            if (w.upCap < total) {
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.up));
                w.upCap = total;
            }
            std::vector<uint8_t> buf(total);
            if (!readN(0, buf.data(), total)) { fprintf(stderr, "CALL read fail\n"); continue; }
            void* m = nullptr; w.up->Map(0, nullptr, &m);
            std::memcpy(m, buf.data(), total);
            w.up->Unmap(0, nullptr);
            // Copy act, scales, biases (use CopyBufferRegion with explicit sizes)
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            list->CopyBufferRegion(w.rAct.Get(), 0, w.up.Get(), 0, szA);
            submit();
            w.rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rAct.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
            submit();
            w.rActSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            list->CopyBufferRegion(w.rS.Get(), 0, w.up.Get(), szA, szS);
            submit();
            w.rSSt = D3D12_RESOURCE_STATE_COPY_DEST;
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rS.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
            submit();
            w.rSSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            list->CopyBufferRegion(w.rB.Get(), 0, w.up.Get(), szA + szS, szB);
            submit();
            w.rBSt = D3D12_RESOURCE_STATE_COPY_DEST;
            alloc->Reset(); list->Reset(alloc.Get(), nullptr);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rB.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
            submit();
            w.rBSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            // cbuffer
            ComPtr<ID3D12Resource> rCb;
            device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&rCb));
            void* cm = nullptr; rCb->Map(0, nullptr, &cm);
            struct { UINT32 K, nb, ns, pad; } cbv = { w.K, w.nb, w.ns, 0 };
            std::memcpy(cm, &cbv, sizeof(cbv));
            rCb->Unmap(0, nullptr);
            // DEBUG removed
            // Dispatch
            alloc->Reset(); list->Reset(alloc.Get(), pso.Get());
            list->SetComputeRootSignature(rs.Get());
            list->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(3, w.rAct->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress());
            list->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress());
            list->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());
            list->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
            if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                list->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
            list->Dispatch((UINT)w.M, 1, 1);
            { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &b);
            w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE; }
            list->CopyResource(w.rRb.Get(), w.rOut.Get());
            submit();
            void* om = nullptr; w.rRb->Map(0, nullptr, &om);
            UINT32 szOut = (UINT32)w.M * 4;
            fprintf(stderr, "  CALL: %s M=%llu outv_bytes (all):", name.c_str(), (unsigned long long)w.M); fflush(stderr);
            for (UINT32 di = 0; di < szOut; di++) fprintf(stderr, " %02x", ((unsigned char*)om)[di]);
            fprintf(stderr, "\n"); fflush(stderr);
            _write(1, &szOut, 4);
            _write(1, om, szOut);
            w.rRb->Unmap(0, nullptr);
            _flushall();
        } else if (cmd == "BATCH_ALL") {
            // BATCH_ALL <act_K> <packed_act_size> <scales_total_size> <biases_total_size>\n
            //   then <act_bytes> <scales_bytes_concat> <biases_bytes_concat>
            // Dispatches every loaded weight in sequence and returns results concatenated
            // (each with 4-byte len + M*4 floats). Total response size = sum of all outputs.
            std::vector<std::string> bt;
            size_t pp = 0;
            while (pp < rest.size()) {
                while (pp < rest.size() && rest[pp] == ' ') pp++;
                size_t ss2 = pp;
                while (pp < rest.size() && rest[pp] != ' ') pp++;
                bt.push_back(rest.substr(ss2, pp - ss2));
            }
            if (bt.size() < 4) { fprintf(stderr, "BATCH_ALL: bad args\n"); continue; }
            UINT32 actK = std::stoul(bt[0]);
            UINT32 szA = std::stoul(bt[1]);
            UINT32 szS = std::stoul(bt[2]);
            UINT32 szB = std::stoul(bt[3]);
            UINT64 totalIn = szA + szS + szB;
            std::vector<uint8_t> buf(totalIn);
            if (!readN(0, buf.data(), totalIn)) { fprintf(stderr, "BATCH_ALL: read fail\n"); continue; }
            for (auto& kv : ws) {
                if (!kv.second.loaded) continue;
                W& w = kv.second;
                UINT32 this_szA = actK * 4;
                UINT32 this_szS = (UINT32)w.M * w.ns * 4;
                UINT32 this_szB = (UINT32)w.M * w.ns * 4;
                UINT32 this_totalIn = this_szA + this_szS + this_szB;
                if (w.upCap < this_totalIn) {
                    device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(this_totalIn, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.up));
                    w.upCap = this_totalIn;
                }
                void* m = nullptr; w.up->Map(0, nullptr, &m);
                // Copy act from shared buffer (first this_szA bytes)
                std::memcpy(m, buf.data(), this_szA);
                // Copy scales/biases (zeros for this batch)
                std::memset((char*)m + this_szA, 0, this_szS + this_szB);
                w.up->Unmap(0, nullptr);
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                // Transition rAct to COPY_DEST if needed
                if (w.rActSt != D3D12_RESOURCE_STATE_COPY_DEST) {
                    { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rAct.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rActSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST; list->ResourceBarrier(1, &b); }
                    submit();
                    w.rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
                }
                list->CopyResource(w.rAct.Get(), w.up.Get());
                submit();
                w.rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rAct.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
                submit();
                w.rActSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                // Dispatch
                ComPtr<ID3D12Resource> rCb;
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&rCb));
                void* cm = nullptr; rCb->Map(0, nullptr, &cm);
                struct { UINT32 K, nb, ns, pad; } cbv = { w.K, w.nb, w.ns, 0 };
                std::memcpy(cm, &cbv, sizeof(cbv));
                rCb->Unmap(0, nullptr);
                alloc->Reset(); list->Reset(alloc.Get(), pso.Get());
                list->SetComputeRootSignature(rs.Get());
                // DEBUG: read rAct bytes
                void* act_dbg = nullptr; w.rAct->Map(0, nullptr, &act_dbg);
                float act_val = ((float*)act_dbg)[0];
                w.rAct->Unmap(0, nullptr);
                fprintf(stderr, "  DISPATCH %s: rAct[0]=%f, rAct GPU=%p, rW GPU=%p\n", kv.first.c_str(), act_val, w.rAct->GetGPUVirtualAddress(), w.rW->GetGPUVirtualAddress()); fflush(stderr);
                list->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());  // slot 0 -> t0 (packed)
                list->SetComputeRootShaderResourceView(1, w.rS->GetGPUVirtualAddress());  // slot 1 -> t1 (unused)
                list->SetComputeRootShaderResourceView(2, w.rB->GetGPUVirtualAddress());  // slot 2 -> t2 (unused)
                list->SetComputeRootShaderResourceView(3, w.rAct->GetGPUVirtualAddress()); // slot 3 -> t3 (act)
                list->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress()); // slot 4 -> t4 (gbl)
                list->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress()); // slot 5 -> t5 (rowBias)
                list->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress());
                list->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress());
                list->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());
                list->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
                if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    list->ResourceBarrier(1, &b);
                    w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                list->Dispatch((UINT)w.M, 1, 1);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                list->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE; }
                list->CopyResource(w.rRb.Get(), w.rOut.Get());
                submit();
                void* om = nullptr; w.rRb->Map(0, nullptr, &om);
                UINT32 szOut = (UINT32)w.M * 4;
                _write(1, &szOut, 4);
                _write(1, om, szOut);
                w.rRb->Unmap(0, nullptr);
                _flushall();
            }
        } else if (cmd == "ALL") {
            std::vector<std::string> at;
            size_t pp = 0;
            while (pp < rest.size()) {
                while (pp < rest.size() && rest[pp] == ' ') pp++;
                size_t ss2 = pp;
                while (pp < rest.size() && rest[pp] != ' ') pp++;
                at.push_back(rest.substr(ss2, pp - ss2));
            }
            if (at.empty()) { fprintf(stderr, "ALL: bad args\n"); continue; }
            int Nall = std::stoi(at[0]);
            if ((int)at.size() - 1 != Nall) { fprintf(stderr, "ALL: count mismatch\n"); continue; }
            std::vector<UINT32> sizes(Nall);
            UINT64 totalActs = 0;
            for (int i = 0; i < Nall; i++) { sizes[i] = std::stoul(at[1 + i]); totalActs += sizes[i]; }
            std::vector<uint8_t> actsBuf(totalActs);
            if (!readN(0, actsBuf.data(), totalActs)) { fprintf(stderr, "ALL: acts read fail\n"); continue; }
            UINT64 off = 0;
            int idx = 0;
            for (auto& kv : ws) {
                if (idx >= Nall) break;
                if (!kv.second.loaded) continue;
                W& w = kv.second;
                UINT32 szA = sizes[idx];
                UINT32 szS = (UINT32)w.M * w.ns * 4;
                UINT32 szB = (UINT32)w.M * w.ns * 4;
                UINT32 thisTotal = szA + szS + szB;
                if (w.upCap < thisTotal) {
                    device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(thisTotal, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&w.up));
                    w.upCap = thisTotal;
                }
                void* m = nullptr; w.up->Map(0, nullptr, &m);
                std::memcpy(m, actsBuf.data() + off, szA);
                std::memset((char*)m + szA, 0, szS + szB);
                w.up->Unmap(0, nullptr);
                off += szA;
                fprintf(stderr, "  ALL iter idx=%d name=%s szA=%u M=%llu K=%u loaded=%d off=%llu totalActs=%llu\n", idx, kv.first.c_str(), szA, (unsigned long long)w.M, w.K, w.loaded ? 1 : 0, (unsigned long long)off, (unsigned long long)totalActs); fflush(stderr);
                // Print first 4 bytes of uploadBuf to check act
                void* tmp = nullptr; w.up->Map(0, nullptr, &tmp);
                fprintf(stderr, "    act_bytes=%02x %02x %02x %02x\n", ((unsigned char*)tmp)[0], ((unsigned char*)tmp)[1], ((unsigned char*)tmp)[2], ((unsigned char*)tmp)[3]);
                w.up->Unmap(0, nullptr);
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                list->CopyResource(w.rAct.Get(), w.up.Get());
                submit();
                w.rActSt = D3D12_RESOURCE_STATE_COPY_DEST;
                alloc->Reset(); list->Reset(alloc.Get(), nullptr);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rAct.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST; b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; list->ResourceBarrier(1, &b); }
                submit();
                w.rActSt = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                ComPtr<ID3D12Resource> rCb;
                device->CreateCommittedResource(&hpUp, D3D12_HEAP_FLAG_NONE, &bd(256, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&rCb));
                void* cm = nullptr; rCb->Map(0, nullptr, &cm);
                struct { UINT32 K, nb, ns, pad; } cbv = { w.K, w.nb, w.ns, 0 };
                std::memcpy(cm, &cbv, sizeof(cbv));
                rCb->Unmap(0, nullptr);
                alloc->Reset(); list->Reset(alloc.Get(), pso.Get());
                list->SetComputeRootSignature(rs.Get());
                void* act_dbg = nullptr; w.rAct->Map(0, nullptr, &act_dbg);
                float act_val = ((float*)act_dbg)[0];
                w.rAct->Unmap(0, nullptr);
                fprintf(stderr, "  DISPATCH CALL: rAct[0]=%f, rAct GPU=%p, rW GPU=%p\n", act_val, w.rAct->GetGPUVirtualAddress(), w.rW->GetGPUVirtualAddress()); fflush(stderr);
                list->SetComputeRootShaderResourceView(0, w.rW->GetGPUVirtualAddress());  // slot 0 -> t0 (packed)
                list->SetComputeRootShaderResourceView(1, w.rS->GetGPUVirtualAddress());  // slot 1 -> t1 (unused)
                list->SetComputeRootShaderResourceView(2, w.rB->GetGPUVirtualAddress());  // slot 2 -> t2 (unused)
                list->SetComputeRootShaderResourceView(3, w.rAct->GetGPUVirtualAddress()); // slot 3 -> t3 (act)
                list->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress()); // slot 4 -> t4 (gbl)
                list->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress()); // slot 5 -> t5 (rowBias)
                list->SetComputeRootShaderResourceView(4, w.rGbl->GetGPUVirtualAddress());
                list->SetComputeRootShaderResourceView(5, w.rRowB->GetGPUVirtualAddress());
                list->SetComputeRootUnorderedAccessView(6, w.rOut->GetGPUVirtualAddress());
                list->SetComputeRootConstantBufferView(7, rCb->GetGPUVirtualAddress());
                if (w.rOutSt != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
                    D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = w.rOutSt; b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                    list->ResourceBarrier(1, &b);
                    w.rOutSt = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                list->Dispatch((UINT)w.M, 1, 1);
                { D3D12_RESOURCE_BARRIER b = {}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition.pResource = w.rOut.Get(); b.Transition.Subresource = 0; b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                list->ResourceBarrier(1, &b);
                w.rOutSt = D3D12_RESOURCE_STATE_COPY_SOURCE; }
                list->CopyResource(w.rRb.Get(), w.rOut.Get());
                submit();
                void* om = nullptr; w.rRb->Map(0, nullptr, &om);
                UINT32 szOut = (UINT32)w.M * 4;
                fprintf(stderr, "  ALL: %s M=%llu K=%u outv[0]=%f\n", kv.first.c_str(), (unsigned long long)w.M, w.K, szOut > 0 ? ((float*)om)[0] : 0.0f); fflush(stderr);
                _write(1, &szOut, 4);
                _write(1, om, szOut);
                w.rRb->Unmap(0, nullptr);
                _flushall();
                idx++;
            }
        } else {
            fprintf(stderr, "unknown: %s\n", cmd.c_str());
        }
    }
    return 0;
}
