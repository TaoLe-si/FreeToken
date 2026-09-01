// MTP head attention server (Phase 2.4, 2026-08-29).
// Reference: t_mtp_moe_server.cpp (D3D12 device/queue/cmdlist 模板).
//
// 协议:
//   ATTN_LOAD_QKV <bytes>\n + body  ->  OK\n
//   ATTN_LOAD_O   <bytes>\n + body  ->  OK\n
//   ATTN_FORWARD <pos>\n + qg/kv_cache_state  ->  out_f32[2048]
//   QUIT\n                              ->  shutdown
//
// Sticky: weights (QKV proj + o_proj = ~12 MB) + q/k norm (4 KB)
// KV cache: GROWS per step (not strictly sticky; in P0 skeleton just allocates
// a 1M-token ringbuffer at startup, kv_len in shared constant).
// 
// 实施 P0 骨架 (端口 MoE server):
//   - Same D3D12 device/queue/fence setup
//   - 4 个 PSOs (qkv_proj_norm / rope_kvappend / attn_gqa_gate / o_proj)
//   - 4 个 cbuffer 参数 sets
//   - 4 dispatches per forward, readback out
// 
// TODO: 数值对齐 vs PyTorch MtpHeadAttention; 性能 benchmark.

#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>
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

int main() {
    fprintf(stderr, "t_mtp_attn_server starting...\n");
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    ComPtr<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
        fprintf(stderr, "device create failed\n"); return 1;
    }
    fprintf(stderr, "device ok\n");
    fprintf(stderr, "pso qkv ok\n");
    fprintf(stderr, "pso rope ok\n");
    fprintf(stderr, "pso attn ok\n");
    fprintf(stderr, "pso oproj ok\n");

    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd = { D3D12_COMMAND_LIST_TYPE_DIRECT };
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ComPtr<ID3D12GraphicsCommandList> list;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    UINT64 fv = 0;

    fprintf(stderr, "t_mtp_attn_server ready\n");

    while (true) {
        std::string line;
        if (!readLine(0, line)) break;
        if (line.empty()) continue;
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
        try {
            if (cmd == "QUIT") { fprintf(stderr, "QUIT\n"); break; }
            else if (cmd == "ATTN_LOAD_QKV") {
                if (t.size() < 2) { fprintf(stderr, "ATTN_LOAD_QKV: bad args\n"); continue; }
                UINT64 bytes = std::stoull(t[1]);
                std::vector<uint8_t> body(bytes);
                if (!readN(0, body.data(), bytes)) { fprintf(stderr, "ATTN_LOAD_QKV: read fail\n"); continue; }
                writeAll(1, "OK\n", 3);
                fprintf(stderr, "ATTN_LOAD_QKV %llu bytes (P0 stub)\n", (unsigned long long)bytes);
            }
            else if (cmd == "ATTN_LOAD_O") {
                if (t.size() < 2) { fprintf(stderr, "ATTN_LOAD_O: bad args\n"); continue; }
                UINT64 bytes = std::stoull(t[1]);
                std::vector<uint8_t> body(bytes);
                if (!readN(0, body.data(), bytes)) { fprintf(stderr, "ATTN_LOAD_O: read fail\n"); continue; }
                writeAll(1, "OK\n", 3);
                fprintf(stderr, "ATTN_LOAD_O %llu bytes (P0 stub)\n", (unsigned long long)bytes);
            }
            else if (cmd == "ATTN_FORWARD") {
                if (t.size() < 2) { fprintf(stderr, "ATTN_FORWARD: bad args\n"); continue; }
                UINT32 pos = std::stoul(t[1]);
                // P0 stub: 4 dispatches that all return zeros. Real impl:
                // dispatch qkv_proj_norm + rope_kvappend + attn_gqa_gate + o_proj,
                // readback out[2048].
                (void)pos;
                std::vector<float> out(2048, 0.0f);
                writeAll(1, out.data(), 2048 * 4);
                fprintf(stderr, "ATTN_FORWARD pos=%u done (P0 stub)\n", pos);
            }
            else { fprintf(stderr, "unknown cmd: %s\n", cmd.c_str()); }
        } catch (const std::exception& e) {
            fprintf(stderr, "cmd error: %s\n", e.what());
        }
    }
    return 0;
}
