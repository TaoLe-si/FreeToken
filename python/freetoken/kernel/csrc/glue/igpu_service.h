// python/freetoken/kernel/csrc/glue/igpu_service.h
// FreeToken iGPU D3D12 service C++ bridge (G/B-line: MTP path IPC port).
// Replaces the Python subprocess.Popen + readN/writeAll pipe dance with a
// C++ CreateProcessW + Windows-overlapped-pipe pair, so MTP head forward
// IPC stops paying Python-side latency.
#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ft::glue {

// iGPU D3D12 service bridge (Windows only; non-Windows builds get an unavailable stub).
class IgpuService {
public:
    // server_path: absolute path to t_mxfp4_gemv_v3_server.exe
    IgpuService(std::string server_path, int max_M, int max_K, int max_ns);
    ~IgpuService();

    IgpuService(const IgpuService&) = delete;
    IgpuService& operator=(const IgpuService&) = delete;

    // Stateless GEMV: caller passes weights + act + scales + biases for each call.
    // Equivalent to the Python IgpuFcClient.forward() path.
    torch::Tensor forward_stateless(
        torch::Tensor packed, torch::Tensor act,
        torch::Tensor scales, torch::Tensor biases);

    // Sticky GEMV: caller pre-loads weights once via update_weight(), then only
    // passes act per call (saves the per-call weight transfer cost).
    torch::Tensor fc_call(torch::Tensor act);

    // Replace the sticky weight (used on model swap or hot-reload).
    void update_weight(
        torch::Tensor packed, torch::Tensor scales, torch::Tensor biases);

    // Close the subprocess + release pipes (idempotent).
    void close();

    // Tail of server stderr (last N lines, for diagnostics).
    std::vector<std::string> get_log(int last_n);

    // G/B-line extension: send a raw ASCII command line + binary body, recv a fixed-size
    // byte response. Used by Python to drive MOE_LOAD/MOE_FORWARD/ATTN_LOAD_*/ATTN_FORWARD/
    // MTP_LAYER commands against the MoE/attn fused servers (t_mtp_moe_server.exe,
    // t_mtp_attn_server.exe). Mirrors the Python IgpuFcClient.send_command + recv path
    // but in C++ for sub-ms IPC. ``line`` is sent ASCII + LF; ``body`` follows verbatim;
    // ``recv_bytes`` outputs the raw reply from the server.
    void send_raw(std::string line, torch::Tensor body);
    torch::Tensor recv_raw(int64_t n);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ft::glue
