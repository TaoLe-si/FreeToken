// python/freetoken/kernel/csrc/glue/igpu_service.cpp
// FreeToken iGPU D3D12 service C++ bridge implementation.
//
// Real Windows impl (Phase 2.1, 2026-08-29): mirrors the Python IgpuFcSticky protocol
// (see freetoken/kernel/igpu_fc.py) -- "FC_LOAD M K szP szS szB\n" then bytes, "FC_CALL
// {bytes}\n" then act_f32, replies "OK\n" / 4-byte size + body. Uses CreateProcessW +
// anonymous pipes + a dedicated stderr-drain thread; fc_call is locked so concurrent
// callers serialize. Non-Windows: throws runtime_error on every call.
#include "igpu_service.h"

#include <torch/extension.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
// Order matters: windows.h before anything that uses HANDLE on the C side.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define FT_PIPE_H_TO_OS(h) _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_BINARY)
#define FT_PIPE_OS_TO_H(fd) reinterpret_cast<HANDLE>(_get_osfhandle(fd))
#else
// non-Windows: no-op
#endif

namespace ft::glue {

#if !defined(_WIN32)
// ============================================================
// Non-Windows stub (Linux / macOS / etc -- D3D12 is Windows-only).
// ============================================================
struct IgpuService::Impl {
    std::string server_path;
};

IgpuService::IgpuService(std::string server_path, int /*max_M*/, int /*max_K*/, int /*max_ns*/)
    : impl_(std::make_unique<Impl>()) {
    impl_->server_path = std::move(server_path);
}

IgpuService::~IgpuService() = default;

torch::Tensor IgpuService::forward_stateless(
    torch::Tensor /*packed*/, torch::Tensor /*act*/,
    torch::Tensor /*scales*/, torch::Tensor /*biases*/) {
    throw std::runtime_error("iGPU D3D12 service is Windows-only (build target != _WIN32)");
}

torch::Tensor IgpuService::fc_call(torch::Tensor /*act*/) {
    throw std::runtime_error("iGPU D3D12 service is Windows-only (build target != _WIN32)");
}

void IgpuService::update_weight(
    torch::Tensor /*packed*/, torch::Tensor /*scales*/, torch::Tensor /*biases*/) {
    throw std::runtime_error("iGPU D3D12 service is Windows-only (build target != _WIN32)");
}

void IgpuService::close() {}

std::vector<std::string> IgpuService::get_log(int /*last_n*/) { return {}; }

void IgpuService::send_raw(std::string, torch::Tensor) {
    throw std::runtime_error("IgpuService::send_raw: iGPU D3D12 service is Windows-only (build target != _WIN32)");
}

torch::Tensor IgpuService::recv_raw(int64_t) {
    throw std::runtime_error("IgpuService::recv_raw: iGPU D3D12 service is Windows-only (build target != _WIN32)");
}

#else  // _WIN32
// ============================================================
// Windows real impl (CreateProcessW + anonymous pipes + stderr thread).
// ============================================================

// RAII handles -- prevent leaks on early throws.
struct PipeHandles {
    HANDLE in_read = nullptr;
    HANDLE in_write = nullptr;
    HANDLE out_read = nullptr;
    HANDLE out_write = nullptr;
    HANDLE err_read = nullptr;
    ~PipeHandles() {
        if (in_read)   CloseHandle(in_read);
        if (in_write)  CloseHandle(in_write);
        if (out_read)  CloseHandle(out_read);
        if (out_write) CloseHandle(out_write);
        if (err_read)  CloseHandle(err_read);
    }
};

static bool CreateAnonPipePair(HANDLE& read_end, HANDLE& write_end, bool inherit_write) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;
    if (!CreatePipe(&read_end, &write_end, &sa, 0)) return false;
    // The write end (or read end) must NOT be inheritable -- only the side the child uses.
    SetHandleInformation(inherit_write ? write_end : read_end, HANDLE_FLAG_INHERIT, 0);
    return true;
}

struct IgpuService::Impl {
    std::string server_path;
    PROCESS_INFORMATION pi{};
    HANDLE in_write = nullptr;   // parent writes here -> child stdin
    HANDLE out_read = nullptr;   // parent reads here  <- child stdout
    HANDLE err_read = nullptr;   // parent reads here  <- child stderr
    bool closed = true;
    std::mutex call_lock;        // serialize fc_call / forward_stateless / update_weight
    std::thread stderr_thread;
    std::vector<std::string> stderr_log;
    int M = 0, K = 0, ns = 0;    // sticky weight dims
    bool sticky_loaded = false;
};

// Append a line to the stderr log (called by the drain thread).
static void AppendStderrLine(std::vector<std::string>& log, const std::string& line) {
    log.push_back(line);
    // Cap at ~4 KiB of lines (memory bound for diagnostics -- we never need more).
    constexpr size_t kMax = 4096;
    if (log.size() > kMax) {
        log.erase(log.begin(), log.begin() + (log.size() - kMax));
    }
}

static void StderrDrainThread(HANDLE err_read, std::vector<std::string>* log, bool* stop) {
    constexpr DWORD kBufSize = 4096;
    char buf[kBufSize];
    std::string carry;
    while (!*stop) {
        DWORD got = 0;
        BOOL ok = ReadFile(err_read, buf, kBufSize, &got, nullptr);
        if (!ok || got == 0) {
            // EOF or pipe closed -> exit
            if (!carry.empty()) AppendStderrLine(*log, carry);
            return;
        }
        carry.append(buf, got);
        // Split on \n, keep the trailing partial in carry.
        size_t nl;
        while ((nl = carry.find('\n')) != std::string::npos) {
            std::string line = carry.substr(0, nl);
            // Trim trailing \r if present.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            AppendStderrLine(*log, line);
            carry.erase(0, nl + 1);
        }
    }
    if (!carry.empty()) AppendStderrLine(*log, carry);
}

static void WriteAllOrThrow(HANDLE h, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    size_t written = 0;
    while (written < n) {
        DWORD w = 0;
        BOOL ok = WriteFile(h, p + written, static_cast<DWORD>(n - written), &w, nullptr);
        if (!ok || w == 0) {
            throw std::runtime_error("IgpuService: WriteFile to iGPU server failed");
        }
        written += w;
    }
}

static void ReadExactOrThrow(HANDLE h, void* data, size_t n) {
    char* p = static_cast<char*>(data);
    size_t got = 0;
    while (got < n) {
        DWORD r = 0;
        BOOL ok = ReadFile(h, p + got, static_cast<DWORD>(n - got), &r, nullptr);
        if (!ok || r == 0) {
            throw std::runtime_error("IgpuService: iGPU server died / pipe closed");
        }
        got += r;
    }
}

IgpuService::IgpuService(std::string server_path, int max_M, int max_K, int max_ns)
    : impl_(std::make_unique<Impl>()) {
    impl_->server_path = std::move(server_path);
    impl_->M = max_M;
    impl_->K = max_K;
    impl_->ns = max_ns;

    // ---- Create anonymous pipes ----
    HANDLE in_read = nullptr, in_write = nullptr;
    HANDLE out_read = nullptr, out_write = nullptr;
    HANDLE err_read = nullptr, err_write = nullptr;
    if (!CreateAnonPipePair(in_read, in_write, /*inherit_write=*/true)) {
        throw std::runtime_error("IgpuService: CreatePipe(stdin) failed");
    }
    if (!CreateAnonPipePair(out_read, out_write, /*inherit_write=*/false)) {
        throw std::runtime_error("IgpuService: CreatePipe(stdout) failed");
    }
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;
    if (!CreatePipe(&err_read, &err_write, &sa, 0)) {
        throw std::runtime_error("IgpuService: CreatePipe(stderr) failed");
    }
    SetHandleInformation(err_read, HANDLE_FLAG_INHERIT, 0);

    // ---- Build STARTUPINFO ----
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.hStdInput = in_read;
    si.hStdOutput = out_write;
    si.hStdError = err_write;
    si.dwFlags |= STARTF_USESTDHANDLES;

    // ---- Resolve exe path (must exist) ----
    std::wstring exe_w;
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, impl_->server_path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) throw std::runtime_error("IgpuService: bad server_path encoding");
        exe_w.resize(wlen - 1);
        MultiByteToWideChar(CP_UTF8, 0, impl_->server_path.c_str(), -1, exe_w.data(), wlen);
    }

    // ---- Spawn child ----
    std::wstring cmdline = L""" + exe_w + L""";
    // cwd = server's directory (the .dxil shaders live next to the .exe; without
    // this set the child inherits our parent cwd and the shaders can't be opened).
    std::wstring server_dir_w;
    {
        std::string dir;
        const char* slash = std::strrchr(impl_->server_path.c_str(), '\\');
        const char* fwdslash = std::strrchr(impl_->server_path.c_str(), '/');
        const char* sep = (slash && fwdslash) ? (slash > fwdslash ? slash : fwdslash)
                        : (slash ? slash : (fwdslash ? fwdslash : nullptr));
        if (sep) dir.assign(impl_->server_path.c_str(), sep - impl_->server_path.c_str());
        int wlen = MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, nullptr, 0);
        if (wlen > 0) {
            server_dir_w.resize(wlen - 1);
            MultiByteToWideChar(CP_UTF8, 0, dir.c_str(), -1, server_dir_w.data(), wlen);
        }
    }
    if (!CreateProcessW(
            exe_w.c_str(),
            cmdline.data(),
            nullptr, nullptr,
            /*bInheritHandles=*/TRUE,
            0,
            nullptr,                                            // lpEnvironment (inherit parent)
            server_dir_w.empty() ? nullptr : server_dir_w.c_str(),  // lpCurrentDirectory
            &si,
            &impl_->pi)) {
        throw std::runtime_error("IgpuService: CreateProcessW failed for " + impl_->server_path);
    }

    // Child uses its ends; close the parent's inherited ends.
    CloseHandle(in_read);
    CloseHandle(out_write);
    CloseHandle(err_write);

    impl_->in_write = in_write;
    impl_->out_read = out_read;
    impl_->err_read = err_read;
    impl_->closed = false;

    // Start stderr drain thread.
    auto drain_stop = std::make_shared<bool>(false);
    impl_->stderr_thread = std::thread(StderrDrainThread, impl_->err_read,
                                       &impl_->stderr_log, drain_stop.get());
    // (We don't join on close -- the thread will exit when err_read EOFs; we mark the stop
    // flag as a hint, but the real signal is the child closing its stderr handle.)

    // ---- Wait for "psoFc ok" / "server ready" on stderr (up to 15s) ----
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (std::chrono::steady_clock::now() < deadline) {
        bool ready = false;
        for (const auto& line : impl_->stderr_log) {
            if (line.find("psoFc ok") != std::string::npos ||
                line.find("server ready") != std::string::npos) {
                ready = true;
                break;
            }
        }
        if (ready) break;
        Sleep(50);
    }
}

IgpuService::~IgpuService() {
    try { close(); } catch (...) {}
    if (impl_->stderr_thread.joinable()) impl_->stderr_thread.join();
}

void IgpuService::close() {
    if (!impl_ || impl_->closed) return;
    std::lock_guard<std::mutex> lk(impl_->call_lock);
    if (impl_->closed) return;
    // Send QUIT gracefully; if the child ignores it, terminate.
    try { WriteAllOrThrow(impl_->in_write, "QUIT\n", 5); } catch (...) {}
    if (impl_->pi.hProcess) {
        DWORD wait = WaitForSingleObject(impl_->pi.hProcess, 2000);
        if (wait == WAIT_TIMEOUT) TerminateProcess(impl_->pi.hProcess, 1);
        CloseHandle(impl_->pi.hProcess);
        CloseHandle(impl_->pi.hThread);
        impl_->pi = PROCESS_INFORMATION{};
    }
    if (impl_->in_write) { CloseHandle(impl_->in_write); impl_->in_write = nullptr; }
    if (impl_->out_read) { CloseHandle(impl_->out_read); impl_->out_read = nullptr; }
    if (impl_->err_read) { CloseHandle(impl_->err_read); impl_->err_read = nullptr; }
    impl_->closed = true;
}

void IgpuService::update_weight(
    torch::Tensor packed, torch::Tensor scales, torch::Tensor biases) {
    if (!impl_ || impl_->closed) throw std::runtime_error("IgpuService: not open");
    if (packed.scalar_type() != torch::kInt32) {
        throw std::runtime_error("IgpuService::update_weight: packed must be int32 (uint32)");
    }
    if (scales.scalar_type() != torch::kFloat32 || biases.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("IgpuService::update_weight: scales/biases must be float32");
    }
    if (!packed.is_contiguous()) packed = packed.contiguous();
    if (!scales.is_contiguous()) scales = scales.contiguous();
    if (!biases.is_contiguous()) biases = biases.contiguous();

    auto M = packed.size(0);
    auto K = packed.size(1) * 8;
    auto ns = K / 32;
    TORCH_CHECK(scales.size(0) == M && scales.size(1) == ns, "scales shape mismatch");
    TORCH_CHECK(biases.size(0) == M && biases.size(1) == ns, "biases shape mismatch");

    impl_->M = M;
    impl_->K = K;
    impl_->ns = ns;

    auto szP = packed.nbytes();
    auto szS = scales.nbytes();
    auto szB = biases.nbytes();
    char cmd[128];
    int n = std::snprintf(cmd, sizeof(cmd), "FC_LOAD %lld %lld %zu %zu %zu\n",
                          (long long)M, (long long)K, szP, szS, szB);
    TORCH_CHECK(n > 0 && n < (int)sizeof(cmd), "IgpuService: FC_LOAD cmd overflow");

    std::lock_guard<std::mutex> lk(impl_->call_lock);
    if (impl_->closed) throw std::runtime_error("IgpuService: not open");

    // Tensor -> bytes via .cpu().data_ptr() after a sync (caller must hold GPU side done).
    auto* p_ptr = packed.cpu().data_ptr();
    auto* s_ptr = scales.cpu().data_ptr();
    auto* b_ptr = biases.cpu().data_ptr();
    WriteAllOrThrow(impl_->in_write, cmd, n);
    WriteAllOrThrow(impl_->in_write, p_ptr, szP);
    WriteAllOrThrow(impl_->in_write, s_ptr, szS);
    WriteAllOrThrow(impl_->in_write, b_ptr, szB);

    // Expect "OK\n"
    char ack[3];
    ReadExactOrThrow(impl_->out_read, ack, 3);
    if (std::memcmp(ack, "OK\n", 3) != 0) {
        std::string msg = std::string("IgpuService::update_weight: FC_LOAD ack failed (server says: ")
                        + std::string(ack, 3) + ")";
        throw std::runtime_error(msg);
    }
    impl_->sticky_loaded = true;
}

torch::Tensor IgpuService::fc_call(torch::Tensor act) {
    if (!impl_ || impl_->closed) throw std::runtime_error("IgpuService: not open");
    if (!impl_->sticky_loaded) {
        throw std::runtime_error("IgpuService::fc_call: weights not loaded -- call update_weight first");
    }
    if (act.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("IgpuService::fc_call: act must be float32");
    }
    auto K = act.numel();
    TORCH_CHECK(K == impl_->K, "IgpuService::fc_call: act.numel() != sticky K");

    auto* a_ptr = act.contiguous().cpu().data_ptr();
    auto szA = act.nbytes();

    char cmd[64];
    int n = std::snprintf(cmd, sizeof(cmd), "FC_CALL %zu\n", szA);
    TORCH_CHECK(n > 0 && n < (int)sizeof(cmd), "IgpuService: FC_CALL cmd overflow");

    std::lock_guard<std::mutex> lk(impl_->call_lock);
    if (impl_->closed) throw std::runtime_error("IgpuService: not open");

    WriteAllOrThrow(impl_->in_write, cmd, n);
    WriteAllOrThrow(impl_->in_write, a_ptr, szA);

    // Read 4-byte LE size + that many bytes of float32 output.
    uint32_t out_bytes = 0;
    ReadExactOrThrow(impl_->out_read, &out_bytes, 4);
    TORCH_CHECK(out_bytes == impl_->M * sizeof(float),
                "IgpuService::fc_call: output size mismatch (", out_bytes, " vs M*4)");

    // Allocate CPU tensor for the output; caller can .to(device) as needed.
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto out = torch::empty({static_cast<int64_t>(impl_->M)}, opts);
    ReadExactOrThrow(impl_->out_read, out.data_ptr(), out_bytes);
    return out;
}

torch::Tensor IgpuService::forward_stateless(
    torch::Tensor packed, torch::Tensor act,
    torch::Tensor scales, torch::Tensor biases) {
    // Stateless path = update_weight() + fc_call() under the same lock so concurrent
    // fc_call callers (rare on the MTP path -- fc_call is the hot loop) don't see
    // half-loaded weights. The Python IgpuFcClient.forward path was the same idiom.
    update_weight(packed, scales, biases);
    return fc_call(act);
}

std::vector<std::string> IgpuService::get_log(int last_n) {
    if (!impl_) return {};
    if (last_n <= 0) return {};
    int n = last_n;
    std::lock_guard<std::mutex> lk(impl_->call_lock);  // also guards stderr_log mutation
    if (n > (int)impl_->stderr_log.size()) n = (int)impl_->stderr_log.size();
    return std::vector<std::string>(impl_->stderr_log.end() - n, impl_->stderr_log.end());
}

void IgpuService::send_raw(std::string line, torch::Tensor body) {
    if (!impl_) throw std::runtime_error("IgpuService::send_raw: not initialized");
    std::lock_guard<std::mutex> lk(impl_->call_lock);
    std::string cmd = line;
    if (cmd.empty() || cmd.back() != 0x0A) cmd.push_back(0x0A);
    WriteAllOrThrow(impl_->in_write, cmd.data(), cmd.size());
    if (body.defined() && body.numel() > 0) {
        size_t n = (size_t)body.numel() * body.element_size();
        WriteAllOrThrow(impl_->in_write, body.data_ptr(), n);
    }
}

torch::Tensor IgpuService::recv_raw(int64_t n) {
    if (!impl_) throw std::runtime_error("IgpuService::recv_raw: not initialized");
    if (n <= 0) return torch::Tensor();
    std::lock_guard<std::mutex> lk(impl_->call_lock);
    auto out = torch::empty({(int64_t)n}, torch::dtype(torch::kUInt8));
    ReadExactOrThrow(impl_->out_read, out.data_ptr(), (size_t)n);
    return out;
}

#endif  // _WIN32

}  // namespace ft::glue
