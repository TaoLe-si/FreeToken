// python/freetoken/scheduler/csrc/sched_index.cpp
#include "sched_index.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <stdexcept>

namespace ft::sched {

static inline int64_t _sum(const std::vector<int64_t>& v) {
    int64_t s = 0;
    for (auto x : v) s += x;
    return s;
}

std::pair<torch::Tensor, torch::Tensor> make_input(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    torch::Device device) {
    TORCH_CHECK(table_idx_per_req.size() == extend_len_per_req.size(),
                "table_idx and extend_len must have the same length");
    TORCH_CHECK(extend_len_per_req.size() == cached_len_per_req.size(),
                "extend_len and cached_len must have the same length");
    const int64_t total = _sum(extend_len_per_req);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto mapping_host = torch::empty({total}, opts_i64);
    auto positions_host = torch::empty({total}, opts_i64);
    int64_t offset = 0;
    for (size_t i = 0; i < table_idx_per_req.size(); ++i) {
        const int64_t tbl = table_idx_per_req[i];
        const int64_t len = extend_len_per_req[i];
        const int64_t cached = cached_len_per_req[i];
        if (len > 0) {
            mapping_host.slice(0, offset, offset + len).fill_(tbl);
            auto pos_view = positions_host.slice(0, offset, offset + len);
            for (int64_t k = 0; k < len; ++k) {
                pos_view[k] = cached + k;
            }
        }
        offset += len;
    }
    auto mapping_dev = mapping_host.to(device, /*non_blocking=*/true);
    auto positions_dev = positions_host.to(device, /*non_blocking=*/true);
    return {mapping_dev, positions_dev};
}

std::pair<torch::Tensor, torch::Tensor> make_write(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& device_len_per_req,
    torch::Device device) {
    TORCH_CHECK(table_idx_per_req.size() == device_len_per_req.size(),
                "table_idx and device_len must have the same length");
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto mapping_host = torch::empty({(int64_t)table_idx_per_req.size()}, opts_i64);
    auto write_host = torch::empty({(int64_t)device_len_per_req.size()}, opts_i64);
    std::memcpy(mapping_host.data_ptr(), table_idx_per_req.data(),
                table_idx_per_req.size() * sizeof(int64_t));
    std::memcpy(write_host.data_ptr(), device_len_per_req.data(),
                device_len_per_req.size() * sizeof(int64_t));
    auto mapping_dev = mapping_host.to(device, /*non_blocking=*/true);
    auto write_dev = write_host.to(device, /*non_blocking=*/true);
    return {mapping_dev, write_dev};
}

std::pair<torch::Tensor, torch::Tensor> make_positions(
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req) {
    TORCH_CHECK(extend_len_per_req.size() == cached_len_per_req.size(),
                "extend_len and cached_len must have the same length");
    const int64_t total = _sum(extend_len_per_req);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto positions_host = torch::empty({total}, opts_i32);
    int64_t offset = 0;
    for (size_t i = 0; i < extend_len_per_req.size(); ++i) {
        const int64_t len = extend_len_per_req[i];
        const int64_t cached = cached_len_per_req[i];
        if (len > 0) {
            auto v = positions_host.slice(0, offset, offset + len);
            for (int64_t k = 0; k < len; ++k) {
                v[k] = static_cast<int32_t>(cached + k);
            }
        }
        offset += len;
    }
    return {positions_host, positions_host};
}

std::pair<torch::Tensor, torch::Tensor> build_decode_fla_meta(
    int64_t bs,
    torch::Tensor linear_table_idx_dev,
    torch::Device device) {
    TORCH_CHECK(bs >= 0, "bs must be non-negative");
    TORCH_CHECK(linear_table_idx_dev.scalar_type() == torch::kInt32,
                "linear_table_idx_dev must be int32");
    auto cu = torch::arange(bs + 1, torch::TensorOptions().dtype(torch::kInt32).device(device));
    return {cu, linear_table_idx_dev};
}

static inline int64_t _slot_or(const std::vector<int64_t>& linear_slot,
                              const std::vector<int64_t>& table_idx,
                              size_t i) {
    return linear_slot[i] >= 0 ? linear_slot[i] : table_idx[i];
}

std::vector<torch::Tensor> build_prefill_fla_meta(
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    const std::vector<int64_t>& linear_slot_per_req,
    const std::vector<int64_t>& table_idx_per_req) {
    const size_t N = extend_len_per_req.size();
    TORCH_CHECK(cached_len_per_req.size() == N, "cached_len size mismatch");
    TORCH_CHECK(linear_slot_per_req.size() == N, "linear_slot size mismatch");
    TORCH_CHECK(table_idx_per_req.size() == N, "table_idx size mismatch");
    // cu_seqlens: cumsum of extend_len, len N+1.
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto opts_b   = torch::TensorOptions().dtype(torch::kBool).pinned_memory(true);
    auto cu_host = torch::empty({(int64_t)(N + 1)}, opts_i64);
    cu_host[0] = 0;
    for (size_t i = 0; i < N; ++i) {
        cu_host[(int64_t)(i + 1)] = cu_host[(int64_t)i] + extend_len_per_req[i];
    }
    // cache_indices: gdn_slot per req (linear_slot if >=0 else table_idx).
    auto idx_host = torch::empty({(int64_t)N}, opts_i32);
    auto has_init_host = torch::empty({(int64_t)N}, opts_b);
    // fresh_state_indices: cached_len == 0 -> linear_slot (or table_idx if linear_slot None).
    std::vector<int64_t> fresh;
    fresh.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        idx_host[(int64_t)i] = (int32_t)_slot_or(linear_slot_per_req, table_idx_per_req, i);
        has_init_host[(int64_t)i] = (cached_len_per_req[i] > 0);
        if (cached_len_per_req[i] == 0) {
            fresh.push_back(_slot_or(linear_slot_per_req, table_idx_per_req, i));
        }
    }
    auto fresh_host = fresh.empty()
        ? torch::empty({0}, opts_i64)
        : torch::from_blob(fresh.data(), {(int64_t)fresh.size()}, opts_i64).clone();
    return {cu_host, idx_host, has_init_host, fresh_host};
}

void restore_linear_states(
    torch::Tensor conv_states,
    torch::Tensor recurrent_states,
    const std::vector<int64_t>& src_slots,
    const std::vector<int64_t>& dst_slots) {
    TORCH_CHECK(src_slots.size() == dst_slots.size(), "src/dst slot count mismatch");
    for (size_t i = 0; i < src_slots.size(); ++i) {
        const int64_t src = src_slots[i];
        const int64_t dst = dst_slots[i];
        if (src == dst) continue;  // no-op when already the same slot
        conv_states.select(1, dst).copy_(conv_states.select(1, src));
        recurrent_states.select(1, dst).copy_(recurrent_states.select(1, src));
    }
}

int64_t accept_count(
    const std::vector<int64_t>& draft_ids,
    const std::vector<int64_t>& verify_ids,
    int64_t base) {
    const size_t N = draft_ids.size();
    int64_t n = 0;
    for (size_t i = 0; i < N; ++i) {
        const size_t idx = base + i;
        if (idx >= verify_ids.size()) break;
        if (verify_ids[idx] != draft_ids[i]) break;
        ++n;
    }
    return n;
}

std::vector<int64_t> gpu_int_to_cpu_list(torch::Tensor gpu_tensor) {
    // The Python .tolist() on a tensor auto-handles both int32 and int64 and uses
    // a single D2H sync. We forward the cast via .to(int64) for uniform output, then
    // use the C++ accessor to pull values without another Python layer.
    auto cpu_t = gpu_tensor.to(torch::kCPU).contiguous();
    const int64_t N = cpu_t.numel();
    std::vector<int64_t> out;
    out.reserve(N);
    if (cpu_t.scalar_type() == torch::kInt32) {
        const int32_t* p = cpu_t.data_ptr<int32_t>();
        for (int64_t i = 0; i < N; ++i) out.push_back(static_cast<int64_t>(p[i]));
    } else if (cpu_t.scalar_type() == torch::kInt64) {
        const int64_t* p = cpu_t.data_ptr<int64_t>();
        for (int64_t i = 0; i < N; ++i) out.push_back(p[i]);
    auto pt_opts = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
        TORCH_CHECK(false, "gpu_int_to_cpu_list: dtype must be int32 or int64");
    }
    return out;
}

// Return 6 tensors (max_seqlen_k / max_seqlen_q packed as size-1 int64 tensors)
// because pybind11 std::tuple with mixed tensor+int fails to convert.
torch::Tensor build_linear_table_idx_decode_hybrid(
    const std::vector<int64_t>& linear_slot_idx_per_req,
    int64_t padding_slot,
    torch::Device device) {
    const int64_t N = (int64_t)linear_slot_idx_per_req.size();
    auto opts_pin = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto host = torch::empty({N}, opts_pin);
    for (int64_t i = 0; i < N; ++i) {
        host[i] = (int32_t)(linear_slot_idx_per_req[i] >= 0 ? linear_slot_idx_per_req[i] : padding_slot);
    }
    return host.to(device, /*non_blocking=*/true);
}

std::vector<torch::Tensor> build_mtp_verify_meta(
    const std::vector<int64_t>& snap_slots_host,
    int64_t verify_rows,
    torch::Device device) {
    const int64_t N = (int64_t)snap_slots_host.size();
    // snap_slots int64 device
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64);
    auto opts_pin_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto opts_pin_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto opts_b = torch::TensorOptions().dtype(torch::kBool);
    torch::Tensor snap_slots_dev;
    torch::Tensor host_snap_slots_pinned;
    if (N > 0) {
        auto host_i64 = torch::empty({N}, opts_pin_i64);
        for (int64_t i = 0; i < N; ++i) host_i64[i] = snap_slots_host[i];
        snap_slots_dev = host_i64.to(device, /*non_blocking=*/true);
        host_snap_slots_pinned = torch::empty({N}, opts_pin_i32);
        for (int64_t i = 0; i < N; ++i) host_snap_slots_pinned[i] = (int32_t)snap_slots_host[i];
    } else {
        snap_slots_dev = torch::empty({0}, opts_i64.device(device));
        host_snap_slots_pinned = torch::empty({0}, opts_pin_i32);
    }
    // cu_seqlens_varlen int32 device, len 2, [0, verify_rows]
    auto cu_seqlens_varlen_dev = torch::tensor({0, (int32_t)verify_rows}, opts_i32.device(device));
    // has_initial_state bool device, len 1, value True
    auto has_init_dev = torch::tensor({true}, opts_b.device(device));
    return {snap_slots_dev, cu_seqlens_varlen_dev, has_init_dev, host_snap_slots_pinned};
}

std::vector<torch::Tensor> prepare_decode_input_indices(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    const std::vector<int64_t>& device_len_per_req,
    const std::vector<int64_t>& linear_slot_idx_per_req,
    int64_t padding_slot,
    torch::Device device) {
    const int64_t N = (int64_t)table_idx_per_req.size();
    TORCH_CHECK((int64_t)extend_len_per_req.size() == N, "extend_len size mismatch");
    TORCH_CHECK((int64_t)cached_len_per_req.size() == N, "cached_len size mismatch");
    TORCH_CHECK((int64_t)device_len_per_req.size() == N, "device_len size mismatch");
    TORCH_CHECK((int64_t)linear_slot_idx_per_req.size() == N, "linear_slot size mismatch");
    auto opts_pin_i32 = torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true);
    auto opts_pin_i64 = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64);
    // ---- make_input ----
    int64_t total_extend = 0;
    for (auto v : extend_len_per_req) total_extend += v;
    auto mapping_host = torch::empty({total_extend}, opts_pin_i64);
    auto positions_host = torch::empty({total_extend}, opts_pin_i32);
    int64_t off = 0;
    for (int64_t i = 0; i < N; ++i) {
        const int64_t len = extend_len_per_req[i];
        const int64_t cached = cached_len_per_req[i];
        if (len > 0) {
            for (int64_t k = 0; k < len; ++k) {
                mapping_host[off + k] = table_idx_per_req[i];
                positions_host[off + k] = (int32_t)(cached + k);
            }
        }
        off += len;
    }
    auto mapping_dev = mapping_host.to(device, /*non_blocking=*/true);
    auto positions_dev = positions_host.to(device, /*non_blocking=*/true);
    // ---- make_write ----
    auto write_host = torch::empty({N}, opts_pin_i64);
    for (int64_t i = 0; i < N; ++i) {
        // can_decode is implicit when device_len_per_req > cached_len (decode path).
        // Caller passes device_len_per_req = -1 for non-decodable reqs (e.g. ChunkedReq).
        write_host[i] = device_len_per_req[i];
    }
    auto write_dev = write_host.to(device, /*non_blocking=*/true);
    // ---- linear_table_idx hybrid ----
    auto lt_host = torch::empty({N}, opts_pin_i32);
    for (int64_t i = 0; i < N; ++i) {
        lt_host[i] = (int32_t)(linear_slot_idx_per_req[i] >= 0
                              ? linear_slot_idx_per_req[i]
                              : padding_slot);
    }
    auto lt_dev = lt_host.to(device, /*non_blocking=*/true);
    return {mapping_dev, positions_dev, write_dev, lt_dev, positions_host};
    // NOTE: positions_host returned so caller can reuse for a future out-of-loop read.
}

std::pair<torch::Tensor, torch::Tensor> view_to_device(
    torch::Tensor mapping_host,
    torch::Tensor write_host,
    torch::Device device) {
    auto mapping_dev = mapping_host.to(device, /*non_blocking=*/true);
    auto write_dev = write_host.to(device, /*non_blocking=*/true);
    return {mapping_dev, write_dev};
}

void write_tokens(
    torch::Tensor token_pool,
    torch::Tensor table_idx_dev,
    torch::Tensor write_idx_dev,
    torch::Tensor next_tokens_gpu) {
    TORCH_CHECK(token_pool.scalar_type() == torch::kInt32, "token_pool must be int32");
    TORCH_CHECK(table_idx_dev.scalar_type() == torch::kInt64, "table_idx_dev must be int64");
    TORCH_CHECK(write_idx_dev.scalar_type() == torch::kInt64, "write_idx_dev must be int64");
    const int64_t N = table_idx_dev.numel();
    TORCH_CHECK(write_idx_dev.numel() == N, "table_idx_dev and write_idx_dev must have the same length");
    TORCH_CHECK(next_tokens_gpu.numel() == N || next_tokens_gpu.numel() == 1,
                "next_tokens_gpu must have length N or 1, got ", next_tokens_gpu.numel(), " vs N=", N);
    std::vector<at::indexing::TensorIndex> indices = {table_idx_dev, write_idx_dev};
    auto flat_next = next_tokens_gpu.reshape({-1}).to(token_pool.device(), /*non_blocking=*/true);
    token_pool.index_put_(indices, flat_next);
}

}  // namespace ft::sched