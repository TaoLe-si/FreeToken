// python/freetoken/scheduler/csrc/sched_index.h
// P0 of the Python->C++ rewrite: tensor index construction + token_pool write-back.
#pragma once
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <utility>
namespace ft::sched {
std::pair<torch::Tensor, torch::Tensor> make_input(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    torch::Device device);
std::pair<torch::Tensor, torch::Tensor> make_write(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& device_len_per_req,
    torch::Device device);
void write_tokens(
    torch::Tensor token_pool,
    torch::Tensor table_idx_dev,
    torch::Tensor write_idx_dev,
    torch::Tensor next_tokens_gpu);
// Build positions (cached_len .. cached_len+extend_len) per request, packed into one
// host pinned tensor. Mirrors scheduler._make_positions which previously ran a
// Python for-loop calling torch.arange with `out=` -- a per-step hot path.
// Returns (positions_host, positions_host) -- caller is expected to .to(device).
std::pair<torch::Tensor, torch::Tensor> make_positions(
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req);

// Decode path: cu_seqlens = arange(bs+1) int32, cache_indices = batch.linear_table_idx int32.
// Returns (cu_seqlens_dev, cache_indices_dev). Single C++ call replaces 2 Python
// list comprehensions + 2 .to(device, non_blocking=True) per decode step.
std::pair<torch::Tensor, torch::Tensor> build_decode_fla_meta(
    int64_t bs,
    torch::Tensor linear_table_idx_dev,
    torch::Device device);

// Prefill path: builds cu_seqlens (cumsum of extend_len), cache_indices (gdn_slot),
// has_initial_state (cached_len>0), fresh_state_indices (cached_len==0). All host pinned,
// returned as host tensors; caller does .to(device). The Python list comprehensions
// over reqs are moved into C++.
// Prefill path: builds cu_seqlens (cumsum of extend_len), cache_indices (gdn_slot),
// has_initial_state (cached_len>0), fresh_state_indices (cached_len==0). All host pinned,
// returned as host tensors; caller does .to(device). The Python list comprehensions
// over reqs are moved into C++. Returns a tuple (cu_seqlens_host, cache_indices_host,
// has_initial_state_host, fresh_state_indices_host) so we do not need to bind a struct.
std::vector<torch::Tensor> build_prefill_fla_meta(
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    const std::vector<int64_t>& linear_slot_per_req,
    const std::vector<int64_t>& table_idx_per_req);

// Restore GDN (conv + recurrent) state from a list of snapshot slots into a list of
// live slots on the engine stream. Replaces the Python for-loop in
// scheduler._restore_linear_states; each pair is a single direct copy of two tensors.
void restore_linear_states(
    torch::Tensor conv_states,           // [n_layers, num_slots, ...]
    torch::Tensor recurrent_states,      // [n_layers, num_slots, ...]
    const std::vector<int64_t>& src_slots,
    const std::vector<int64_t>& dst_slots);

// MTP accept count: how many drafts match the verify logits in sequence.
// Replaces MtpDriver.accept_count Python for-loop (called per decode round).
int64_t accept_count(
    const std::vector<int64_t>& draft_ids,
    const std::vector<int64_t>& verify_ids,
    int64_t base);

// One D2H sync that materialises an int32/int64 GPU tensor into a Python list<int64_t>.
// Replaces `tensor.cpu().tolist()` patterns scattered through the hot path. The .tolist()
// itself already produces a list of Python ints, so wrapping the GPU tensor in a single
// C++ -> pybind cast lets us skip the intermediate Python list builder overhead.
std::vector<int64_t> gpu_int_to_cpu_list(torch::Tensor gpu_tensor);

// Per-step attention metadata builder (FA backend path). Replaces the Python list
// comprehensions + 4 torch.tensor/.cumsum_/.stack ops + 3 .to(device) calls in
// attention/fa.py:prepare_metadata. All work consolidates into one C++ call.
// Inputs (Python lists) -> outputs (host + device tensors packed into the return tuple):
//   returns {cu_seqlens_k_dev, cu_seqlens_q_dev, cache_seqlens_dev, page_table_dev,
//            max_seqlen_k, max_seqlen_q}
struct FAMetaOut {
    torch::Tensor cu_seqlens_k;       // int32 [bs+1] device
    torch::Tensor cu_seqlens_q;       // int32 [bs+1] device
    torch::Tensor cache_seqlens;      // int32 [bs]   device
    torch::Tensor page_table;         // int32 [bs, page_indices] device
    int64_t max_seqlen_k;
    int64_t max_seqlen_q;
};
// fa_prepare_metadata returns a 6-tuple: (cu_k, cu_q, cache_seqlens, page_table,
// max_seqlen_k, max_seqlen_q) -- packed into a tuple to avoid binding a struct.
// Build batch.linear_table_idx for the decode path (per-step hot path).
// Hybrid mode: linear_slot_idx if set, else padding_slot. Naive mode: input_mapping[0]
// (the table_idx column from the C++ make_input). Replaces two Python list comprehensions
// + a torch.tensor alloc + an H2D copy.
torch::Tensor build_linear_table_idx_decode_hybrid(
    const std::vector<int64_t>& linear_slot_idx_per_req,
    int64_t padding_slot,
    torch::Device device);

// MTP verify batch meta tensors. Per verify-batch construction in scheduler._prepare_batch.
// Replaces 4 small torch.tensor() calls in a Python for-loop with one C++ call.
// Returns 4 tensors: snap_slots (int64 device), cu_seqlens_varlen (int32 device,
//   len=2, values [0, verify_rows]), has_initial_state (bool device, len=1),
//   host_snap_slots (int32 pinned, len=N) for stable ownership.
std::vector<torch::Tensor> build_mtp_verify_meta(
    const std::vector<int64_t>& snap_slots_host,
    int64_t verify_rows,
    torch::Device device);

// One-stop per-decode-step tensor builder. Bundles 4 hot operations that were previously
// 4 separate Python calls into one C++ call:
//   1. make_input  (input_mapping_dev, positions_dev)
//   2. make_write  (write_mapping_dev)
//   3. make_positions (positions_host)
//   4. linear_table_idx hybrid (linear_table_idx_dev)
// All from a single batch of (table_idx, extend_len, cached_len, linear_slot_idx, padding_slot).
// Returns 5 tensors in a vector.
std::vector<torch::Tensor> prepare_decode_input_indices(
    const std::vector<int64_t>& table_idx_per_req,
    const std::vector<int64_t>& extend_len_per_req,
    const std::vector<int64_t>& cached_len_per_req,
    const std::vector<int64_t>& device_len_per_req,
    const std::vector<int64_t>& linear_slot_idx_per_req,
    int64_t padding_slot,
    torch::Device device);

std::pair<torch::Tensor, torch::Tensor> view_to_device(
    torch::Tensor mapping_host,
    torch::Tensor write_host,
    torch::Device device);
}  // namespace ft::sched