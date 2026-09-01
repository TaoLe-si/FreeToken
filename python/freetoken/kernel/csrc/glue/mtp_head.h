// python/freetoken/kernel/csrc/glue/mtp_head.h
// FreeToken MTP head C++ class (Phase 2.5, real impl via Python delegation, 2026-08-30).
//
// Single-step fused MTP head forward: the C++ MtpHead class is a thin wrapper
// that delegates the actual forward computation to a Python Qwen3_5MtpHead
// instance (held as a pybind11::object callback). The Python side already
// implements the production path (PyTorch SDPA + iGPU FC + G.3 graph + bf16
// weights), so the C++ wrapper is fully functional with zero duplication.
//
// This removes the Phase 2.5 P0 stub -- forward_with_state now actually runs
// (via Python delegation) rather than throwing.
#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>

namespace ft::glue {

class MtpHead {
public:
    // cfg: ModelConfig snapshot (Python dict with hidden_size, num_qo_heads,
    // num_kv_heads, head_dim, vocab_size, num_experts, num_experts_per_tok,
    // moe_intermediate, etc). pybind11::dict auto-converts from a Python dict.
    MtpHead(pybind11::dict cfg);
    ~MtpHead();

    MtpHead(const MtpHead&) = delete;
    MtpHead& operator=(const MtpHead&) = delete;

    // Single-step forward: take prev_token_id, prev_hidden, position.
    // Returns (logits [1, V] bf16, state [1, H] bf16). Delegates to Python
    // Qwen3_5MtpHead.forward_with_state (set via set_forward_callback).
    std::tuple<torch::Tensor, torch::Tensor> forward_with_state(
        int64_t prev_token_id,
        torch::Tensor prev_hidden,
        int64_t position);

    // KV cache management (mirrors MtpHeadAttention in Python).
    void extend_context(torch::Tensor tokens, torch::Tensor hiddens, int64_t start_pos);
    void truncate_kv(int64_t n);
    void reset_draft_cache();
    int64_t kv_len() const;

    // Setters for the LM head (callback to dGPU logits) and the Python
    // forward_with_state delegation. Both are set by the Python loader.
    void set_lm_head_callback(pybind11::object cb);
    void set_forward_callback(pybind11::object cb);
    void set_extend_context_callback(pybind11::object cb);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ft::glue
