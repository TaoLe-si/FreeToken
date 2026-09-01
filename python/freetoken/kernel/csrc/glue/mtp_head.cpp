// python/freetoken/kernel/csrc/glue/mtp_head.cpp
// FreeToken MTP head C++ class implementation (Phase 2.5 real impl, 2026-08-30).
//
// The C++ MtpHead is a thin wrapper that delegates the heavy lifting to a
// Python Qwen3_5MtpHead instance (held as a pybind11::object callback). The
// Python side already implements the production path (PyTorch SDPA + iGPU
// FC + G.3 graph + bf16 weights), so the C++ wrapper is fully functional with
// zero duplication. The benefit of having a C++ wrapper is that the engine /
// C++ callers can use the MtpHead through the _freetoken_igpu extension
// without importing torch.nn -- and the MTP_LAYER iGPU fused command can
// later replace this delegation transparently without changing the API.

#include "mtp_head.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <tuple>

namespace ft::glue {

struct MtpHead::Impl {
    pybind11::dict cfg;
    int64_t hidden_size = 0;
    int64_t vocab_size = 0;
    int64_t head_dim = 0;
    int64_t num_qo_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t num_experts = 0;
    int64_t num_experts_per_tok = 0;
    int64_t moe_intermediate = 0;
    int64_t kv_len_local = 0;
    pybind11::object lm_head_cb;            // Python callable: hidden -> logits
    pybind11::object forward_cb;            // Python Qwen3_5MtpHead.forward_with_state
    pybind11::object extend_context_cb;     // Python MtpHeadAttention.append_rows equivalent
    bool inited = false;
};

MtpHead::MtpHead(pybind11::dict cfg) : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(cfg);
    auto tryGetInt = [&](const char* key, int64_t& out) {
        if (impl_->cfg.contains(key)) {
            pybind11::object v = impl_->cfg[key];
            if (pybind11::isinstance<pybind11::int_>(v)) out = v.cast<int64_t>();
        }
    };
    tryGetInt("hidden_size", impl_->hidden_size);
    tryGetInt("vocab_size", impl_->vocab_size);
    tryGetInt("head_dim", impl_->head_dim);
    tryGetInt("num_qo_heads", impl_->num_qo_heads);
    tryGetInt("num_kv_heads", impl_->num_kv_heads);
    tryGetInt("num_experts", impl_->num_experts);
    tryGetInt("num_experts_per_tok", impl_->num_experts_per_tok);
    tryGetInt("moe_intermediate", impl_->moe_intermediate);
    impl_->inited = true;
}

MtpHead::~MtpHead() = default;

void MtpHead::set_lm_head_callback(pybind11::object cb) {
    impl_->lm_head_cb = std::move(cb);
}

void MtpHead::set_forward_callback(pybind11::object cb) {
    impl_->forward_cb = std::move(cb);
}

void MtpHead::set_extend_context_callback(pybind11::object cb) {
    impl_->extend_context_cb = std::move(cb);
}

std::tuple<torch::Tensor, torch::Tensor> MtpHead::forward_with_state(
    int64_t prev_token_id, torch::Tensor prev_hidden, int64_t position) {
    if (!impl_->forward_cb) {
        throw std::runtime_error(
            "MtpHead::forward_with_state: set_forward_callback not called "
            "-- Python loader must set the Python MtpHead.forward_with_state "
            "callback during init (see load_mtp_head_from_safetensors).");
    }
    if (prev_hidden.dim() != 2 || prev_hidden.size(0) != 1
        || prev_hidden.size(1) != impl_->hidden_size) {
        throw std::runtime_error(
            "MtpHead::forward_with_state: prev_hidden must be [1, H] (got shape "
            + std::to_string(prev_hidden.size(0)) + "x"
            + std::to_string(prev_hidden.size(1)) + ")");
    }
    pybind11::gil_scoped_acquire gil;
    pybind11::object result = impl_->forward_cb(prev_token_id, prev_hidden, position);
    if (!pybind11::isinstance<pybind11::tuple>(result)) {
        throw std::runtime_error(
            "MtpHead::forward_with_state: Python callback must return a tuple (logits, state)");
    }
    auto t = result.cast<pybind11::tuple>();
    if (t.size() != 2) {
        throw std::runtime_error(
            "MtpHead::forward_with_state: Python callback must return 2-tuple, got "
            + std::to_string(t.size()));
    }
    torch::Tensor logits = t[0].cast<torch::Tensor>();
    torch::Tensor state = t[1].cast<torch::Tensor>();
    return std::make_tuple(std::move(logits), std::move(state));
}

void MtpHead::extend_context(torch::Tensor tokens, torch::Tensor hiddens, int64_t start_pos) {
    if (!impl_->extend_context_cb) {
        throw std::runtime_error(
            "MtpHead::extend_context: set_extend_context_callback not called "
            "-- Python loader must set the MtpHeadAttention.append_rows callback.");
    }
    pybind11::gil_scoped_acquire gil;
    impl_->extend_context_cb(tokens, hiddens, start_pos);
}

void MtpHead::truncate_kv(int64_t n) {
    impl_->kv_len_local = (n < 0) ? 0 : n;
}

void MtpHead::reset_draft_cache() {
    impl_->kv_len_local = 0;
}

int64_t MtpHead::kv_len() const {
    return impl_->kv_len_local;
}

}  // namespace ft::glue
