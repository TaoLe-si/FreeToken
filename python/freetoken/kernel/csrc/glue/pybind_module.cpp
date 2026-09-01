// python/freetoken/kernel/csrc/glue/pybind_module.cpp
// FreeToken C++ glue module entry point (pybind11).
//
// Registers: ft.glue.IgpuService (Windows-only D3D12 service bridge; non-Windows
// raises at call time) and ft.glue.MtpHead (Phase 2.5 real impl via Python
// delegation -- forward_with_state is now functional, no longer a stub).
#include <torch/extension.h>
#include "igpu_service.h"
#include "mtp_head.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FreeToken C++ glue: iGPU D3D12 service + MTP head (Phase 2.5)";
    m.attr("__version__") = "0.4.0";

    auto igpu_submodule = m.def_submodule("igpu", "iGPU D3D12 MXFP4 GEMV service bridge");
    pybind11::class_<ft::glue::IgpuService>(igpu_submodule, "IgpuService",
        "C++ bridge to the iGPU D3D12 MXFP4 GEMV server subprocess (Windows). "
        "Non-Windows: throws runtime_error on every call.")
        .def(pybind11::init<const std::string&, int, int, int>(),
             pybind11::arg("server_path"),
             pybind11::arg("max_M") = 0,
             pybind11::arg("max_K") = 0,
             pybind11::arg("max_ns") = 0)
        .def("forward_stateless", &ft::glue::IgpuService::forward_stateless,
             "Stateless MXFP4 GEMV: caller passes weights + act + scales + biases each call.")
        .def("fc_call", &ft::glue::IgpuService::fc_call,
             "Sticky MXFP4 GEMV: weights loaded once via update_weight(); only act per call.")
        .def("update_weight", &ft::glue::IgpuService::update_weight,
             "Replace the sticky weight (hot-reload / model swap).")
        .def("close", &ft::glue::IgpuService::close, "Kill subprocess + release pipes.")
        .def("get_log", &ft::glue::IgpuService::get_log, pybind11::arg("last_n") = 64,
             "Tail of server stderr (last_n lines) for diagnostics.")
        .def("send_raw", &ft::glue::IgpuService::send_raw,
             pybind11::arg("line"), pybind11::arg("body") = torch::Tensor(),
             "Send an arbitrary ASCII command line + optional binary body. Used to drive "
             "MOE_LOAD/MOE_FORWARD/ATTN_LOAD_*/ATTN_FORWARD/MTP_LAYER commands against the "
             "MoE/attn fused iGPU servers. Non-Windows: throws.")
        .def("recv_raw", &ft::glue::IgpuService::recv_raw, pybind11::arg("n"),
             "Read n raw bytes from the server. Use after send_raw for binary responses.");

    auto mtp_submodule = m.def_submodule("mtp", "MTP head fused forward (Phase 2.5 real impl)");
    pybind11::class_<ft::glue::MtpHead>(mtp_submodule, "MtpHead",
        "Single-step fused MTP head forward (C++). Real implementation: delegates to "
        "the Python Qwen3_5MtpHead instance registered via set_forward_callback. "
        "The Python side runs the production path (PyTorch SDPA + iGPU FC + G.3 graph). "
        "Future MTP_LAYER iGPU command can replace the delegation transparently without "
        "changing the API.")
        .def(pybind11::init<pybind11::dict>(),
             pybind11::arg("cfg"),
             "Construct from a Python dict-shaped config (hidden_size, vocab_size, "
             "head_dim, num_qo_heads, num_kv_heads, num_experts, num_experts_per_tok, "
             "moe_intermediate).")
        .def("set_lm_head_callback", &ft::glue::MtpHead::set_lm_head_callback,
             pybind11::arg("cb"),
             "Register a Python callable (hidden) -> logits for the LM head (stays on dGPU).")
        .def("set_forward_callback", &ft::glue::MtpHead::set_forward_callback,
             pybind11::arg("cb"),
             "Register the Python Qwen3_5MtpHead.forward_with_state callable. "
             "Required before the first forward_with_state() call.")
        .def("set_extend_context_callback", &ft::glue::MtpHead::set_extend_context_callback,
             pybind11::arg("cb"),
             "Register the Python MtpHeadAttention.append_rows callable. "
             "Required before extend_context() calls.")
        .def("forward_with_state", &ft::glue::MtpHead::forward_with_state,
             pybind11::arg("prev_token_id"),
             pybind11::arg("prev_hidden"),
             pybind11::arg("position"),
             "Single-step fused forward: returns (logits [1, V] bf16, state [1, H] bf16). "
             "Delegates to the registered Python forward callback.")
        .def("extend_context", &ft::glue::MtpHead::extend_context,
             "Append context rows to the draft KV cache. Delegates to the registered "
             "Python extend_context callback.")
        .def("truncate_kv", &ft::glue::MtpHead::truncate_kv,
             pybind11::arg("n"),
             "Keep only the first n rows (post-verify rollback).")
        .def("reset_draft_cache", &ft::glue::MtpHead::reset_draft_cache,
             "Drop the draft KV cache entirely.")
        .def("kv_len", &ft::glue::MtpHead::kv_len,
             "Current number of rows in the draft KV cache.");
}
