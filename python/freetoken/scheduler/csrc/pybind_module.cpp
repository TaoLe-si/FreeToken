// python/freetoken/scheduler/csrc/pybind_module.cpp
#include <torch/extension.h>
#include "sched_index.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FreeToken scheduler hot-path C++ glue (P0 of rewrite)";
    m.attr("__version__") = "0.1.0";

    m.def("make_input", &ft::sched::make_input,
          pybind11::arg("table_idx_per_req"),
          pybind11::arg("extend_len_per_req"),
          pybind11::arg("cached_len_per_req"),
          pybind11::arg("device"),
          "Build (table_idx, position) device pair from per-req fields.");

    m.def("make_write", &ft::sched::make_write,
          pybind11::arg("table_idx_per_req"),
          pybind11::arg("device_len_per_req"),
          pybind11::arg("device"),
          "Build (table_idx, device_len) device pair.");

    m.def("make_positions", &ft::sched::make_positions,
          pybind11::arg("extend_len_per_req"),
          pybind11::arg("cached_len_per_req"),
          "Build positions host pinned tensor. Returns (host, host); caller does .to(device).");

    m.def("build_decode_fla_meta", &ft::sched::build_decode_fla_meta,
          pybind11::arg("bs"),
          pybind11::arg("linear_table_idx_dev"),
          pybind11::arg("device"),
          "Decode: cu_seqlens=arange(bs+1) int32, cache_indices=linear_table_idx.");

    m.def("build_prefill_fla_meta", &ft::sched::build_prefill_fla_meta,
          pybind11::arg("extend_len_per_req"),
          pybind11::arg("cached_len_per_req"),
          pybind11::arg("linear_slot_per_req"),
          pybind11::arg("table_idx_per_req"),
          "Prefill: cu_seqlens cumsum + cache_indices + has_initial_state + fresh.");

    m.def("restore_linear_states", &ft::sched::restore_linear_states,
          pybind11::arg("conv_states"),
          pybind11::arg("recurrent_states"),
          pybind11::arg("src_slots"),
          pybind11::arg("dst_slots"),
          "Restore GDN conv + recurrent state from src slots into dst slots in place.");

    m.def("accept_count", &ft::sched::accept_count,
          pybind11::arg("draft_ids"),
          pybind11::arg("verify_ids"),
          pybind11::arg("base"),
          "How many drafts match verify in sequence (stop at first mismatch).");

    m.def("gpu_int_to_cpu_list", &ft::sched::gpu_int_to_cpu_list,
          pybind11::arg("gpu_tensor"),
          "D2H sync int32/int64 GPU tensor -> std::vector<int64_t>. One sync, no PyList.");

    

    m.def("build_linear_table_idx_decode_hybrid", &ft::sched::build_linear_table_idx_decode_hybrid,
          pybind11::arg("linear_slot_idx_per_req"),
          pybind11::arg("padding_slot"),
          pybind11::arg("device"),
          "Decode path linear_table_idx for hybrid cache (one int32 device tensor).");

    m.def("build_mtp_verify_meta", &ft::sched::build_mtp_verify_meta,
          pybind11::arg("snap_slots_host"),
          pybind11::arg("verify_rows"),
          pybind11::arg("device"),
          "Build MTP verify batch meta tensors. Returns [snap_slots_dev, cu_seqlens_varlen_dev, has_init_dev, host_snap_slots_pinned].");

        m.def("prepare_decode_input_indices", &ft::sched::prepare_decode_input_indices,
          pybind11::arg("table_idx_per_req"),
          pybind11::arg("extend_len_per_req"),
          pybind11::arg("cached_len_per_req"),
          pybind11::arg("device_len_per_req"),
          pybind11::arg("linear_slot_idx_per_req"),
          pybind11::arg("padding_slot"),
          pybind11::arg("device"),
          "One-stop per-decode-step tensor builder. Returns 5 tensors.");

m.def("view_to_device", &ft::sched::view_to_device,
          pybind11::arg("mapping_host"),
          pybind11::arg("write_host"),
          pybind11::arg("device"),
          "Move two pinned-host int64 tensors to device (non-blocking).");

    m.def("write_tokens", &ft::sched::write_tokens,
          pybind11::arg("token_pool"),
          pybind11::arg("table_idx_dev"),
          pybind11::arg("write_idx_dev"),
          pybind11::arg("next_tokens_gpu"),
          "Scatter next_tokens into token_pool.");
}