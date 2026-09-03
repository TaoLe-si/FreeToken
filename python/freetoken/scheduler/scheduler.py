from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import os
import time as _time

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"




# Module-level export: the scheduler hot path is C++-only (P0 of the rewrite).
# If the .pyd is missing, surface a clear build error pointing at the rebuild step.
try:
    from freetoken.scheduler import _freetoken_sched as _sched_cpp
    _SCHED_CPP_OK = True
except ImportError as _sched_import_err:
    raise RuntimeError(
        "freetoken.scheduler._freetoken_sched.pyd missing -- the scheduler hot path is C++-only.\n"
        "Rebuild with E:\\FreeToken\\_build_sched.bat\n"
        "Underlying error: " + repr(_sched_import_err)
    ) from _sched_import_err

class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # MTP (Multi-Token Prediction) speculative decoding. The driver owns the Qwen3.5/3.6 MTP
        # head + an iGPU MXFP4 FC sticky when ``--mtp-igpu-fc`` is set; it is a no-op stub when
        # MTP is disabled. A load failure must NOT block serve startup -- the engine keeps serving
        # vanilla decode, and a warning explains why MTP is off.
        self.mtp: object | None = None
        self.mtp_stats: dict[str, int] = {
            "drafted": 0, "accepted_tokens": 0, "verify_calls": 0, "misses": 0,
        }
        if config.mtp:
            try:
                from freetoken.engine.mtp_driver import MtpDriver

                self.mtp = MtpDriver(
                    self.engine,
                    config.model_path,
                    k=config.mtp_k,
                    use_igpu_fc=config.mtp_igpu_fc,
                    fc_backend=("igpu" if config.mtp_igpu_fc else "dgpu"),
                )
                logger.info_rank0(
                    f"MTP enabled: K={config.mtp_k}, igpu_fc={config.mtp_igpu_fc}"
                )
                # Init-time draft warmup: run one dry K-token draft round NOW so every
                # Triton kernel the draft path touches (moe_align_block_size, dequant
                # gathers, bmm autotune ...) compiles here, while free VRAM still
                # exists. At request time the post-graph-capture GPU is full and a
                # lazy cuModuleLoadData goes through the WDDM shared pool (minutes).
                self._mtp_warmup_draft()
            except Exception as exc:  # noqa: BLE001
                logger.warning_rank0(
                    f"MTP driver failed to initialize ({exc!r}); continuing without MTP."
                )
                self.mtp = None

    def _mtp_warmup_draft(self) -> None:
        """One-time init-time MTP draft warmup (never fatal).

        Runs a dry draft round for a dummy uid with a zero prev_hidden so the
        MTP head's Triton/cublas compile cost lands at init, not at the first
        request. commit_round + reset afterwards leave the head KV cache clean.
        """
        if self.mtp is None:
            return
        try:
            t0 = _time.perf_counter()
            hidden = int(getattr(self.mtp.cfg, "hidden_size", 0))
            if hidden <= 0:
                return
            zero_hidden = torch.zeros(1, hidden, dtype=torch.bfloat16, device=self.device)
            with torch.inference_mode():
                torch.cuda.synchronize(self.device)
                self.mtp.draft(-2, 0, zero_hidden, position=0)
                # Reset the head KV: roll back the dry round's rows, then clear
                # the persistent draft cache entirely so the first real request
                # starts from a clean (owner-less) head cache.
                self.mtp.commit_round(-2, None, None)
                self.mtp.head.attn.reset_draft_cache()
                self.mtp._cache_owner_uid = None
                torch.cuda.synchronize(self.device)
            logger.info_rank0(f"[MTP] draft warmup done in {(_time.perf_counter() - t0) * 1e3:.1f} ms")
        except Exception as exc:  # noqa: BLE001 -- warmup must never block serve startup
            logger.warning_rank0(f"[MTP] draft warmup failed (non-fatal): {exc!r}")

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    def _flush_error_replies(self, message: str) -> None:
        """Best-effort terminal error reply for EVERY live request. Called when the
        scheduler loop is about to die so clients get an explicit error instead of
        hanging until their HTTP timeout (the "empty 200" failure mode)."""
        try:
            uids = set()
            for r in getattr(self.prefill_manager, "pending_list", []):
                uids.add(r.uid)
            for r in getattr(self.decode_manager, "running_reqs", []):
                uids.add(r.uid)
            if uids:
                self.send_result([ErrorReplyMsg(uid=u, error=message) for u in sorted(uids)])
        except Exception as exc:  # noqa: BLE001 -- flushing must never mask the crash
            logger.error(f"error-reply flush failed: {exc!r}")

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        try:
            if ENV.DISABLE_OVERLAP_SCHEDULING:
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    while True:
                        self.normal_loop()
            else:
                assert torch.cuda.current_stream() == self.stream
                data = None
                while True:
                    data = self.overlap_loop(data)
        except BaseException:
            # Scheduler is dying: deliver terminal errors to every live client BEFORE
            # the process exits, then re-raise so the supervisor still records the death.
            self._flush_error_replies("scheduler crashed: see server logs")
            raise

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        if self.mtp is not None:
            try:
                self.mtp.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning_rank0(f"MTP driver close failed: {exc!r}")
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done, *_rest) = last_data[0].batch, last_data[1]
        copy_done.synchronize()

        # MTP verify batch: the engine returned per-position argmax (NOT a sample); the accept
        # path runs through the cache_manager (cache_req_to_len) instead of the generic
        # sampler-based reply loop. This branch is taken for one req, one prefill-style forward
        # over K+1 candidates; we publish drafts[1:n] + correction as DetokenizeMsg's and
        # restore normal decode for next iteration.
        if getattr(batch, "mtp_verify", False):
            # Drain point for the in-flight guard: cleared here (not inside
            # _mtp_process_verify) so EVERY exit path -- including an aborted req
            # that frees below -- re-arms the next verify build.
            self._mtp_verify_inflight = None
            self._mtp_process_verify(batch, last_data[1])
            return

        # MTP context seeding: stash per-req prompt hiddens (prefill batches
        # only; verify batches were reconciled by commit_round above).
        if (
            self.mtp is not None
            and batch.is_prefill
            and getattr(last_data[1], "all_hidden", None) is not None
        ):
            self._mtp_stash_prefill_hiddens(batch, last_data[1].all_hidden)

        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        # C++ only -- no Python fallback by user request:
        # materialise the whole next_tokens_cpu as a Python list in ONE C++ call instead
        # of repeated `.item()` syncs per req (the previous code did `int(t.item())` inside
        # the for-loop, which is `bs` separate GPU syncs).
        next_tokens_list = _sched_cpp.gpu_int_to_cpu_list(next_tokens_cpu)
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in self.finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                # C++ only -- no Python fallback by user request: pull the single token
                # from the pre-materialised Python list (no per-req .item() GPU sync).
                next_token_int = next_tokens_list[i]
                req.append_host(next_tokens_cpu[i].unsqueeze(0))
                next_token = next_token_int
                if os.environ.get("FT_MTP_DEBUG"):
                    _pool = getattr(self.engine, "linear_state_pool", None)
                    if _pool is not None:
                        _slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
                        torch.cuda.synchronize()  # state-hash race fix
                        _rec = _pool.recurrent_states[:, _slot].float()
                        _cv = _pool.conv_states[:, _slot].float()
                        logger.info_rank0("[MTP-dbg] state-hash: slot=%s cached=%s rec=%.6f/%.3f cv=%.6f/%.3f",
                                          _slot, req.cached_len,
                                          float(_rec.sum()), float(_rec.abs().max()),
                                          float(_cv.sum()), float(_cv.abs().max()))
                    if os.environ.get("FT_MTP_STATE_LAYERS") and req.cached_len == 23:
                        _per = " ".join("L%d=%.4f/%.4f" % (li,
                            float(_pool.recurrent_states[li, _slot].float().sum()),
                            float(_pool.conv_states[li, _slot].float().sum()))
                            for li in range(_pool.recurrent_states.shape[0]))
                        logger.info_rank0("[MTP-dbg] per-layer: %s", _per)
                logger.info_rank0(
                    "[dbg-reply] phase=%s tok=%d eos=%s can_decode=%s out_len=%d",
                    "prefill" if batch.is_prefill else "decode",
                    next_token, next_token in self.eos_token_ids,
                    req.can_decode, len(req.output_ids) if hasattr(req,"output_ids") else -1,
                )
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                hit_length = not req.can_decode
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # MTP draft hook (decode only). After we shipped the sampled token, if the
                # request is still alive and greedy, ask the MTP head to draft K candidates
                # starting from this token; write them into the request's table_idx + host
                # input_ids so the NEXT scheduling round can push them as one prefill
                # (the verify batch). Greedy-only because drafts and verification must
                # use the same sampling semantics to be safely comparable (EAGLE-style
                # speculation). Hidden state comes from forward_output.prev_hidden[i].
                if (
                    batch.is_decode
                    and not finished
                    and self.mtp is not None
                    and req.sampling_params.is_greedy
                ):
                    prev_hidden_row = (
                        last_data[1].prev_hidden[i]
                        if (last_data is not None and last_data[1].prev_hidden is not None)
                        else None
                    )
                    # Head convention: input (x_p, h_{p-1}) per DeepSeek/Qwen3-Next MTP.
                    # The batch's prev_hidden = h(p) (state AFTER the just-processed
                    # token at position p), so we need h_{p-1}. For round 1 that's
                    # stash[-1] = h(last_prompt); for round N>1 it's the previous
                    # round's batch.prev_hidden (= h(p_{N-1})).
                    # Head convention (x_p, h_{p-1}): drafting from the shipped token
                    # t_i needs h at position i-1 = the hidden AFTER the token this
                    # decode batch just processed = prev_hidden_row of THIS drain.
                    # The old code used the PREVIOUS round's saved hidden (one round
                    # stale) -- every draft was conditioned on the wrong context
                    # hidden and matched nothing (m=0 every round).
                    head_prev_h = prev_hidden_row
                    # --- Head-cache reconcile (draft-quality fix): the head cache
                    # must mirror the main model's committed tokens one row each
                    # (row j pairs (x_j, h_{j-1})). Per round the main context
                    # advances by TWO tokens (the decode-processed u_k and the
                    # just-sampled t), but commit_round only ever appended the
                    # correction row -- the head cache lagged two rows behind, so
                    # every draft attended a context missing its two most recent
                    # tokens and matched nothing (m=0 forever). Backfill the
                    # missing rows here, before drafting:
                    if prev_hidden_row is not None and self.mtp is not None:
                        kv = self.mtp.head.attn.kv_len()
                        miss = req.cached_len - kv
                        if 0 < miss <= 2:
                            h_prev2 = getattr(req, "_mtp_prev_h", None)  # h(cached-3), saved last round
                            ids = req.input_ids
                            toks_bf = []
                            hid_bf = []
                            if miss == 2 and h_prev2 is not None:
                                toks_bf.append(int(ids[req.cached_len - 2]))
                                hid_bf.append(h_prev2.to(self.device).view(1, -1))
                            toks_bf.append(int(ids[req.cached_len - 1]))
                            hid_bf.append(prev_hidden_row.to(self.device).view(1, -1).to(torch.bfloat16))
                            if len(toks_bf) > 0:
                                self.mtp.reconcile(
                                    req.uid,
                                    torch.tensor(toks_bf, dtype=torch.long, device=self.device),
                                    torch.cat(hid_bf, dim=0).to(torch.bfloat16),
                                    kv - 1,
                                )
                    # Save this round's h(p) for the next round's backfill
                    if prev_hidden_row is not None:
                        req._mtp_prev_h = prev_hidden_row
                    if os.environ.get("FT_MTP_DEBUG"):
                        logger.info_rank0(
                            f"[MTP-dbg] hook: ids_numel={req.input_ids.numel()} cached_len={req.cached_len} "
                            f"device_len={req.device_len} next_tok={next_token} head_kv={self.mtp.head.attn.kv_len()}")
                    self._mtp_maybe_draft(req, next_token, head_prev_h, req.cached_len - 1)

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        if reply:
            logger.info_rank0("[dbg-reply] sending %d msgs", len(reply))
        self.send_result(reply)

    def _build_mtp_verify_batch(self) -> Batch | None:
        """Build a single-request prefill batch over the K+1 MTP candidates, or None.

        A drafted req (mtp_verify=True) lives in decode_manager.running_reqs. The
        verify batch REUSES the request's existing table_idx -- the host input_ids
        and the token_pool at that index already hold prompt + t + K drafts staged
        by _mtp_maybe_draft; the verify forward's extend_len = K+1 runs the main
        model on [t, d1..dK] (the prompt prefix is already in KV).

        Hybrid (GDN) models: the linear-attention state is advanced by the verify
        forward, and a partial accept must roll it back. We snapshot the live slot
        BEFORE the forward (on the scheduling stream -- ordered before the engine
        stream's forward via the usual wait_stream) and restore it in
        _mtp_process_verify when drafts are rejected. Ping-pong chunk tracking is
        disabled for this one forward so a mid-verify boundary snapshot of
        rejected drafts can never be donated into the radix tree.
        """
        verify_reqs = [
            r for r in self.decode_manager.running_reqs
            if getattr(r, "mtp_verify", False)
        ]
        if not verify_reqs:
            return None
        req = verify_reqs[0]
        if getattr(self, "_mtp_verify_inflight", None) is not None:
            # Overlap scheduling: the previous verify batch is still in flight and
            # this req's mtp_* scratch reflects ITS staged state (complete_one has
            # already advanced cached_len past the drafts). Building a second
            # verify batch here would forward a garbage suffix. The drain point in
            # _process_last_data clears the marker before the next build.
            return None
        # Hybrid GDN snapshot. K+1 slots: snap[0]=pre-verify (used by reject_all rollback
        # and partial accept n=0), snap[1..K]=per-step state at the end of each verify step
        # (filled by C.5 GDN forward hook during the verify forward; snap[n] is used by
        # partial_accept_hybrid[n] in _mtp_process_verify to roll GDN state back to base+2+n
        # without replay). Mirrors llama.cpp PR #22400's "GDN intermediates" path. snap[K]
        # equals the post-verify state already in live slot, so partial accept n=K needs no
        # rollback. (alloc failure -> skip this verify round entirely; on discard, snap[K+1
        # free list stays clean.)
        req._mtp_gdn_snap = None
        req._mtp_gdn_snap_per_step = None
        pool = getattr(self.engine, "linear_state_pool", None)
        if pool is not None:
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            # K = len(mtp_drafts) -- already staged by _mtp_maybe_draft before this builder.
            _k_snap = len(getattr(req, "mtp_drafts", []) or [])
            try:
                snap_slots = pool.alloc(_k_snap + 1)  # K+1
                pool.copy_from(slot, snap_slots[0])
                req._mtp_gdn_snap = snap_slots[0]            # compat: reject_all uses snap[0]
                req._mtp_gdn_snap_per_step = snap_slots      # C.7 partial_accept uses snap[n]
                # Hand the snap slots to the GDN forward hook via batch attribute; the engine
                # build_fla_metadata call later promotes it onto fla.mtp_verify_snap_slots
                # and sets mtp_verify_step_idx=0 so the GDN op writes per-step h/conv.
                # Stored on req (not batch) because the batch hasn't been constructed yet.
                req._mtp_verify_snap_slots_for_hook = snap_slots
            except Exception as exc:  # noqa: BLE001 -- pool exhausted: skip verify
                logger.warning_rank0(f"MTP verify skipped (no GDN snapshot slot: {exc})")
                self._mtp_discard_drafts(req)
                return None
            # Disable mid-chunk tracking for the verify forward only.
            req._mtp_pp_saved = req.mamba_ping_pong
            req.mamba_ping_pong = None
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        # Snapshot the staged state ON THE BATCH: under overlap scheduling the
        # next decode drain (which records pred0 on the req) runs AFTER this build
        # but BEFORE this batch drains -- so we snapshot the SHAPE (was the seed
        # token already forwarded by a decode batch?) and read pred0 from the req
        # at drain time.
        base = int(getattr(req, "mtp_base", 0))
        batch.mtp_drafts = list(getattr(req, "mtp_drafts", []) or [])
        batch.mtp_base = base
        batch.mtp_seed_hidden = getattr(req, "mtp_seed_hidden", None)
        # K-row shape iff the seed token's KV is already committed (cached past it).
        batch.mtp_overlap = req.cached_len > base
        # mtp_verify flags consumed by engine.forward_batch (skip sampler; argmax the
        # per-position logits) and by lm_head.forward (skip the last-token gather -- we
        # need all K+1 rows). no_lm_head_gather is read by every lm_head subclass.
        batch.mtp_verify = True
        batch.no_lm_head_gather = True
        # C.5 hook: hand the per-step snap slots to the GDN forward path via batch attr.
        # _forward calls build_fla_metadata; that path detects batch.mtp_verify_snap_slots
        # and promotes the slot list onto fla.mtp_verify_snap_slots (int64 tensor) plus
        # fla.mtp_verify_step_idx=0 (the kernel walks 0..K per token, snapshotting after
        # each step). Mirrors track_dst/track_h_row plumbing in build_fla_metadata.
        snap_slots_for_hook = getattr(req, "_mtp_verify_snap_slots_for_hook", None)
        if snap_slots_for_hook is not None:
            batch.mtp_verify_snap_slots = snap_slots_for_hook  # list[int], promoted in _forward
            batch.mtp_verify_k = len(snap_slots_for_hook) - 1  # K for per-step iteration
        if os.environ.get("FT_MTP_DEBUG"):
            logger.info_rank0(
                f"[MTP-dbg] build: cached_len={req.cached_len} device_len={req.device_len} "
                f"base={base} K={len(batch.mtp_drafts)} overlap={batch.mtp_overlap} "
                f"ids={req.input_ids[max(0, base - 2): base + len(batch.mtp_drafts) + 3].tolist()}")
        self._mtp_verify_inflight = req
        return batch

    def _mtp_stash_prefill_hiddens(self, batch: Batch, all_hidden: torch.Tensor) -> None:
        if os.environ.get("FT_MTP_DEBUG"):
            print(f"[MTP-dbg] stash: batch={getattr(batch, 'phase', '?')} extends={getattr(batch, 'mtp_sched_extends', None) is not None} all_hidden={all_hidden is not None}", flush=True)
        """Per-req: append this batch's hidden rows to req._mtp_seed_hiddens.

        req._mtp_seed_start is set on the FIRST stash (the row range begins at
        the request's schedule-time cached_len for that forward) and reused by
        subsequent chunks; subsequent chunks only append rows. Cache-hit prefills
        only contribute rows for the computed suffix -- the head then attends
        over the recent context only (degraded but not broken)."""
        extends = getattr(batch, "mtp_sched_extends", None)
        if not extends:
            return
        offset = 0
        # Padded reqs lead real reqs in order (pad_batch appends dummies); index
        # both lists together.
        padded = batch.padded_reqs
        for i, req in enumerate(batch.reqs):
            ext = extends[i] if i < len(extends) else 0
            if ext > 0 and req.table_idx != -1 and not req.aborted                     and req.sampling_params.is_greedy:
                rows = all_hidden[offset : offset + ext]
                prev = getattr(req, "_mtp_seed_hiddens", None)
                if prev is None:
                    # req.cached_len has been advanced by complete_one; subtract ext
                    # to recover the schedule-time cached_len (= row-range start).
                    req._mtp_seed_hiddens = rows.contiguous()
                    req._mtp_seed_start = req.cached_len - ext
                else:
                    req._mtp_seed_hiddens = torch.cat([prev, rows], dim=0)
            offset += ext

    def _mtp_discard_drafts(self, req: Req) -> None:
        drafts = getattr(req, "mtp_drafts", None) or []
        keep = req.input_ids.numel() - len(drafts)
        if keep >= req.cached_len:
            req.input_ids = req._ids_buf[:keep]
            req.device_len = keep
        req.mtp_verify = False
        req.mtp_drafts = None
        req.mtp_base = None

    def _mtp_release_inflight_and_scratch(self, req: Req) -> None:
        """Verify round fully done (accept or reject): clear the req's mtp_* scratch
        (the in-flight marker itself is cleared at the drain entry in
        _process_last_data)."""
        req.mtp_verify = False
        req.mtp_drafts = None
        req.mtp_base = None
        req.mtp_pred0 = None

    def _mtp_release_gdn_snap(self, req: Req) -> None:
        """Restore the ping-pong pointer and free the verify snapshot slots.

        After C.6, ``req._mtp_gdn_snap_per_step`` is the full K+1 snap-slot list
        (snap[0]=pre-verify, snap[1..K]=post per-step). Free the whole list at once
        (``pool.free`` accepts a list). ``req._mtp_gdn_snap`` (= snap[0]) is kept
        as a compatibility pointer for any caller still reading the single-slot
        attribute; we drop the ref after the bulk free but do NOT call
        ``pool.free`` on it again (it was the first element of the list just freed).
        """
        pool = getattr(self.engine, "linear_state_pool", None)
        snap_slots = getattr(req, "_mtp_gdn_snap_per_step", None)
        if snap_slots is not None and pool is not None:
            pool.free(snap_slots)  # pool.free accepts int | list[int] | Tensor
            req._mtp_gdn_snap_per_step = None
        req._mtp_gdn_snap = None  # just drop the compat ref (was snap_slots[0])
        if getattr(req, "_mtp_pp_saved", "__unset__") != "__unset__":
            req.mamba_ping_pong = req._mtp_pp_saved
            req._mtp_pp_saved = "__unset__"

    def _mtp_process_verify(
        self, batch: Batch, forward_output: "ForwardOutput"
    ) -> None:
        """Drain one MTP verify forward: compare main-model per-position argmax vs the K
        MTP drafts, accept the longest common prefix, ship drafts[1:n] + correction as
        DetokenizeMsg's, advance the request to position p+n and roll back the rejected
        drafts above it. EOS / stop / length are honored at the SAME semantic points as
        the vanilla decode loop (correction wins EOS, but any accepted draft that's EOS
        still terminates the request and the bonus correction is dropped)."""
        self.mtp_stats["verify_calls"] += 1
        if self.mtp_stats["verify_calls"] % 64 == 0:
            st = self.mtp_stats
            acc = st["accepted_tokens"]
            drafted = max(st["drafted"] - st["misses"], 0)
            logger.info_rank0(
                "MTP stats: verify=%d drafted=%d misses=%d accepted=%d "
                "(accept rate %.2f tok/verify)"
                % (st["verify_calls"], st["drafted"], st["misses"], acc,
                   (acc / st["verify_calls"]) if st["verify_calls"] else 0.0)
            )
        req = batch.reqs[0]
        if req.aborted:
            # Aborted while the verify forward was in flight: mirror the decode
            # drain path (free + finish, NO DetokenizeMsg -- the abort ack is the
            # terminal reply). The staged drafts die with the request.
            self._mtp_release_gdn_snap(req)
            req.mtp_verify = False
            req.mtp_drafts = None
            req.mtp_base = None
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            self.finished_reqs.add(req)
            return
        # Snapshot semantics (the batch carries the staged state as of its BUILD;
        # the req's scratch may have been touched by a later decode drain since):
        #   base   = rope position of the seed token t' (the token the drafts extend)
        #   drafts = [d1..dK], d_i at rope position base+i
        # Two protocol shapes exist, depending on overlap scheduling:
        #   K-row (overlap): the decode batch that sampled t' ALREADY forwarded t'
        #     (its KV is in), so the verify forward ran [d1..dK] only. The row that
        #     verifies d1 is the decode batch's own sampled token (pred0, recorded by
        #     the NEXT hook call and snapshotted here).
        #   K+1-row (non-overlap): t' was NOT forwarded yet; the verify forward ran
        #     [t', d1..dK] and its first argmax row verifies d1.
        # In both shapes preds[i] = the main model's argmax prediction at position
        # base+i, and preds[i] verifies drafts[i] (= d_{i+1}).
        if req.table_idx == -1:
            # The request finished/was freed while this verify was in flight (e.g.
            # the next decode drain hit EOS and removed it) -- nothing to publish.
            self._mtp_release_gdn_snap(req)
            return
        drafts: list[int] = list(getattr(batch, "mtp_drafts", []) or [])
        base: int = int(getattr(batch, "mtp_base", 0))
        overlap = bool(getattr(batch, "mtp_overlap", False))
        pred0 = getattr(req, "mtp_pred0", None) if overlap else None
        K = len(drafts)
                # C++ only -- no Python fallback by user request:
        # one C++ call converts the int64 GPU host tensor to int32 std::vector<int64_t>
        # instead of `.to(torch.int32).tolist()` (two PyTorch dispatch ops).
        rows = _sched_cpp.gpu_int_to_cpu_list(forward_output.next_tokens_cpu)
        # ----------------------------------------------------------------------
        # PATH 3 (EAGLE-style): re-run the MTP head K times using the VERIFY forward's
        # main-model hidden states as prev_hidden (teacher forcing). Hook-time drafts
        # use the head's self-fed hidden (step i's output -> step i+1's prev_hidden),
        # which differs from training where prev_hidden is the main model's ground-truth
        # hidden at the same position. The mismatch collapses drafts[1:] acceptance
        # to 0% on hybrid (n<K -> reject_all -> 0 accepted tokens). Re-running the
        # head with the verify's true main hidden gives teacher-forcing accuracy
        # (~80% per slot -> ~50% for n=K=3 with three independent slots), unlocking
        # actual MTP speedup.
        # ----------------------------------------------------------------------
        # (path3 head re-eval REMOVED: its argmax fed the acceptance off-by-one and it
        # never belongs in the acceptance path. Draft quality improvements belong in
        # the draft hook, not here.)
# Build targets[i] = the main model's argmax at position base+i+1 -- the
        # distribution that VERIFYING draft d_{i+1}.
        #   K-row (overlap, pred0 set): the verify consumed [d1..dK] at base+1..base+K;
        #     output row k is the argmax AFTER consuming d_{k+1}, i.e. the target for
        #     position base+k+2. So d1 is verified by pred0 (the decode step's own
        #     sample at base+1), d2 by rows[0], d3 by rows[1], ... dK by rows[K-2].
        #   K+1-row shape: the verify consumed [t, d1..dK]; row k is the argmax at
        #     base+k+1, so d1..dK are verified by rows[0..K-1] directly.
        # (The previous path3 wiring compared d_{k+1} against rows[k] -- an off-by-one
        # that shifted every acceptance target by one position and published a
        # correction conditioned on a REJECTED draft, permanently desyncing the KV and
        # GDN state. That is the root cause of the MTP output corruption.)
        if pred0 is not None:
            targets = ([int(pred0)] + rows[:K - 1]) if K > 0 else []
            bonus = rows[K - 1] if len(rows) >= K else None      # argmax at base+K+1
        else:
            targets = rows[:K]
            bonus = rows[K] if len(rows) >= K + 1 else None      # argmax at base+K+1
        if len(targets) < K:
            targets = targets + [targets[-1]] * (K - len(targets))
        # Greedy accept: longest prefix where the main-model argmax matches the draft.
        m = 0
        while m < K and targets[m] == drafts[m]:
            m += 1
        # Degenerate-repeat guard: a full accept of K identical drafts locks the
        # generation into a repetition loop (the verify targets are conditioned on
        # the drafted repeats, so the main model endorses them; pure decode would
        # never enter the loop). Reject all-identical full accepts -- costs at
        # most one round of speed, prevents the a-a-a lock-in.
        if m == K and K >= 2 and all(d == drafts[0] for d in drafts[:K]):
            m = 0
            if os.environ.get("FT_MTP_DEBUG"):
                logger.info_rank0("[MTP-dbg] degenerate full-accept rejected (identical drafts)")
        # Hybrid varlen verify: the kernel writes only the FINAL recurrent/conv state;
        # snap[1..K] all hold that final state, so ONLY all-or-nothing acceptance is
        # recoverable (snap[0] restores the pre-verify state on reject; full accept
        # keeps the live post-verify state). Partial accepts on this path would roll
        # back to an over-advanced state -- clamp to {0, K}. (Non-hybrid paths only
        # roll KV back, so true partial accept is safe there.)
        hybrid = getattr(self.engine, 'linear_state_pool', None) is not None
        n = m if (m == K or not hybrid) else 0
        if os.environ.get('FT_MTP_DEBUG'):
            logger.info_rank0(f'[MTP-dbg] m={m} n={n} K={K} drafts={drafts[:4]} targets={targets[:4]} rows={rows[:4] if len(rows) else []}')
        # correction = the main model's token at position base+n+1 (the next token
        # after the accepted prefix). It comes from the verify's OWN argmax row:
        #   n >= 1: argmax after consuming d_n = rows[n-1] (K-row) / rows[n] (K+1-row)
        #   n == 0: K-row -> pred0 (already shipped by the decode drain; nothing new);
        #           K+1-row -> rows[0] (never shipped).
        # The old path3 'correction = preds[n]' used a row conditioned on REJECTED
        # drafts when n < K -- publishing tokens the model never proposed for that
        # position. bonuses/rows are re-derived below at the publish site.
        mtp_targets = targets
        mtp_bonus = bonus
        mtp_n = n
        n = n  # kept name for downstream code
        # Hybrid (GDN) models: the linear-attention state was advanced by the verify
        # forward itself. Pre-C.7 the only stored snapshot was the pre-verify state, so
        # partial accepts could not be recovered without re-running the GDN layers
        # (expensive) -- and the "ship-ready" workaround was mtp_skip_next, which
        # produced the visible loop-thinking pathology. After C.4-C.7 the GDN op writes
        # a per-step snap trace (snap[0..K]) during the verify forward; partial accepts
        # roll back via copy_from(snap[n], slot) with no replay, byte-exact alignment.
        # All-or-nothing fallback removed: K=2 + per-step snap mirrors llama.cpp #22400.
        hybrid = getattr(self.engine, "linear_state_pool", None) is not None
        eos_ids = self.eos_token_ids
        ignore_eos = req.sampling_params.ignore_eos

        def _eos(tok: int) -> bool:
            return (not ignore_eos) and tok in eos_ids

        # Truncate an accepted tail at the first EOS draft (the request ends there).
        eos_in_accepted = next((j for j in range(n) if _eos(drafts[j])), n)

        reject_all = False
        # rollback_snap = which snap slot to copy_from(live_slot, ...) on the accept path;
        # None means no rollback (full accept n == K: live slot already IS the post-verify
        # state == snap[K]). C.4 + C.7 re-enabled 2026-08-29: snap[0..K-1] are populated
        # per-step during the verify forward, so partial accept rolls back byte-exactly.
        rollback_snap = None
        if hybrid:
            if n == K:
                # Full accept: live slot is the post-verify state (= snap[K]); no rollback.
                n = eos_in_accepted  # only truncate for an EOS draft
            elif n == 0:
                # Reject all: preds[0] != draft[0] -> snap[0] rollback + re-sample base+1.
                # Matches the legacy all-or-nothing behavior on hybrid K=2.
                reject_all = True
            else:
                # Partial accept (0 < n < K): snap[n] rollback (C.7 re-enabled). The verify
                # forward wrote snap[1..K-1] on the engine stream (C.4 _forward_mtp_verify);
                # snap[n] is the GDN state at the position right after the last accepted
                # draft (= the state required for KV position base+n). Truncating the KV
                # from base+1+K back to base+1+n (done by cache_req_to_len below) leaves the
                # GDN state over-advanced by (K-n) steps; copy_from(snap[n], slot) restores
                # the byte-exact pre-accept state without re-running any GDN layers. Mirrors
                # llama.cpp PR #22400 (GDN intermediates) and EAGLE-style partial accept.
                snap_list = getattr(req, "_mtp_gdn_snap_per_step", None) or [None] * (K + 1)
                rollback_snap = snap_list[n] if 0 <= n < len(snap_list) else None

        if reject_all:
            # Restore the pre-verify GDN state and free the verify's KV pages back
            # to the seed token's committed prefix (t' KV stays -- the decode batch
            # computed it). The correction (= preds[0], the model's own next token)
            # is re-staged at the seed position for the next decode step: in the
            # K-row shape it was ALREADY shipped by the decode drain (pred0), so
            # nothing is published; in the K+1-row shape it was never shipped, so
            # publish it now.
            pool = self.engine.linear_state_pool
            snap = getattr(req, "_mtp_gdn_snap", None)
            slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
            if snap is not None:
                pool.copy_from(snap, slot)
            # The rejection correction must be the main model's OWN token at base+1:
            # K-row shape -> pred0 (the decode drain's sample, already shipped, whose
            # KV the next decode step will write). K+1-row shape -> rows[0] (the
            # argmax after consuming t; not yet shipped). Using rows conditioned on
            # REJECTED drafts here desynced KV/state from the published tokens.
            correction = int(pred0) if pred0 is not None else rows[0]
            rollback = base + 1  # KV committed through the seed token t' (base)
            if rollback < req.cached_len:
                self.cache_manager.cache_req_to_len(req, rollback)
            req.cached_len = rollback
            req.input_ids = req._ids_buf[:rollback]
            req.append_host(torch.tensor([int(correction)], dtype=torch.int32))
            req.device_len = req.input_ids.numel()  # == rollback + 1
            tp = self.token_pool[req.table_idx]
            if tp.shape[0] >= req.device_len:
                tp[req.device_len - 1] = int(correction)
            self._mtp_release_gdn_snap(req)
            # Commit the correction's own head row ((x_{t_pos+1}, h(t_pos))) so the
            # next round's cache has no gap at the seed position; the K+1 shape's
            # verify already computed h(t_pos) itself (its first row).
            all_hidden_r = getattr(forward_output, "all_hidden", None)
            seed_hidden_r = getattr(batch, "mtp_seed_hidden", None)
            if pred0 is not None and seed_hidden_r is not None:
                self.mtp.commit_round(
                    req.uid,
                    torch.tensor([int(correction)], dtype=torch.long,
                                  device=seed_hidden_r.device),
                    seed_hidden_r.view(1, -1).to(seed_hidden_r.dtype),
                )
            elif all_hidden_r is not None and all_hidden_r.shape[0] > 0:
                self.mtp.commit_round(
                    req.uid,
                    torch.tensor([int(correction)], dtype=torch.long,
                                  device=all_hidden_r.device),
                    all_hidden_r[:1],
                )
            req.mtp_verify = False
            req.mtp_drafts = None
            req.mtp_base = None
            req.mtp_pred0 = None
            self.mtp_stats["misses"] += 1
            self._mtp_release_inflight_and_scratch(req)
            if pred0 is None:
                # K+1-row shape: the correction is a NEW token for the client.
                self._publish_verify_reply(req, [
                    DetokenizeMsg(
                        uid=req.uid, next_token=int(correction), finished=False,
                        finish_reason=None, matched_stop=None,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                ], batch)
            else:
                self._publish_verify_reply(req, [], batch)
            return

        # Hybrid partial-accept rollback (C.7 re-enabled). Restore the GDN state to
        # snap[n] (= state right after the last accepted draft). pool.copy_from copies
        # BOTH conv + recurrent state across all linear layers, byte-exact. No kernel
        # replay needed -- the snap was written during the verify forward itself.
        if hybrid and rollback_snap is not None:
            pool_a = self.engine.linear_state_pool
            pool_a.copy_from(rollback_snap, req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx)
        # correction = the main model's prediction at the position FOLLOWING the last
        # accepted token = preds[n] (when n == K this is the bonus row). The old code
        # used preds[K] unconditionally, which skipped positions whenever n < K.
        if n != eos_in_accepted:
            # An accepted draft was EOS: end exactly there. The EOS draft itself is
            # the terminal token; no correction is sampled.
            n = eos_in_accepted
            correction = drafts[n]
        else:
            # The token at base+n+1 = the argmax AFTER consuming the last accepted
            # draft d_n (rows[n-1] in K-row shape, rows[n] in K+1 shape). For n == 0
            # the only position not yet filled is base+1: pred0 (K-row, already
            # shipped by the decode drain) or rows[0] (K+1 shape). NEVER a row
            # conditioned on a rejected draft.
            if n >= 1:
                correction = rows[n - 1] if pred0 is not None else rows[n]
            else:
                correction = int(pred0) if pred0 is not None else rows[0]

        # Snapshot no longer needed on the accept path: release it NOW (before any
        # finish branch calls _free_req_resources -> cache_req(finished=True)) so the
        # ping-pong pointer is restored before the hybrid donate/free reads it.
        self._mtp_release_gdn_snap(req)
        if hybrid and n != eos_in_accepted:
            # EOS draft inside the accepted prefix: cached_len stops short of the
            # live GDN slot's post-verify depth (base+2+K). Donating that
            # over-advanced state into the radix tree would COW-restore a
            # future hit past its node boundary -- suppress the live donate.
            req.mtp_no_live_donate = True
        # MTP commit: re-seed the head KV with exact rows for the newly committed
        # tokens using true main-model hiddens. Head row j pairs (x_j, h_{j-1}):
        #   K-row shape: the verify's all_hidden rows are h(t_pos+1)..h(t_pos+K);
        #     d1's row needs h(t_pos) = the seed hidden snapshotted on the batch.
        #     committed = d1..dn + correction pair with [h(t_pos), h(t_pos+1..t_pos+n)].
        #   K+1-row shape: the verify processed t' first, so all_hidden rows are
        #     h(t_pos)..h(t_pos+K) -- committed tokens pair with all_hidden[:n+1].
        all_hidden = getattr(forward_output, "all_hidden", None)
        seed_hidden = getattr(batch, "mtp_seed_hidden", None)
        committed_toks_list = drafts[:n] + [int(correction)]
        if all_hidden is not None and len(committed_toks_list) > 0:
            if pred0 is not None:
                if seed_hidden is not None:
                    hid = torch.cat(
                        [seed_hidden.to(all_hidden.device).view(1, -1).to(all_hidden.dtype),
                         all_hidden[:n]], dim=0,
                    )
                else:
                    hid = all_hidden[:n]  # defensive: no seed hidden, skip d1's exact row
                toks = torch.tensor(
                    committed_toks_list[: hid.shape[0]], dtype=torch.long,
                    device=all_hidden.device,
                )
            else:
                hid = all_hidden[: n + 1]
                toks = torch.tensor(
                    committed_toks_list[: hid.shape[0]], dtype=torch.long,
                    device=all_hidden.device,
                )
            if toks.shape[0] == hid.shape[0] and toks.shape[0] > 0:
                self.mtp.commit_round(req.uid, toks, hid)
        reply: List[DetokenizeMsg] = []
        # Commit the accepted prefix: the seed token t' sits at rope position base
        # (its KV came from this verify in the K+1-row shape, or from the decode
        # batch in the K-row shape), and the n accepted drafts occupy base+1..base+n.
        # target cached_len = base+1+n; the orphaned draft pages above it are freed.
        target_cached_len = base + 1 + n
        if target_cached_len < req.cached_len:
            self.cache_manager.cache_req_to_len(req, target_cached_len)
        req.cached_len = target_cached_len
        req.input_ids = req._ids_buf[:target_cached_len]

        # The "correction" is the model's prediction at position p+n: ship it as the
        # next sampled token, then commit it (writes the token to the token_pool + host
        # buffer) so the next decode iteration sees the request at position p+n+1.
        if correction in self.eos_token_ids and not req.sampling_params.ignore_eos:
            # Correction is EOS: terminate immediately. Do NOT extend the request.
            finished = True
            finish_reason = "stop"
            reply.append(
                DetokenizeMsg(
                    uid=req.uid, next_token=int(correction), finished=True,
                    finish_reason="stop", matched_stop=None,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
            )
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            self.finished_reqs.add(req)
        else:
            # Normal case: ship the accepted drafts as reply tokens, then the
            # correction. Publication shape depends on the protocol shape:
            #   K-row (overlap): the decode drain already shipped pred0 -- and
            #     pred0 == d1 whenever n >= 1 (that is what acceptance means), so
            #     only d2..dn are NEW tokens here.
            #   K+1-row: none of the drafts were shipped yet (d1..dn all new).
            first_new = 1 if pred0 is not None else 0
            for tok in drafts[first_new:n]:
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid, next_token=int(tok), finished=False,
                        finish_reason=None, matched_stop=None,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )
            # Ship the correction as the next reply token, then append it to the
            # request so device_len = cached_len + 1 and the next decode step
            # starts from there.
            reply.append(
                DetokenizeMsg(
                    uid=req.uid, next_token=int(correction), finished=False,
                    finish_reason=None, matched_stop=None,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
            )
            req.append_host(torch.tensor([int(correction)], dtype=torch.int32))
            req.device_len = req.input_ids.numel()
            # Mirror the correction into token_pool at the request's table_idx so the
            # next decode forward reads it via token_pool[table_idx][cached_len].
            tp = self.token_pool[req.table_idx]
            if tp.shape[0] >= req.device_len:
                tp[req.device_len - 1] = int(correction)
            # Stats + length-finish check: tokens published this round = n + 1 in the
            # K+1 shape (n drafts + correction) and n in the K shape (the correction
            # plus n-1 new drafts; pred0 was counted by its own decode drain).
            self.mtp_stats["accepted_tokens"] += n + 1 if pred0 is None else n
            finished = not req.can_decode  # hit output budget
            if finished:
                finish_reason = "length"
                # Override the last correction reply's finished flag.
                reply[-1] = DetokenizeMsg(
                    uid=req.uid, next_token=int(correction), finished=True,
                    finish_reason="length", matched_stop=None,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                self.finished_reqs.add(req)

        # Cleanup MTP scratch on the request (always, even when we bail early on a miss).
        if os.environ.get("FT_MTP_DEBUG"):
            _ids = req.input_ids
            logger.info_rank0("[MTP-dbg] ids-tail: cached=%s ids=%s", req.cached_len,
                              _ids[max(0, req.cached_len - 6):req.cached_len].tolist())
        req.mtp_verify = False
        req.mtp_drafts = None
        req.mtp_base = None
        req.mtp_pred0 = None
        self._mtp_release_gdn_snap(req)

        if os.environ.get("FT_MTP_DEBUG"):
            logger.info_rank0("[MTP-dbg] publish: m=%s n=%s pred0=%s drafts=%s corr=%s ship=%s",
                              m, n, pred0, drafts[:4], correction,
                              [mm.next_token for mm in reply])
        self._publish_verify_reply(req, reply, batch)

    def _publish_verify_reply(
        self, req: Req, reply: List[DetokenizeMsg], batch: Batch | None = None
    ) -> None:
        """Reply + status report (mirrors the bottom of _process_last_data's main path).
        Shared by the accept path and the hybrid all-reject path (empty reply)."""
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        if batch is not None:
            self.status_reporter.report_batch(
                batch,
                running_reqs=len(self.decode_manager.running_reqs),
                queue_reqs=len(self.prefill_manager.pending_list),
                kv_used_pages=used,
                kv_total_pages=total,
                page_size=self.config.page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        if reply:
            self.send_result(reply)

    def _mtp_maybe_draft(
        self,
        req: Req,
        prev_token_id: int,
        prev_hidden: torch.Tensor | None,  # 当前 batch 的 h(p) -- 我们需要的是 h(p-1)
        base_pos: int,
    ) -> None:
        # Ship-ready partial accept (hybrid): the previous round had n < K drafts
        # accepted. We let vanilla decode resync GDN state once before resuming MTP,
        # so the *next* decode step is a vanilla decode (no draft, no verify batch).
        if getattr(req, "mtp_skip_next", False):
            req.mtp_skip_next = False
            return
        """Seed the head KV (first call only), draft K candidates, and stage the K
        drafts in the request's table_idx + host input_ids so the NEXT round can
        verify them as one prefill."""
        if getattr(req, "mtp_verify", False):
            # Overlap scheduling: this decode batch's forward already ran (its
            # token is sampled), but the STAGED drafts from the previous hook are
            # still awaiting their verify. Record this batch's sampled token as
            # pred0 (the row that verifies draft #1) and skip staging -- a second
            # stage would append new drafts past the in-flight ones and clobber
            # req.mtp_drafts, mismatching the verify batch.
            req.mtp_pred0 = prev_token_id
            return
        # First draft for this request: seed the head's persistent KV cache with
        # the prefill's prompt hiddens so each draft step attends over the full
        # context (the trained Qwen MTP head expects this -- window-only drafts
        # are out-of-distribution).
        if os.environ.get("FT_MTP_NO_VERIFY"):
            return
        stash = getattr(req, "_mtp_seed_hiddens", None)
        if stash is not None:
            start = getattr(req, "_mtp_seed_start", 0)
            # Seed head-KV rows j = start+1 .. t_pos-1 only. Row j pairs
            # (x_j, h_{j-1}); stash[i] = h_{start+i}. The LAST stash row (h at the
            # seed token's position) is deliberately NOT seeded: draft step 1
            # appends the seed token's own row at j=t_pos, and seeding it too would
            # double-count that position in the head's attention.
            t_pos_pre = int(req.input_ids.numel()) - 1
            n_rows = max(0, min(stash.shape[0], t_pos_pre - (start + 1)))
            if os.environ.get("FT_MTP_DEBUG"):
                print(f"[MTP-dbg] seed: stash={tuple(stash.shape)} start={start} "
                      f"t_pos={t_pos_pre} n_rows={n_rows}", flush=True)
            if n_rows > 0:
                toks = req.input_ids[start + 1 : start + 1 + n_rows].to(self.device, dtype=torch.long)
                self.mtp.seed_context(req.uid, toks, stash[:n_rows].to(self.device), start)
            # Always seed the j=0 row with a zero-paired dummy so the head's
            # attention has the full context. Without this the head attends a
            # context starting at j=1 while the main model has j=0 too, leaving
            # the trained MTP module off-distribution.
            if getattr(req, "_mtp_seeded_zero", None) is not True:
                req._mtp_seeded_zero = True
                try:
                    zero_row = torch.zeros(1, self.mtp.head.attn.cfg.hidden_size,
                                           dtype=self.mtp.head.attn.cfg.dtype if hasattr(self.mtp.head.attn.cfg, 'dtype') else torch.bfloat16,
                                           device=self.mtp.head._device if hasattr(self.mtp.head, '_device') else self.device)
                    self.mtp.seed_context(req.uid, req.input_ids[0:1].to(self.device, dtype=torch.long),
                                          zero_row, start_pos=-1)
                except Exception as _e:
                    logger.warning_rank0("[MTP] seed_context failed for uid=%s (%r); continuing with window-only drafts", req.uid, _e)
            req._mtp_seed_hiddens = None
            if os.environ.get("FT_MTP_DUMP") and not getattr(self, "_mtp_dumped", False):
                self._mtp_dumped = True
                try:
                    torch.save({
                        "ids": req.input_ids[: start + 1 + n + 2].cpu(),
                        "hiddens": stash.cpu(),
                        "start": start,
                        "n_rows": n,
                    }, "E:/_mtp_seed_dump.pt")
                    print(f"[MTP-dump] seed saved: start={start} n={n} stash={tuple(stash.shape)}", flush=True)
                except Exception as _exc:
                    print(f"[MTP-dump] seed failed: {_exc!r}", flush=True)
        if prev_hidden is None:
            return
        if req.remain_len < self.mtp.k + 2:  # +2 = correction + slack for length-finish logic
            return
        self.mtp_stats["drafted"] += 1
        # position = the rope position of prev_token_id itself (the just-sampled
        # token, already appended at input_ids[cached_len]) -- the MTP head layer
        # processing that token sits at exactly this position. (The old code passed
        # cached_len-1, shifting every draft's rope by one.)
        # Rope position of prev_token_id (the just-appended seed token): the head
        # layer processes it at its own position. This is ids.numel()-1 AFTER the
        # append -- which equals base_pos (= cached_len-1) under overlap (the seed
        # was already forwarded) but base_pos+1 in the non-overlap shape (the seed
        # is still un-forwarded, cached points AT it). Compute it directly so both
        # shapes share one convention: mtp_base = the seed token's own position.
        t_pos = req.input_ids.numel() - 1
        # FT_MTP_PROF: per-round timing
        _prof_on = os.environ.get("FT_MTP_PROF") == "1"
        if _prof_on:
            if not hasattr(self, "_mtp_prof"):
                self._mtp_prof = {"drafts": 0, "draft_total": 0.0, "sched_total": 0.0,
                                  "verify_total": 0.0, "commit_total": 0.0,
                                  "accept_per_total": 0.0}
            _t_d0 = _time.perf_counter()
        drafts = self.mtp.draft(
            req.uid, prev_token_id, prev_hidden.view(1, -1), position=t_pos
        )
        if _prof_on:
            torch.cuda.synchronize()
            dt = _time.perf_counter() - _t_d0
            self._mtp_prof["drafts"] += 1
            self._mtp_prof["draft_total"] += dt
            # Pull per-stage times from the head's perf accumulator (filled by mtp.py)
            head_prof = getattr(self.mtp.head, "_perf", None)
            if self._mtp_prof["drafts"] % 32 == 0:
                p = self._mtp_prof
                if p["drafts"] > 0:
                    print(f"[MTP-prof] rounds={p['drafts']} draft_avg={p['draft_total']/p['drafts']*1000:.1f}ms accept_avg={p['accept_per_total']/p['drafts']:.2f}", flush=True)
                if head_prof is not None and head_prof.get("steps", 0) > 0:
                    n = head_prof["steps"]
                    print(f"[MTP-prof]   fc_avg={head_prof['fc']/max(1,head_prof['fc_n'])*1000:.2f}ms attn_avg={head_prof['attn']/n*1000:.2f}ms moe_avg={head_prof['moe']/n*1000:.2f}ms lmh_avg={head_prof['lmh']/n*1000:.2f}ms", flush=True)
                head_prof["fc"] = 0.0; head_prof["fc_n"] = 0
                head_prof["attn"] = 0.0; head_prof["moe"] = 0.0
                head_prof["lmh"] = 0.0; head_prof["steps"] = 0
        if not drafts:
            self.mtp_stats["misses"] += 1
            return
        K = len(drafts)
        # Stage ALL K drafts after the just-sampled token. The verify forward then
        # extends over [t, d1..dK] == K+1 tokens from cached_len (t already sits at
        # input_ids[cached_len] via the decode step's append_host + token_pool
        # scatter), producing exactly K+1 per-position argmax rows:
        # [pred@t+1 (verifies d1), ..., pred@dK (bonus)].
        # Use shape[0] (no sync) instead of .numel() (forces CPU-GPU sync).
        pre_draft_len = req.input_ids.shape[0]
        # Convert drafts (Python list[int] of length K) to int32 tensor in one shot.
        K = len(drafts)
        drafts_t = torch.tensor(drafts, dtype=torch.int32) if K > 0 else None
        req.append_host(drafts_t)
        req.device_len = req.input_ids.shape[0]
        tp = self.token_pool[req.table_idx]
        if tp.shape[0] < req.device_len:
            # Pool row cannot hold the drafts (should not happen: remain_len gate
            # above bounds device_len) -- undo the host staging and skip verify.
            req.input_ids = req._ids_buf[:pre_draft_len]
            req.device_len = pre_draft_len
            self.mtp_stats["misses"] += 1
            return
        # Reuse drafts_t (avoid a second Python-list->tensor alloc + H2D copy).
        tp[pre_draft_len: req.device_len] = drafts_t.to(dtype=tp.dtype, device=tp.device)
        # Mark for verify on the next scheduling round.
        req.mtp_verify = True
        req.mtp_drafts = drafts
        req.mtp_pred0 = None  # filled by the next decode drain (overlap) if needed
        # mtp_base = the seed token's own rope position; drafts occupy
        # base+1..base+K. The accept path commits KV through base+1+n.
        req.mtp_base = t_pos
        req.mtp_seed_hidden = prev_hidden.detach().clone()
        if os.environ.get("FT_MTP_DUMP") and not getattr(self, "_mtp_dumped", False):
            self._mtp_dumped = True
            try:
                _stash = getattr(req, "_mtp_seed_hiddens", None)
                torch.save({
                    "ids": req.input_ids[: t_pos + 2].cpu(),
                    "hiddens": _stash.cpu() if _stash is not None else None,
                    "t_pos": t_pos,
                    "drafts": drafts,
                    "prev_hidden": prev_hidden.detach().cpu(),
                    "prev_token": int(prev_token_id),
                }, "E:/_mtp_dump.pt")
                print(f"[MTP-dump] saved: t_pos={t_pos} stash={None if _stash is None else tuple(_stash.shape)} drafts={drafts}", flush=True)
            except Exception as _exc:
                print(f"[MTP-dump] failed: {_exc!r}", flush=True)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            # Log-and-continue: raising here kills the scheduler process and strands
            # every in-flight request (clients then hang until HTTP timeout). A
            # forward-compat/skewed message type must not be fatal.
            self._unknown_msg_count = getattr(self, "_unknown_msg_count", 0) + 1
            if self._unknown_msg_count <= 5 or self._unknown_msg_count % 100 == 0:
                logger.error(f"Unknown message type: {type(msg)}; ignoring (seen {self._unknown_msg_count}x)")

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot.

        C++ only -- no Python fallback by user request: the per-req for-loop that called
        `pool.copy_from(src, dst)` is now one C++ call over the full src/dst slot lists.
        """
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        src_list = [r.mamba_restore_src for r in batch.reqs if r.mamba_restore_src is not None]
        dst_list = [r.linear_slot_idx for r in batch.reqs if r.mamba_restore_src is not None]
        if not src_list:
            return
        _sched_cpp.restore_linear_states(
            pool.conv_states, pool.recurrent_states, src_list, dst_list)
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        # MTP: snapshot each req's extend_len BEFORE complete_one (the forward
        # consumes the row range [old_cached, old_cached+extend)) -- needed at
        # the prefill drain to slice all_hidden into per-req rows for context
        # seeding (see _mtp_stash_prefill_hiddens).
        if batch.is_prefill:
            batch.mtp_sched_extends = [r.extend_len for r in batch.padded_reqs]
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    # C++ only -- no Python fallback by user request:
                    # one C++ call replaces the list comprehension + pinned alloc + H2D.
                    slot_inputs = [int(r.linear_slot_idx) if r.linear_slot_idx is not None else -1
                                   for r in batch.padded_reqs]
                    batch.linear_table_idx = _sched_cpp.build_linear_table_idx_decode_hybrid(
                        slot_inputs, int(pool.padding_slot), self.device)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
            # C.5 hook: promote batch.mtp_verify_snap_slots (list[int], set by
            # _build_mtp_verify_batch) onto fla so the GDN forward writes per-step h/conv
            # into the right pool slots. mtp_verify_step_idx=0 tells the kernel to start
            # its per-step iteration from 0; the kernel walks 0..K, snapshotting after
            # each verify step. Mirrors track_dst/track_h_row promotion logic below.
            if (getattr(batch, "mtp_verify", False)
                    and getattr(batch, "mtp_verify_snap_slots", None) is not None):
                slot_list = batch.mtp_verify_snap_slots
                K = batch.mtp_verify_k
                device = self.device
                # Existing C.5 promotion: per-step snap slots as int64 device tensor.
                # C++ only -- no Python fallback by user request: the 4 torch.tensor()
                # allocations (snap_slots/cu_seqlens/has_init/host_snap_slots) are bundled
                # into one C++ call below.
                batch.fla_metadata.mtp_verify_step_idx = 0
                # G.2: persistent buffers + stable host ints so the C.4 per-step decode path
                # (gdn.py _forward_mtp_verify) does NOT allocate new tensors or call .item()
                # during forward -- required for CUDA-graph capture. All values are stable
                # for the lifetime of this req (snap_slots list allocated in
                # _build_mtp_verify_batch and not reassigned until _mtp_release_gdn_snap).
                # RC1 fix: cu_seqlens MUST match the verify forward's actual row count.
                # Overlap (K-row): engine.forward_batch calls req.complete_one() at
                # launch, so cached_len is already past the seed token and the verify
                # extends exactly K rows [d1..dK]. Non-overlap (K+1-row): the seed
                # token is un-forwarded and the verify runs [t, d1..dK] = K+1 rows.
                # The old hardcoded [0, K+1] made the FLA kernel loop T=K+1 steps over
                # a K-row tensor: OOB q/k/v read, OOB output write, and a corrupted
                # over-advanced final state written back to the LIVE GDN slot.
                _verify_rows = K if getattr(batch, "mtp_overlap", False) else K + 1
                snap_slots_dev, cu_varlen_dev, has_init_dev, host_snap_pinned = _sched_cpp.build_mtp_verify_meta(
                    [int(s) for s in slot_list], int(_verify_rows), device)
                batch.fla_metadata.mtp_verify_snap_slots = snap_slots_dev
                batch.fla_metadata.mtp_verify_cu_seqlens_varlen = cu_varlen_dev
                batch.fla_metadata.mtp_verify_has_initial_state = has_init_dev
                # snap_slots is a Python list[int]; copy it (cheap) for stable ownership
                # so a downstream mutation of req._mtp_verify_snap_slots_for_hook can't
                # bleed into the captured graph. The C++ wrapper above already returns a
                # pinned-host int32 clone that we re-materialise as a Python list --
                # int() reads are cheap so this stays out of the hot loop.
                batch.fla_metadata.mtp_verify_snap_host_slots = host_snap_pinned.tolist()
                # live slot = req's linear slot (hybrid-radix) or table_idx (naive). Stable
                # for req lifetime -- see _mtp_verify_snap_slots_for_hook ownership above.
                verify_req = batch.reqs[0] if batch.reqs else None
                batch.fla_metadata.mtp_verify_live_slot = (
                    verify_req.linear_slot_idx
                    if verify_req is not None and verify_req.linear_slot_idx is not None
                    else (verify_req.table_idx if verify_req is not None else -1)
                )
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate per-request vision soft tokens (in request order) for a prefill
        batch so the model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first

        # MTP verify batch: when a request has been drafted (req.mtp_verify=True), the K
        # speculative tokens are already resident in its table_idx + the host input_ids
        # tensor; we just need to push the main model on the K+1 candidates as a prefill
        # and let _mtp_process_verify pick the accept prefix. We schedule the verify
        # before any prefill/decode so a verify-ing request doesn't get starved by a
        # newly-arriving prompt.
        if self.mtp is not None:
            verify_batch = self._build_mtp_verify_batch()
            if verify_batch is not None:
                forward_input = self._prepare_batch(verify_batch)
                self._report_prompt_admissions(verify_batch)
                return forward_input

        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        # MTP verify forward: the engine already wrote per-position argmax into next_tokens_gpu
        # (no sampler, no candidate to ship to the next decode step -- we route the accept/rollback
        # through cache_manager in _mtp_process_verify instead). Skipping the token_pool write
        # also keeps the verify batch from clobbering the live decode table_idx it shares with the
        # request's normal decode row.
        if not getattr(batch, "mtp_verify", False):
            # C++ only -- no Python fallback by user request.
            # Historical context: this is the line that crashed the API call
            # before P0 of the C++ rewrite (broadcast error from
            # output_mapping=None when the Python return was truncated). The
            # C++ variant uses index_put_ with explicit shape checks, so a
            # bad output_mapping now raises a clean RuntimeError instead of
            # silently corrupting the token pool.
            _sched_cpp.write_tokens(
                self.token_pool,
                output_mapping[0],  # table_idx_dev
                output_mapping[1],  # device_len_dev
                forward_output.next_tokens_gpu,
            )
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # C++ only -- no Python fallback by user request.
    # Python originally had `torch.arange(cached, device_len, out=host_slice])`
    # per req in a for-loop; C++ version packs everything in one tight loop over
    # a single pinned-host tensor then H2D-copies once.
    positions_host, _ = _sched_cpp.make_positions(
        [int(r.extend_len) for r in batch.padded_reqs],
        [int(r.cached_len) for r in batch.padded_reqs],
    )
    return positions_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # C++ only -- no Python fallback by user request.
    return _sched_cpp.make_input(
        [int(r.table_idx) for r in batch.padded_reqs],
        [int(r.extend_len) for r in batch.padded_reqs],
        [int(r.cached_len) for r in batch.padded_reqs],
        device,
    )

def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # C++ only -- no Python fallback by user request.
    return _sched_cpp.make_write(
        [int(req.table_idx) for req in batch.reqs],
        [(int(req.device_len) if req.can_decode else -1) for req in batch.reqs],
        device,
    )
