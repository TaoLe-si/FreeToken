from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from freetoken.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        # Skip reqs that are mid-MTP-verify (req.mtp_verify=True). They were drafted by the
        # MTP head in the previous decode step and are waiting to be verified as one prefill;
        # including them in the next decode batch would (a) sample a token from a stale KV
        # cache that doesn't yet cover the speculative tokens, and (b) re-publish tokens
        # that _mtp_process_verify is about to ship on its own channel. The scheduler
        # routes them through _build_mtp_verify_batch -> _schedule_next_batch instead.
        decode_reqs = [r for r in self.running_reqs if not getattr(r, "mtp_verify", False)]
        if not decode_reqs:
            # Every running req is mid-MTP-verify: its verify batch (built by
            # _build_mtp_verify_batch) is either in flight or about to be -- an
            # empty decode Batch here would trip pad_batch's next() (size 0,
            # can_use_cuda_graph True) and crash the scheduler.
            return None
        return Batch(reqs=sorted(decode_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
