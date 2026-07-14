# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-owned scheduled-forward-pass projection and operation walk.

The input is the scheduler's aggregate ``ForwardPassMetrics`` workload, but the
module deliberately depends only on its stable mapping shape.  Dynamo owns the
scheduler and transport; AIC owns how those five scheduled counters project to
operation queries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from aiconfigurator.sdk.resolution.session import ResolutionSession


ForwardPassPhase = Literal["prefill", "decode", "mixed"]

_SCHEDULED_FIELDS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)


@dataclass(frozen=True)
class ScheduledForwardPass:
    """Validated single-rank scheduled workload consumed by the AIC walker."""

    num_prefill_requests: int
    sum_prefill_tokens: int
    sum_prefill_kv_tokens: int
    num_decode_requests: int
    sum_decode_kv_tokens: int
    source_metrics: Mapping[str, object] = field(repr=False, compare=False)

    @classmethod
    def from_metrics_by_rank(cls, metrics_by_rank: Sequence[Mapping[str, object]]) -> ScheduledForwardPass:
        """Validate and project one attention-DP rank's FPM workload.

        Queued counters, wall time, and variance fields are intentionally
        ignored: they are telemetry, not operation-shape inputs.
        """

        if len(metrics_by_rank) != 1:
            raise ValueError(
                "AIC scheduled-forward-pass prediction requires exactly one "
                f"attention-DP rank, got {len(metrics_by_rank)}"
            )
        metrics = metrics_by_rank[0]
        if not isinstance(metrics, Mapping):
            raise TypeError("ForwardPassMetrics rank entry must be a mapping")
        if int(metrics.get("version", 1)) != 1:
            raise ValueError(f"unsupported ForwardPassMetrics version {metrics.get('version')!r}")
        scheduled = metrics.get("scheduled_requests")
        if not isinstance(scheduled, Mapping):
            raise TypeError("ForwardPassMetrics scheduled_requests must be a mapping")

        values: dict[str, int] = {}
        for name in _SCHEDULED_FIELDS:
            raw = scheduled.get(name, 0)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError(f"scheduled_requests.{name} must be a non-negative integer")
            values[name] = raw

        if values["num_prefill_requests"] == 0 and (
            values["sum_prefill_tokens"] > 0 or values["sum_prefill_kv_tokens"] > 0
        ):
            raise ValueError("prefill token sums require num_prefill_requests > 0")
        if values["num_decode_requests"] == 0 and values["sum_decode_kv_tokens"] > 0:
            raise ValueError("decode KV token sum requires num_decode_requests > 0")
        if values["num_prefill_requests"] == 0 and values["num_decode_requests"] == 0:
            raise ValueError("scheduled ForwardPassMetrics workload must not be empty")

        return cls(source_metrics=metrics, **values)

    @property
    def phase(self) -> ForwardPassPhase:
        if self.num_prefill_requests > 0 and self.num_decode_requests > 0:
            return "mixed"
        if self.num_prefill_requests > 0:
            return "prefill"
        return "decode"

    def as_scheduled_requests(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _SCHEDULED_FIELDS}


class ForwardPassWalker:
    """Walk AIC operations for one validated scheduled aggregate.

    Supplying no resolution session follows every operation's ordinary query
    path.  Supplying a session follows the same operation order and shapes while
    allowing exact-hit/miss recording.  Callback replay remains owned by the
    caller's ``ResolutionSession`` or ``OnlineResolutionCoordinator``.
    """

    def __init__(self, *, model: Any, database: Any, model_name: str) -> None:
        self._model = model
        self._database = database
        self._model_name = model_name

    @staticmethod
    def prepare(
        metrics_by_rank: Sequence[Mapping[str, object]],
    ) -> ScheduledForwardPass:
        return ScheduledForwardPass.from_metrics_by_rank(metrics_by_rank)

    @staticmethod
    def _op_name(op: Any) -> str:
        return str(getattr(op, "_name", ""))

    def walk(
        self,
        workload: ScheduledForwardPass,
        *,
        session: ResolutionSession | None = None,
    ) -> float:
        if workload.phase == "mixed":
            return self._walk_mixed(workload, session=session)
        if workload.phase == "prefill":
            return self._walk_prefill(workload, session=session)
        return self._walk_decode(workload, session=session)

    def _query_context_op(
        self,
        op: Any,
        *,
        x: int,
        batch_size: int,
        sequence_length: int,
        prefix: int,
        session: ResolutionSession | None,
    ) -> float:
        kwargs = {
            "x": x,
            "batch_size": batch_size,
            "beam_width": 1,
            "s": sequence_length,
            "prefix": prefix,
            "model_name": self._model_name,
            "seq_imbalance_correction_scale": 1.0,
        }
        if session is None:
            return float(op.query(self._database, **kwargs))
        return float(op.query_with_resolution(self._database, session=session, **kwargs))

    def _query_generation_op(
        self,
        op: Any,
        *,
        batch_size: int,
        sequence_length: int,
        session: ResolutionSession | None,
    ) -> float:
        kwargs = {
            "x": batch_size,
            "batch_size": batch_size,
            "beam_width": 1,
            "s": sequence_length,
            "model_name": self._model_name,
            "gen_seq_imbalance_correction_scale": 1.0,
        }
        if session is None:
            return float(op.query(self._database, **kwargs))
        return float(op.query_with_resolution(self._database, session=session, **kwargs))

    def _walk_prefill(
        self,
        workload: ScheduledForwardPass,
        *,
        session: ResolutionSession | None,
    ) -> float:
        batch_size = workload.num_prefill_requests
        new_tokens_per_request = max(workload.sum_prefill_tokens // batch_size, 1)
        prefix_per_request = workload.sum_prefill_kv_tokens // batch_size
        total = 0.0
        for op in self._model.context_ops:
            x = batch_size if "logits_gemm" in self._op_name(op) else batch_size * new_tokens_per_request
            total += self._query_context_op(
                op,
                x=x,
                batch_size=batch_size,
                sequence_length=new_tokens_per_request,
                prefix=prefix_per_request,
                session=session,
            )
        return total

    def _walk_decode(
        self,
        workload: ScheduledForwardPass,
        *,
        session: ResolutionSession | None,
    ) -> float:
        batch_size = workload.num_decode_requests
        kv_per_request = workload.sum_decode_kv_tokens // batch_size
        return sum(
            self._query_generation_op(
                op,
                batch_size=batch_size,
                sequence_length=kv_per_request,
                session=session,
            )
            for op in self._model.generation_ops
        )

    def _walk_mixed(
        self,
        workload: ScheduledForwardPass,
        *,
        session: ResolutionSession | None,
    ) -> float:
        context_tokens = workload.sum_prefill_tokens
        decode_batch = workload.num_decode_requests
        combined_prefix = workload.sum_prefill_kv_tokens
        combined_tokens = max(context_tokens + decode_batch, 1)
        total = 0.0

        # Pass 1: prefill and decode tokens share non-attention kernels.
        for op in self._model.context_ops:
            name = self._op_name(op)
            if name == "context_attention":
                continue
            x = 1 if "logits_gemm" in name else combined_tokens
            total += self._query_context_op(
                op,
                x=x,
                batch_size=1,
                sequence_length=combined_tokens,
                prefix=combined_prefix,
                session=session,
            )

        # Pass 2: context attention retains the scheduled prefill shape.
        prefill_batch = workload.num_prefill_requests
        new_tokens_per_request = max(context_tokens // prefill_batch, 1)
        prefix_per_request = combined_prefix // prefill_batch
        for op in self._model.context_ops:
            if self._op_name(op) != "context_attention":
                continue
            total += self._query_context_op(
                op,
                x=context_tokens,
                batch_size=prefill_batch,
                sequence_length=new_tokens_per_request,
                prefix=prefix_per_request,
                session=session,
            )

        # Pass 3: generation non-attention work was included in pass 1.
        kv_per_request = workload.sum_decode_kv_tokens // decode_batch
        for op in self._model.generation_ops:
            if self._op_name(op) != "generation_attention":
                continue
            total += self._query_generation_op(
                op,
                batch_size=decode_batch,
                sequence_length=max(kv_per_request, 1),
                session=session,
            )
        return total


__all__ = ["ForwardPassPhase", "ForwardPassWalker", "ScheduledForwardPass"]
