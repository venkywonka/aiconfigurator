# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aiconfigurator.sdk.forward_pass import ForwardPassWalker, ScheduledForwardPass
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult


def _fpm(**scheduled: int) -> dict[str, object]:
    return {
        "version": 1,
        "worker_id": "worker-0",
        "dp_rank": 0,
        "counter_id": 7,
        "scheduled_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
            **scheduled,
        },
        "queued_requests": {
            "num_prefill_requests": 999,
            "sum_prefill_tokens": 999,
            "num_decode_requests": 999,
            "sum_decode_kv_tokens": 999,
        },
        "wall_time": 999.0,
    }


@dataclass
class _Model:
    context_ops: list[Operation]
    generation_ops: list[Operation]


class _RecordingOp(Operation):
    def __init__(self, name: str, latency_ms: float) -> None:
        super().__init__(name, 1.0)
        self.latency_ms = latency_ms
        self.calls: list[dict[str, object]] = []

    def query(self, database, **kwargs) -> PerformanceResult:
        del database
        self.calls.append(dict(kwargs))
        return PerformanceResult(self.latency_ms)


class _SessionAwareOp(_RecordingOp):
    def __init__(self, name: str, latency_ms: float) -> None:
        super().__init__(name, latency_ms)
        self.sessions: list[object | None] = []

    def query_with_resolution(self, database, *, session=None, **kwargs) -> PerformanceResult:
        del database
        self.sessions.append(session)
        self.calls.append(dict(kwargs))
        return PerformanceResult(self.latency_ms)


def _walker(*, context_ops: list[Operation], generation_ops: list[Operation]) -> ForwardPassWalker:
    return ForwardPassWalker(
        model=_Model(context_ops=context_ops, generation_ops=generation_ops),
        database=object(),
        model_name="test/model",
    )


@pytest.mark.parametrize(
    ("metrics", "match"),
    [
        ([], "exactly one attention-DP rank"),
        ([_fpm(num_prefill_requests=1), _fpm(num_prefill_requests=1)], "exactly one attention-DP rank"),
        ([{**_fpm(num_prefill_requests=1), "version": 2}], "unsupported ForwardPassMetrics version"),
        ([_fpm(sum_prefill_tokens=1)], "prefill token sums require"),
        ([_fpm(sum_decode_kv_tokens=1)], "decode KV token sum requires"),
        ([_fpm()], "must not be empty"),
    ],
)
def test_scheduled_forward_pass_rejects_invalid_workloads(metrics, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ScheduledForwardPass.from_metrics_by_rank(metrics)


def test_prefill_walk_uses_scheduled_shape_and_ordinary_queries_without_session() -> None:
    context = _RecordingOp("context_moe", 2.0)
    logits = _RecordingOp("logits_gemm", 3.0)
    generation = _RecordingOp("generation_attention", 99.0)
    workload = ScheduledForwardPass.from_metrics_by_rank(
        [_fpm(num_prefill_requests=2, sum_prefill_tokens=10, sum_prefill_kv_tokens=4)]
    )

    latency_ms = _walker(context_ops=[context, logits], generation_ops=[generation]).walk(workload)

    assert workload.phase == "prefill"
    assert latency_ms == 5.0
    assert context.calls == [
        {
            "x": 10,
            "batch_size": 2,
            "beam_width": 1,
            "s": 5,
            "prefix": 2,
            "model_name": "test/model",
            "seq_imbalance_correction_scale": 1.0,
        }
    ]
    assert logits.calls == [{**context.calls[0], "x": 2}]
    assert generation.calls == []


def test_decode_walk_uses_scheduled_batch_and_average_kv_shape() -> None:
    context = _RecordingOp("context_attention", 99.0)
    generation = _RecordingOp("generation_attention", 4.0)
    workload = ScheduledForwardPass.from_metrics_by_rank([_fpm(num_decode_requests=3, sum_decode_kv_tokens=27)])

    latency_ms = _walker(context_ops=[context], generation_ops=[generation]).walk(workload)

    assert workload.phase == "decode"
    assert latency_ms == 4.0
    assert context.calls == []
    assert generation.calls == [
        {
            "x": 3,
            "batch_size": 3,
            "beam_width": 1,
            "s": 9,
            "model_name": "test/model",
            "gen_seq_imbalance_correction_scale": 1.0,
        }
    ]


def test_mixed_walk_is_three_pass_and_excludes_generation_non_attention_ops() -> None:
    context_moe = _RecordingOp("context_moe", 10.0)
    logits = _RecordingOp("logits_gemm", 2.0)
    context_attention = _RecordingOp("context_attention", 3.0)
    generation_moe = _RecordingOp("generation_moe", 100.0)
    generation_attention = _RecordingOp("generation_attention", 4.0)
    workload = ScheduledForwardPass.from_metrics_by_rank(
        [
            _fpm(
                num_prefill_requests=3,
                sum_prefill_tokens=10,
                sum_prefill_kv_tokens=6,
                num_decode_requests=3,
                sum_decode_kv_tokens=27,
            )
        ]
    )

    latency_ms = _walker(
        context_ops=[context_moe, logits, context_attention],
        generation_ops=[generation_moe, generation_attention],
    ).walk(workload)

    assert workload.phase == "mixed"
    assert latency_ms == 19.0
    assert context_moe.calls == [
        {
            "x": 13,
            "batch_size": 1,
            "beam_width": 1,
            "s": 13,
            "prefix": 6,
            "model_name": "test/model",
            "seq_imbalance_correction_scale": 1.0,
        }
    ]
    assert logits.calls == [{**context_moe.calls[0], "x": 1}]
    assert context_attention.calls == [
        {
            "x": 10,
            "batch_size": 3,
            "beam_width": 1,
            "s": 3,
            "prefix": 2,
            "model_name": "test/model",
            "seq_imbalance_correction_scale": 1.0,
        }
    ]
    assert generation_moe.calls == []
    assert generation_attention.calls == [
        {
            "x": 3,
            "batch_size": 3,
            "beam_width": 1,
            "s": 9,
            "model_name": "test/model",
            "gen_seq_imbalance_correction_scale": 1.0,
        }
    ]


def test_pure_and_resolving_walks_emit_identical_operation_shapes() -> None:
    pure_context = _SessionAwareOp("context_moe", 2.0)
    pure_attention = _SessionAwareOp("context_attention", 3.0)
    pure_generation = _SessionAwareOp("generation_attention", 4.0)
    resolving_context = _SessionAwareOp("context_moe", 2.0)
    resolving_attention = _SessionAwareOp("context_attention", 3.0)
    resolving_generation = _SessionAwareOp("generation_attention", 4.0)
    workload = ScheduledForwardPass.from_metrics_by_rank(
        [
            _fpm(
                num_prefill_requests=1,
                sum_prefill_tokens=8,
                sum_prefill_kv_tokens=4,
                num_decode_requests=2,
                sum_decode_kv_tokens=20,
            )
        ]
    )
    resolution_session = object()

    pure_latency = _walker(context_ops=[pure_context, pure_attention], generation_ops=[pure_generation]).walk(workload)
    resolving_latency = _walker(
        context_ops=[resolving_context, resolving_attention],
        generation_ops=[resolving_generation],
    ).walk(workload, session=resolution_session)

    assert pure_latency == resolving_latency
    assert [pure_context.calls, pure_attention.calls, pure_generation.calls] == [
        resolving_context.calls,
        resolving_attention.calls,
        resolving_generation.calls,
    ]
    assert pure_context.sessions == []
    assert pure_attention.sessions == []
    assert pure_generation.sessions == []
    assert resolving_context.sessions == [resolution_session]
    assert resolving_attention.sessions == [resolution_session]
    assert resolving_generation.sessions == [resolution_session]
