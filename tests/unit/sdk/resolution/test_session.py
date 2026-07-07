# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import (
    MissSet,
    ResolutionBudget,
    ResolutionFailed,
    ResolutionSession,
)
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    ResolutionPolicy,
    UnresolvedCode,
)

pytestmark = pytest.mark.unit


def _protocol(**overrides: object) -> MeasurementProtocol:
    values = {
        "revision": "microbench-v1",
        "warmups": 3,
        "samples": 1,
        "statistic": "median",
        "timer": "cuda_event",
        "tuning_revision": "none",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="h100_sxm",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="h100",
        runtime_versions={"cuda": "12.9"},
    )


def _request(*, op_id: str = "gemm", m: int = 8, protocol: MeasurementProtocol | None = None) -> MeasurementRequest:
    query = {"m": m, "n": 16, "k": 32}
    environment = _environment()
    semantic = {"dtype": "bf16"}
    return MeasurementRequest(
        op_id=op_id,
        key=PerfKey.build("gemm/v1", query, environment, semantic),
        query=query,
        environment=environment,
        semantic_descriptor=semantic,
        protocol=protocol or _protocol(),
    )


def _valid_record(request: MeasurementRequest, latency_ms: float) -> MeasurementRecord:
    return MeasurementRecord.valid(
        key=request.key,
        latency_ms=latency_ms,
        energy_wms=0.0,
        samples_ms=(latency_ms,),
        protocol=request.protocol,
        perf_row={"latency": latency_ms},
        provenance={"collector_revision": "r1"},
    )


def _valid_record_with_protocol(
    request: MeasurementRequest,
    protocol: MeasurementProtocol,
    latency_ms: float,
) -> MeasurementRecord:
    return MeasurementRecord.valid(
        key=request.key,
        latency_ms=latency_ms,
        energy_wms=0.0,
        samples_ms=(latency_ms,),
        protocol=protocol,
        perf_row={"latency": latency_ms},
        provenance={"collector_revision": "r1"},
    )


def _failed_record(
    request: MeasurementRequest,
    code: UnresolvedCode,
    *,
    status: RecordStatus = RecordStatus.FAILED,
) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=status,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=request.protocol,
        perf_row={},
        provenance={"collector_revision": "r1"},
        failure_code=code,
        failure_reason=f"injected {code.value}",
    )


class _Executor:
    def __init__(
        self,
        responder: Callable[[Sequence[MeasurementRequest]], Sequence[MeasurementRecord]],
    ) -> None:
        self.responder = responder
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []
        self.deadlines: list[float] = []
        self.cancellation_tokens: list[object] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        batch = tuple(requests)
        self.request_batches.append(batch)
        self.deadlines.append(deadline_monotonic)
        self.cancellation_tokens.append(cancellation)
        return self.responder(batch)


def _session(
    tmp_path,
    executor: _Executor,
    *,
    protocol: MeasurementProtocol | None = None,
    budget: ResolutionBudget | None = None,
    clock=None,
    cancellation=None,
) -> ResolutionSession:
    selected_protocol = protocol or _protocol()
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    if cancellation is not None:
        kwargs["cancellation"] = cancellation
    return ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        budget or ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        selected_protocol,
        **kwargs,
    )


def _execute_request(session: ResolutionSession, request: MeasurementRequest) -> float:
    def query() -> float:
        record = session.lookup(request.key)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        session.record_miss(request, request.op_id)
        return 0.0

    return session.execute_callback(query)


class _Cancellation:
    def __init__(self, value: bool = False) -> None:
        self.value = value

    def cancelled(self) -> bool:
        return self.value


class _SequencedCancellation:
    def __init__(self, *values: bool) -> None:
        self.values = list(values)

    def cancelled(self) -> bool:
        return self.values.pop(0) if self.values else False


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_miss_set_deduplicates_same_perf_identity_across_request_op_ids() -> None:
    first = _request(op_id="layer.0.qkv")
    second = _request(op_id="layer.1.qkv")
    misses = MissSet()

    misses.record(first, "layer.0.qkv")
    misses.record(second, "layer.1.qkv")

    assert misses.requests() == (first,)
    assert misses.entries()[0].consumers == ["layer.0.qkv", "layer.1.qkv"]


def test_miss_set_rejects_same_key_with_conflicting_protocol() -> None:
    first = _request(protocol=_protocol())
    conflicting = _request(protocol=_protocol(warmups=4))
    misses = MissSet()
    misses.record(first, "first")

    with pytest.raises(ValueError, match="conflicting requests share PerfKey"):
        misses.record(conflicting, "conflicting")

    assert len(misses) == 1
    assert misses


def test_execute_callback_collects_then_requeries_exactly_once(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor)
    calls = 0

    def query() -> float:
        nonlocal calls
        calls += 1
        record = session.lookup(request.key)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        session.record_miss(request, "gemm")
        return 0.0

    assert session.execute_callback(query) == pytest.approx(1.25)
    assert calls == 2
    assert executor.request_batches == [(request,)]
    assert session.report.unique_misses == 1
    assert session.report.consumer_misses == 1
    payload = session.report.to_dict()
    assert isinstance(payload["collection_seconds"], float)
    assert payload["collection_seconds"] >= 0.0
    assert payload == {
        "overlay_hits": 1,
        "unique_misses": 1,
        "consumer_misses": 1,
        "accepted_records": 1,
        "rejected_records": 0,
        "collection_seconds": payload["collection_seconds"],
        "unresolved": [],
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_nested_execute_callback_joins_outer_collection_cycle(tmp_path) -> None:
    outer_request = _request(op_id="outer", m=8)
    nested_request = _request(op_id="nested", m=16)
    executor = _Executor(
        lambda requests: [
            _valid_record(request, 1.0 if request.key == outer_request.key else 2.0) for request in requests
        ]
    )
    session = _session(tmp_path, executor)
    outer_calls = 0
    nested_calls = 0

    def nested_query() -> float:
        nonlocal nested_calls
        nested_calls += 1
        record = session.lookup(nested_request.key)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        session.record_miss(nested_request, "nested")
        return 0.0

    def outer_query() -> float:
        nonlocal outer_calls
        outer_calls += 1
        record = session.lookup(outer_request.key)
        if record is None:
            session.record_miss(outer_request, "outer")
            outer_latency = 0.0
        else:
            assert record.latency_ms is not None
            outer_latency = record.latency_ms
        return outer_latency + session.execute_callback(nested_query)

    assert session.execute_callback(outer_query) == pytest.approx(3.0)
    assert outer_calls == 2
    assert nested_calls == 2
    assert executor.request_batches == [(outer_request, nested_request)]
    assert session.report.unique_misses == 2
    assert session.report.consumer_misses == 2


def test_execute_callback_restores_outermost_lifecycle_after_query_error(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor)

    def fail_after_recording_miss() -> float:
        session.record_miss(request, "gemm")
        raise RuntimeError("injected query failure")

    with pytest.raises(RuntimeError, match="injected query failure"):
        session.execute_callback(fail_after_recording_miss)

    session.resolve_pending()
    assert executor.request_batches == []
    assert _execute_request(session, request) == pytest.approx(1.25)
    assert executor.request_batches == [(request,)]


def test_duplicate_valid_executor_records_fail_without_becoming_overlay_hits(tmp_path) -> None:
    request = _request()
    executor = _Executor(
        lambda requests: [
            _valid_record(requests[0], 1.25),
            _valid_record(requests[0], 1.0),
        ]
    )
    session = _session(tmp_path, executor)

    def query() -> float:
        record = session.lookup(request.key)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        session.record_miss(request, "gemm")
        return 0.0

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(query)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.INVALID_MEASUREMENT]
    assert session.overlay.lookup(request.key, request.protocol) is None
    reopened = OverlayStore(session.overlay.path)
    assert reopened.lookup(request.key, request.protocol) is None


def test_deterministic_failure_is_negatively_cached_within_session(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_failed_record(requests[0], UnresolvedCode.UNSUPPORTED_SHAPE)])
    session = _session(tmp_path, executor)

    with pytest.raises(ResolutionFailed) as first:
        _execute_request(session, request)
    with pytest.raises(ResolutionFailed) as second:
        _execute_request(session, request)

    assert [reason.code for reason in first.value.reasons] == [UnresolvedCode.UNSUPPORTED_SHAPE]
    assert [reason.code for reason in second.value.reasons] == [UnresolvedCode.UNSUPPORTED_SHAPE]
    assert len(executor.request_batches) == 1
    assert session.overlay.lookup(request.key, request.protocol) is None


def test_transient_retry_reuses_key_charge_then_exhausts_and_new_key_exceeds_budget(tmp_path) -> None:
    request = _request(m=8)
    other = _request(m=16)
    executor = _Executor(lambda requests: [_failed_record(requests[0], UnresolvedCode.TIMEOUT)])
    session = _session(
        tmp_path,
        executor,
        budget=ResolutionBudget(max_new_keys=1, max_wall_seconds=30.0, max_transient_attempts_per_key=2),
    )

    with pytest.raises(ResolutionFailed) as first:
        _execute_request(session, request)
    with pytest.raises(ResolutionFailed) as other_failure:
        _execute_request(session, other)
    with pytest.raises(ResolutionFailed) as second:
        _execute_request(session, request)
    with pytest.raises(ResolutionFailed) as exhausted:
        _execute_request(session, request)

    assert [reason.code for reason in first.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert [reason.code for reason in other_failure.value.reasons] == [UnresolvedCode.BUDGET_EXHAUSTED]
    assert [reason.code for reason in second.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert [reason.code for reason in exhausted.value.reasons] == [UnresolvedCode.RETRY_EXHAUSTED]
    assert executor.request_batches == [(request,), (request,)]


def test_resolve_pending_consumes_snapshot_before_transient_retry(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_failed_record(requests[0], UnresolvedCode.TIMEOUT)])
    session = _session(
        tmp_path,
        executor,
        budget=ResolutionBudget(max_new_keys=1, max_wall_seconds=30.0, max_transient_attempts_per_key=2),
    )
    session.record_miss(request, "gemm")

    with pytest.raises(ResolutionFailed) as first:
        session.resolve_pending()
    session.resolve_pending()
    session.record_miss(request, "gemm")
    with pytest.raises(ResolutionFailed) as second:
        session.resolve_pending()
    session.resolve_pending()
    session.record_miss(request, "gemm")
    with pytest.raises(ResolutionFailed) as exhausted:
        session.resolve_pending()

    assert [reason.code for reason in first.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert [reason.code for reason in second.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert [reason.code for reason in exhausted.value.reasons] == [UnresolvedCode.RETRY_EXHAUSTED]
    assert executor.request_batches == [(request,), (request,)]


def test_pre_dispatch_cancellation_does_not_charge_or_poison_key(tmp_path) -> None:
    request = _request()
    cancellation = _Cancellation(True)
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    cancellation.value = False

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert _execute_request(session, request) == pytest.approx(1.25)
    assert executor.request_batches == [(request,)]


def test_cancellation_during_empty_executor_result_does_not_poison_key(tmp_path) -> None:
    request = _request()
    cancellation = _Cancellation()

    def cancel_during_execution(requests):
        del requests
        cancellation.value = True
        return []

    executor = _Executor(cancel_during_execution)
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    cancellation.value = False
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert _execute_request(session, request) == pytest.approx(1.25)
    assert len(executor.request_batches) == 2


def test_cancellation_during_failed_result_does_not_consume_transient_attempt(tmp_path) -> None:
    request = _request()
    cancellation = _Cancellation()

    def cancel_with_timeout(requests):
        cancellation.value = True
        return [_failed_record(requests[0], UnresolvedCode.TIMEOUT)]

    executor = _Executor(cancel_with_timeout)
    session = _session(
        tmp_path,
        executor,
        budget=ResolutionBudget(max_new_keys=1, max_wall_seconds=30.0, max_transient_attempts_per_key=1),
        cancellation=cancellation,
    )

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    cancellation.value = False
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert _execute_request(session, request) == pytest.approx(1.25)


def test_cancellation_dominates_protocol_mismatch_without_negative_poison(tmp_path) -> None:
    request = _request()
    unrequested = _request(m=16)
    cancellation = _Cancellation()
    other_protocol = _protocol(warmups=4)

    def cancel_with_mismatch(requests):
        cancellation.value = True
        return [
            _valid_record_with_protocol(requests[0], other_protocol, 1.0),
            _valid_record(unrequested, 0.75),
        ]

    executor = _Executor(cancel_with_mismatch)
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    cancellation.value = False
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert session.overlay.lookup(request.key, other_protocol) is None
    assert session.overlay.lookup(unrequested.key, unrequested.protocol) is None
    assert _execute_request(session, request) == pytest.approx(1.25)
    assert len(executor.request_batches) == 2


def test_cancellation_dominates_duplicate_output_without_persistent_hit(tmp_path) -> None:
    request = _request()
    cancellation = _Cancellation()

    def cancel_with_duplicates(requests):
        cancellation.value = True
        return [_valid_record(requests[0], 1.0), _valid_record(requests[0], 1.25)]

    executor = _Executor(cancel_with_duplicates)
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    cancellation.value = False
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert session.overlay.lookup(request.key, request.protocol) is None
    assert _execute_request(session, request) == pytest.approx(1.25)
    assert len(executor.request_batches) == 2


def test_cancellation_arriving_during_validation_dominates_and_allows_retry(tmp_path) -> None:
    request = _request()
    other_protocol = _protocol(warmups=4)
    cancellation = _SequencedCancellation(False, False, True)
    executor = _Executor(lambda requests: [_valid_record_with_protocol(requests[0], other_protocol, 1.0)])
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as cancelled:
        _execute_request(session, request)
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in cancelled.value.reasons] == [UnresolvedCode.CANCELLED]
    assert _execute_request(session, request) == pytest.approx(1.25)


def test_late_executor_result_is_persisted_but_current_callback_exhausts_wall_budget(tmp_path) -> None:
    request = _request()
    clock = _Clock()

    def finish_late(requests):
        clock.advance(2.0)
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(finish_late)
    session = _session(
        tmp_path,
        executor,
        budget=ResolutionBudget(max_new_keys=1, max_wall_seconds=1.0),
        clock=clock,
    )

    with pytest.raises(ResolutionFailed) as late:
        _execute_request(session, request)

    assert [reason.code for reason in late.value.reasons] == [UnresolvedCode.BUDGET_EXHAUSTED]
    hit = session.overlay.lookup(request.key, request.protocol)
    assert hit is not None
    assert hit.latency_ms == pytest.approx(1.25)


def test_partial_valid_result_remains_queryable_when_sibling_is_missing(tmp_path) -> None:
    first = _request(op_id="first", m=8)
    missing = _request(op_id="missing", m=16)
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor)

    def query() -> float:
        total = 0.0
        for request in (first, missing):
            record = session.lookup(request.key)
            if record is None:
                session.record_miss(request, request.op_id)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(query)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.INVALID_MEASUREMENT]
    assert session.overlay.lookup(first.key, first.protocol) is not None
    assert session.overlay.lookup(missing.key, missing.protocol) is None


def test_two_cold_callbacks_share_one_collection_behind_callback_lock(tmp_path) -> None:
    request = _request()
    entered_executor = threading.Event()
    release_executor = threading.Event()

    def block_first_execution(requests):
        entered_executor.set()
        assert release_executor.wait(timeout=5.0)
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(block_first_execution)
    session = _session(tmp_path, executor)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_execute_request, session, request)
        assert entered_executor.wait(timeout=5.0)
        second = pool.submit(_execute_request, session, request)
        release_executor.set()
        results = [first.result(), second.result()]

    assert results == pytest.approx([1.25, 1.25])
    assert executor.request_batches == [(request,)]


def test_protocol_mismatch_rejects_entire_key_without_persisting_matching_record(tmp_path) -> None:
    request = _request()
    other_protocol = _protocol(warmups=4)
    executor = _Executor(
        lambda requests: [
            _valid_record(requests[0], 1.25),
            _valid_record_with_protocol(requests[0], other_protocol, 1.0),
        ]
    )
    session = _session(tmp_path, executor)

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.IDENTITY_MISMATCH]
    assert session.overlay.lookup(request.key, request.protocol) is None
    assert session.overlay.lookup(request.key, other_protocol) is None


def test_unrequested_record_is_rejected_and_never_persisted(tmp_path) -> None:
    request = _request(m=8)
    unrequested = _request(m=16)
    executor = _Executor(lambda requests: [_valid_record(unrequested, 1.0)])
    session = _session(tmp_path, executor)

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    assert [reason.code for reason in failure.value.reasons] == [
        UnresolvedCode.INVALID_MEASUREMENT,
        UnresolvedCode.INVALID_MEASUREMENT,
    ]
    assert session.overlay.lookup(request.key, request.protocol) is None
    assert session.overlay.lookup(unrequested.key, unrequested.protocol) is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (RecordStatus.REJECTED, UnresolvedCode.INVALID_MEASUREMENT),
        (RecordStatus.FAILED, UnresolvedCode.TIMEOUT),
    ],
)
def test_nonvalid_records_are_persisted_for_audit_but_never_become_hits(tmp_path, status, code) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_failed_record(requests[0], code, status=status)])
    session = _session(tmp_path, executor)

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    row = session.overlay._connection.execute("SELECT status, failure_code FROM measurement_records").fetchone()
    assert [reason.code for reason in failure.value.reasons] == [code]
    assert tuple(row) == (status.value, code.value)
    assert session.overlay.lookup(request.key, request.protocol) is None


def test_executor_exception_is_transient_collector_failure(tmp_path) -> None:
    request = _request()

    def raise_worker_loss(requests):
        del requests
        raise RuntimeError("worker lost")

    executor = _Executor(raise_worker_loss)
    session = _session(tmp_path, executor)

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.COLLECTOR_FAILED]
    assert session.overlay.lookup(request.key, request.protocol) is None


def test_executor_exception_after_cancellation_is_not_negative_or_transient_poison(tmp_path) -> None:
    request = _request()
    cancellation = _Cancellation()

    def cancel_then_raise(requests):
        del requests
        cancellation.value = True
        raise RuntimeError("cancelled worker")

    executor = _Executor(cancel_then_raise)
    session = _session(tmp_path, executor, cancellation=cancellation)

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)
    cancellation.value = False
    executor.responder = lambda requests: [_valid_record(requests[0], 1.25)]

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.CANCELLED]
    assert _execute_request(session, request) == pytest.approx(1.25)


def test_second_walk_that_still_records_miss_fails_without_third_query(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor)
    calls = 0

    def always_missing() -> float:
        nonlocal calls
        calls += 1
        session.record_miss(request, "gemm")
        return 0.0

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(always_missing)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.REQUERY_STILL_MISSING]
    assert calls == 2


def test_unique_miss_metric_does_not_recount_transient_attempts_for_same_key(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_failed_record(requests[0], UnresolvedCode.TIMEOUT)])
    session = _session(tmp_path, executor)

    for _ in range(2):
        with pytest.raises(ResolutionFailed):
            _execute_request(session, request)

    assert session.report.unique_misses == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_keys": -1, "max_wall_seconds": 1.0},
        {"max_new_keys": 1, "max_wall_seconds": -1.0},
        {"max_new_keys": 1, "max_wall_seconds": float("nan")},
        {"max_new_keys": 1, "max_wall_seconds": float("inf")},
        {"max_new_keys": 1, "max_wall_seconds": 1.0, "max_transient_attempts_per_key": -1},
    ],
)
def test_resolution_budget_rejects_invalid_limits(kwargs) -> None:
    with pytest.raises(ValueError):
        ResolutionBudget(**kwargs)


def test_executor_receives_absolute_remaining_deadline_and_cancellation_token(tmp_path) -> None:
    request = _request()
    clock = _Clock()
    clock.value = 10.0
    cancellation = _Cancellation()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(
        tmp_path,
        executor,
        budget=ResolutionBudget(max_new_keys=1, max_wall_seconds=3.0),
        clock=clock,
        cancellation=cancellation,
    )

    assert _execute_request(session, request) == pytest.approx(1.25)
    assert executor.deadlines == [pytest.approx(13.0)]
    assert executor.cancellation_tokens == [cancellation]


def test_request_protocol_mismatch_fails_before_executor_dispatch(tmp_path) -> None:
    request = _request(protocol=_protocol(warmups=4))
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = _session(tmp_path, executor, protocol=_protocol())

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.IDENTITY_MISMATCH]
    assert executor.request_batches == []


def test_observe_only_reports_miss_without_executor_dispatch(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=1, max_wall_seconds=1.0),
        request.protocol,
        policy=ResolutionPolicy.OBSERVE_ONLY,
    )

    with pytest.raises(ResolutionFailed) as failure:
        _execute_request(session, request)

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.OBSERVE_ONLY]
    assert executor.request_batches == []
    payload = session.report.to_dict()
    assert payload == {
        "overlay_hits": 0,
        "unique_misses": 1,
        "consumer_misses": 1,
        "accepted_records": 0,
        "rejected_records": 0,
        "collection_seconds": 0.0,
        "unresolved": [
            {
                "code": "observe_only",
                "operation": "gemm",
                "detail": f"observed unresolved key {request.key.digest}",
            }
        ],
    }
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_pure_policy_never_constructs_resolution_session(tmp_path) -> None:
    executor = _Executor(lambda requests: [])
    overlay = OverlayStore(tmp_path / "overlay.sqlite")

    with pytest.raises(ValueError, match="pure prediction"):
        ResolutionSession(
            overlay,
            executor,
            ResolutionBudget(max_new_keys=0, max_wall_seconds=0.0),
            _protocol(),
            policy=ResolutionPolicy.PURE,
        )


def test_missing_adapter_and_checkpoint_expose_structured_change(tmp_path) -> None:
    executor = _Executor(lambda requests: [])
    session = _session(tmp_path, executor)
    checkpoint = session.checkpoint()

    assert not session.changed_since(checkpoint)
    session.record_missing_adapter("attention", RuntimeError("no lazy adapter"))
    assert session.changed_since(checkpoint)

    with pytest.raises(ResolutionFailed) as failure:
        session.resolve_pending()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.MISSING_ADAPTER]


def test_resolve_pending_is_noop_when_nothing_was_discovered(tmp_path) -> None:
    executor = _Executor(lambda requests: [])
    session = _session(tmp_path, executor)

    session.resolve_pending()

    assert executor.request_batches == []


def test_overlay_fill_between_discovery_and_dispatch_avoids_executor(tmp_path) -> None:
    request = _request()
    executor = _Executor(lambda requests: pytest.fail("executor must not run after overlay race hit"))
    session = _session(tmp_path, executor)
    session.record_miss(request, "gemm")
    other = OverlayStore(session.overlay.path)
    other.append(_valid_record(request, 1.25))

    session.resolve_pending()

    assert executor.request_batches == []
    assert session.report.unique_misses == 1
    assert session.report.overlay_hits == 1
