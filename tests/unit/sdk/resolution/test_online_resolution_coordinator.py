# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import pytest

from aiconfigurator.sdk import resolution as resolution_api
from aiconfigurator.sdk.resolution.coordinator import OnlineResolutionCoordinator
from aiconfigurator.sdk.resolution.fallback import FallbackStore
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import (
    HybridFallbackValue,
    ResolutionBudget,
    ResolutionFailed,
    ResolutionSession,
)
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementFailureKind,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
)

pytestmark = pytest.mark.unit


def test_resolution_package_exports_online_coordinator() -> None:
    assert resolution_api.OnlineResolutionCoordinator is OnlineResolutionCoordinator
    assert resolution_api.MeasurementFailureKind is MeasurementFailureKind


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="microbench-v1",
        warmups=3,
        samples=1,
        statistic="median",
        timer="cuda_event",
        tuning_revision="none",
    )


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="h100_sxm",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="h100",
        runtime_versions={"cuda": "12.9"},
    )


def _request(*, op_id: str, m: int) -> MeasurementRequest:
    query = {"m": m, "n": 16, "k": 32}
    environment = _environment()
    return MeasurementRequest(
        op_id=op_id,
        key=PerfKey.build("gemm/v1", query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"dtype": "bf16"},
        protocol=_protocol(),
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


def _reopened_lookup(path, key: PerfKey, protocol: MeasurementProtocol):
    overlay = OverlayStore(path)
    try:
        return overlay.lookup(key, protocol)
    finally:
        overlay.close()


def _failed_record(request: MeasurementRequest, code: UnresolvedCode) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=RecordStatus.FAILED,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=request.protocol,
        perf_row={},
        provenance={"collector_revision": "r1"},
        failure_code=code,
        failure_reason=f"injected {code.value}",
    )


def _operational_timeout_record(request: MeasurementRequest) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=RecordStatus.FAILED,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=request.protocol,
        perf_row={},
        provenance={"collector_revision": "deadline-aware-r1"},
        failure_code=UnresolvedCode.TIMEOUT,
        failure_reason="request reached the cumulative boundary",
        failure_kind=MeasurementFailureKind.OPERATIONAL,
    )


class _Executor:
    def __init__(
        self,
        responder: Callable[[Sequence[MeasurementRequest]], Sequence[MeasurementRecord]],
    ) -> None:
        self.responder = responder
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        return self.responder(batch)


class _DeadlineAwareExecutor:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []
        self.deadline_monotonic: float | None = None
        self.cancellation = None
        self.close_calls = 0

    def execute(self, requests, *, deadline_monotonic, cancellation):
        batch = tuple(requests)
        self.request_batches.append(batch)
        self.deadline_monotonic = deadline_monotonic
        self.cancellation = cancellation
        self.entered.set()
        assert self.release.wait(timeout=5.0)
        return [_valid_record(batch[0], 1.25)]

    def close(self) -> None:
        self.close_calls += 1


class _QueuedBudgetExecutor:
    def __init__(
        self,
        clock: _ManualClock,
        first_entered: threading.Event,
        release_first: threading.Event,
        second_finished: threading.Event,
    ) -> None:
        self.clock = clock
        self.first_entered = first_entered
        self.release_first = release_first
        self.second_finished = second_finished
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []
        self.starts: list[float] = []
        self.deadlines: list[float] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        del cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        self.starts.append(self.clock())
        self.deadlines.append(deadline_monotonic)
        if len(self.request_batches) == 1:
            self.first_entered.set()
            assert self.release_first.wait(timeout=5.0)
        else:
            self.clock.value = 1.75
            self.second_finished.set()
        return [_valid_record(batch[0], 1.25)]


class _MixedCumulativeBoundaryExecutor:
    def __init__(
        self,
        clock: _ManualClock,
        *,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.clock = clock
        self.entered = entered
        self.release = release
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        del cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            assert self.release.wait(timeout=5.0)
        self.clock.value = deadline_monotonic
        return [
            _valid_record(batch[0], 1.25),
            MeasurementRecord(
                key=batch[1].key,
                status=RecordStatus.FAILED,
                latency_ms=None,
                energy_wms=0.0,
                samples_ms=(),
                protocol=batch[1].protocol,
                perf_row={},
                provenance={"collector_revision": "deadline-aware-r1"},
                failure_code=UnresolvedCode.TIMEOUT,
                failure_reason="second request reached the cumulative boundary",
                failure_kind=MeasurementFailureKind.OPERATIONAL,
            ),
        ]


class _SignalingOverlay(OverlayStore):
    def __init__(self, path, exact_appended: threading.Event) -> None:
        super().__init__(path)
        self._exact_appended = exact_appended

    def append(self, record: MeasurementRecord) -> int:
        sequence = super().append(record)
        if record.status is RecordStatus.VALID:
            self._exact_appended.set()
        return sequence


class _KeySignalingOverlay(OverlayStore):
    def __init__(self, path, key: PerfKey, appended: threading.Event) -> None:
        super().__init__(path)
        self._key = key
        self._appended = appended

    def append(self, record: MeasurementRecord) -> int:
        sequence = super().append(record)
        if record.key == self._key and record.status is RecordStatus.VALID:
            self._appended.set()
        return sequence


class _BlockingAppendOverlay(OverlayStore):
    def __init__(self, path, append_entered: threading.Event, release_append: threading.Event) -> None:
        super().__init__(path)
        self._append_entered = append_entered
        self._release_append = release_append
        self.close_calls = 0

    def append(self, record: MeasurementRecord) -> int:
        self._append_entered.set()
        assert self._release_append.wait(timeout=5.0)
        return super().append(record)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class _BlockingValidAppendOverlay(_BlockingAppendOverlay):
    def append(self, record: MeasurementRecord) -> int:
        if record.status is RecordStatus.VALID:
            self._append_entered.set()
            assert self._release_append.wait(timeout=5.0)
        return OverlayStore.append(self, record)


class _CloseTrackingOverlay(OverlayStore):
    def __init__(self, path, order: list[str]) -> None:
        super().__init__(path)
        self._order = order

    def close(self) -> None:
        self._order.append("overlay")
        super().close()


class _CloseTrackingExecutor(_Executor):
    def __init__(self, order: list[str]) -> None:
        super().__init__(lambda requests: [])
        self._order = order

    def close(self) -> None:
        self._order.append("executor")


class _FailingCloseExecutor(_CloseTrackingExecutor):
    def close(self) -> None:
        super().close()
        raise RuntimeError("injected executor cleanup failure")


class _BlockingCloseExecutor(_CloseTrackingExecutor):
    def __init__(
        self,
        order: list[str],
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        super().__init__(order)
        self._entered = entered
        self._release = release

    def close(self) -> None:
        self._order.append("executor:start")
        self._entered.set()
        assert self._release.wait(timeout=5.0)
        self._order.append("executor:done")


class _LateCommitSession(ResolutionSession):
    def __init__(self, *args, late_commit_entered: threading.Event, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._late_commit_entered = late_commit_entered

    def commit_late_valid_records(self, requests, records):
        self._late_commit_entered.set()
        return super().commit_late_valid_records(requests, records)


class _ExactBeforeFallbackCoordinator(OnlineResolutionCoordinator):
    def __init__(self, *args, race_writer: OverlayStore, race_record: MeasurementRecord, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._race_writer = race_writer
        self._race_record = race_record

    def _publish_hybrid_fallbacks(self, entries, failure) -> None:
        self._race_writer.append(self._race_record)
        super()._publish_hybrid_fallbacks(entries, failure)


class _ExactDuringFallbackPublicationSession(ResolutionSession):
    def __init__(self, *args, race_writer: OverlayStore, race_record: MeasurementRecord, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._race_writer = race_writer
        self._race_record = race_record

    def publish_fallback(self, request, value, failure):
        self._race_writer.append(self._race_record)
        return super().publish_fallback(request, value, failure)


class _ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class _ScriptedWaiter:
    def __init__(
        self,
        clock: _ManualClock,
        *,
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        self.clock = clock
        self.entered = entered
        self.release = release
        self.deadlines: list[float] = []

    def __call__(
        self,
        futures: set[Future[tuple[MeasurementRecord, ...]]],
        *,
        deadline_monotonic: float,
    ) -> set[Future[tuple[MeasurementRecord, ...]]]:
        self.deadlines.append(deadline_monotonic)
        assert self.entered.wait(timeout=5.0)
        if len(self.deadlines) == 1:
            self.clock.value = deadline_monotonic
            return set()
        self.release.set()
        for future in futures:
            future.result(timeout=5.0)
        return set(futures)


class _DeadlineOnlyWaiter:
    def __init__(self, clock: _ManualClock, entered: threading.Event) -> None:
        self.clock = clock
        self.entered = entered
        self.deadlines: list[float] = []

    def __call__(
        self,
        futures: set[Future[tuple[MeasurementRecord, ...]]],
        *,
        deadline_monotonic: float,
    ) -> set[Future[tuple[MeasurementRecord, ...]]]:
        assert futures
        assert self.entered.wait(timeout=5.0)
        self.deadlines.append(deadline_monotonic)
        self.clock.value = deadline_monotonic
        return set()


class _InjectedWaiterAbort(BaseException):
    pass


class _InjectedExecutorAbort(BaseException):
    pass


class _AbortAfterFutureCompletesWaiter:
    def __call__(
        self,
        futures: set[Future[tuple[MeasurementRecord, ...]]],
        *,
        deadline_monotonic: float,
    ) -> set[Future[tuple[MeasurementRecord, ...]]]:
        del deadline_monotonic
        for future in futures:
            future.result(timeout=5.0)
        raise _InjectedWaiterAbort("injected waiter abort")


def test_callback_block_deadline_starts_before_discovery_and_expired_callback_does_not_dispatch(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    clock = _ManualClock()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
        clock=clock,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
    )

    def query() -> float:
        clock.value = 1.0
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert session.report.callbacks[0]["collection"]["deadline_source"] == "callback_block"
    assert session.report.measurement_attempts == 0
    assert executor.request_batches == []


def test_remaining_cumulative_deadline_starts_before_discovery_and_expired_callback_does_not_dispatch(
    tmp_path,
) -> None:
    request = _request(op_id="a", m=8)
    clock = _ManualClock()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request.protocol,
        clock=clock,
    )
    coordinator = OnlineResolutionCoordinator(session, clock=clock)

    def query() -> float:
        clock.value = 2.0
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.BUDGET_EXHAUSTED]
    assert session.report.callbacks[0]["collection"]["deadline_source"] == "cumulative_budget"
    assert session.report.measurement_attempts == 0
    assert executor.request_batches == []


def test_waiter_base_exception_abandons_waiter_and_allows_safe_late_finalization(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    exact_appended = threading.Event()
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    overlay = _SignalingOverlay(tmp_path / "overlay.sqlite", exact_appended)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        wait_for_futures=_AbortAfterFutureCompletesWaiter(),
    )

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(_InjectedWaiterAbort, match="injected waiter abort"):
            coordinator.execute_callback(query)
        assert exact_appended.wait(timeout=1.0)
        assert coordinator._flights == {}
        assert coordinator._batches == {}
        assert session.report.late_completions == 1
    finally:
        coordinator.close()


def test_executor_base_exception_at_cumulative_deadline_is_preserved(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    clock = _ManualClock()

    def abort_at_deadline(requests):
        assert tuple(requests) == (request,)
        clock.value = 2.0
        raise _InjectedExecutorAbort("injected executor abort")

    executor = _Executor(abort_at_deadline)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request.protocol,
        clock=clock,
    )

    def wait_until_done(futures, *, deadline_monotonic):
        del deadline_monotonic
        for future in futures:
            future.result(timeout=5.0)
        return set(futures)

    coordinator = OnlineResolutionCoordinator(
        session,
        clock=clock,
        wait_for_futures=wait_until_done,
    )

    def query() -> float:
        session.record_miss(request, "a")
        return 0.0

    try:
        with pytest.raises(_InjectedExecutorAbort, match="injected executor abort"):
            coordinator.execute_callback(query)
        assert coordinator._flights == {}
        assert coordinator._batches == {}
        assert session.report.accepted_records == 0
    finally:
        coordinator.close()

    assert _reopened_lookup(overlay.path, request.key, request.protocol) is None


def test_coordinator_batches_complete_a_b_a_miss_set_and_replays_once(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    executor = _Executor(
        lambda requests: [_valid_record(request, 1.0 if request.key == request_a.key else 2.0) for request in requests]
    )
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        _protocol(),
    )
    coordinator = OnlineResolutionCoordinator(session)
    calls = 0

    def query() -> float:
        nonlocal calls
        calls += 1
        total = 0.0
        for request, consumer in (
            (request_a, "a.left"),
            (request_b, "b"),
            (request_a, "a.right"),
        ):
            record = session.lookup(request.key)
            if record is None:
                session.record_miss(request, consumer)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    assert coordinator.execute_callback(query) == pytest.approx(4.0)
    assert calls == 2
    assert executor.request_batches == [(request_a, request_b)]
    assert session.report.unique_misses == 2
    assert session.report.consumer_misses == 3


def test_active_key_protocol_flight_is_joined_after_first_caller_deadline(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    clock = _ManualClock()

    def block_until_released(requests):
        entered.set()
        assert release.wait(timeout=5.0)
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(block_until_released)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
        clock=clock,
    )
    waiter = _ScriptedWaiter(clock, entered=entered, release=release)
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=waiter,
    )

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as first_failure:
            coordinator.execute_callback(query)
        assert "timeout" in str(first_failure.value)

        assert coordinator.execute_callback(query) == pytest.approx(1.25)
    finally:
        coordinator.close()

    assert executor.request_batches == [(request,)]
    assert session.report.measurement_attempts == 1
    assert session.report.single_flight_joins == 1
    assert waiter.deadlines == pytest.approx([1.0, 2.0])


def test_timed_out_a_b_batch_survives_a_only_delivery_for_later_b_lookup(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    entered = threading.Event()
    release = threading.Event()
    clock = _ManualClock()

    def finish_batch(requests):
        entered.set()
        assert release.wait(timeout=5.0)
        return [
            _valid_record(requests[0], 1.25),
            _valid_record(requests[1], 2.5),
        ]

    executor = _Executor(finish_batch)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request_a.protocol,
        clock=clock,
    )
    waiter = _ScriptedWaiter(clock, entered=entered, release=release)
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=waiter,
    )

    def query(requests: tuple[MeasurementRequest, ...]) -> float:
        total = 0.0
        for request in requests:
            record = session.lookup(request.key, request.protocol)
            if record is None:
                session.record_miss(request, request.op_id)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    try:
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(lambda: query((request_a, request_b)))

        assert coordinator.execute_callback(lambda: query((request_a,))) == pytest.approx(1.25)
        persisted_b = overlay.lookup(request_b.key, request_b.protocol)
        assert persisted_b is not None
        assert persisted_b.latency_ms == pytest.approx(2.5)
        assert coordinator._flights == {}
        assert coordinator._batches == {}

        assert coordinator.execute_callback(lambda: query((request_b,))) == pytest.approx(2.5)

        assert coordinator._flights == {}
        assert coordinator._batches == {}
    finally:
        release.set()
        coordinator.close()

    assert executor.request_batches == [(request_a, request_b)]
    assert session.report.measurement_attempts == 2
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is not None


def test_same_key_with_different_protocol_does_not_join_active_flight(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    protocol_b = MeasurementProtocol(
        revision="microbench-v2",
        warmups=request_a.protocol.warmups,
        samples=request_a.protocol.samples,
        statistic=request_a.protocol.statistic,
        timer=request_a.protocol.timer,
        tuning_revision=request_a.protocol.tuning_revision,
    )
    request_b = MeasurementRequest(
        op_id=request_a.op_id,
        key=request_a.key,
        query=request_a.query,
        environment=request_a.environment,
        semantic_descriptor=request_a.semantic_descriptor,
        protocol=protocol_b,
    )
    entered = threading.Event()
    release = threading.Event()
    second_finished = threading.Event()
    clock = _ManualClock()

    def block_until_released(requests):
        entered.set()
        assert release.wait(timeout=5.0)
        request = requests[0]
        if request.protocol == protocol_b:
            second_finished.set()
        return [_valid_record(request, 1.25)]

    executor = _Executor(block_until_released)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request_a.protocol,
        clock=clock,
    )
    waiter = _DeadlineOnlyWaiter(clock, entered)
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=waiter,
    )

    def query(request: MeasurementRequest) -> float:
        session.record_miss(request, request.op_id)
        return 0.0

    try:
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(lambda: query(request_a))
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(lambda: query(request_b))

        assert session.report.single_flight_joins == 0
        assert set(coordinator._flights) == {
            (request_a.key, request_a.protocol),
            (request_b.key, request_b.protocol),
        }
        assert len(coordinator._batches) == 2
    finally:
        release.set()
        assert second_finished.wait(timeout=5.0)
        coordinator.close()

    assert executor.request_batches == [(request_a,), (request_b,)]


def test_late_valid_result_is_charged_once_and_persisted_before_next_callback(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    exact_appended = threading.Event()
    clock = _ManualClock()

    def finish_inside_cumulative_budget(requests):
        entered.set()
        assert release.wait(timeout=5.0)
        clock.value = 3.0
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(finish_inside_cumulative_budget)
    overlay = _SignalingOverlay(tmp_path / "overlay.sqlite", exact_appended)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
        clock=clock,
    )
    waiter = _DeadlineOnlyWaiter(clock, entered)
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=waiter,
    )
    calls = 0

    def query() -> float:
        nonlocal calls
        calls += 1
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(query)
        assert calls == 1

        release.set()
        assert exact_appended.wait(timeout=1.0)
        persisted = overlay.lookup(request.key, request.protocol)
        assert persisted is not None
        assert persisted.latency_ms == pytest.approx(1.25)
        assert session.report.collection_seconds == pytest.approx(3.0)
        assert session.report.late_completions == 1

        assert coordinator.execute_callback(query) == pytest.approx(1.25)
    finally:
        coordinator.close()

    assert calls == 2
    assert executor.request_batches == [(request,)]
    assert session.report.measurement_attempts == 1
    assert session.report.single_flight_joins == 0


def test_cumulative_deadline_cancels_active_flight_and_rejects_late_exact(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    clock = _ManualClock()
    executor = _DeadlineAwareExecutor(entered, release)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request.protocol,
        clock=clock,
    )
    waiter = _DeadlineOnlyWaiter(clock, entered)
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=waiter,
    )

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
        assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.TIMEOUT]
        assert executor.deadline_monotonic == pytest.approx(2.0)

        clock.value = 2.5
        assert executor.cancellation is not None
        assert executor.cancellation.cancelled()
    finally:
        release.set()
        coordinator.close()

    assert _reopened_lookup(overlay.path, request.key, request.protocol) is None
    assert session.report.late_completions == 0


@pytest.mark.parametrize("max_block_seconds", [None, 2.0, 3.0])
def test_effective_cumulative_wait_deadline_reports_budget_not_operational_timeout(
    tmp_path,
    max_block_seconds,
) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    clock = _ManualClock()
    executor = _DeadlineAwareExecutor(entered, release)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request.protocol,
        clock=clock,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=max_block_seconds,
        clock=clock,
        wait_for_futures=_DeadlineOnlyWaiter(clock, entered),
    )

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
        assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.BUDGET_EXHAUSTED]
        assert session.report.callbacks[0]["collection"]["deadline_source"] == "cumulative_budget"
        assert executor.cancellation is not None
        assert executor.cancellation.cancelled()
    finally:
        release.set()
        coordinator.close()


def test_cumulative_boundary_preserves_valid_a_and_only_degrades_timed_out_b(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.resolution.session")
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    clock = _ManualClock()
    executor = _MixedCumulativeBoundaryExecutor(clock)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request_a.protocol,
        clock=clock,
    )
    hybrid_calls = {"a": 0, "b": 0}

    def wait_until_done(futures, *, deadline_monotonic):
        del deadline_monotonic
        for future in futures:
            future.result(timeout=5.0)
        return set(futures)

    coordinator = OnlineResolutionCoordinator(
        session,
        clock=clock,
        wait_for_futures=wait_until_done,
        fallback_store=FallbackStore(tmp_path / "fallbacks"),
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )

    def query_one(request: MeasurementRequest, fallback_latency: float) -> float:
        record = session.lookup(request.key, request.protocol)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        fallback = session.lookup_fallback(request.key, request.op_id)
        if fallback is not None:
            return fallback.latency_ms

        def resolve_hybrid() -> HybridFallbackValue:
            hybrid_calls[request.op_id] += 1
            return HybridFallbackValue(
                latency_ms=fallback_latency,
                source="empirical",
                provenance={"operation": request.op_id},
            )

        session.record_miss(request, request.op_id, hybrid_resolver=resolve_hybrid)
        return 0.0

    try:
        result = coordinator.execute_callback(lambda: query_one(request_a, 100.0) + query_one(request_b, 4.0))
    finally:
        coordinator.close()

    assert result == pytest.approx(5.25)
    assert hybrid_calls == {"a": 0, "b": 1}
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is None
    assert session.report.callbacks[0]["collection"]["deadline_source"] == "cumulative_budget"
    fallback_warnings = [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    assert len(fallback_warnings) == 1
    assert fallback_warnings[0].key_digest == request_b.key.digest


def test_concurrent_exact_writer_wins_before_hybrid_publication(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.resolution.session")
    request = _request(op_id="a", m=8)
    overlay_path = tmp_path / "overlay.sqlite"
    overlay = OverlayStore(overlay_path)
    race_writer = OverlayStore(overlay_path)
    executor = _Executor(lambda requests: [_operational_timeout_record(requests[0])])
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    fallback_dir = tmp_path / "fallbacks"
    hybrid_calls = 0
    query_calls = 0
    coordinator = _ExactBeforeFallbackCoordinator(
        session,
        fallback_store=FallbackStore(fallback_dir),
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
        race_writer=race_writer,
        race_record=_valid_record(request, 1.25),
    )

    def query() -> float:
        nonlocal hybrid_calls, query_calls
        query_calls += 1
        record = session.lookup(request.key, request.protocol)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        fallback = session.lookup_fallback(request.key, request.op_id)
        if fallback is not None:
            return fallback.latency_ms

        def resolve_hybrid() -> HybridFallbackValue:
            nonlocal hybrid_calls
            hybrid_calls += 1
            return HybridFallbackValue(
                latency_ms=100.0,
                source="empirical",
                provenance={"operation": request.op_id},
            )

        session.record_miss(request, request.op_id, hybrid_resolver=resolve_hybrid)
        return 0.0

    try:
        result = coordinator.execute_callback(query)
    finally:
        coordinator.close()
        race_writer.close()

    assert result == pytest.approx(1.25)
    assert query_calls == 2
    assert hybrid_calls == 0
    assert session.report.hybrid_publications == 0
    assert not fallback_dir.exists() or list(fallback_dir.iterdir()) == []
    assert not [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    assert _reopened_lookup(overlay_path, request.key, request.protocol) is not None


def test_exact_published_by_hybrid_resolver_wins_before_sidecar_publication(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.resolution.session")
    request = _request(op_id="a", m=8)
    overlay_path = tmp_path / "overlay.sqlite"
    overlay = OverlayStore(overlay_path)
    race_writer = OverlayStore(overlay_path)
    executor = _Executor(lambda requests: [_operational_timeout_record(requests[0])])
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    fallback_dir = tmp_path / "fallbacks"
    hybrid_calls = 0
    query_calls = 0
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=FallbackStore(fallback_dir),
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )

    def query() -> float:
        nonlocal hybrid_calls, query_calls
        query_calls += 1
        record = session.lookup(request.key, request.protocol)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        fallback = session.lookup_fallback(request.key, request.op_id)
        if fallback is not None:
            return fallback.latency_ms

        def resolve_hybrid() -> HybridFallbackValue:
            nonlocal hybrid_calls
            hybrid_calls += 1
            race_writer.append(_valid_record(request, 1.25))
            return HybridFallbackValue(
                latency_ms=100.0,
                source="empirical",
                provenance={"operation": request.op_id},
            )

        session.record_miss(request, request.op_id, hybrid_resolver=resolve_hybrid)
        return 0.0

    try:
        result = coordinator.execute_callback(query)
    finally:
        coordinator.close()
        race_writer.close()

    assert result == pytest.approx(1.25)
    assert query_calls == 2
    assert hybrid_calls == 1
    assert session.report.hybrid_publications == 0
    assert not fallback_dir.exists() or list(fallback_dir.iterdir()) == []
    assert not [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    assert _reopened_lookup(overlay_path, request.key, request.protocol) is not None


def test_exact_after_post_resolver_recheck_safely_coexists_and_wins_replay(tmp_path, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="aiconfigurator.sdk.resolution.session")
    request = _request(op_id="a", m=8)
    overlay_path = tmp_path / "overlay.sqlite"
    overlay = OverlayStore(overlay_path)
    race_writer = OverlayStore(overlay_path)
    executor = _Executor(lambda requests: [_operational_timeout_record(requests[0])])
    session = _ExactDuringFallbackPublicationSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
        race_writer=race_writer,
        race_record=_valid_record(request, 1.25),
    )
    fallback_dir = tmp_path / "fallbacks"
    hybrid_calls = 0
    query_calls = 0
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=FallbackStore(fallback_dir),
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )

    def query() -> float:
        nonlocal hybrid_calls, query_calls
        query_calls += 1
        record = session.lookup(request.key, request.protocol)
        if record is not None:
            assert record.latency_ms is not None
            return record.latency_ms
        fallback = session.lookup_fallback(request.key, request.op_id)
        if fallback is not None:
            return fallback.latency_ms

        def resolve_hybrid() -> HybridFallbackValue:
            nonlocal hybrid_calls
            hybrid_calls += 1
            return HybridFallbackValue(
                latency_ms=100.0,
                source="empirical",
                provenance={"operation": request.op_id},
            )

        session.record_miss(request, request.op_id, hybrid_resolver=resolve_hybrid)
        return 0.0

    try:
        result = coordinator.execute_callback(query)
    finally:
        coordinator.close()
        race_writer.close()

    assert result == pytest.approx(1.25)
    assert query_calls == 2
    assert hybrid_calls == 1
    assert session.report.hybrid_publications == 1
    assert len(list(fallback_dir.glob("*.json"))) == 1
    fallback_warnings = [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    assert len(fallback_warnings) == 1
    assert _reopened_lookup(overlay_path, request.key, request.protocol) is not None


def test_post_deadline_mixed_result_commits_only_valid_a_without_changing_timed_out_caller(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    entered = threading.Event()
    release = threading.Event()
    exact_a_appended = threading.Event()
    clock = _ManualClock()
    executor = _MixedCumulativeBoundaryExecutor(
        clock,
        entered=entered,
        release=release,
    )
    overlay = _KeySignalingOverlay(tmp_path / "overlay.sqlite", request_a.key, exact_a_appended)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request_a.protocol,
        clock=clock,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        clock=clock,
        wait_for_futures=_DeadlineOnlyWaiter(clock, entered),
    )
    query_calls = 0

    def query() -> float:
        nonlocal query_calls
        query_calls += 1
        for request in (request_a, request_b):
            if session.lookup(request.key, request.protocol) is None:
                session.record_miss(request, request.op_id)
        return 0.0

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
        observed_codes = [reason.code for reason in failure.value.reasons]
        assert observed_codes == [
            UnresolvedCode.BUDGET_EXHAUSTED,
            UnresolvedCode.BUDGET_EXHAUSTED,
        ]
        assert query_calls == 1
        callback_snapshot = session.report.to_dict()["callbacks"][0]

        release.set()
        assert exact_a_appended.wait(timeout=1.0)
        assert [reason.code for reason in failure.value.reasons] == observed_codes
        assert query_calls == 1
        assert session.report.to_dict()["callbacks"][0] == callback_snapshot
    finally:
        release.set()
        coordinator.close()

    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is None
    assert session.report.callbacks[0]["replay"]["attempted"] is False


@pytest.mark.parametrize("malformed_result", ["unrequested_timeout", "duplicate_timeout"])
def test_malformed_deadline_aware_batch_cannot_authorize_post_deadline_valid_records(
    tmp_path,
    malformed_result,
) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    request_c = _request(op_id="c", m=32)
    clock = _ManualClock()

    def return_malformed_batch(requests):
        assert tuple(requests) == (request_a, request_b)
        clock.value = 2.0
        if malformed_result == "unrequested_timeout":
            return [
                _valid_record(request_a, 1.25),
                _valid_record(request_b, 2.5),
                _operational_timeout_record(request_c),
            ]
        return [
            _valid_record(request_a, 1.25),
            _operational_timeout_record(request_b),
            _operational_timeout_record(request_b),
        ]

    executor = _Executor(return_malformed_batch)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request_a.protocol,
        clock=clock,
    )

    def wait_until_done(futures, *, deadline_monotonic):
        del deadline_monotonic
        for future in futures:
            future.result(timeout=5.0)
        return set(futures)

    coordinator = OnlineResolutionCoordinator(
        session,
        clock=clock,
        wait_for_futures=wait_until_done,
    )

    def query() -> float:
        total = 0.0
        for request in (request_a, request_b):
            record = session.lookup(request.key, request.protocol)
            if record is None:
                session.record_miss(request, request.op_id)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [
        UnresolvedCode.BUDGET_EXHAUSTED,
        UnresolvedCode.BUDGET_EXHAUSTED,
    ]
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is None
    assert session.report.accepted_records == 0


def test_post_deadline_valid_plus_invalid_measurement_fails_entire_batch_closed(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    clock = _ManualClock()

    def return_valid_plus_invalid(requests):
        assert tuple(requests) == (request_a, request_b)
        clock.value = 2.0
        return [
            _valid_record(request_a, 1.25),
            _failed_record(request_b, UnresolvedCode.INVALID_MEASUREMENT),
        ]

    executor = _Executor(return_valid_plus_invalid)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request_a.protocol,
        clock=clock,
    )
    coordinator = OnlineResolutionCoordinator(session, clock=clock)

    def query() -> float:
        for request in (request_a, request_b):
            if session.lookup(request.key, request.protocol) is None:
                session.record_miss(request, request.op_id)
        return 0.0

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [
        UnresolvedCode.BUDGET_EXHAUSTED,
        UnresolvedCode.BUDGET_EXHAUSTED,
    ]
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is None
    assert session.report.accepted_records == 0


def test_queued_batch_uses_actual_execution_time_and_remaining_session_budget(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    second_appended = threading.Event()
    clock = _ManualClock()
    executor = _QueuedBudgetExecutor(
        clock,
        first_entered,
        release_first,
        second_finished,
    )
    overlay = _KeySignalingOverlay(tmp_path / "overlay.sqlite", request_b.key, second_appended)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request_a.protocol,
        clock=clock,
    )
    wait_calls = 0

    def timeout_without_releasing_first(futures, *, deadline_monotonic):
        nonlocal wait_calls
        assert futures
        wait_calls += 1
        if wait_calls == 1:
            assert first_entered.wait(timeout=5.0)
        clock.value = deadline_monotonic
        return set()

    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=0.5,
        clock=clock,
        wait_for_futures=timeout_without_releasing_first,
    )

    def query(request: MeasurementRequest) -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, request.op_id)
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(lambda: query(request_a))
        with pytest.raises(ResolutionFailed):
            coordinator.execute_callback(lambda: query(request_b))

        clock.value = 1.5
        release_first.set()
        assert second_finished.wait(timeout=5.0)
        assert second_appended.wait(timeout=5.0)
    finally:
        release_first.set()
        coordinator.close()

    assert executor.starts == pytest.approx([0.0, 1.5])
    assert executor.deadlines == pytest.approx([2.0, 2.0])
    assert session.report.collection_seconds == pytest.approx(1.75)
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is not None


def test_done_future_after_cumulative_deadline_is_budget_failure_not_exact(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    clock = _ManualClock()

    def return_valid_after_budget(requests):
        clock.value = 2.5
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(return_valid_after_budget)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=2.0),
        request.protocol,
        clock=clock,
    )

    def wait_until_done(futures, *, deadline_monotonic):
        del deadline_monotonic
        for future in futures:
            future.result(timeout=5.0)
        return set(futures)

    coordinator = OnlineResolutionCoordinator(
        session,
        clock=clock,
        wait_for_futures=wait_until_done,
    )

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.BUDGET_EXHAUSTED]
    assert _reopened_lookup(overlay.path, request.key, request.protocol) is None
    assert session.report.accepted_records == 0


def test_executor_exception_releases_delivered_single_flight_state(tmp_path) -> None:
    request = _request(op_id="a", m=8)

    def raise_executor_failure(requests):
        assert tuple(requests) == (request,)
        raise RuntimeError("injected executor failure")

    executor = _Executor(raise_executor_failure)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)

    def query() -> float:
        session.record_miss(request, "a")
        return 0.0

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)

        assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.COLLECTOR_FAILED]
        assert [reason.failure_kind for reason in failure.value.reasons] == [MeasurementFailureKind.INVARIANT]
        assert coordinator._flights == {}
        assert coordinator._batches == {}
    finally:
        coordinator.close()


def test_late_commit_waits_for_unrelated_callback_and_does_not_contaminate_trace(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    executor_entered = threading.Event()
    release_executor = threading.Event()
    exact_appended = threading.Event()
    late_commit_entered = threading.Event()
    unrelated_active = threading.Event()
    release_unrelated = threading.Event()
    clock = _ManualClock()

    def finish_late(requests):
        executor_entered.set()
        assert release_executor.wait(timeout=5.0)
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(finish_late)
    overlay = _SignalingOverlay(tmp_path / "overlay.sqlite", exact_appended)
    session = _LateCommitSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
        clock=clock,
        late_commit_entered=late_commit_entered,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=_DeadlineOnlyWaiter(clock, executor_entered),
    )

    def cold_query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(cold_query)

    unrelated_result: list[float] = []

    def run_unrelated_callback() -> None:
        def unrelated_query() -> float:
            unrelated_active.set()
            assert release_unrelated.wait(timeout=5.0)
            return 7.0

        unrelated_result.append(coordinator.execute_callback(unrelated_query))

    unrelated_thread = threading.Thread(target=run_unrelated_callback)
    unrelated_thread.start()
    try:
        assert unrelated_active.wait(timeout=5.0)
        release_executor.set()
        assert late_commit_entered.wait(timeout=5.0)
        assert not exact_appended.wait(timeout=0.05)
    finally:
        release_unrelated.set()
        unrelated_thread.join(timeout=5.0)
        release_executor.set()
        coordinator.close()

    assert not unrelated_thread.is_alive()
    assert unrelated_result == [7.0]
    assert exact_appended.wait(timeout=5.0)
    assert session.report.callbacks[1]["collection"]["accepted_records"] == 0
    assert session.report.callbacks[1]["final_evidence"] == []
    assert session.report.accepted_records == 1
    assert session.report.late_completions == 1


def test_late_finalization_cannot_block_measurement_worker_for_next_cold_callback(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    first_started = threading.Event()
    release_first = threading.Event()
    second_waiting = threading.Event()
    release_second_waiter = threading.Event()
    second_started = threading.Event()
    late_commit_entered = threading.Event()
    clock = _ManualClock()

    def execute(requests):
        assert len(requests) == 1
        request = requests[0]
        if request == request_a:
            first_started.set()
            assert release_first.wait(timeout=5.0)
            return [_valid_record(request, 1.25)]
        assert request == request_b
        second_started.set()
        return [_valid_record(request, 2.5)]

    waiter_calls = 0

    def wait_for_futures(futures, *, deadline_monotonic):
        nonlocal waiter_calls
        waiter_calls += 1
        if waiter_calls == 1:
            assert first_started.wait(timeout=5.0)
            clock.value = deadline_monotonic
            return set()
        assert waiter_calls == 2
        second_waiting.set()
        assert release_second_waiter.wait(timeout=5.0)
        return {future for future in futures if future.done()}

    executor = _Executor(execute)
    overlay = _SignalingOverlay(tmp_path / "overlay.sqlite", threading.Event())
    session = _LateCommitSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request_a.protocol,
        clock=clock,
        late_commit_entered=late_commit_entered,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        clock=clock,
        wait_for_futures=wait_for_futures,
    )

    def query(request: MeasurementRequest) -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, request.op_id)
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(lambda: query(request_a))

    second_results: list[float] = []
    second_errors: list[BaseException] = []

    def run_second_callback() -> None:
        try:
            second_results.append(coordinator.execute_callback(lambda: query(request_b)))
        except BaseException as error:
            second_errors.append(error)

    second_thread = threading.Thread(target=run_second_callback)
    second_thread.start()
    try:
        assert second_waiting.wait(timeout=5.0)
        release_first.set()
        assert late_commit_entered.wait(timeout=5.0)
        assert second_started.wait(timeout=0.1)
    finally:
        release_second_waiter.set()
        release_first.set()
        second_thread.join(timeout=5.0)
        coordinator.close()

    assert not second_thread.is_alive()
    assert second_errors == []
    assert second_results == [pytest.approx(2.5)]
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is not None
    assert session.report.collection_seconds == pytest.approx(1.0)
    assert session.report.late_completions == 1
    assert coordinator._finalizations == set()
    assert coordinator._finalizer_shutdown
    assert not any(thread.is_alive() for thread in coordinator._finalizer._threads)


def test_close_signals_active_flight_and_returns_within_bound(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    executor = _DeadlineAwareExecutor(entered, release)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )

    def timeout_waiter(futures, *, deadline_monotonic):
        del deadline_monotonic
        assert futures
        assert entered.wait(timeout=5.0)
        return set()

    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=0.01,
        wait_for_futures=timeout_waiter,
    )

    def query() -> float:
        session.record_miss(request, "a")
        return 0.0

    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(query)

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="did not stop within"):
            coordinator.close(timeout_seconds=0.05)
        assert time.monotonic() - started < 0.5
        assert executor.cancellation is not None
        assert executor.cancellation.cancelled()
        with pytest.raises(RuntimeError, match="closed"):
            coordinator.execute_callback(lambda: 0.0)
    finally:
        release.set()

    coordinator.close(timeout_seconds=1.0)
    coordinator.close(timeout_seconds=1.0)
    assert executor.close_calls == 1


def test_cancel_active_signals_inflight_work_without_waiting_for_resource_cleanup(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    entered = threading.Event()
    release = threading.Event()
    executor = _DeadlineAwareExecutor(entered, release)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)
    callback_errors: list[BaseException] = []

    def query() -> float:
        session.record_miss(request, "a")
        return 1.0

    def run_callback() -> None:
        try:
            coordinator.execute_callback(query)
        except BaseException as error:
            callback_errors.append(error)

    callback = threading.Thread(
        target=run_callback,
        daemon=True,
    )
    callback.start()
    assert entered.wait(timeout=5.0)

    started = time.monotonic()
    coordinator.cancel_active()
    assert time.monotonic() - started < 0.5
    assert executor.cancellation is not None
    assert executor.cancellation.cancelled()
    assert executor.close_calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.execute_callback(lambda: 0.0)

    release.set()
    callback.join(timeout=5.0)
    assert not callback.is_alive()
    assert callback_errors
    coordinator.close(timeout_seconds=1.0)
    assert executor.close_calls == 1


def test_close_waits_for_active_callback_writer_before_closing_overlay(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    append_entered = threading.Event()
    release_append = threading.Event()
    overlay = _BlockingAppendOverlay(
        tmp_path / "overlay.sqlite",
        append_entered,
        release_append,
    )
    executor = _Executor(lambda requests: [_valid_record(requests[0], 1.25)])
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)
    callback_results: list[float] = []
    callback_errors: list[BaseException] = []

    def query() -> float:
        record = session.lookup(request.key, request.protocol)
        if record is None:
            session.record_miss(request, "a")
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    def run_callback() -> None:
        try:
            callback_results.append(coordinator.execute_callback(query))
        except BaseException as error:
            callback_errors.append(error)

    callback_thread = threading.Thread(target=run_callback)
    callback_thread.start()
    assert append_entered.wait(timeout=5.0)
    assert coordinator._flights == {}
    assert coordinator._batches == {}

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="resource cleanup"):
            coordinator.close(timeout_seconds=0.05)
        assert time.monotonic() - started < 0.5
        assert overlay.close_calls == 0
    finally:
        release_append.set()
        callback_thread.join(timeout=5.0)
        coordinator.close(timeout_seconds=1.0)

    assert not callback_thread.is_alive()
    assert callback_errors == []
    assert callback_results == [pytest.approx(1.25)]
    assert overlay.close_calls == 1


def test_close_bounds_autonomous_late_finalization_before_resource_cleanup(tmp_path) -> None:
    request = _request(op_id="a", m=8)
    executor_entered = threading.Event()
    release_executor = threading.Event()
    append_entered = threading.Event()
    release_append = threading.Event()
    overlay = _BlockingValidAppendOverlay(
        tmp_path / "overlay.sqlite",
        append_entered,
        release_append,
    )

    def finish_late(requests):
        executor_entered.set()
        assert release_executor.wait(timeout=5.0)
        return [_valid_record(requests[0], 1.25)]

    executor = _Executor(finish_late)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request.protocol,
    )

    def timeout_waiter(futures, *, deadline_monotonic):
        del deadline_monotonic
        assert futures
        assert executor_entered.wait(timeout=5.0)
        return set()

    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=0.01,
        wait_for_futures=timeout_waiter,
    )

    def query() -> float:
        session.record_miss(request, request.op_id)
        return 0.0

    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(query)

    release_executor.set()
    assert append_entered.wait(timeout=5.0)
    safety_release = threading.Timer(0.5, release_append.set)
    safety_release.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="finalization did not stop"):
            coordinator.close(timeout_seconds=0.05)
        assert time.monotonic() - started < 0.25
        assert overlay.close_calls == 0
    finally:
        release_append.set()
        safety_release.cancel()

    coordinator.close(timeout_seconds=1.0)
    assert overlay.close_calls == 1
    assert coordinator._finalizations == set()
    assert not any(thread.is_alive() for thread in coordinator._finalizer._threads)


def test_close_orders_executor_before_overlay_and_is_idempotent(tmp_path) -> None:
    order: list[str] = []
    protocol = _protocol()
    overlay = _CloseTrackingOverlay(tmp_path / "overlay.sqlite", order)
    session = ResolutionSession(
        overlay,
        _CloseTrackingExecutor(order),
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)

    coordinator.close()
    coordinator.close()

    assert order == ["executor", "overlay"]
    with pytest.raises(RuntimeError, match="closed"):
        overlay.lookup(_request(op_id="a", m=8).key, protocol)


def test_close_timeout_bounds_resource_cleanup_and_retry_joins_cleanup_thread(tmp_path) -> None:
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()
    protocol = _protocol()
    overlay = _CloseTrackingOverlay(tmp_path / "overlay.sqlite", order)
    session = ResolutionSession(
        overlay,
        _BlockingCloseExecutor(order, entered, release),
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)
    safety_release = threading.Timer(0.5, release.set)
    safety_release.start()

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="resource cleanup"):
            coordinator.close(timeout_seconds=0.05)
        assert time.monotonic() - started < 0.25
        assert entered.is_set()
        assert order == ["executor:start"]
        assert overlay.lookup(_request(op_id="a", m=8).key, protocol) is None
    finally:
        release.set()
        safety_release.cancel()

    coordinator.close(timeout_seconds=1.0)
    assert order == ["executor:start", "executor:done", "overlay"]
    assert coordinator._cleanup_thread is not None
    assert not coordinator._cleanup_thread.is_alive()


def test_close_cancels_queued_batch_without_leaking_flights_or_background_thread(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    first_entered = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    clock = _ManualClock()
    executor = _QueuedBudgetExecutor(
        clock,
        first_entered,
        release_first,
        second_finished,
    )
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request_a.protocol,
        clock=clock,
    )
    wait_calls = 0

    def timeout_waiter(futures, *, deadline_monotonic):
        nonlocal wait_calls
        del deadline_monotonic
        assert futures
        wait_calls += 1
        if wait_calls == 1:
            assert first_entered.wait(timeout=5.0)
        return set()

    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=0.5,
        clock=clock,
        wait_for_futures=timeout_waiter,
    )

    def query(request: MeasurementRequest) -> float:
        session.record_miss(request, request.op_id)
        return 0.0

    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(lambda: query(request_a))
    with pytest.raises(ResolutionFailed):
        coordinator.execute_callback(lambda: query(request_b))

    try:
        with pytest.raises(RuntimeError, match="did not stop within"):
            coordinator.close(timeout_seconds=0.05)
        assert (request_b.key, request_b.protocol) not in coordinator._flights
        assert len(coordinator._batches) == 1
    finally:
        release_first.set()

    coordinator.close(timeout_seconds=1.0)
    assert coordinator._flights == {}
    assert coordinator._batches == {}
    assert not second_finished.is_set()
    assert not any(thread.is_alive() for thread in coordinator._background._threads)


def test_context_manager_preserves_primary_error_and_logs_cleanup_failure(tmp_path, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="aiconfigurator.sdk.resolution.coordinator")
    order: list[str] = []
    protocol = _protocol()
    session = ResolutionSession(
        _CloseTrackingOverlay(tmp_path / "overlay.sqlite", order),
        _FailingCloseExecutor(order),
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)

    def fail_primary() -> float:
        raise ValueError("primary callback failure")

    with pytest.raises(ValueError, match="primary callback failure"), coordinator:
        coordinator.execute_callback(fail_primary)

    assert order == ["executor", "overlay"]
    cleanup_failures = [
        record for record in caplog.records if getattr(record, "event", None) == "aic_online_resolution_cleanup_failed"
    ]
    assert len(cleanup_failures) == 1
    assert cleanup_failures[0].levelno == logging.ERROR
    assert cleanup_failures[0].cleanup_error == "injected executor cleanup failure"


def test_partial_valid_exact_is_committed_before_typed_sibling_failure(tmp_path) -> None:
    request_a = _request(op_id="a", m=8)
    request_b = _request(op_id="b", m=16)
    executor = _Executor(
        lambda requests: [
            _valid_record(requests[0], 1.25),
            _failed_record(requests[1], UnresolvedCode.TIMEOUT),
        ]
    )
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        request_a.protocol,
    )
    coordinator = OnlineResolutionCoordinator(session)
    calls = 0

    def query() -> float:
        nonlocal calls
        calls += 1
        total = 0.0
        for request in (request_a, request_b):
            record = session.lookup(request.key, request.protocol)
            if record is None:
                session.record_miss(request, request.op_id)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(query)
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.TIMEOUT]
    assert calls == 1
    assert _reopened_lookup(overlay.path, request_a.key, request_a.protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, request_b.protocol) is None
    assert session.report.accepted_records == 1
    assert session.report.rejected_records == 1
