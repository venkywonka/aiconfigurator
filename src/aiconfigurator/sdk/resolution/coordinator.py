# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live online resolution coordination."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from aiconfigurator.sdk.resolution.fallback import FallbackStore
from aiconfigurator.sdk.resolution.session import (
    CancellationToken,
    MissEntry,
    ResolutionFailed,
    ResolutionSession,
)
from aiconfigurator.sdk.resolution.types import (
    MeasurementFailureKind,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
    UnresolvedReason,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)
_FlightIdentity: TypeAlias = tuple[PerfKey, MeasurementProtocol]
_BatchFuture: TypeAlias = Future["_BatchResult"]
_CleanupFuture: TypeAlias = Future[tuple[BaseException, ...]]
_FinalizationFuture: TypeAlias = Future[None]
_FutureWaiter: TypeAlias = Callable[..., set[_BatchFuture]]
_HYBRID_ELIGIBLE = frozenset(
    {
        UnresolvedCode.MISSING_ADAPTER,
        UnresolvedCode.UNSUPPORTED_SHAPE,
        UnresolvedCode.RESOURCE_UNAVAILABLE,
        UnresolvedCode.COLLECTOR_FAILED,
        UnresolvedCode.TIMEOUT,
        UnresolvedCode.INVALID_MEASUREMENT,
        UnresolvedCode.BUDGET_EXHAUSTED,
        UnresolvedCode.RETRY_EXHAUSTED,
        UnresolvedCode.REQUERY_STILL_MISSING,
    }
)


@dataclass(frozen=True, slots=True)
class _Flight:
    request: MeasurementRequest
    future: _BatchFuture


@dataclass(frozen=True, slots=True)
class _BatchResult:
    records: tuple[MeasurementRecord, ...]
    started_monotonic: float
    completed_monotonic: float
    deadline_monotonic: float
    cancelled_at_completion: bool
    error: BaseException | None = None


class _DeadlineCancellation:
    def __init__(
        self,
        parent: CancellationToken,
        *,
        deadline_monotonic: float,
        clock: Callable[[], float],
    ) -> None:
        self._parent = parent
        self._deadline_monotonic = deadline_monotonic
        self._clock = clock
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def set_deadline(self, deadline_monotonic: float) -> None:
        self._deadline_monotonic = deadline_monotonic

    def cancelled(self) -> bool:
        return self._cancelled.is_set() or self._parent.cancelled() or self._clock() >= self._deadline_monotonic


@dataclass(slots=True)
class _BatchState:
    future: _BatchFuture
    requests: tuple[MeasurementRequest, ...]
    cancellation: _DeadlineCancellation
    waiters: int = 0
    timed_out: bool = False
    delivered: bool = False
    accounted: bool = False
    late_finalized: bool = False


def _timeout_record(request: MeasurementRequest) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=RecordStatus.FAILED,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=request.protocol,
        perf_row={},
        provenance={"coordinator": "caller_deadline"},
        failure_code=UnresolvedCode.TIMEOUT,
        failure_reason="online resolution caller deadline expired",
        failure_kind=MeasurementFailureKind.OPERATIONAL,
    )


def _budget_record(request: MeasurementRequest) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=RecordStatus.FAILED,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=request.protocol,
        perf_row={},
        provenance={"coordinator": "cumulative_budget"},
        failure_code=UnresolvedCode.BUDGET_EXHAUSTED,
        failure_reason="online resolution cumulative collection budget exhausted before execution",
        failure_kind=MeasurementFailureKind.BUDGET,
    )


class OnlineResolutionCoordinator:
    """Coordinate one online resolution callback over its complete miss set."""

    def __init__(
        self,
        session: ResolutionSession,
        *,
        max_block_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        wait_for_futures: _FutureWaiter | None = None,
        fallback_store: FallbackStore | None = None,
        prediction_revision: str | None = None,
        on_measurement_failure: str = "error",
        force_remeasure: bool = False,
    ) -> None:
        if max_block_seconds is not None and (not math.isfinite(max_block_seconds) or max_block_seconds <= 0):
            raise ValueError("max_block_seconds must be finite and positive")
        if on_measurement_failure not in {"error", "hybrid"}:
            raise ValueError("on_measurement_failure must be 'error' or 'hybrid'")
        if on_measurement_failure == "hybrid" and fallback_store is None:
            raise ValueError("HYBRID measurement failure policy requires a fallback store")
        self._session = session
        self._max_block_seconds = max_block_seconds
        self._clock = clock
        self._wait_for_futures = wait_for_futures or self._default_wait_for_futures
        self._background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aic-online-resolution")
        self._finalizer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aic-online-finalization")
        self._flights: dict[_FlightIdentity, _Flight] = {}
        self._batches: dict[_BatchFuture, _BatchState] = {}
        self._finalizations: set[_FinalizationFuture] = set()
        self._finalization_errors: list[BaseException] = []
        self._lock = threading.RLock()
        self._callbacks_drained = threading.Condition(self._lock)
        self._active_callbacks = 0
        self._closed = False
        self._shutdown_complete = False
        self._background_shutdown = False
        self._finalizer_shutdown = False
        self._executor_closed = False
        self._overlay_closed = False
        self._cleanup_future: _CleanupFuture | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._on_measurement_failure = on_measurement_failure
        self._session.configure_fallback(
            fallback_store,
            prediction_revision=prediction_revision,
            force_remeasure=force_remeasure,
        )

    def execute_callback(
        self,
        query: Callable[[], T],
        *,
        context: Mapping[str, object] | None = None,
    ) -> T:
        with self._lock:
            if self._closed:
                raise RuntimeError("online resolution coordinator is closed")
            self._active_callbacks += 1
        try:
            callback_started = self._clock()
            caller_deadline = callback_started + self._session.remaining_collection_seconds()
            deadline_source = "cumulative_budget"
            if self._max_block_seconds is not None:
                block_deadline = callback_started + self._max_block_seconds
                if block_deadline < caller_deadline:
                    caller_deadline = block_deadline
                    deadline_source = "callback_block"

            def resolve_pending() -> None:
                self._resolve_pending(
                    caller_deadline_monotonic=caller_deadline,
                    deadline_source=deadline_source,
                )

            return self._session._execute_callback_with_resolver(
                query,
                resolve_pending,
                context=context,
            )
        finally:
            with self._callbacks_drained:
                self._active_callbacks -= 1
                self._callbacks_drained.notify_all()

    def __enter__(self) -> OnlineResolutionCoordinator:
        with self._lock:
            if self._closed:
                raise RuntimeError("online resolution coordinator is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            logger.exception(
                "online resolution cleanup failed while preserving primary error",
                extra={
                    "event": "aic_online_resolution_cleanup_failed",
                    "cleanup_error": str(cleanup_error),
                    "primary_error": str(exc),
                },
            )
        return False

    def cancel_active(self) -> None:
        """Stop admission and cooperatively cancel active measurement batches.

        This is the non-blocking half of teardown for callers such as Live
        Mocker that must roll back scheduler state immediately. ``close`` owns
        the subsequent bounded joins and resource cleanup.
        """

        self._request_cancellation()

    def _request_cancellation(self) -> set[_BatchFuture]:
        with self._lock:
            if self._shutdown_complete:
                return set()
            self._closed = True
            states = tuple(self._batches.values())
            for state in states:
                state.cancellation.cancel()
            futures = {state.future for state in states}
        for future in futures:
            future.cancel()
        return futures

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        shutdown_deadline = time.monotonic() + timeout_seconds
        with self._lock:
            if self._shutdown_complete:
                return
        futures = self._request_cancellation()
        remaining_seconds = max(0.0, shutdown_deadline - time.monotonic())
        _, not_done = wait(futures, timeout=remaining_seconds) if futures else (set(), set())
        with self._lock:
            if not self._background_shutdown:
                self._background.shutdown(wait=False, cancel_futures=True)
                self._background_shutdown = True
        if not_done:
            raise RuntimeError(
                f"online resolution teardown did not stop within {timeout_seconds:.3f}s; "
                f"{len(not_done)} batch(es) remain active"
            )
        self._background.shutdown(wait=True, cancel_futures=True)

        with self._callbacks_drained:
            while self._active_callbacks:
                remaining_seconds = max(0.0, shutdown_deadline - time.monotonic())
                if remaining_seconds <= 0:
                    raise RuntimeError(
                        "online resolution resource cleanup did not start within "
                        f"{timeout_seconds:.3f}s; {self._active_callbacks} callback(s) remain active"
                    )
                self._callbacks_drained.wait(timeout=remaining_seconds)

        with self._lock:
            finalizations = set(self._finalizations)
        remaining_seconds = max(0.0, shutdown_deadline - time.monotonic())
        _, not_finalized = wait(finalizations, timeout=remaining_seconds) if finalizations else (set(), set())
        if not_finalized:
            raise RuntimeError(
                f"online resolution finalization did not stop within {timeout_seconds:.3f}s; "
                f"{len(not_finalized)} task(s) remain active"
            )
        with self._lock:
            if not self._finalizer_shutdown:
                self._finalizer.shutdown(wait=False, cancel_futures=False)
                self._finalizer_shutdown = True
        self._finalizer.shutdown(wait=True, cancel_futures=False)

        start_cleanup = False
        with self._lock:
            cleanup_future = self._cleanup_future
            if cleanup_future is None:
                cleanup_future = Future()
                cleanup_thread = threading.Thread(
                    target=self._run_resource_cleanup,
                    args=(cleanup_future,),
                    name="aic-online-resolution-cleanup",
                    daemon=True,
                )
                self._cleanup_future = cleanup_future
                self._cleanup_thread = cleanup_thread
                start_cleanup = True
            else:
                cleanup_thread = self._cleanup_thread
        if start_cleanup:
            assert cleanup_thread is not None
            cleanup_thread.start()

        remaining_seconds = max(0.0, shutdown_deadline - time.monotonic())
        try:
            cleanup_errors = cleanup_future.result(timeout=remaining_seconds)
        except FutureTimeoutError as error:
            raise RuntimeError(
                f"online resolution resource cleanup did not stop within {timeout_seconds:.3f}s"
            ) from error

        if cleanup_thread is not None:
            cleanup_thread.join()
        with self._lock:
            if self._cleanup_future is cleanup_future:
                self._cleanup_future = None
            self._shutdown_complete = self._executor_closed and self._overlay_closed
            finalization_errors = tuple(self._finalization_errors)
        if finalization_errors:
            raise finalization_errors[0]
        if cleanup_errors:
            raise cleanup_errors[0]

    def _run_resource_cleanup(self, cleanup_future: _CleanupFuture) -> None:
        cleanup_errors: list[BaseException] = []
        close_executor = getattr(self._session.executor, "close", None)
        if callable(close_executor) and not self._executor_closed:
            try:
                close_executor()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                with self._lock:
                    self._executor_closed = True
        elif not callable(close_executor):
            with self._lock:
                self._executor_closed = True
        if not self._overlay_closed:
            try:
                self._session.overlay.close()
            except BaseException as error:
                cleanup_errors.append(error)
            else:
                with self._lock:
                    self._overlay_closed = True
        cleanup_future.set_result(tuple(cleanup_errors))

    def _resolve_pending(
        self,
        *,
        caller_deadline_monotonic: float,
        deadline_source: str,
    ) -> None:
        entries = self._session.pending_entries()

        def execute_measurements(
            requests: Sequence[MeasurementRequest],
            *,
            deadline_monotonic: float,
            cancellation,
        ) -> tuple[MeasurementRecord, ...]:
            return self._execute_measurements(
                requests,
                deadline_monotonic=deadline_monotonic,
                caller_deadline_monotonic=caller_deadline_monotonic,
                deadline_source=deadline_source,
                cancellation=cancellation,
            )

        try:
            self._session.resolve_pending(
                execute=execute_measurements,
                account_collection_time=False,
            )
        except ResolutionFailed as failure:
            if self._on_measurement_failure != "hybrid":
                raise
            self._publish_hybrid_fallbacks(entries, failure)

    def _publish_hybrid_fallbacks(
        self,
        entries: tuple[MissEntry, ...],
        failure: ResolutionFailed,
    ) -> None:
        handled_reason_indexes: set[int] = set()
        for entry in entries:
            request = entry.request
            matching = [(index, reason) for index, reason in enumerate(failure.reasons) if reason.key == request.key]
            if self._session.overlay.lookup(request.key, request.protocol) is not None:
                if len(matching) > 1:
                    raise failure
                if matching:
                    handled_reason_indexes.add(matching[0][0])
                continue
            if len(matching) != 1:
                raise failure
            index, reason = matching[0]
            if (
                reason.code not in _HYBRID_ELIGIBLE
                or reason.failure_kind not in {MeasurementFailureKind.OPERATIONAL, MeasurementFailureKind.BUDGET}
                or entry.hybrid_resolver is None
            ):
                raise failure
            try:
                value = entry.hybrid_resolver()
                # Narrow the resolver window without claiming a distributed
                # transaction: exact evidence that arrives later still wins on replay.
                if self._session.overlay.lookup(request.key, request.protocol) is None:
                    self._session.publish_fallback(request, value, reason)
            except Exception as error:
                hybrid_reason = UnresolvedReason(
                    UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    f"HYBRID fallback failed after {reason.code.value}: {type(error).__name__}: {error}",
                    key=request.key,
                    failure_kind=MeasurementFailureKind.INVARIANT,
                )
                self._session.observe_unresolved_reason(hybrid_reason)
                raise ResolutionFailed((*failure.reasons, hybrid_reason)) from error
            handled_reason_indexes.add(index)
        if handled_reason_indexes != set(range(len(failure.reasons))):
            raise failure

    def _execute_measurements(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        caller_deadline_monotonic: float,
        deadline_source: str,
        cancellation,
    ) -> tuple[MeasurementRecord, ...]:
        requested = tuple(requests)
        if self._clock() >= caller_deadline_monotonic:
            self._session.observe_deadline_source(deadline_source)
            factory = _timeout_record if deadline_source == "callback_block" else _budget_record
            return tuple(factory(request) for request in requested)
        flights: dict[_FlightIdentity, _Flight] = {}
        with self._lock:
            if self._closed:
                raise RuntimeError("online resolution coordinator is closed")
            newly_owned: list[MeasurementRequest] = []
            for request in requested:
                identity = (request.key, request.protocol)
                flight = self._flights.get(identity)
                if flight is None:
                    newly_owned.append(request)
                else:
                    flights[identity] = flight
                    self._session.report.single_flight_joins += 1
            if newly_owned:
                owned_requests = tuple(newly_owned)
                batch_cancellation = _DeadlineCancellation(
                    cancellation,
                    deadline_monotonic=deadline_monotonic,
                    clock=self._clock,
                )
                future = self._background.submit(
                    self._run_batch,
                    owned_requests,
                    deadline_monotonic,
                    batch_cancellation,
                )
                self._batches[future] = _BatchState(
                    future=future,
                    requests=owned_requests,
                    cancellation=batch_cancellation,
                )
                for request in newly_owned:
                    identity = (request.key, request.protocol)
                    flight = _Flight(request=request, future=future)
                    self._flights[identity] = flight
                    flights[identity] = flight
                future.add_done_callback(self._schedule_finalization)

            unique_futures = {flight.future for flight in flights.values()}
            for future in unique_futures:
                self._batches[future].waiters += 1
        done: set[_BatchFuture] = set()
        wait_completed = False
        wait_error: BaseException | None = None
        finalization_error: BaseException | None = None
        try:
            try:
                done = self._wait_for_futures(
                    unique_futures,
                    deadline_monotonic=caller_deadline_monotonic,
                )
                if not done.issubset(unique_futures):
                    raise ValueError("future waiter returned an unknown future")
                if unique_futures - done:
                    self._session.observe_deadline_source(deadline_source)
                wait_completed = True
            except BaseException as error:
                wait_error = error
        finally:
            with self._lock:
                for future in unique_futures:
                    state = self._batches.get(future)
                    if state is None:
                        continue
                    state.waiters -= 1
                    if wait_completed and future in done:
                        state.delivered = True
                    else:
                        state.timed_out = True
            for future in unique_futures:
                if future.done():
                    try:
                        self._finalize_future(future)
                    except BaseException as error:
                        if finalization_error is None:
                            finalization_error = error
        if wait_error is not None:
            raise wait_error.with_traceback(wait_error.__traceback__)
        if finalization_error is not None:
            raise finalization_error.with_traceback(finalization_error.__traceback__)

        completed_records: dict[_FlightIdentity, list[MeasurementRecord]] = {}
        batch_results: dict[_BatchFuture, _BatchResult] = {}
        execution_error: BaseException | None = None
        for future in done:
            result = future.result()
            batch_results[future] = result
            if result.error is not None:
                if execution_error is None:
                    execution_error = result.error
                continue
            if any(record.failure_code is UnresolvedCode.BUDGET_EXHAUSTED for record in result.records):
                self._session.observe_deadline_source("cumulative_budget")
            for record in result.records:
                identity = (record.key, record.protocol)
                completed_records.setdefault(identity, []).append(record)

        requested_identities = {(request.key, request.protocol) for request in requested}
        for future, result in batch_results.items():
            with self._lock:
                state = self._batches.get(future)
                sibling_requests = (
                    tuple(
                        request
                        for request in state.requests
                        if (request.key, request.protocol) not in requested_identities
                    )
                    if state is not None
                    else ()
                )
            if sibling_requests and result.error is None:
                self._session.commit_late_valid_records(sibling_requests, result.records)
            with self._lock:
                for request in sibling_requests:
                    identity = (request.key, request.protocol)
                    flight = self._flights.get(identity)
                    if flight is not None and flight.future is future:
                        del self._flights[identity]

        records: list[MeasurementRecord] = []
        with self._lock:
            for request in requested:
                identity = (request.key, request.protocol)
                flight = flights[identity]
                if flight.future not in done:
                    records.append(
                        _timeout_record(request) if deadline_source == "callback_block" else _budget_record(request)
                    )
                    continue
                records.extend(completed_records.get(identity, ()))
                if self._flights.get(identity) is flight:
                    del self._flights[identity]
            for future in done:
                state = self._batches.get(future)
                has_remaining_flights = any(flight.future is future for flight in self._flights.values())
                if state is not None and state.delivered and not has_remaining_flights:
                    del self._batches[future]
        if execution_error is not None:
            raise execution_error
        return tuple(records)

    def _schedule_finalization(self, future: _BatchFuture) -> None:
        self._account_future(future)
        with self._lock:
            finalization = self._finalizer.submit(self._finalize_future, future)
            self._finalizations.add(finalization)
        finalization.add_done_callback(self._retire_finalization)

    def _retire_finalization(self, finalization: _FinalizationFuture) -> None:
        error = None if finalization.cancelled() else finalization.exception()
        with self._lock:
            self._finalizations.discard(finalization)
            if error is not None:
                self._finalization_errors.append(error)

    def _account_future(self, future: _BatchFuture) -> None:
        if future.cancelled():
            return
        result = future.result()
        elapsed_seconds: float | None = None
        with self._lock:
            state = self._batches.get(future)
            if state is not None and not state.accounted:
                state.accounted = True
                elapsed_seconds = max(
                    0.0,
                    result.completed_monotonic - result.started_monotonic,
                )
        if elapsed_seconds is not None:
            self._session.charge_collection_seconds(elapsed_seconds)

    def _finalize_future(self, future: _BatchFuture) -> None:
        late_requests: tuple[MeasurementRequest, ...] | None = None
        with self._lock:
            state = self._batches.get(future)
            if state is None:
                return
            if state.timed_out and state.waiters == 0 and not state.delivered and not state.late_finalized:
                state.late_finalized = True
                late_requests = state.requests

        if future.cancelled():
            with self._lock:
                state = self._batches.pop(future, None)
                if state is not None:
                    for request in state.requests:
                        identity = (request.key, request.protocol)
                        flight = self._flights.get(identity)
                        if flight is not None and flight.future is future:
                            del self._flights[identity]
            return
        self._account_future(future)
        result = future.result()
        if late_requests is None:
            return
        try:
            if result.error is None:
                self._session.commit_late_valid_records(late_requests, result.records)
        finally:
            with self._lock:
                for request in late_requests:
                    identity = (request.key, request.protocol)
                    flight = self._flights.get(identity)
                    if flight is not None and flight.future is future:
                        del self._flights[identity]
                self._batches.pop(future, None)

    def _run_batch(
        self,
        requests: tuple[MeasurementRequest, ...],
        deadline_monotonic: float,
        cancellation: _DeadlineCancellation,
    ) -> _BatchResult:
        del deadline_monotonic
        started_monotonic = self._clock()
        remaining_seconds = self._session.remaining_collection_seconds()
        execution_deadline = started_monotonic + max(0.0, remaining_seconds)
        cancellation.set_deadline(execution_deadline)
        if remaining_seconds <= 0:
            return _BatchResult(
                records=tuple(_budget_record(request) for request in requests),
                started_monotonic=started_monotonic,
                completed_monotonic=started_monotonic,
                deadline_monotonic=execution_deadline,
                cancelled_at_completion=True,
            )
        self._session.observe_measurement_attempts(len(requests))
        error: BaseException | None = None
        try:
            records = tuple(
                self._session.executor.execute(
                    requests,
                    deadline_monotonic=execution_deadline,
                    cancellation=cancellation,
                )
            )
        except BaseException as caught:
            records = ()
            error = caught
        completed_monotonic = self._clock()
        if completed_monotonic >= execution_deadline and error is None:
            records_by_identity: dict[_FlightIdentity, list[MeasurementRecord]] = {}
            for record in records:
                records_by_identity.setdefault((record.key, record.protocol), []).append(record)
            requested_identities = tuple((request.key, request.protocol) for request in requests)
            complete_identity_match = (
                len(set(requested_identities)) == len(requested_identities)
                and len(records) == len(requests)
                and set(records_by_identity) == set(requested_identities)
                and all(len(records_by_identity[identity]) == 1 for identity in requested_identities)
            )
            deadline_aware = complete_identity_match and any(
                matches[0].status is not RecordStatus.VALID
                and matches[0].failure_code
                in {
                    UnresolvedCode.TIMEOUT,
                    UnresolvedCode.BUDGET_EXHAUSTED,
                    UnresolvedCode.CANCELLED,
                }
                for matches in records_by_identity.values()
            )
            if deadline_aware:
                bounded_records: list[MeasurementRecord] = []
                for request in requests:
                    matches = records_by_identity.get((request.key, request.protocol), ())
                    if len(matches) == 1 and matches[0].status is RecordStatus.VALID:
                        bounded_records.append(matches[0])
                    else:
                        bounded_records.append(_budget_record(request))
                records = tuple(bounded_records)
            else:
                records = tuple(_budget_record(request) for request in requests)
            error = None
        return _BatchResult(
            records=records,
            started_monotonic=started_monotonic,
            completed_monotonic=completed_monotonic,
            deadline_monotonic=execution_deadline,
            cancelled_at_completion=cancellation.cancelled(),
            error=error,
        )

    def _default_wait_for_futures(
        self,
        futures: set[_BatchFuture],
        *,
        deadline_monotonic: float,
    ) -> set[_BatchFuture]:
        remaining = max(0.0, deadline_monotonic - self._clock())
        done, _ = wait(futures, timeout=remaining)
        return set(done)
