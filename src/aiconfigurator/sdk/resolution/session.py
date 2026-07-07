# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NoReturn, Protocol, TypeVar

from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.types import (
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    ResolutionPolicy,
    UnresolvedCode,
    UnresolvedReason,
)

T = TypeVar("T")


class CancellationToken(Protocol):
    def cancelled(self) -> bool: ...


class _NeverCancelled:
    def cancelled(self) -> bool:
        return False


class MeasurementExecutor(Protocol):
    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> Sequence[MeasurementRecord]: ...


@dataclass(slots=True)
class MissEntry:
    request: MeasurementRequest
    consumers: list[str] = field(default_factory=list)


class MissSet:
    def __init__(self) -> None:
        self._entries: OrderedDict[PerfKey, MissEntry] = OrderedDict()

    def record(self, request: MeasurementRequest, consumer: str) -> None:
        entry = self._entries.get(request.key)
        if entry is None:
            self._entries[request.key] = MissEntry(request, [consumer])
            return
        if not self._same_work_order(entry.request, request):
            raise ValueError(f"conflicting requests share PerfKey {request.key.digest}")
        entry.consumers.append(consumer)

    @staticmethod
    def _same_work_order(left: MeasurementRequest, right: MeasurementRequest) -> bool:
        return (
            left.key == right.key
            and left.query == right.query
            and left.environment == right.environment
            and left.semantic_descriptor == right.semantic_descriptor
            and left.protocol == right.protocol
        )

    def requests(self) -> tuple[MeasurementRequest, ...]:
        return tuple(entry.request for entry in self._entries.values())

    def entries(self) -> tuple[MissEntry, ...]:
        return tuple(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class ResolutionBudget:
    max_new_keys: int
    max_wall_seconds: float
    max_transient_attempts_per_key: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.max_new_keys, bool) or not isinstance(self.max_new_keys, int) or self.max_new_keys < 0:
            raise ValueError("max_new_keys must be a non-negative integer")
        if not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds < 0:
            raise ValueError("max_wall_seconds must be finite and non-negative")
        if (
            isinstance(self.max_transient_attempts_per_key, bool)
            or not isinstance(self.max_transient_attempts_per_key, int)
            or self.max_transient_attempts_per_key < 0
        ):
            raise ValueError("max_transient_attempts_per_key must be a non-negative integer")


@dataclass(slots=True)
class ResolutionReport:
    overlay_hits: int = 0
    unique_misses: int = 0
    consumer_misses: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    collection_seconds: float = 0.0
    unresolved: list[UnresolvedReason] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe snapshot of resolution evidence and failures."""
        return {
            "overlay_hits": self.overlay_hits,
            "unique_misses": self.unique_misses,
            "consumer_misses": self.consumer_misses,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "collection_seconds": self.collection_seconds,
            "unresolved": [
                {
                    "code": reason.code.value,
                    "operation": reason.operation,
                    "detail": reason.detail,
                }
                for reason in self.unresolved
            ],
        }


class ResolutionFailed(RuntimeError):  # noqa: N818 - public contract names the failed resolution state
    def __init__(self, reasons: Sequence[UnresolvedReason]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(f"{reason.code}: {reason.operation}: {reason.detail}" for reason in reasons))


class ResolutionSession:
    def __init__(
        self,
        overlay: OverlayStore,
        executor: MeasurementExecutor,
        budget: ResolutionBudget,
        protocol: MeasurementProtocol,
        *,
        policy: ResolutionPolicy = ResolutionPolicy.MEASURE_ON_MISS,
        clock: Callable[[], float] = time.monotonic,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if policy is ResolutionPolicy.PURE:
            raise ValueError("pure prediction does not construct a ResolutionSession")
        self.overlay = overlay
        self.executor = executor
        self.budget = budget
        self.protocol = protocol
        self.policy = policy
        self.report = ResolutionReport()
        self._clock = clock
        self._cancellation = cancellation or _NeverCancelled()
        self._charged_keys: set[PerfKey] = set()
        self._observed_miss_keys: set[PerfKey] = set()
        self._callback_lock = threading.RLock()
        self._callback_depth = 0
        self._misses = MissSet()
        self._unresolved: list[UnresolvedReason] = []
        self._negative: dict[PerfKey, UnresolvedReason] = {}
        self._transient_attempts: dict[PerfKey, int] = {}

    def lookup(self, key: PerfKey) -> MeasurementRecord | None:
        record = self.overlay.lookup(key, self.protocol)
        if record is not None:
            self.report.overlay_hits += 1
        return record

    def record_miss(self, request: MeasurementRequest, consumer: str) -> None:
        if request.protocol != self.protocol:
            self.record_unresolved(
                UnresolvedReason(
                    UnresolvedCode.IDENTITY_MISMATCH,
                    consumer,
                    "request protocol does not match resolution session",
                )
            )
            return
        previous_failure = self._negative.get(request.key)
        if previous_failure is not None:
            self.record_unresolved(previous_failure)
            return
        if self._transient_attempts.get(request.key, 0) >= self.budget.max_transient_attempts_per_key:
            self.record_unresolved(
                UnresolvedReason(
                    UnresolvedCode.RETRY_EXHAUSTED,
                    consumer,
                    f"transient retry budget exhausted for {request.key.digest}",
                )
            )
            return
        self._misses.record(request, consumer)
        self.report.consumer_misses += 1

    def record_missing_adapter(self, operation: str, error: Exception) -> None:
        self.record_unresolved(UnresolvedReason(UnresolvedCode.MISSING_ADAPTER, operation, str(error)))

    def record_unresolved(self, reason: UnresolvedReason) -> None:
        self._unresolved.append(reason)

    def checkpoint(self) -> tuple[int, int]:
        return len(self._misses), len(self._unresolved)

    def changed_since(self, checkpoint: tuple[int, int]) -> bool:
        return self.checkpoint() != checkpoint

    def _fail(self, reasons: Sequence[UnresolvedReason]) -> NoReturn:
        self.report.unresolved.extend(reasons)
        raise ResolutionFailed(reasons)

    def _remember_failure(self, key: PerfKey, reason: UnresolvedReason) -> None:
        deterministic = {
            UnresolvedCode.MISSING_ADAPTER,
            UnresolvedCode.UNSUPPORTED_SHAPE,
            UnresolvedCode.TOPOLOGY_MISMATCH,
            UnresolvedCode.IDENTITY_MISMATCH,
            UnresolvedCode.INVALID_MEASUREMENT,
        }
        if reason.code in deterministic:
            self._negative[key] = reason
        elif reason.code not in {UnresolvedCode.CANCELLED, UnresolvedCode.BUDGET_EXHAUSTED}:
            self._transient_attempts[key] = self._transient_attempts.get(key, 0) + 1

    def resolve_pending(self) -> None:
        if self._unresolved:
            reasons = tuple(self._unresolved)
            self._unresolved.clear()
            self._misses.clear()
            self._fail(reasons)
        discovered = self._misses.requests()
        self._misses.clear()
        if not discovered:
            return
        newly_observed = {request.key for request in discovered} - self._observed_miss_keys
        self.report.unique_misses += len(newly_observed)
        self._observed_miss_keys.update(newly_observed)
        requests = tuple(request for request in discovered if self.lookup(request.key) is None)
        if not requests:
            return

        if self._cancellation.cancelled():
            self._fail(
                [UnresolvedReason(UnresolvedCode.CANCELLED, request.op_id, "session cancelled") for request in requests]
            )

        if self.policy is ResolutionPolicy.OBSERVE_ONLY:
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.OBSERVE_ONLY,
                        request.op_id,
                        f"observed unresolved key {request.key.digest}",
                    )
                    for request in requests
                ]
            )

        newly_charged = tuple(request for request in requests if request.key not in self._charged_keys)
        remaining_keys = self.budget.max_new_keys - len(self._charged_keys)
        remaining_seconds = self.budget.max_wall_seconds - self.report.collection_seconds
        if len(newly_charged) > remaining_keys or remaining_seconds <= 0:
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.BUDGET_EXHAUSTED,
                        request.op_id,
                        f"need {len(newly_charged)} new keys/{remaining_seconds:.3f}s remaining; "
                        f"budget has {remaining_keys} keys",
                    )
                    for request in requests
                ]
            )

        self._charged_keys.update(request.key for request in newly_charged)
        started = self._clock()
        try:
            records = tuple(
                self.executor.execute(
                    requests,
                    deadline_monotonic=started + remaining_seconds,
                    cancellation=self._cancellation,
                )
            )
        except Exception as error:
            self.report.collection_seconds += self._clock() - started
            code = UnresolvedCode.CANCELLED if self._cancellation.cancelled() else UnresolvedCode.COLLECTOR_FAILED
            reasons = [UnresolvedReason(code, request.op_id, str(error)) for request in requests]
            for request, reason in zip(requests, reasons, strict=True):
                self._remember_failure(request.key, reason)
            self._fail(reasons)
        elapsed = self._clock() - started
        self.report.collection_seconds += elapsed
        cancelled_after_dispatch = self._cancellation.cancelled()

        requested = {request.key: request for request in requests}
        records_by_key: dict[PerfKey, list[MeasurementRecord]] = {}
        failures: list[UnresolvedReason] = []
        failures_to_remember: list[tuple[PerfKey, UnresolvedReason]] = []
        invalid_keys: set[PerfKey] = set()
        validation_rejections = 0
        for record in records:
            request = requested.get(record.key)
            if request is None:
                if not cancelled_after_dispatch:
                    failures.append(
                        UnresolvedReason(
                            UnresolvedCode.INVALID_MEASUREMENT,
                            record.key.namespace,
                            f"executor returned unrequested key {record.key.digest}",
                        )
                    )
                continue
            if record.protocol != request.protocol:
                invalid_keys.add(record.key)
                if not cancelled_after_dispatch:
                    reason = UnresolvedReason(
                        UnresolvedCode.IDENTITY_MISMATCH,
                        request.op_id,
                        "record protocol does not exactly match request",
                    )
                    failures.append(reason)
                    failures_to_remember.append((record.key, reason))
                    validation_rejections += 1
                continue
            records_by_key.setdefault(record.key, []).append(record)

        for request in requests:
            if request.key in invalid_keys:
                continue
            matches = records_by_key.get(request.key, [])
            if len(matches) != 1 and cancelled_after_dispatch:
                continue
            if len(matches) != 1:
                reason = UnresolvedReason(
                    UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    f"expected one record for {request.key.digest}, got {len(matches)}",
                )
                failures.append(reason)
                failures_to_remember.append((request.key, reason))
                continue

            record = matches[0]
            self.overlay.append(record)
            self.report.accepted_records += int(record.status is RecordStatus.VALID)
            self.report.rejected_records += int(record.status is not RecordStatus.VALID)
            if cancelled_after_dispatch:
                continue
            if record.status is RecordStatus.FAILED:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.COLLECTOR_FAILED,
                    request.op_id,
                    record.failure_reason or "collector failed",
                )
                failures.append(reason)
                failures_to_remember.append((request.key, reason))
            elif record.status is not RecordStatus.VALID or record.latency_ms is None:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    record.failure_reason or "measurement rejected",
                )
                failures.append(reason)
                failures_to_remember.append((request.key, reason))

        cancelled_after_dispatch = cancelled_after_dispatch or self._cancellation.cancelled()
        if cancelled_after_dispatch:
            failures.clear()
            failures.append(
                UnresolvedReason(
                    UnresolvedCode.CANCELLED,
                    "resolution_session",
                    "session cancelled during collection",
                )
            )
        else:
            self.report.rejected_records += validation_rejections
            for key, reason in failures_to_remember:
                self._remember_failure(key, reason)
        if elapsed > remaining_seconds and not cancelled_after_dispatch:
            failures.extend(
                UnresolvedReason(
                    UnresolvedCode.BUDGET_EXHAUSTED,
                    request.op_id,
                    f"collection exceeded wall budget by {elapsed - remaining_seconds:.3f}s",
                )
                for request in requests
            )
        if failures:
            self._fail(failures)

    def execute_callback(self, query: Callable[[], T]) -> T:
        with self._callback_lock:
            if self._callback_depth:
                return query()
            self._callback_depth += 1
            try:
                return self._execute_callback_locked(query)
            except BaseException:
                self._misses.clear()
                self._unresolved.clear()
                raise
            finally:
                self._callback_depth -= 1

    def _execute_callback_locked(self, query: Callable[[], T]) -> T:
        self._misses.clear()
        self._unresolved.clear()
        first = query()
        if not self._misses and not self._unresolved:
            return first
        self.resolve_pending()
        self._misses.clear()
        self._unresolved.clear()
        result = query()
        if self._misses or self._unresolved:
            reasons = list(self._unresolved)
            reasons.extend(
                UnresolvedReason(
                    UnresolvedCode.REQUERY_STILL_MISSING,
                    entry.request.op_id,
                    f"key {entry.request.key.digest} remained absent after collection",
                )
                for entry in self._misses.entries()
            )
            self._fail(reasons)
        return result
