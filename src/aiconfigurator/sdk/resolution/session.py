# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol, TypeVar

from aiconfigurator.sdk.resolution.fallback import FallbackRecord, FallbackStore
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.types import (
    MeasurementFailureKind,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    ProtocolMismatchError,
    RecordStatus,
    ResolutionPolicy,
    UnresolvedCode,
    UnresolvedReason,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)

_EXACT_SOURCES = ("overlay", "curated_exact")
_FINAL_SOURCES = (*_EXACT_SOURCES, "fallback")
_OBSERVED_SOURCES = (*_FINAL_SOURCES, "miss")


def _source_counts() -> dict[str, int]:
    return dict.fromkeys(_OBSERVED_SOURCES, 0)


def _final_source_counts() -> dict[str, int]:
    return dict.fromkeys(_FINAL_SOURCES, 0)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("resolution evidence object keys must be strings")
            copied[key] = _json_value(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _json_snapshot(value: Any) -> Any:
    try:
        encoded = json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("resolution evidence must contain JSON-safe finite values") from error
    return json.loads(encoded)


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


class MeasurementExecute(Protocol):
    def __call__(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> Sequence[MeasurementRecord]: ...


@dataclass(frozen=True, slots=True)
class HybridFallbackValue:
    latency_ms: float
    source: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not math.isfinite(self.latency_ms) or self.latency_ms <= 0:
            raise ValueError("HYBRID fallback latency must be finite and positive")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("HYBRID fallback source must be non-empty")
        provenance = _json_snapshot(self.provenance)
        if not isinstance(provenance, dict):
            raise TypeError("HYBRID fallback provenance must be a mapping")
        object.__setattr__(self, "provenance", provenance)


HybridResolver = Callable[[], HybridFallbackValue]


@dataclass(slots=True)
class MissEntry:
    request: MeasurementRequest
    consumers: list[str] = field(default_factory=list)
    hybrid_resolver: HybridResolver | None = None


class MissSet:
    def __init__(self) -> None:
        self._entries: OrderedDict[PerfKey, MissEntry] = OrderedDict()

    def record(
        self,
        request: MeasurementRequest,
        consumer: str,
        *,
        hybrid_resolver: HybridResolver | None = None,
    ) -> None:
        entry = self._entries.get(request.key)
        if entry is None:
            self._entries[request.key] = MissEntry(request, [consumer], hybrid_resolver)
            return
        if not self._same_work_order(entry.request, request):
            raise ValueError(f"conflicting requests share PerfKey {request.key.digest}")
        entry.consumers.append(consumer)
        if entry.hybrid_resolver is None:
            entry.hybrid_resolver = hybrid_resolver

    @staticmethod
    def _same_work_order(left: MeasurementRequest, right: MeasurementRequest) -> bool:
        return (
            left.key == right.key
            and left.query == right.query
            and left.environment == right.environment
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
class _PerfKeyEvidence:
    key: PerfKey
    consumers: list[str] = field(default_factory=list)
    source_counts: dict[str, int] = field(default_factory=_source_counts)
    final_source_counts: dict[str, int] = field(default_factory=_final_source_counts)

    def to_dict(self) -> dict[str, object]:
        return {
            "digest": self.key.digest,
            "namespace": self.key.namespace,
            "query": json.loads(self.key.query_json),
            "environment": json.loads(self.key.environment_json),
            "consumers": list(self.consumers),
            "consumer_count": len(self.consumers),
            "source_counts": {source: count for source, count in self.source_counts.items() if source != "fallback"},
            "final_source_counts": {
                source: count for source, count in self.final_source_counts.items() if source in _EXACT_SOURCES
            },
        }


@dataclass(slots=True)
class ResolutionReport:
    overlay_hits: int = 0
    unique_misses: int = 0
    consumer_misses: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    collection_seconds: float = 0.0
    measurement_attempts: int = 0
    single_flight_joins: int = 0
    late_completions: int = 0
    fallback_hits: int = 0
    hybrid_publications: int = 0
    hybrid_fallbacks: list[dict[str, object]] = field(default_factory=list)
    unresolved: list[UnresolvedReason] = field(default_factory=list)
    registry_routes: list[dict[str, object]] = field(default_factory=list)
    resources: list[dict[str, object]] = field(default_factory=list)
    waves: list[dict[str, object]] = field(default_factory=list)
    assignments: list[dict[str, object]] = field(default_factory=list)
    workers: list[dict[str, object]] = field(default_factory=list)
    invocations: list[dict[str, object]] = field(default_factory=list)
    inventories: list[dict[str, object]] = field(default_factory=list)
    leases: list[dict[str, object]] = field(default_factory=list)
    records: list[dict[str, object]] = field(default_factory=list)
    evidence_links: list[dict[str, object]] = field(default_factory=list)
    callbacks: list[dict[str, object]] = field(default_factory=list)
    _perf_keys: OrderedDict[PerfKey, _PerfKeyEvidence] = field(default_factory=OrderedDict, repr=False)
    _persisted_record_ids: set[tuple[str, str, int, str]] = field(default_factory=set, repr=False)
    _evidence_link_ids: set[str] = field(default_factory=set, repr=False)
    _execution_evidence_ids: set[str] = field(default_factory=set, repr=False)

    def _key_evidence(self, key: PerfKey) -> _PerfKeyEvidence:
        evidence = self._perf_keys.get(key)
        if evidence is None:
            evidence = _PerfKeyEvidence(key)
            self._perf_keys[key] = evidence
        return evidence

    def observe_source(
        self,
        key: PerfKey,
        source: str,
        *,
        consumer: str | None = None,
    ) -> None:
        if source not in _OBSERVED_SOURCES:
            raise ValueError(f"unsupported resolution evidence source {source!r}")
        evidence = self._key_evidence(key)
        evidence.source_counts[source] += 1
        if consumer is not None:
            if not isinstance(consumer, str) or not consumer.strip():
                raise ValueError("resolution evidence consumer must be a non-empty string")
            evidence.consumers.append(consumer)

    def observe_final_sources(self, sources: Mapping[PerfKey, str]) -> None:
        for key, source in sources.items():
            if source not in _FINAL_SOURCES:
                raise ValueError(f"unsupported final resolution evidence source {source!r}")
            self._key_evidence(key).final_source_counts[source] += 1

    def observe_record(
        self,
        record: MeasurementRecord,
        *,
        persisted: bool,
        sequence: int | None,
        include_execution: bool,
        rejection_reason: str | None = None,
    ) -> None:
        if persisted:
            if sequence is None:
                raise ValueError("persisted resolution records require an overlay sequence")
            identity = (record.key.digest, record.protocol.digest, sequence, record.status.value)
            if identity in self._persisted_record_ids:
                return
            self._persisted_record_ids.add(identity)
        failure = None
        if record.failure_code is not None or record.failure_reason is not None:
            failure = {
                "code": record.failure_code.value if record.failure_code is not None else None,
                "reason": record.failure_reason,
            }
        self.records.append(
            {
                "key_digest": record.key.digest,
                "status": record.status.value,
                "latency_ms": record.latency_ms,
                "latency_units": "ms",
                "energy_wms": record.energy_wms,
                "energy_units": "watt_milliseconds",
                "samples_ms": list(record.samples_ms),
                "protocol": json.loads(record.protocol.canonical),
                "perf_row": _json_snapshot(record.perf_row),
                "provenance": _json_snapshot(record.provenance),
                "failure": failure,
                "persisted": persisted,
                "sequence": sequence,
                "rejection_reason": rejection_reason,
            }
        )
        if include_execution:
            self._observe_resolution_execution(record)

    def observe_evidence_link(self, link: Mapping[str, object]) -> dict[str, object]:
        snapshot = _json_snapshot(link)
        if not isinstance(snapshot, dict):
            raise TypeError("resolution evidence links must be mappings")
        identity = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if identity not in self._evidence_link_ids:
            self._evidence_link_ids.add(identity)
            self.evidence_links.append(snapshot)
        return snapshot

    def _observe_resolution_execution(self, record: MeasurementRecord) -> None:
        execution = record.provenance.get("resolution_execution")
        if execution is None:
            return
        if not isinstance(execution, Mapping):
            raise TypeError("resolution_execution provenance must be a mapping")
        categories = (
            ("registry_route", "registry_routes", False),
            ("resource", "resources", True),
            ("wave", "waves", False),
            ("assignment", "assignments", True),
            ("worker", "workers", True),
            ("invocation", "invocations", True),
            ("inventory", "inventories", False),
            ("lease", "leases", True),
        )
        for field_name, report_name, include_key in categories:
            value = execution.get(field_name)
            if value is None:
                continue
            snapshot = _json_snapshot(value)
            if not isinstance(snapshot, dict):
                raise TypeError(f"resolution_execution {field_name} must be a mapping")
            if include_key:
                snapshot = {"key_digest": record.key.digest, **snapshot}
            identity = f"{report_name}:{json.dumps(snapshot, sort_keys=True, separators=(',', ':'), allow_nan=False)}"
            if identity in self._execution_evidence_ids:
                continue
            self._execution_evidence_ids.add(identity)
            getattr(self, report_name).append(snapshot)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe snapshot of resolution evidence and failures."""
        perf_keys = sorted(self._perf_keys.values(), key=lambda evidence: evidence.key.digest)
        source_counts = _source_counts()
        final_source_counts = _final_source_counts()
        exact_source_counts = dict.fromkeys((*_EXACT_SOURCES, "miss"), 0)
        final_exact_source_counts = dict.fromkeys(_EXACT_SOURCES, 0)
        for evidence in perf_keys:
            for source, count in evidence.source_counts.items():
                source_counts[source] += count
                if source in exact_source_counts:
                    exact_source_counts[source] += count
            for source, count in evidence.final_source_counts.items():
                final_source_counts[source] += count
                if source in final_exact_source_counts:
                    final_exact_source_counts[source] += count
        return {
            "overlay_hits": self.overlay_hits,
            "unique_misses": self.unique_misses,
            "consumer_misses": self.consumer_misses,
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "collection_seconds": self.collection_seconds,
            "measurement_attempts": self.measurement_attempts,
            "single_flight_joins": self.single_flight_joins,
            "late_completions": self.late_completions,
            "fallback_hits": self.fallback_hits,
            "hybrid_publications": self.hybrid_publications,
            "hybrid_fallbacks": _json_snapshot(self.hybrid_fallbacks),
            "unresolved": [
                {
                    "code": reason.code.value,
                    "operation": reason.operation,
                    "detail": reason.detail,
                    **({"key_digest": reason.key.digest} if reason.key is not None else {}),
                    **({"failure_kind": reason.failure_kind.value} if reason.failure_kind is not None else {}),
                }
                for reason in self.unresolved
            ],
            "perf_keys": [evidence.to_dict() for evidence in perf_keys],
            "consumer_counts": {evidence.key.digest: len(evidence.consumers) for evidence in perf_keys},
            "exact_source_counts": exact_source_counts,
            "final_exact_source_counts": final_exact_source_counts,
            "source_counts": source_counts,
            "final_source_counts": final_source_counts,
            "registry_routes": _json_snapshot(self.registry_routes),
            "resources": _json_snapshot(self.resources),
            "waves": _json_snapshot(self.waves),
            "assignments": _json_snapshot(self.assignments),
            "workers": _json_snapshot(self.workers),
            "invocations": _json_snapshot(self.invocations),
            "inventories": _json_snapshot(self.inventories),
            "leases": _json_snapshot(self.leases),
            "records": _json_snapshot(self.records),
            "evidence_links": _json_snapshot(self.evidence_links),
            "callbacks": _json_snapshot(self.callbacks),
        }


class ResolutionFailed(RuntimeError):  # noqa: N818 - public contract names the failed resolution state
    def __init__(self, reasons: Sequence[UnresolvedReason]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(f"{reason.code}: {reason.operation}: {reason.detail}" for reason in reasons))


@dataclass(slots=True)
class _CallbackTrace:
    sequence: int
    context: dict[str, object]
    accepted_records_start: int
    rejected_records_start: int
    collection_seconds_start: float
    consumer_counts: OrderedDict[PerfKey, int] = field(default_factory=OrderedDict)
    source_counts: OrderedDict[PerfKey, dict[str, int]] = field(default_factory=OrderedDict)
    current_pass_sources: OrderedDict[PerfKey, str] = field(default_factory=OrderedDict)
    current_pass_links: dict[PerfKey, dict[str, object]] = field(default_factory=dict)
    collection_attempted: bool = False
    replay_attempted: bool = False
    replay_phase: bool = False
    deadline_source: str | None = None


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
        self._report_lock = threading.RLock()
        self._callback_depth = 0
        self._misses = MissSet()
        self._unresolved: list[UnresolvedReason] = []
        self._tainted_operations: list[str] = []
        self._negative: dict[PerfKey, UnresolvedReason] = {}
        self._transient_attempts: dict[PerfKey, int] = {}
        self._callback_sequence = 0
        self._callback_trace: _CallbackTrace | None = None
        self._fallback_store: FallbackStore | None = None
        self._prediction_revision: str | None = None
        self._force_remeasure = False
        self._active_fallbacks: dict[PerfKey, FallbackRecord] = {}

    def bind_request(self, request: MeasurementRequest) -> MeasurementRequest:
        """Apply executor-owned route identity while preserving sampling policy."""

        binder = getattr(self.executor, "bind_request", None)
        if binder is None:
            return request
        bound = binder(request)
        if not isinstance(bound, MeasurementRequest):
            raise TypeError("executor request binder must return MeasurementRequest")
        if bound.key != request.key or bound.query != request.query or bound.environment != request.environment:
            raise ProtocolMismatchError("executor request binder changed physical request identity")
        if (
            bound.protocol.warmups != request.protocol.warmups
            or bound.protocol.samples != request.protocol.samples
            or bound.protocol.statistic != request.protocol.statistic
        ):
            raise ProtocolMismatchError("executor request binder changed session sampling policy")
        return bound

    def _observe_source(
        self,
        key: PerfKey,
        source: str,
        *,
        consumer: str | None,
        evidence_link: dict[str, object] | None = None,
    ) -> None:
        trace = self._callback_trace
        report_consumer = consumer
        if trace is not None and trace.replay_phase:
            report_consumer = None
        self.report.observe_source(key, source, consumer=report_consumer)
        if trace is None:
            return
        counts = trace.source_counts.setdefault(key, _source_counts())
        counts[source] += 1
        if report_consumer is not None:
            trace.consumer_counts[key] = trace.consumer_counts.get(key, 0) + 1
        if source in _FINAL_SOURCES:
            trace.current_pass_sources[key] = source
            if evidence_link is not None:
                trace.current_pass_links[key] = evidence_link

    def _observe_persisted_record(
        self,
        record: MeasurementRecord,
        sequence: int,
        *,
        include_execution: bool,
    ) -> dict[str, object]:
        self.report.observe_record(
            record,
            persisted=True,
            sequence=sequence,
            include_execution=include_execution,
        )
        link: dict[str, object] = {
            "key_digest": record.key.digest,
            "source": "overlay" if record.status is RecordStatus.VALID else "overlay_record",
            "sequence": sequence,
        }
        overlay_path = getattr(self.overlay, "path", None)
        if overlay_path is not None:
            link["overlay_path"] = str(overlay_path)
        return self.report.observe_evidence_link(link)

    def lookup(
        self,
        key: PerfKey,
        protocol: MeasurementProtocol | None = None,
        *,
        consumer: str | None = None,
    ) -> MeasurementRecord | None:
        record = self.overlay.lookup(key, protocol or self.protocol)
        if record is not None:
            self.report.overlay_hits += 1
            if record.sequence is not None:
                link = self._observe_persisted_record(
                    record,
                    record.sequence,
                    include_execution=False,
                )
            else:
                link = next(
                    (
                        candidate
                        for candidate in reversed(self.report.evidence_links)
                        if candidate.get("key_digest") == key.digest and candidate.get("source") == "overlay"
                    ),
                    None,
                )
                if link is None:
                    link = {"key_digest": key.digest, "source": "overlay"}
                    overlay_path = getattr(self.overlay, "path", None)
                    if overlay_path is not None:
                        link["overlay_path"] = str(overlay_path)
                    link = self.report.observe_evidence_link(link)
            self._observe_source(key, "overlay", consumer=consumer, evidence_link=link)
        return record

    def record_curated_hit(
        self,
        key: PerfKey,
        consumer: str,
        *,
        evidence_link: Mapping[str, object] | None = None,
    ) -> None:
        """Record one compatible literal curated row used by the active walk."""

        if evidence_link is not None and not isinstance(evidence_link, Mapping):
            raise TypeError("curated evidence_link must be a mapping")
        link = dict(_json_snapshot(evidence_link or {}))
        link["key_digest"] = key.digest
        link["source"] = "curated_exact"
        snapshot = self.report.observe_evidence_link(link)
        self._observe_source(key, "curated_exact", consumer=consumer, evidence_link=snapshot)

    def configure_fallback(
        self,
        store: FallbackStore | None,
        *,
        prediction_revision: str | None,
        force_remeasure: bool,
    ) -> None:
        if not isinstance(force_remeasure, bool):
            raise TypeError("force_remeasure must be a bool")
        if (store is None) != (prediction_revision is None):
            raise ValueError("fallback store and prediction revision must be configured together")
        if prediction_revision is not None and (
            not isinstance(prediction_revision, str) or not prediction_revision.strip()
        ):
            raise ValueError("prediction_revision must be a non-empty string")
        self._fallback_store = store
        self._prediction_revision = prediction_revision
        self._force_remeasure = force_remeasure

    def lookup_fallback(self, key: PerfKey, consumer: str) -> FallbackRecord | None:
        record = self._active_fallbacks.get(key)
        if record is None:
            if self._fallback_store is None or self._prediction_revision is None:
                return None
            record = self._fallback_store.lookup(
                key,
                prediction_revision=self._prediction_revision,
                force_remeasure=self._force_remeasure,
            )
        if record is not None:
            self.report.fallback_hits += 1
            logger.info(
                "using durable AIC HYBRID fallback",
                extra={
                    "event": "aic_hybrid_fallback_hit",
                    "key_digest": key.digest,
                    "namespace": key.namespace,
                    "identity_digest": record.identity_digest,
                    "prediction_revision": record.prediction_revision,
                    "sidecar_path": str(record.path),
                    "hybrid_source": record.hybrid_source,
                    "latency_ms": record.latency_ms,
                    "latency_units": "ms",
                },
            )
            link = self.report.observe_evidence_link(
                {
                    "key_digest": key.digest,
                    "source": "fallback",
                    "path": str(record.path),
                    "identity_digest": record.identity_digest,
                    "prediction_revision": record.prediction_revision,
                }
            )
            self._observe_source(key, "fallback", consumer=consumer, evidence_link=link)
        return record

    def publish_fallback(
        self,
        request: MeasurementRequest,
        value: HybridFallbackValue,
        failure: UnresolvedReason,
    ) -> FallbackRecord:
        if self._fallback_store is None or self._prediction_revision is None:
            raise RuntimeError("HYBRID fallback publication requires a configured durable store")
        record, created = self._fallback_store.publish_with_status(
            request.key,
            prediction_revision=self._prediction_revision,
            latency_ms=value.latency_ms,
            hybrid_source=value.source,
            measurement_failure=failure,
        )
        self._active_fallbacks[request.key] = record
        self.report.hybrid_publications += int(created)
        self.report.hybrid_fallbacks.append(
            {
                "key_digest": request.key.digest,
                "latency_ms": record.latency_ms,
                "latency_units": "ms",
                "path": str(record.path),
                "hybrid_provenance": {
                    "source": record.hybrid_source,
                    "prediction_revision": record.prediction_revision,
                    "metadata": dict(value.provenance) if created else {},
                },
                "measurement_failure": {
                    "code": record.measurement_failure.code.value,
                    "operation": record.measurement_failure.operation,
                    "detail": record.measurement_failure.detail,
                },
            }
        )
        event = "aic_hybrid_fallback_published" if created else "aic_hybrid_fallback_reused"
        log = logger.warning if created else logger.info
        trace = self._callback_trace
        log(
            "published durable AIC HYBRID fallback"
            if created
            else "reused existing durable AIC HYBRID fallback winner",
            extra={
                "event": event,
                "key_digest": request.key.digest,
                "namespace": request.key.namespace,
                "failure_code": record.measurement_failure.code.value,
                "hybrid_source": record.hybrid_source,
                "latency_ms": record.latency_ms,
                "latency_units": "ms",
                "identity_digest": record.identity_digest,
                "prediction_revision": record.prediction_revision,
                "sidecar_path": str(record.path),
                "operation": request.op_id,
                "callback_context": dict(trace.context) if trace is not None else {},
            },
        )
        return record

    def pending_entries(self) -> tuple[MissEntry, ...]:
        return self._misses.entries()

    def record_miss(
        self,
        request: MeasurementRequest,
        consumer: str,
        *,
        hybrid_resolver: HybridResolver | None = None,
    ) -> None:
        self._observe_source(request.key, "miss", consumer=consumer)
        if (
            request.protocol.warmups != self.protocol.warmups
            or request.protocol.samples != self.protocol.samples
            or request.protocol.statistic != self.protocol.statistic
        ):
            self.record_unresolved(
                UnresolvedReason(
                    UnresolvedCode.IDENTITY_MISMATCH,
                    consumer,
                    "request sampling policy does not match resolution session",
                    key=request.key,
                    failure_kind=MeasurementFailureKind.INVARIANT,
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
                    key=request.key,
                    failure_kind=MeasurementFailureKind.OPERATIONAL,
                )
            )
            return
        self._misses.record(request, consumer, hybrid_resolver=hybrid_resolver)
        self.report.consumer_misses += 1

    def record_missing_adapter(self, operation: str, error: Exception) -> None:
        self.record_unresolved(
            UnresolvedReason(
                UnresolvedCode.MISSING_ADAPTER,
                operation,
                str(error),
                failure_kind=MeasurementFailureKind.INVARIANT,
            )
        )

    def record_binding_error(self, operation: str, error: Exception) -> None:
        code = (
            UnresolvedCode.IDENTITY_MISMATCH
            if isinstance(error, ProtocolMismatchError)
            else UnresolvedCode.MISSING_ADAPTER
        )
        self.record_unresolved(
            UnresolvedReason(
                code,
                operation,
                str(error),
                failure_kind=MeasurementFailureKind.INVARIANT,
            )
        )

    def record_unresolved(self, reason: UnresolvedReason) -> None:
        self._unresolved.append(reason)

    def mark_tainted(self, operation: str) -> None:
        """Mark the active operation walk as containing provisional evidence."""
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("tainted operation must be a non-empty string")
        if operation not in self._tainted_operations:
            self._tainted_operations.append(operation)

    def checkpoint(self) -> tuple[int, int]:
        return len(self._misses), len(self._unresolved)

    def changed_since(self, checkpoint: tuple[int, int]) -> bool:
        return self.checkpoint() != checkpoint

    def _begin_replay(self) -> None:
        trace = self._callback_trace
        if trace is None:
            raise RuntimeError("resolution replay requires an active callback trace")
        trace.replay_attempted = True
        trace.replay_phase = True
        trace.current_pass_sources.clear()
        trace.current_pass_links.clear()

    def _finish_callback(
        self,
        *,
        result: object | None,
        error: BaseException | None,
    ) -> None:
        trace = self._callback_trace
        if trace is None:
            return
        resolved = error is None
        final_sources = trace.current_pass_sources if resolved else {}
        if resolved:
            self.report.observe_final_sources(final_sources)
        failures = (
            [
                {
                    "code": reason.code.value,
                    "operation": reason.operation,
                    "detail": reason.detail,
                }
                for reason in error.reasons
            ]
            if isinstance(error, ResolutionFailed)
            else []
        )
        error_payload = (
            {
                "type": type(error).__name__,
                "message": str(error),
            }
            if error is not None
            else None
        )
        result_source = getattr(result, "source", None)
        if not isinstance(result_source, str):
            result_source = None
        final_evidence = [
            trace.current_pass_links.get(
                key,
                {
                    "key_digest": key.digest,
                    "source": source,
                },
            )
            for key, source in sorted(final_sources.items(), key=lambda item: item[0].digest)
        ]
        collection_payload: dict[str, object] = {
            "attempted": trace.collection_attempted,
            "accepted_records": self.report.accepted_records - trace.accepted_records_start,
            "rejected_records": self.report.rejected_records - trace.rejected_records_start,
            "wall_seconds": self.report.collection_seconds - trace.collection_seconds_start,
        }
        if trace.deadline_source is not None:
            collection_payload["deadline_source"] = trace.deadline_source
        self.report.callbacks.append(
            {
                "sequence": trace.sequence,
                "context": trace.context,
                "outcome": "resolved" if resolved else "failed",
                "collection": collection_payload,
                "replay": {
                    "attempted": trace.replay_attempted,
                    "outcome": (
                        "succeeded"
                        if resolved and trace.replay_attempted
                        else "failed"
                        if trace.replay_attempted
                        else "not_attempted"
                    ),
                },
                "result_source": result_source,
                "consumer_counts": {
                    key.digest: count
                    for key, count in sorted(trace.consumer_counts.items(), key=lambda item: item[0].digest)
                },
                "source_counts": {
                    key.digest: {source: count for source, count in counts.items() if source != "fallback"}
                    for key, counts in sorted(trace.source_counts.items(), key=lambda item: item[0].digest)
                },
                "final_exact_sources": {
                    key.digest: source
                    for key, source in sorted(final_sources.items(), key=lambda item: item[0].digest)
                    if source in _EXACT_SOURCES
                },
                "final_evidence": final_evidence,
                "failures": failures,
                "error": error_payload,
            }
        )
        self._callback_trace = None

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

    def charge_collection_seconds(self, elapsed_seconds: float) -> None:
        """Charge physical collection time to the session's sole cumulative ledger."""

        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("collection elapsed time must be finite and non-negative")
        with self._report_lock:
            self.report.collection_seconds += elapsed_seconds

    def remaining_collection_seconds(self) -> float:
        """Return unspent physical collection time from the sole session ledger."""

        with self._report_lock:
            return max(0.0, self.budget.max_wall_seconds - self.report.collection_seconds)

    def observe_measurement_attempts(self, count: int) -> None:
        """Count requests only when their physical executor batch actually starts."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("measurement attempt count must be a non-negative integer")
        with self._report_lock:
            self.report.measurement_attempts += count

    def observe_unresolved_reason(self, reason: UnresolvedReason) -> None:
        """Add a post-measurement structured failure to the cumulative report."""

        if not isinstance(reason, UnresolvedReason):
            raise TypeError("reason must be an UnresolvedReason")
        with self._report_lock:
            self.report.unresolved.append(reason)

    def observe_deadline_source(self, source: str) -> None:
        """Attach the effective deadline owner to the active callback report."""

        if source not in {"callback_block", "cumulative_budget"}:
            raise ValueError("unknown resolution deadline source")
        trace = self._callback_trace
        if trace is not None:
            trace.deadline_source = source

    def commit_late_valid_records(
        self,
        requests: Sequence[MeasurementRequest],
        records: Sequence[MeasurementRecord],
    ) -> tuple[MeasurementRecord, ...]:
        """Append valid late exact siblings without mutating a completed callback trace."""

        requested = {(request.key, request.protocol): request for request in requests}
        candidates: dict[tuple[PerfKey, MeasurementProtocol], list[MeasurementRecord]] = {}
        for record in records:
            identity = (record.key, record.protocol)
            if identity in requested:
                candidates.setdefault(identity, []).append(record)

        committed: list[MeasurementRecord] = []
        with self._callback_lock, self._report_lock:
            for identity, request in requested.items():
                matches = candidates.get(identity, ())
                if len(matches) != 1:
                    continue
                record = matches[0]
                if record.status is not RecordStatus.VALID or record.latency_ms is None:
                    continue
                if self.overlay.lookup(request.key, request.protocol) is not None:
                    continue
                sequence = self.overlay.append(record)
                self._observe_persisted_record(record, sequence, include_execution=True)
                self.report.accepted_records += 1
                self.report.late_completions += 1
                committed.append(record)
        return tuple(committed)

    def resolve_pending(
        self,
        *,
        execute: MeasurementExecute | None = None,
        account_collection_time: bool = True,
    ) -> None:
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
        requests = tuple(request for request in discovered if self.lookup(request.key, request.protocol) is None)
        if not requests:
            return

        if self._cancellation.cancelled():
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.CANCELLED,
                        request.op_id,
                        "session cancelled",
                        key=request.key,
                        failure_kind=MeasurementFailureKind.CANCELLATION,
                    )
                    for request in requests
                ]
            )

        if self.policy is ResolutionPolicy.OBSERVE_ONLY:
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.OBSERVE_ONLY,
                        request.op_id,
                        f"observed unresolved key {request.key.digest}",
                        key=request.key,
                        failure_kind=MeasurementFailureKind.OBSERVATION,
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
                        key=request.key,
                        failure_kind=MeasurementFailureKind.BUDGET,
                    )
                    for request in requests
                ]
            )

        self._charged_keys.update(request.key for request in newly_charged)
        trace = self._callback_trace
        if trace is not None:
            trace.collection_attempted = True
        started = self._clock()
        try:
            execute_measurements = execute or self.executor.execute
            records = tuple(
                execute_measurements(
                    requests,
                    deadline_monotonic=started + remaining_seconds,
                    cancellation=self._cancellation,
                )
            )
        except Exception as error:
            if account_collection_time:
                self.charge_collection_seconds(self._clock() - started)
            code = UnresolvedCode.CANCELLED if self._cancellation.cancelled() else UnresolvedCode.COLLECTOR_FAILED
            failure_kind = (
                MeasurementFailureKind.CANCELLATION
                if code is UnresolvedCode.CANCELLED
                else MeasurementFailureKind.INVARIANT
            )
            reasons = [
                UnresolvedReason(
                    code,
                    request.op_id,
                    str(error),
                    key=request.key,
                    failure_kind=failure_kind,
                )
                for request in requests
            ]
            for request, reason in zip(requests, reasons, strict=True):
                self._remember_failure(request.key, reason)
            self._fail(reasons)
        elapsed = self._clock() - started
        if account_collection_time:
            self.charge_collection_seconds(elapsed)
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
                self.report.observe_record(
                    record,
                    persisted=False,
                    sequence=None,
                    include_execution=True,
                    rejection_reason="executor returned an unrequested key",
                )
                if not cancelled_after_dispatch:
                    failures.append(
                        UnresolvedReason(
                            UnresolvedCode.INVALID_MEASUREMENT,
                            record.key.namespace,
                            f"executor returned unrequested key {record.key.digest}",
                            key=record.key,
                            failure_kind=MeasurementFailureKind.INVARIANT,
                        )
                    )
                continue
            if record.protocol != request.protocol:
                self.report.observe_record(
                    record,
                    persisted=False,
                    sequence=None,
                    include_execution=True,
                    rejection_reason="record protocol does not exactly match request",
                )
                invalid_keys.add(record.key)
                if not cancelled_after_dispatch:
                    reason = UnresolvedReason(
                        UnresolvedCode.IDENTITY_MISMATCH,
                        request.op_id,
                        "record protocol does not exactly match request",
                        key=request.key,
                        failure_kind=MeasurementFailureKind.INVARIANT,
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
                for record in matches:
                    self.report.observe_record(
                        record,
                        persisted=False,
                        sequence=None,
                        include_execution=True,
                        rejection_reason=f"expected one record for key, got {len(matches)}",
                    )
                reason = UnresolvedReason(
                    UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    f"expected one record for {request.key.digest}, got {len(matches)}",
                    key=request.key,
                    failure_kind=MeasurementFailureKind.INVARIANT,
                )
                failures.append(reason)
                failures_to_remember.append((request.key, reason))
                continue

            record = matches[0]
            sequence = self.overlay.append(record)
            self._observe_persisted_record(
                record,
                sequence,
                include_execution=True,
            )
            self.report.accepted_records += int(record.status is RecordStatus.VALID)
            self.report.rejected_records += int(record.status is not RecordStatus.VALID)
            if cancelled_after_dispatch:
                continue
            if record.status is RecordStatus.FAILED:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.COLLECTOR_FAILED,
                    request.op_id,
                    record.failure_reason or "collector failed",
                    key=request.key,
                    failure_kind=record.failure_kind or MeasurementFailureKind.INVARIANT,
                )
                failures.append(reason)
                failures_to_remember.append((request.key, reason))
            elif record.status is not RecordStatus.VALID or record.latency_ms is None:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    record.failure_reason or "measurement rejected",
                    key=request.key,
                    failure_kind=record.failure_kind or MeasurementFailureKind.INVARIANT,
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
                    failure_kind=MeasurementFailureKind.CANCELLATION,
                )
            )
        else:
            self.report.rejected_records += validation_rejections
            for key, reason in failures_to_remember:
                self._remember_failure(key, reason)
        if account_collection_time and elapsed > remaining_seconds and not cancelled_after_dispatch:
            failures.extend(
                UnresolvedReason(
                    UnresolvedCode.BUDGET_EXHAUSTED,
                    request.op_id,
                    f"collection exceeded wall budget by {elapsed - remaining_seconds:.3f}s",
                    key=request.key,
                    failure_kind=MeasurementFailureKind.BUDGET,
                )
                for request in requests
            )
        if failures:
            self._fail(failures)

    def execute_callback(
        self,
        query: Callable[[], T],
        *,
        context: Mapping[str, object] | None = None,
    ) -> T:
        return self._execute_callback_with_resolver(
            query,
            self.resolve_pending,
            context=context,
        )

    def _execute_callback_with_resolver(
        self,
        query: Callable[[], T],
        resolve_pending: Callable[[], None],
        *,
        context: Mapping[str, object] | None = None,
    ) -> T:
        with self._callback_lock:
            if self._callback_depth:
                if context is not None:
                    raise ValueError("nested resolution callbacks cannot replace the outer callback context")
                return query()
            context_snapshot = _json_snapshot(context or {})
            if not isinstance(context_snapshot, dict):
                raise TypeError("resolution callback context must be a mapping")
            self._callback_depth += 1
            self._active_fallbacks.clear()
            self._callback_sequence += 1
            self._callback_trace = _CallbackTrace(
                sequence=self._callback_sequence,
                context=context_snapshot,
                accepted_records_start=self.report.accepted_records,
                rejected_records_start=self.report.rejected_records,
                collection_seconds_start=self.report.collection_seconds,
            )
            try:
                result = self._execute_callback_locked(query, resolve_pending)
            except BaseException as error:
                self._finish_callback(result=None, error=error)
                self._misses.clear()
                self._unresolved.clear()
                self._tainted_operations.clear()
                raise
            else:
                self._finish_callback(result=result, error=None)
                return result
            finally:
                self._active_fallbacks.clear()
                self._callback_depth -= 1

    def _execute_callback_locked(
        self,
        query: Callable[[], T],
        resolve_pending: Callable[[], None],
    ) -> T:
        self._misses.clear()
        self._unresolved.clear()
        self._tainted_operations.clear()
        first = query()
        if not self._misses and not self._unresolved:
            if self._tainted_operations:
                self._fail(
                    [
                        UnresolvedReason(
                            UnresolvedCode.REQUERY_STILL_MISSING,
                            operation,
                            "provisional evidence was produced without a recorded miss",
                        )
                        for operation in self._tainted_operations
                    ]
                )
            return first
        first_pass_entries = self._misses.entries()
        resolve_pending()
        self._misses.clear()
        self._unresolved.clear()
        self._tainted_operations.clear()
        self._begin_replay()
        result = query()
        replay_tainted_operations = tuple(self._tainted_operations)
        replay_tainted = bool(replay_tainted_operations) or getattr(result, "source", None) == "unresolved"
        if replay_tainted and not self._misses and not self._unresolved:
            operations = replay_tainted_operations or tuple(entry.request.op_id for entry in first_pass_entries)
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.REQUERY_STILL_MISSING,
                        operation,
                        "provisional evidence remained after collection",
                    )
                    for operation in operations
                ]
            )
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
