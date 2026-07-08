# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aiconfigurator.sdk.performance_result import PerformanceResult


def _copy_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            copied[key] = _copy_json(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_json(item) for item in value]
    return value


def _encode_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the stable JSON encoding used by resolution identities."""
    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON values must be mappings")
    return _encode_json(_copy_json(value))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class _FrozenMapping(Mapping[str, Any]):
    """Small, pickle-safe immutable mapping for copied JSON evidence."""

    __slots__ = ("_data", "_hash")

    def __init__(self, value: Mapping[str, Any]) -> None:
        self._data = {key: _freeze_json(item) for key, item in value.items()}
        self._hash: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(canonical_json(self))
        return self._hash

    def __reduce__(self) -> tuple[type[_FrozenMapping], tuple[dict[str, Any]]]:
        return _FrozenMapping, (_copy_json(self),)


def _snapshot_json_mapping(value: Mapping[str, Any], *, field_name: str) -> tuple[Mapping[str, Any], str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        canonical = _encode_json(_copy_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must contain JSON-safe finite values") from error
    return _FrozenMapping(json.loads(canonical)), canonical


def _immutable_json_mapping(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    return _snapshot_json_mapping(value, field_name=field_name)[0]


@dataclass(frozen=True, slots=True)
class PerfKey:
    namespace: str
    query_json: str
    environment_json: str

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        for field_name in ("query_json", "environment_json"):
            raw_json = getattr(self, field_name)
            try:
                value = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"{field_name} must contain a JSON object") from error
            if not isinstance(value, dict):
                raise TypeError(f"{field_name} must contain a JSON object")
            object.__setattr__(self, field_name, canonical_json(value))

    @classmethod
    def build(
        cls,
        namespace: str,
        query: Mapping[str, Any],
        environment: Mapping[str, Any] | MeasurementEnvironment,
    ) -> PerfKey:
        environment_json = (
            environment.canonical if isinstance(environment, MeasurementEnvironment) else canonical_json(environment)
        )
        return cls(
            namespace,
            canonical_json(query),
            environment_json,
        )

    @property
    def canonical(self) -> str:
        return canonical_json(
            {
                "namespace": self.namespace,
                "query": json.loads(self.query_json),
                "environment": json.loads(self.environment_json),
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    revision: str
    warmups: int
    samples: int
    statistic: str = "median"
    timer: str = "cuda_event"
    tuning_revision: str = "none"

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("revision must not be empty")
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")
        if self.samples <= 0:
            raise ValueError("samples must be positive")
        if not self.statistic:
            raise ValueError("statistic must not be empty")
        if not self.timer:
            raise ValueError("timer must not be empty")
        if not self.tuning_revision:
            raise ValueError("tuning_revision must not be empty")

    @property
    def canonical(self) -> str:
        return canonical_json(
            {
                "revision": self.revision,
                "warmups": self.warmups,
                "samples": self.samples,
                "statistic": self.statistic,
                "timer": self.timer,
                "tuning_revision": self.tuning_revision,
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementEnvironment:
    system: str
    backend: str
    backend_version: str
    gpu_class: str
    runtime_versions: Mapping[str, str]
    topology_schema: str | None = None
    topology_fingerprint: str | None = None
    profile_compatibility: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_versions",
            _immutable_json_mapping(self.runtime_versions, field_name="runtime_versions"),
        )
        if self.profile_compatibility is not None:
            object.__setattr__(
                self,
                "profile_compatibility",
                _immutable_json_mapping(
                    self.profile_compatibility,
                    field_name="profile_compatibility",
                ),
            )

    @property
    def canonical(self) -> str:
        identity = {
            "system": self.system,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "gpu_class": self.gpu_class,
            "runtime_versions": self.runtime_versions,
            "topology_schema": self.topology_schema,
            "topology_fingerprint": self.topology_fingerprint,
        }
        if self.profile_compatibility is not None:
            identity["profile_compatibility"] = self.profile_compatibility
        return canonical_json(identity)


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    op_id: str
    key: PerfKey
    query: Mapping[str, Any]
    environment: MeasurementEnvironment
    semantic_descriptor: Mapping[str, Any]
    protocol: MeasurementProtocol

    def __post_init__(self) -> None:
        query, query_json = _snapshot_json_mapping(self.query, field_name="query")
        semantic_descriptor, _ = _snapshot_json_mapping(
            self.semantic_descriptor,
            field_name="semantic_descriptor",
        )
        if self.key.query_json != query_json:
            raise ValueError("request query does not match PerfKey query")
        if self.key.environment_json != self.environment.canonical:
            raise ValueError("request environment does not match PerfKey environment")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "semantic_descriptor", semantic_descriptor)


class _StringEnum(str, Enum):
    """Python 3.10-compatible equivalent of ``enum.StrEnum``."""

    def __str__(self) -> str:
        return self.value


class RecordStatus(_StringEnum):
    VALID = "valid"
    REJECTED = "rejected"
    FAILED = "failed"


class ResolutionPolicy(_StringEnum):
    PURE = "pure"
    OBSERVE_ONLY = "observe_only"
    MEASURE_ON_MISS = "measure_on_miss"


class UnresolvedCode(_StringEnum):
    MISSING_ADAPTER = "missing_adapter"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    COLLECTOR_FAILED = "collector_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    IDENTITY_MISMATCH = "identity_mismatch"
    INVALID_MEASUREMENT = "invalid_measurement"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RETRY_EXHAUSTED = "retry_exhausted"
    REQUERY_STILL_MISSING = "requery_still_missing"
    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    key: PerfKey
    status: RecordStatus
    latency_ms: float | None
    energy_wms: float
    samples_ms: tuple[float, ...]
    protocol: MeasurementProtocol
    perf_row: Mapping[str, Any]
    provenance: Mapping[str, Any]
    failure_code: UnresolvedCode | None = None
    failure_reason: str | None = None
    sequence: int | None = field(default=None, compare=False)

    @classmethod
    def valid(cls, **kwargs: Any) -> MeasurementRecord:
        return cls(status=RecordStatus.VALID, failure_code=None, failure_reason=None, **kwargs)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples_ms", tuple(self.samples_ms))
        object.__setattr__(
            self,
            "perf_row",
            _immutable_json_mapping(self.perf_row, field_name="perf_row"),
        )
        object.__setattr__(
            self,
            "provenance",
            _immutable_json_mapping(self.provenance, field_name="provenance"),
        )

        if not math.isfinite(self.energy_wms) or self.energy_wms < 0:
            raise ValueError("energy_wms must be finite and non-negative")
        if any(not math.isfinite(sample) or sample < 0 for sample in self.samples_ms):
            raise ValueError("measurement samples must be finite and non-negative")
        if self.status is RecordStatus.VALID:
            if self.latency_ms is None or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
                raise ValueError("valid records require finite non-negative latency_ms")
            if len(self.samples_ms) != self.protocol.samples:
                raise ValueError("sample count must match the measurement protocol")
            if self.failure_code is not None or self.failure_reason is not None:
                raise ValueError("valid records cannot provide failure details")
        elif self.latency_ms is not None:
            raise ValueError("non-valid records cannot provide latency_ms")

    def performance_result(self, scale_factor: float = 1.0) -> PerformanceResult:
        if self.status is not RecordStatus.VALID or self.latency_ms is None:
            raise ValueError("only valid measurement records produce performance results")
        return PerformanceResult(
            self.latency_ms * scale_factor,
            energy=self.energy_wms * scale_factor,
            source="overlay",
        )


@dataclass(frozen=True, slots=True)
class UnresolvedReason:
    code: UnresolvedCode
    operation: str
    detail: str
