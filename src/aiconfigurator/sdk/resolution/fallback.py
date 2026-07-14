# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable per-``PerfKey`` HYBRID fallback evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from aiconfigurator.sdk.resolution.types import PerfKey, UnresolvedCode, UnresolvedReason, canonical_json

FALLBACK_SCHEMA_VERSION = 1
_LATENCY_UNITS = "ms"
_TIMING_SOURCE = "hybrid"
_RECORD_FIELDS = frozenset(
    {
        "canonical_perf_key",
        "created_at",
        "exact",
        "hybrid_provenance",
        "identity_digest",
        "key_digest",
        "latency_ms",
        "latency_units",
        "measurement_failure",
        "schema_version",
        "timing_source",
    }
)
_PERF_KEY_FIELDS = frozenset({"namespace", "query", "environment"})
_PROVENANCE_FIELDS = frozenset({"prediction_revision", "source"})
_FAILURE_FIELDS = frozenset({"code", "operation", "detail"})
_RFC3339 = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z")
_ExactT = TypeVar("_ExactT")


class FallbackCorruptionError(RuntimeError):
    """A published sidecar is malformed or does not match its requested identity."""

    def __init__(self, path: Path, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"corrupt fallback sidecar {path}: {detail}")


def fallback_identity_digest(
    key: PerfKey,
    prediction_revision: str,
    *,
    schema_revision: int = FALLBACK_SCHEMA_VERSION,
) -> str:
    """Return the canonical identity for one per-key HYBRID fallback."""

    if not isinstance(key, PerfKey):
        raise TypeError("key must be a PerfKey")
    if not isinstance(prediction_revision, str) or not prediction_revision.strip():
        raise ValueError("prediction_revision must be a non-empty string")
    if isinstance(schema_revision, bool) or not isinstance(schema_revision, int) or schema_revision <= 0:
        raise ValueError("schema_revision must be a positive integer")
    identity = canonical_json(
        {
            "canonical_perf_key": json.loads(key.canonical),
            "fallback_schema_revision": schema_revision,
            "prediction_revision": prediction_revision,
        }
    )
    return f"sha256:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class FallbackRecord:
    """One durable, explicitly non-exact HYBRID result."""

    identity_digest: str
    key: PerfKey
    latency_ms: float
    hybrid_source: str
    prediction_revision: str
    measurement_failure: UnresolvedReason
    created_at: str
    path: Path
    schema_version: int = FALLBACK_SCHEMA_VERSION
    latency_units: str = _LATENCY_UNITS
    timing_source: str = _TIMING_SOURCE
    exact: bool = False


class FallbackStore:
    """Filesystem store containing one immutable JSON file per fallback identity."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            raise ValueError("fallback store directory must be absolute")

    @staticmethod
    def identity_digest(key: PerfKey, prediction_revision: str) -> str:
        return fallback_identity_digest(key, prediction_revision)

    def path_for(self, key: PerfKey, *, prediction_revision: str) -> Path:
        digest = self.identity_digest(key, prediction_revision)
        return self.directory / f"{digest.removeprefix('sha256:')}.json"

    def publish(
        self,
        key: PerfKey,
        *,
        prediction_revision: str,
        latency_ms: float,
        hybrid_source: str,
        measurement_failure: UnresolvedReason,
    ) -> FallbackRecord:
        record, _ = self.publish_with_status(
            key,
            prediction_revision=prediction_revision,
            latency_ms=latency_ms,
            hybrid_source=hybrid_source,
            measurement_failure=measurement_failure,
        )
        return record

    def publish_with_status(
        self,
        key: PerfKey,
        *,
        prediction_revision: str,
        latency_ms: float,
        hybrid_source: str,
        measurement_failure: UnresolvedReason,
    ) -> tuple[FallbackRecord, bool]:
        """Publish one immutable record and report whether this call created it."""

        record = self._new_record(
            key,
            prediction_revision=prediction_revision,
            latency_ms=latency_ms,
            hybrid_source=hybrid_source,
            measurement_failure=measurement_failure,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(self._payload(record)).encode("utf-8") + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{record.identity_digest.removeprefix('sha256:')}.",
            suffix=".tmp",
            dir=self.directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, record.path)
            except FileExistsError:
                winner = self.lookup(key, prediction_revision=prediction_revision)
                if winner is None:
                    raise RuntimeError("fallback winner disappeared during publication") from None
                return winner, False
            temporary_path.unlink()
            self._fsync_directory()
            return record, True
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def lookup(
        self,
        key: PerfKey,
        *,
        prediction_revision: str,
        force_remeasure: bool = False,
    ) -> FallbackRecord | None:
        path = self.path_for(key, prediction_revision=prediction_revision)
        if not isinstance(force_remeasure, bool):
            raise TypeError("force_remeasure must be a bool")
        if force_remeasure:
            return None
        raw = self._read_sidecar(path)
        if raw is None:
            return None
        try:
            payload = json.loads(
                raw,
                parse_constant=self._reject_json_constant,
                object_pairs_hook=self._unique_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise FallbackCorruptionError(path, f"invalid JSON: {error}") from error
        return self._record_from_payload(
            payload,
            path=path,
            expected_key=key,
            expected_prediction_revision=prediction_revision,
        )

    def lookup_after_exact(
        self,
        key: PerfKey,
        *,
        prediction_revision: str,
        overlay_lookup: Callable[[], _ExactT | None],
        curated_lookup: Callable[[], _ExactT | None],
        force_remeasure: bool = False,
    ) -> _ExactT | FallbackRecord | None:
        """Probe exact overlay and curated evidence before a fallback sidecar."""

        if not callable(overlay_lookup) or not callable(curated_lookup):
            raise TypeError("exact evidence lookups must be callable")
        overlay = overlay_lookup()
        if overlay is not None:
            return overlay
        curated = curated_lookup()
        if curated is not None:
            return curated
        return self.lookup(
            key,
            prediction_revision=prediction_revision,
            force_remeasure=force_remeasure,
        )

    @staticmethod
    def _read_sidecar(path: Path) -> str | None:
        try:
            path_stat = os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise FallbackCorruptionError(path, f"cannot inspect sidecar: {error}") from error
        if not stat.S_ISREG(path_stat.st_mode):
            raise FallbackCorruptionError(path, "sidecar must be a regular immutable file")

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise FallbackCorruptionError(path, f"sidecar must be a regular immutable file: {error}") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise FallbackCorruptionError(path, "sidecar must be a regular immutable file")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                raw = stream.read()
        except FallbackCorruptionError:
            raise
        except OSError as error:
            raise FallbackCorruptionError(path, f"cannot read sidecar: {error}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(raw, bytes):
            raise FallbackCorruptionError(path, "sidecar read returned no bytes")
        try:
            return raw.decode("utf-8")
        except UnicodeError as error:
            raise FallbackCorruptionError(path, f"invalid JSON encoding: {error}") from error

    def _new_record(
        self,
        key: PerfKey,
        *,
        prediction_revision: str,
        latency_ms: float,
        hybrid_source: str,
        measurement_failure: UnresolvedReason,
    ) -> FallbackRecord:
        latency = self._positive_latency(latency_ms)
        self._non_empty_string(prediction_revision, field_name="prediction_revision")
        self._non_empty_string(hybrid_source, field_name="hybrid_source")
        if not isinstance(measurement_failure, UnresolvedReason):
            raise TypeError("measurement_failure must be an UnresolvedReason")
        if not isinstance(measurement_failure.code, UnresolvedCode):
            raise TypeError("measurement_failure code must be an UnresolvedCode")
        self._non_empty_string(measurement_failure.operation, field_name="measurement_failure operation")
        self._non_empty_string(measurement_failure.detail, field_name="measurement_failure detail")
        identity_digest = self.identity_digest(key, prediction_revision)
        return FallbackRecord(
            identity_digest=identity_digest,
            key=key,
            latency_ms=latency,
            hybrid_source=hybrid_source,
            prediction_revision=prediction_revision,
            measurement_failure=measurement_failure,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            path=self.path_for(key, prediction_revision=prediction_revision),
        )

    @staticmethod
    def _payload(record: FallbackRecord) -> dict[str, object]:
        return {
            "canonical_perf_key": json.loads(record.key.canonical),
            "created_at": record.created_at,
            "exact": record.exact,
            "hybrid_provenance": {
                "prediction_revision": record.prediction_revision,
                "source": record.hybrid_source,
            },
            "identity_digest": record.identity_digest,
            "key_digest": record.key.digest,
            "latency_ms": record.latency_ms,
            "latency_units": record.latency_units,
            "measurement_failure": {
                "code": record.measurement_failure.code.value,
                "detail": record.measurement_failure.detail,
                "operation": record.measurement_failure.operation,
            },
            "schema_version": record.schema_version,
            "timing_source": record.timing_source,
        }

    def _record_from_payload(
        self,
        payload: object,
        *,
        path: Path,
        expected_key: PerfKey,
        expected_prediction_revision: str,
    ) -> FallbackRecord:
        try:
            return self._validate_payload(
                payload,
                path=path,
                expected_key=expected_key,
                expected_prediction_revision=expected_prediction_revision,
            )
        except FallbackCorruptionError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            detail = str(error) or type(error).__name__
            raise FallbackCorruptionError(path, detail) from error

    def _validate_payload(
        self,
        payload: object,
        *,
        path: Path,
        expected_key: PerfKey,
        expected_prediction_revision: str,
    ) -> FallbackRecord:
        record_value = self._exact_object(payload, _RECORD_FIELDS, field_name="sidecar")
        schema_version = record_value["schema_version"]
        if type(schema_version) is not int or schema_version != FALLBACK_SCHEMA_VERSION:
            raise ValueError(f"unsupported fallback schema_version {schema_version!r}")

        key_value = self._exact_object(
            record_value["canonical_perf_key"],
            _PERF_KEY_FIELDS,
            field_name="canonical_perf_key",
        )
        key = PerfKey.build(
            self._non_empty_string(key_value["namespace"], field_name="PerfKey namespace"),
            self._mapping(key_value["query"], field_name="PerfKey query"),
            self._mapping(key_value["environment"], field_name="PerfKey environment"),
        )
        if key != expected_key:
            raise ValueError("canonical PerfKey does not match the requested key")
        if record_value["key_digest"] != expected_key.digest:
            raise ValueError("key_digest does not match the canonical PerfKey")

        provenance = self._exact_object(
            record_value["hybrid_provenance"],
            _PROVENANCE_FIELDS,
            field_name="hybrid_provenance",
        )
        prediction_revision = self._non_empty_string(
            provenance["prediction_revision"],
            field_name="prediction_revision",
        )
        if prediction_revision != expected_prediction_revision:
            raise ValueError("prediction_revision does not match the requested identity")
        hybrid_source = self._non_empty_string(provenance["source"], field_name="HYBRID source")
        expected_digest = self.identity_digest(expected_key, expected_prediction_revision)
        if record_value["identity_digest"] != expected_digest:
            raise ValueError("identity_digest does not match the requested fallback identity")

        latency_ms = self._positive_latency(record_value["latency_ms"])
        if record_value["latency_units"] != _LATENCY_UNITS:
            raise ValueError("fallback latency_units must be 'ms'")
        if record_value["timing_source"] != _TIMING_SOURCE:
            raise ValueError("fallback timing_source must be 'hybrid'")
        if record_value["exact"] is not False:
            raise ValueError("fallback exact field must be literal false")

        failure_value = self._exact_object(
            record_value["measurement_failure"],
            _FAILURE_FIELDS,
            field_name="measurement_failure",
        )
        code = UnresolvedCode(failure_value["code"])
        operation = self._non_empty_string(
            failure_value["operation"],
            field_name="measurement_failure operation",
        )
        detail = self._non_empty_string(
            failure_value["detail"],
            field_name="measurement_failure detail",
        )
        created_at = self._rfc3339(record_value["created_at"])
        return FallbackRecord(
            identity_digest=expected_digest,
            key=key,
            latency_ms=latency_ms,
            hybrid_source=hybrid_source,
            prediction_revision=prediction_revision,
            measurement_failure=UnresolvedReason(code, operation, detail),
            created_at=created_at,
            path=path,
        )

    @staticmethod
    def _positive_latency(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("fallback latency_ms must be a number")
        latency = float(value)
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("fallback latency_ms must be positive and finite")
        return latency

    @staticmethod
    def _non_empty_string(value: object, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        if not value.strip():
            raise ValueError(f"{field_name} must be non-empty")
        return value

    @staticmethod
    def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError(f"{field_name} must be a JSON object")
        return value

    @classmethod
    def _exact_object(
        cls,
        value: object,
        expected_fields: frozenset[str],
        *,
        field_name: str,
    ) -> dict[str, Any]:
        result = cls._mapping(value, field_name=field_name)
        actual_fields = frozenset(result)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            extra = sorted(actual_fields - expected_fields)
            raise ValueError(f"{field_name} fields mismatch: missing={missing}, extra={extra}")
        return result

    @staticmethod
    def _rfc3339(value: object) -> str:
        created_at = FallbackStore._non_empty_string(value, field_name="created_at")
        if _RFC3339.fullmatch(created_at) is None:
            raise ValueError("created_at must be an RFC3339 timestamp")
        normalized = created_at[:-1] + "+00:00" if created_at.endswith("Z") else created_at
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at must include an RFC3339 timezone")
        return created_at

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r}")

    @staticmethod
    def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.directory, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
