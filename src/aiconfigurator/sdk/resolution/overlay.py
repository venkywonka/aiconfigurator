# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeVar

from aiconfigurator.sdk.resolution.types import (
    MeasurementProtocol,
    MeasurementRecord,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
    canonical_json,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS measurement_records (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    key_digest TEXT NOT NULL,
    key_json TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    energy_wms REAL NOT NULL,
    samples_json TEXT NOT NULL,
    statistic TEXT NOT NULL,
    protocol_digest TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    perf_row_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    failure_code TEXT,
    failure_reason TEXT
);
CREATE INDEX IF NOT EXISTS measurement_key_sequence
ON measurement_records(key_digest, sequence DESC);
"""

_T = TypeVar("_T")
_INITIALIZATION_TIMEOUT_SECONDS = 30.0


class OverlayStore:
    """Process-local connection to an append-only measurement overlay."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._owner_pid = os.getpid()
        self._closed = False
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA busy_timeout=100")
            journal_mode = self._retry_locked(lambda: self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            if str(journal_mode).lower() != "wal":
                raise RuntimeError(f"overlay requires SQLite WAL mode, got {journal_mode!r}")
            self._retry_locked(lambda: self._connection.executescript(_SCHEMA))
            self._connection.execute("PRAGMA busy_timeout=30000")
        except BaseException:
            self._connection.close()
            self._closed = True
            raise
        self._cache: dict[tuple[str, str], MeasurementRecord] = {}
        self._data_version = self._read_data_version()

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("OverlayStore connections cannot be pickled")

    def _ensure_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError("OverlayStore connection was inherited by a different process")
        if self._closed:
            raise RuntimeError("OverlayStore is closed")

    @staticmethod
    def _retry_locked(action: Callable[[], _T]) -> _T:
        deadline = time.monotonic() + _INITIALIZATION_TIMEOUT_SECONDS
        delay = 0.01
        while True:
            try:
                return action()
            except sqlite3.OperationalError as error:
                message = str(error).lower()
                if not ("locked" in message or "busy" in message):
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.1)

    def _read_data_version(self) -> int:
        return int(self._connection.execute("PRAGMA data_version").fetchone()[0])

    def _refresh_cache(self) -> None:
        data_version = self._read_data_version()
        if data_version != self._data_version:
            self._cache.clear()
            self._data_version = data_version

    def append(self, record: MeasurementRecord) -> int:
        """Append one immutable record and return its monotonic sequence."""
        self._ensure_owner()
        with self._lock:
            self._ensure_owner()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                sequence = self._insert_record(record)
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

            committed = replace(record, sequence=sequence)
            if committed.status is RecordStatus.VALID:
                cache_key = (committed.key.digest, committed.protocol.digest)
                cached = self._cache.get(cache_key)
                if cached is None or cached.sequence is None or cached.sequence < sequence:
                    self._cache[cache_key] = committed
            return sequence

    def _insert_record(self, record: MeasurementRecord) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO measurement_records (
                key_digest,
                key_json,
                status,
                latency_ms,
                energy_wms,
                samples_json,
                statistic,
                protocol_digest,
                protocol_json,
                perf_row_json,
                provenance_json,
                failure_code,
                failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.key.digest,
                record.key.canonical,
                record.status.value,
                record.latency_ms,
                record.energy_wms,
                json.dumps(record.samples_ms, separators=(",", ":"), allow_nan=False),
                record.protocol.statistic,
                record.protocol.digest,
                record.protocol.canonical,
                canonical_json(record.perf_row),
                canonical_json(record.provenance),
                record.failure_code.value if record.failure_code is not None else None,
                record.failure_reason,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite append did not return a sequence")
        return int(cursor.lastrowid)

    def lookup(
        self,
        key: PerfKey,
        protocol: MeasurementProtocol,
    ) -> MeasurementRecord | None:
        """Return the latest valid exact record for the full key and protocol."""
        self._ensure_owner()
        with self._lock:
            self._ensure_owner()
            self._refresh_cache()
            cache_key = (key.digest, protocol.digest)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            record = self._lookup_database(key, protocol)
            if record is not None:
                self._cache[cache_key] = record
            return record

    def _lookup_database(
        self,
        key: PerfKey,
        protocol: MeasurementProtocol,
    ) -> MeasurementRecord | None:
        rows = self._connection.execute(
            """
            SELECT *
            FROM measurement_records
            WHERE key_digest = ?
              AND status = 'valid'
              AND protocol_digest = ?
            ORDER BY sequence DESC
            """,
            (key.digest, protocol.digest),
        )
        for row in rows:
            if row["key_json"] != key.canonical:
                continue
            if row["protocol_json"] != protocol.canonical:
                continue
            return self._record_from_row(row)
        return None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> MeasurementRecord:
        key_value = json.loads(row["key_json"])
        protocol_value = json.loads(row["protocol_json"])
        return MeasurementRecord(
            key=PerfKey.build(
                namespace=key_value["namespace"],
                query=key_value["query"],
                environment=key_value["environment"],
                semantic=key_value["semantic"],
            ),
            status=RecordStatus(row["status"]),
            latency_ms=row["latency_ms"],
            energy_wms=row["energy_wms"],
            samples_ms=tuple(json.loads(row["samples_json"])),
            protocol=MeasurementProtocol(**protocol_value),
            perf_row=json.loads(row["perf_row_json"]),
            provenance=json.loads(row["provenance_json"]),
            failure_code=(UnresolvedCode(row["failure_code"]) if row["failure_code"] is not None else None),
            failure_reason=row["failure_reason"],
            sequence=row["sequence"],
        )

    def close(self) -> None:
        """Close this process-local connection."""
        if self._closed:
            return
        self._ensure_owner()
        with self._lock:
            self._ensure_owner()
            self._connection.close()
            self._closed = True
