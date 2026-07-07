# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.types import (
    MeasurementProtocol,
    MeasurementRecord,
    PerfKey,
    RecordStatus,
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


def _key(m: int = 8) -> PerfKey:
    return PerfKey.build("gemm/v1", {"m": m}, {"system": "h100"})


def _record(
    key: PerfKey,
    latency: float,
    *,
    status: RecordStatus = RecordStatus.VALID,
    protocol: MeasurementProtocol | None = None,
) -> MeasurementRecord:
    valid = status is RecordStatus.VALID
    return MeasurementRecord(
        key=key,
        status=status,
        latency_ms=latency if valid else None,
        energy_wms=0.0,
        samples_ms=(latency,) if valid else (),
        protocol=protocol or _protocol(),
        perf_row={"m": json.loads(key.query_json)["m"], "latency": latency},
        provenance={"collector_revision": "r1"},
        failure_code=None if valid else UnresolvedCode.INVALID_MEASUREMENT,
        failure_reason=None if valid else "boom",
    )


def test_latest_valid_commit_wins(tmp_path) -> None:
    key = _key()
    protocol = _protocol()
    store = OverlayStore(tmp_path / "overlay.sqlite")

    first = store.append(_record(key, 0.2))
    rejected = store.append(_record(key, 9.9, status=RecordStatus.REJECTED))
    last = store.append(_record(key, 0.1))
    hit = store.lookup(key, protocol)

    assert first < rejected < last
    assert hit is not None
    assert hit.sequence == last
    assert hit.latency_ms == pytest.approx(0.1)


def test_rejected_and_failed_records_never_become_hits(tmp_path) -> None:
    key = _key()
    store = OverlayStore(tmp_path / "overlay.sqlite")

    store.append(_record(key, 1.0, status=RecordStatus.REJECTED))
    store.append(_record(key, 1.0, status=RecordStatus.FAILED))

    assert store.lookup(key, _protocol()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"revision": "microbench-v2"},
        {"warmups": 4},
        {"samples": 2},
        {"statistic": "mean"},
        {"timer": "wall_clock"},
        {"tuning_revision": "search-v2"},
    ],
)
def test_complete_protocol_identity_is_required(tmp_path, overrides: dict[str, object]) -> None:
    key = _key()
    store = OverlayStore(tmp_path / "overlay.sqlite")
    store.append(_record(key, 0.1))

    assert store.lookup(key, _protocol(**overrides)) is None


def test_second_unchanged_lookup_uses_memory_cache(tmp_path, monkeypatch) -> None:
    key = _key()
    protocol = _protocol()
    path = tmp_path / "overlay.sqlite"
    store = OverlayStore(path)
    store.append(_record(key, 0.1))
    store.close()
    store = OverlayStore(path)
    store.lookup(key, protocol)

    def unexpected_query(*args, **kwargs):
        raise AssertionError("database lookup should be cached")

    monkeypatch.setattr(store, "_lookup_database", unexpected_query)

    assert store.lookup(key, protocol) is not None


def test_store_allows_serialized_thread_handoff(tmp_path) -> None:
    key = _key()
    protocol = _protocol()
    store = OverlayStore(tmp_path / "overlay.sqlite")
    store.append(_record(key, 0.1))

    with ThreadPoolExecutor(max_workers=1) as executor:
        hit = executor.submit(store.lookup, key, protocol).result()

    assert hit is not None
    assert hit.latency_ms == pytest.approx(0.1)


def test_other_connection_commit_invalidates_memory_cache(tmp_path) -> None:
    path = tmp_path / "overlay.sqlite"
    key = _key()
    protocol = _protocol()
    first = OverlayStore(path)
    second = OverlayStore(path)
    first.append(_record(key, 0.2))
    assert first.lookup(key, protocol).latency_ms == pytest.approx(0.2)

    sequence = second.append(_record(key, 0.1))
    refreshed = first.lookup(key, protocol)

    assert refreshed is not None
    assert refreshed.sequence == sequence
    assert refreshed.latency_ms == pytest.approx(0.1)


def test_close_and_reopen_reuses_committed_record(tmp_path) -> None:
    path = tmp_path / "overlay.sqlite"
    key = _key()
    protocol = _protocol()
    store = OverlayStore(path)
    sequence = store.append(_record(key, 0.1))
    store.close()

    reopened = OverlayStore(path)
    hit = reopened.lookup(key, protocol)

    assert hit is not None
    assert hit.sequence == sequence
    assert hit.latency_ms == pytest.approx(0.1)


def test_append_exception_rolls_back_and_leaves_connection_usable(tmp_path, monkeypatch) -> None:
    key = _key()
    protocol = _protocol()
    store = OverlayStore(tmp_path / "overlay.sqlite")
    insert = store._insert_record

    def insert_then_fail(record):
        insert(record)
        raise RuntimeError("injected append failure")

    monkeypatch.setattr(store, "_insert_record", insert_then_fail)
    with pytest.raises(RuntimeError, match="injected append failure"):
        store.append(_record(key, 0.2))

    assert not store._connection.in_transaction
    assert store.lookup(key, protocol) is None

    monkeypatch.setattr(store, "_insert_record", insert)
    store.append(_record(key, 0.1))
    assert store.lookup(key, protocol).latency_ms == pytest.approx(0.1)


def test_full_canonical_key_is_checked_after_digest_match(tmp_path) -> None:
    key = _key()
    other = _key(16)
    protocol = _protocol()
    store = OverlayStore(tmp_path / "overlay.sqlite")
    expected = store.append(_record(key, 0.2))
    forged = store.append(_record(other, 0.1))
    store._connection.execute(
        "UPDATE measurement_records SET key_digest = ? WHERE sequence = ?",
        (key.digest, forged),
    )
    store._cache.clear()

    hit = store.lookup(key, protocol)

    assert hit is not None
    assert hit.sequence == expected
    assert hit.key == key


def test_full_canonical_protocol_is_checked_after_digest_match(tmp_path) -> None:
    key = _key()
    protocol = _protocol()
    other_protocol = _protocol(warmups=4)
    store = OverlayStore(tmp_path / "overlay.sqlite")
    expected = store.append(_record(key, 0.2, protocol=protocol))
    forged = store.append(_record(key, 0.1, protocol=other_protocol))
    store._connection.execute(
        "UPDATE measurement_records SET protocol_digest = ? WHERE sequence = ?",
        (protocol.digest, forged),
    )
    store._cache.clear()

    hit = store.lookup(key, protocol)

    assert hit is not None
    assert hit.sequence == expected
    assert hit.protocol == protocol


def test_store_enables_wal_and_busy_timeout(tmp_path) -> None:
    store = OverlayStore(tmp_path / "overlay.sqlite")

    journal_mode = store._connection.execute("PRAGMA journal_mode").fetchone()[0]
    busy_timeout = store._connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert journal_mode.lower() == "wal"
    assert busy_timeout == 30_000


def test_concurrent_initialization_is_safe(tmp_path) -> None:
    path = tmp_path / "overlay.sqlite"
    barrier = threading.Barrier(8)

    def open_and_close(_: int) -> None:
        barrier.wait()
        store = OverlayStore(path)
        store.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(open_and_close, range(16)))


def test_store_connection_is_not_pickleable(tmp_path) -> None:
    store = OverlayStore(tmp_path / "overlay.sqlite")

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(store)


def test_inherited_connection_is_rejected(tmp_path, monkeypatch) -> None:
    store = OverlayStore(tmp_path / "overlay.sqlite")
    monkeypatch.setattr("aiconfigurator.sdk.resolution.overlay.os.getpid", lambda: store._owner_pid + 1)

    with pytest.raises(RuntimeError, match="different process"):
        store.lookup(_key(), _protocol())
