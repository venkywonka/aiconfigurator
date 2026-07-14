# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import aiconfigurator.sdk.resolution as resolution
from aiconfigurator.sdk.resolution import fallback as fallback_module
from aiconfigurator.sdk.resolution.fallback import (
    FallbackCorruptionError,
    FallbackStore,
    fallback_identity_digest,
)
from aiconfigurator.sdk.resolution.types import PerfKey, UnresolvedCode, UnresolvedReason, canonical_json

pytestmark = pytest.mark.unit


def _key() -> PerfKey:
    return PerfKey.build(
        "gemm_perf.txt/v1",
        {"dtype": "bfloat16", "k": 1024, "m": 8, "n": 4096},
        {
            "backend": "sglang",
            "backend_version": "0.5.10",
            "gpu_class": "NVIDIA GB200",
            "runtime_versions": {"sglang": "0.5.10"},
            "system": "gb200",
        },
    )


def _failure() -> UnresolvedReason:
    return UnresolvedReason(
        UnresolvedCode.TIMEOUT,
        "attention.context",
        "collector deadline expired",
    )


def _process_publish(
    cache_dir: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
    index: int,
) -> None:
    try:
        start.wait()
        record = FallbackStore(cache_dir).publish(
            _key(),
            prediction_revision="aic-hybrid-r3",
            latency_ms=float(index + 1),
            hybrid_source=f"empirical-{index}",
            measurement_failure=UnresolvedReason(
                UnresolvedCode.TIMEOUT,
                "attention.context",
                f"process-{index}:" + "x" * 500_000,
            ),
        )
        results.put(
            (
                "ok",
                record.identity_digest,
                record.latency_ms,
                record.hybrid_source,
                record.created_at,
            )
        )
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)[:500]))


def test_identity_is_canonical_full_key_plus_schema_and_prediction_revision() -> None:
    key = _key()
    reordered_key = PerfKey.build(
        key.namespace,
        {"n": 4096, "m": 8, "k": 1024, "dtype": "bfloat16"},
        {
            "system": "gb200",
            "runtime_versions": {"sglang": "0.5.10"},
            "gpu_class": "NVIDIA GB200",
            "backend_version": "0.5.10",
            "backend": "sglang",
        },
    )
    expected_identity = canonical_json(
        {
            "canonical_perf_key": json.loads(key.canonical),
            "fallback_schema_revision": 1,
            "prediction_revision": "aic-hybrid-r3",
        }
    )
    expected = f"sha256:{hashlib.sha256(expected_identity.encode('utf-8')).hexdigest()}"

    assert fallback_identity_digest(key, "aic-hybrid-r3", schema_revision=1) == expected
    assert fallback_identity_digest(reordered_key, "aic-hybrid-r3", schema_revision=1) == expected
    assert fallback_identity_digest(key, "aic-hybrid-r4", schema_revision=1) != expected
    assert fallback_identity_digest(key, "aic-hybrid-r3", schema_revision=2) != expected
    assert (
        fallback_identity_digest(
            PerfKey.build(
                key.namespace, json.loads(key.query_json), {**json.loads(key.environment_json), "system": "b200"}
            ),
            "aic-hybrid-r3",
            schema_revision=1,
        )
        != expected
    )


def test_fallback_store_contract_is_exported_from_resolution_package() -> None:
    assert resolution.FALLBACK_SCHEMA_VERSION == 1
    assert resolution.FallbackCorruptionError is FallbackCorruptionError
    assert resolution.FallbackRecord is fallback_module.FallbackRecord
    assert resolution.FallbackStore is FallbackStore
    assert resolution.fallback_identity_digest is fallback_identity_digest


def test_publish_and_reopen_reuses_one_immutable_per_key_fallback(tmp_path) -> None:
    key = _key()
    cache_dir = tmp_path / "fallbacks"

    published = FallbackStore(cache_dir).publish(
        key,
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    reopened = FallbackStore(cache_dir).lookup(
        key,
        prediction_revision="aic-hybrid-r3",
    )

    assert reopened == published
    assert reopened is not None
    assert reopened.key == key
    assert reopened.latency_ms == pytest.approx(1.234)
    assert reopened.hybrid_source == "empirical"
    assert reopened.prediction_revision == "aic-hybrid-r3"
    assert reopened.measurement_failure == _failure()
    assert reopened.exact is False
    assert reopened.timing_source == "hybrid"
    assert reopened.latency_units == "ms"
    assert tuple(cache_dir.glob("*.json")) == (published.path,)

    with pytest.raises(FrozenInstanceError):
        published.latency_ms = 9.0


def test_published_json_has_the_complete_canonical_hybrid_schema(tmp_path) -> None:
    key = _key()
    record = FallbackStore(tmp_path).publish(
        key,
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    raw = record.path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert raw == canonical_json(payload) + "\n"
    assert payload == {
        "canonical_perf_key": json.loads(key.canonical),
        "created_at": record.created_at,
        "exact": False,
        "hybrid_provenance": {
            "prediction_revision": "aic-hybrid-r3",
            "source": "empirical",
        },
        "identity_digest": fallback_identity_digest(key, "aic-hybrid-r3"),
        "key_digest": key.digest,
        "latency_ms": 1.234,
        "latency_units": "ms",
        "measurement_failure": {
            "code": "timeout",
            "detail": "collector deadline expired",
            "operation": "attention.context",
        },
        "schema_version": 1,
        "timing_source": "hybrid",
    }


@pytest.mark.parametrize("latency_ms", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True])
def test_publish_rejects_non_positive_or_non_finite_latency(tmp_path, latency_ms) -> None:
    with pytest.raises((TypeError, ValueError), match="latency_ms"):
        FallbackStore(tmp_path).publish(
            _key(),
            prediction_revision="aic-hybrid-r3",
            latency_ms=latency_ms,
            hybrid_source="empirical",
            measurement_failure=_failure(),
        )


@pytest.mark.parametrize(
    ("prediction_revision", "hybrid_source", "measurement_failure"),
    [
        ("", "empirical", _failure()),
        ("aic-hybrid-r3", "", _failure()),
        (
            "aic-hybrid-r3",
            "empirical",
            UnresolvedReason("timeout", "attention.context", "collector deadline expired"),
        ),
        ("aic-hybrid-r3", "empirical", UnresolvedReason(UnresolvedCode.TIMEOUT, "", "detail")),
        ("aic-hybrid-r3", "empirical", UnresolvedReason(UnresolvedCode.TIMEOUT, "operation", "")),
    ],
)
def test_publish_rejects_incomplete_hybrid_provenance_or_typed_failure(
    tmp_path,
    prediction_revision,
    hybrid_source,
    measurement_failure,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FallbackStore(tmp_path).publish(
            _key(),
            prediction_revision=prediction_revision,
            latency_ms=1.0,
            hybrid_source=hybrid_source,
            measurement_failure=measurement_failure,
        )


def _replace(field: str, value: object) -> Callable[[dict[str, object]], None]:
    def replace(payload: dict[str, object]) -> None:
        payload[field] = value

    return replace


def _replace_nested(parent: str, field: str, value: object) -> Callable[[dict[str, object]], None]:
    def replace(payload: dict[str, object]) -> None:
        nested = payload[parent]
        assert isinstance(nested, dict)
        nested[field] = value

    return replace


def _remove(field: str) -> Callable[[dict[str, object]], None]:
    def remove(payload: dict[str, object]) -> None:
        del payload[field]

    return remove


def _add_extra(payload: dict[str, object]) -> None:
    payload["unexpected"] = True


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_remove("key_digest"), id="missing-field"),
        pytest.param(_add_extra, id="extra-field"),
        pytest.param(_replace("schema_version", 2), id="schema-version"),
        pytest.param(_replace("key_digest", "forged"), id="key-digest"),
        pytest.param(_replace("identity_digest", "sha256:forged"), id="identity-digest"),
        pytest.param(_replace_nested("canonical_perf_key", "namespace", "other/v1"), id="canonical-key"),
        pytest.param(_replace("latency_ms", 0.0), id="zero-latency"),
        pytest.param(_replace("latency_ms", float("inf")), id="infinite-latency"),
        pytest.param(_replace("latency_units", "seconds"), id="latency-units"),
        pytest.param(_replace("exact", 0), id="exact-not-literal-false"),
        pytest.param(_replace("timing_source", "silicon"), id="timing-source"),
        pytest.param(_replace("created_at", "not-rfc3339"), id="created-at"),
        pytest.param(_replace_nested("hybrid_provenance", "prediction_revision", "other"), id="prediction"),
        pytest.param(_replace_nested("hybrid_provenance", "source", ""), id="source"),
        pytest.param(_replace_nested("measurement_failure", "code", "not-a-code"), id="failure-code"),
        pytest.param(_replace_nested("measurement_failure", "operation", ""), id="failure-operation"),
        pytest.param(_replace_nested("measurement_failure", "detail", ""), id="failure-detail"),
    ],
)
def test_lookup_rejects_corrupt_or_identity_mismatched_sidecars(tmp_path, mutate) -> None:
    store = FallbackStore(tmp_path)
    record = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    payload = json.loads(record.path.read_text(encoding="utf-8"))
    mutate(payload)
    record.path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(FallbackCorruptionError) as raised:
        store.lookup(_key(), prediction_revision="aic-hybrid-r3")

    assert raised.value.path == record.path


def test_lookup_normalizes_malformed_json_to_structured_corruption(tmp_path) -> None:
    store = FallbackStore(tmp_path)
    record = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    record.path.write_text("{not json", encoding="utf-8")

    with pytest.raises(FallbackCorruptionError, match="invalid JSON") as raised:
        store.lookup(_key(), prediction_revision="aic-hybrid-r3")

    assert raised.value.path == record.path


def test_lookup_rejects_symlink_even_when_target_contains_valid_json(tmp_path) -> None:
    store = FallbackStore(tmp_path / "store")
    record = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    target = tmp_path / "mutable-target.json"
    record.path.replace(target)
    record.path.symlink_to(target)

    with pytest.raises(FallbackCorruptionError, match="regular immutable file"):
        store.lookup(_key(), prediction_revision="aic-hybrid-r3")


@pytest.mark.parametrize(
    ("read_outcome", "expected_detail"),
    [
        (None, "returned no bytes"),
        (BlockingIOError(11, "temporarily unavailable"), "cannot read sidecar"),
    ],
)
def test_lookup_normalizes_low_level_read_failures_to_structured_corruption(
    tmp_path,
    monkeypatch,
    read_outcome,
    expected_detail: str,
) -> None:
    store = FallbackStore(tmp_path)
    record = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.234,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    real_fdopen = os.fdopen

    class ControlledReader:
        def __init__(self, descriptor: int) -> None:
            self._stream = real_fdopen(descriptor, "rb")

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def read(self):
            if isinstance(read_outcome, BaseException):
                raise read_outcome
            return read_outcome

    monkeypatch.setattr(
        fallback_module.os,
        "fdopen",
        lambda descriptor, _mode: ControlledReader(descriptor),
    )

    with pytest.raises(FallbackCorruptionError, match=expected_detail) as raised:
        store.lookup(_key(), prediction_revision="aic-hybrid-r3")

    assert raised.value.path == record.path


def test_existing_valid_winner_is_returned_without_overwrite(tmp_path) -> None:
    store = FallbackStore(tmp_path)
    first = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.0,
        hybrid_source="empirical-first",
        measurement_failure=_failure(),
    )
    first_bytes = first.path.read_bytes()

    winner = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=9.0,
        hybrid_source="empirical-conflict",
        measurement_failure=UnresolvedReason(
            UnresolvedCode.COLLECTOR_FAILED,
            "gemm",
            "later conflicting failure",
        ),
    )

    assert winner == first
    assert first.path.read_bytes() == first_bytes
    assert tuple(tmp_path.glob("*.json")) == (first.path,)


def test_existing_corrupt_winner_is_rejected_without_overwrite(tmp_path) -> None:
    store = FallbackStore(tmp_path)
    first = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.0,
        hybrid_source="empirical-first",
        measurement_failure=_failure(),
    )
    first.path.write_bytes(b"{corrupt winner")

    with pytest.raises(FallbackCorruptionError, match="invalid JSON"):
        store.publish(
            _key(),
            prediction_revision="aic-hybrid-r3",
            latency_ms=9.0,
            hybrid_source="empirical-conflict",
            measurement_failure=_failure(),
        )

    assert first.path.read_bytes() == b"{corrupt winner"


def test_publish_fsyncs_complete_unique_same_directory_temp_then_links_no_replace(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    created_temps: list[Path] = []
    linked: list[tuple[Path, Path]] = []
    fsynced_modes: list[int] = []
    real_mkstemp = tempfile.mkstemp
    real_link = os.link
    real_fsync = os.fsync

    def tracked_mkstemp(*args, **kwargs):
        fd, raw_path = real_mkstemp(*args, **kwargs)
        created_temps.append(Path(raw_path))
        return fd, raw_path

    def tracked_link(source, destination, *args, **kwargs):
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == cache_dir
        assert destination_path.parent == cache_dir
        assert source_path != destination_path
        assert not destination_path.exists()
        json.loads(source_path.read_text(encoding="utf-8"))
        linked.append((source_path, destination_path))
        return real_link(source, destination, *args, **kwargs)

    def tracked_fsync(fd):
        fsynced_modes.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(fallback_module.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(fallback_module.os, "link", tracked_link)
    monkeypatch.setattr(fallback_module.os, "fsync", tracked_fsync)

    first = FallbackStore(cache_dir).publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.0,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    second_key = PerfKey.build(
        _key().namespace, {**json.loads(_key().query_json), "m": 16}, json.loads(_key().environment_json)
    )
    second = FallbackStore(cache_dir).publish(
        second_key,
        prediction_revision="aic-hybrid-r3",
        latency_ms=2.0,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )

    assert len(created_temps) == 2
    assert len(set(created_temps)) == 2
    assert linked == [(created_temps[0], first.path), (created_temps[1], second.path)]
    assert all(not path.exists() for path in created_temps)
    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_publish_validates_the_other_writer_that_wins_link_race(tmp_path, monkeypatch) -> None:
    key = _key()
    source = FallbackStore(tmp_path / "source").publish(
        key,
        prediction_revision="aic-hybrid-r3",
        latency_ms=4.0,
        hybrid_source="empirical-winner",
        measurement_failure=UnresolvedReason(
            UnresolvedCode.RESOURCE_UNAVAILABLE,
            "gemm",
            "winner failure",
        ),
    )
    winner_bytes = source.path.read_bytes()
    target = FallbackStore(tmp_path / "target")

    def win_link_race(_source, destination, *args, **kwargs):
        destination_path = Path(destination)
        descriptor = os.open(destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, winner_bytes)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise FileExistsError(destination_path)

    monkeypatch.setattr(fallback_module.os, "link", win_link_race)

    winner = target.publish(
        key,
        prediction_revision="aic-hybrid-r3",
        latency_ms=9.0,
        hybrid_source="empirical-loser",
        measurement_failure=_failure(),
    )

    assert winner.latency_ms == pytest.approx(4.0)
    assert winner.hybrid_source == "empirical-winner"
    assert winner.measurement_failure.code is UnresolvedCode.RESOURCE_UNAVAILABLE
    assert winner.path.read_bytes() == winner_bytes
    assert not tuple((tmp_path / "target").glob("*.tmp"))


def test_publish_rejects_corrupt_link_race_winner_and_cleans_temp(tmp_path, monkeypatch) -> None:
    store = FallbackStore(tmp_path)

    def corrupt_link_race(_source, destination, *args, **kwargs):
        destination_path = Path(destination)
        descriptor = os.open(destination_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"{corrupt")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise FileExistsError(destination_path)

    monkeypatch.setattr(fallback_module.os, "link", corrupt_link_race)

    with pytest.raises(FallbackCorruptionError, match="invalid JSON"):
        store.publish(
            _key(),
            prediction_revision="aic-hybrid-r3",
            latency_ms=1.0,
            hybrid_source="empirical",
            measurement_failure=_failure(),
        )

    assert store.path_for(_key(), prediction_revision="aic-hybrid-r3").read_bytes() == b"{corrupt"
    assert not tuple(tmp_path.glob("*.tmp"))


def test_concurrent_threads_converge_on_one_first_valid_writer(tmp_path) -> None:
    store = FallbackStore(tmp_path)

    def publish(index: int):
        return store.publish(
            _key(),
            prediction_revision="aic-hybrid-r3",
            latency_ms=float(index + 1),
            hybrid_source=f"empirical-{index}",
            measurement_failure=_failure(),
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        records = tuple(executor.map(publish, range(32)))

    assert len({record.latency_ms for record in records}) == 1
    assert len({record.hybrid_source for record in records}) == 1
    assert tuple(tmp_path.glob("*.json")) == (records[0].path,)
    assert not tuple(tmp_path.glob("*.tmp"))


def test_concurrent_spawned_processes_converge_without_partial_reads(tmp_path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(target=_process_publish, args=(str(tmp_path), start, results, index)) for index in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    outcomes = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert {outcome[0] for outcome in outcomes} == {"ok"}, outcomes
    assert len({outcome[1:] for outcome in outcomes}) == 1
    assert len(tuple(tmp_path.glob("*.json"))) == 1
    assert not tuple(tmp_path.glob("*.tmp"))


def test_lookup_after_exact_enforces_overlay_then_curated_then_sidecar_precedence(tmp_path, monkeypatch) -> None:
    store = FallbackStore(tmp_path)
    sidecar = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.0,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    calls: list[str] = []
    overlay_exact = object()
    curated_exact = object()
    original_lookup = store.lookup

    def overlay_hit():
        calls.append("overlay")
        return overlay_exact

    def curated_must_not_run():
        raise AssertionError("curated lookup must not run after an overlay hit")

    def sidecar_must_not_run(*args, **kwargs):
        raise AssertionError("sidecar lookup must not run after exact evidence")

    monkeypatch.setattr(store, "lookup", sidecar_must_not_run)
    assert (
        store.lookup_after_exact(
            _key(),
            prediction_revision="aic-hybrid-r3",
            overlay_lookup=overlay_hit,
            curated_lookup=curated_must_not_run,
        )
        is overlay_exact
    )
    assert calls == ["overlay"]

    monkeypatch.setattr(store, "lookup", original_lookup)
    calls.clear()

    def overlay_miss():
        calls.append("overlay")
        return None

    def curated_hit():
        calls.append("curated")
        return curated_exact

    assert (
        store.lookup_after_exact(
            _key(),
            prediction_revision="aic-hybrid-r3",
            overlay_lookup=overlay_miss,
            curated_lookup=curated_hit,
        )
        is curated_exact
    )
    assert calls == ["overlay", "curated"]

    calls.clear()

    def curated_miss():
        calls.append("curated")
        return None

    assert (
        store.lookup_after_exact(
            _key(),
            prediction_revision="aic-hybrid-r3",
            overlay_lookup=overlay_miss,
            curated_lookup=curated_miss,
        )
        == sidecar
    )
    assert calls == ["overlay", "curated"]


def test_force_remeasure_skips_only_the_sidecar_lookup(tmp_path) -> None:
    store = FallbackStore(tmp_path)
    sidecar = store.publish(
        _key(),
        prediction_revision="aic-hybrid-r3",
        latency_ms=1.0,
        hybrid_source="empirical",
        measurement_failure=_failure(),
    )
    sidecar.path.write_bytes(b"{corrupt but force-skipped")
    calls: list[str] = []

    def overlay_miss():
        calls.append("overlay")
        return None

    def curated_miss():
        calls.append("curated")
        return None

    assert (
        store.lookup_after_exact(
            _key(),
            prediction_revision="aic-hybrid-r3",
            overlay_lookup=overlay_miss,
            curated_lookup=curated_miss,
            force_remeasure=True,
        )
        is None
    )
    assert calls == ["overlay", "curated"]
    assert sidecar.path.read_bytes() == b"{corrupt but force-skipped"
