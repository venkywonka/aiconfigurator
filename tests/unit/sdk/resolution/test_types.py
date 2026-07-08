# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Iterator, Mapping
from dataclasses import fields
from inspect import signature

import pytest

import aiconfigurator.sdk.resolution as resolution_api
import aiconfigurator.sdk.resolution.types as resolution_types
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
    canonical_json,
)

pytestmark = pytest.mark.unit


class _ChangingValueMapping(Mapping[str, object]):
    """Return one value for identity validation and another on later reads."""

    def __init__(self) -> None:
        self.reads = 0

    def __getitem__(self, key: str) -> object:
        if key != "m":
            raise KeyError(key)
        self.reads += 1
        return 8 if self.reads <= 2 else 16

    def __iter__(self) -> Iterator[str]:
        return iter(("m",))

    def __len__(self) -> int:
        return 1


class _ChangingItemsMapping(Mapping[object, object]):
    """Expose a string key on the first items pass and an integer later."""

    def __init__(self) -> None:
        self.items_calls = 0

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        return iter(())

    def __len__(self) -> int:
        return 1

    def items(self):
        self.items_calls += 1
        if self.items_calls == 1:
            return (("safe", "first"),)
        return ((1, "second"),)


def _protocol(**overrides: object) -> MeasurementProtocol:
    values = {
        "revision": "microbench-v1",
        "warmups": 3,
        "samples": 3,
        "statistic": "median",
        "timer": "cuda_event",
        "tuning_revision": "trtllm-linear-v1",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)


def _environment(runtime_versions: dict[str, str] | None = None) -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="h100_sxm",
        backend="trtllm",
        backend_version="1.2.0",
        gpu_class="h100-sxm-80gb",
        runtime_versions=runtime_versions or {"cuda": "12.8", "torch": "2.8"},
    )


def _environment_dict() -> dict[str, object]:
    return {
        "system": "h100_sxm",
        "backend": "trtllm",
        "backend_version": "1.2.0",
        "gpu_class": "h100-sxm-80gb",
        "runtime_versions": {"cuda": "12.8", "torch": "2.8"},
        "topology_schema": None,
        "topology_fingerprint": None,
    }


def _key(query: dict[str, object] | None = None) -> PerfKey:
    return PerfKey.build(
        namespace="trtllm/gemm/v1",
        query=query or {"m": 8},
        environment=_environment_dict(),
    )


def test_perf_key_namespace_is_the_only_persisted_dataset_identity() -> None:
    key = _key()

    assert set(json.loads(key.canonical)) == {
        "namespace",
        "query",
        "environment",
    }
    persisted_fields = {
        field.name
        for evidence_type in (PerfKey, MeasurementRequest, MeasurementRecord)
        for field in fields(evidence_type)
    }
    assert persisted_fields.isdisjoint({"dataset_id", "collector_ref"})
    for module in (resolution_api, resolution_types):
        assert not hasattr(module, "EvidenceQuery")
        assert not hasattr(module, "PerfNamespace")


def test_perf_key_is_order_independent() -> None:
    left = PerfKey.build(
        namespace="trtllm/gemm/v1",
        query={"m": 8, "n": 4096, "k": 4096, "dtype": "fp8"},
        environment={"system": "h100_sxm", "backend_version": "1.2.0"},
    )
    right = PerfKey.build(
        namespace="trtllm/gemm/v1",
        query={"dtype": "fp8", "k": 4096, "n": 4096, "m": 8},
        environment={"backend_version": "1.2.0", "system": "h100_sxm"},
    )

    assert left == right
    assert left.digest == right.digest


def test_perf_key_copies_nested_identity_inputs() -> None:
    query = {"shape": [8, 4096], "options": {"layout": "row_major"}}
    environment = {"system": "h100_sxm", "versions": {"cuda": "12.8"}}
    key = PerfKey.build("trtllm/gemm/v1", query, environment)
    canonical = key.canonical
    digest = key.digest

    query["shape"][0] = 16
    query["options"]["layout"] = "column_major"
    environment["versions"]["cuda"] = "13.0"

    assert key.canonical == canonical
    assert key.digest == digest


def test_direct_perf_key_construction_normalizes_json_encoding() -> None:
    built = PerfKey.build("n", {"a": 1, "b": 2}, {})
    direct = PerfKey("n", '{"b": 2, "a": 1}', "{ }")

    assert direct == built
    assert direct.digest == built.digest


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"namespace": "trtllm/gemm/v2"}, "namespace"),
        ({"query": {"m": 16}}, "query"),
        ({"environment": {**_environment_dict(), "system": "h200_sxm"}}, "environment"),
    ],
)
def test_each_perf_key_identity_component_changes_digest(overrides: dict[str, object], field: str) -> None:
    values: dict[str, object] = {
        "namespace": "trtllm/gemm/v1",
        "query": {"m": 8},
        "environment": _environment_dict(),
    }
    values.update(overrides)

    changed = PerfKey.build(**values)
    baseline = PerfKey.build("trtllm/gemm/v1", {"m": 8}, _environment_dict())

    assert changed.digest != baseline.digest, field


def test_perf_key_api_has_no_semantic_identity_component() -> None:
    assert "semantic" not in signature(PerfKey.build).parameters
    assert "semantic_json" not in {field.name for field in fields(PerfKey)}


@pytest.mark.parametrize(
    "query",
    [
        {1: "top-level"},
        {"outer": {1: "nested"}},
        {"outer": [{1: "nested-in-list"}]},
    ],
)
def test_canonical_identity_rejects_non_string_object_keys(query: dict[object, object]) -> None:
    with pytest.raises(TypeError, match="JSON object keys must be strings"):
        PerfKey.build("trtllm/gemm/v1", query, _environment_dict())


def test_canonical_json_validates_and_copies_one_mapping_snapshot() -> None:
    value = _ChangingItemsMapping()

    assert canonical_json(value) == '{"safe":"first"}'
    assert value.items_calls == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "microbench-v2"),
        ("warmups", 4),
        ("samples", 5),
        ("statistic", "mean"),
        ("timer", "wall_clock"),
        ("tuning_revision", "trtllm-linear-v2"),
    ],
)
def test_complete_protocol_participates_in_identity(field: str, value: object) -> None:
    assert _protocol(**{field: value}).digest != _protocol().digest


def test_protocol_is_canonical_across_equivalent_instances() -> None:
    assert _protocol().canonical == _protocol().canonical
    assert _protocol().digest == _protocol().digest


def test_string_enums_preserve_value_string_semantics_on_python_310() -> None:
    assert str(RecordStatus.VALID) == "valid"
    assert str(UnresolvedCode.COLLECTOR_FAILED) == "collector_failed"


def test_environment_copies_runtime_versions() -> None:
    versions = {"cuda": "12.8", "torch": "2.8"}
    environment = _environment(versions)
    canonical = environment.canonical

    versions["cuda"] = "13.0"

    assert environment.canonical == canonical
    with pytest.raises(TypeError):
        environment.runtime_versions["cuda"] = "13.0"


def test_request_accepts_semantic_metadata_outside_physical_identity() -> None:
    query = {"m": 8}
    semantic = {"length_bucket": [128, 256]}
    environment = _environment()
    key = PerfKey.build("trtllm/gemm/v1", query, environment)

    request = MeasurementRequest(
        op_id="gemm",
        key=key,
        query=query,
        environment=environment,
        semantic_descriptor=semantic,
        protocol=_protocol(),
    )

    assert request.key == key


def test_perf_key_accepts_canonical_measurement_environment() -> None:
    environment = _environment()

    key = PerfKey.build("trtllm/gemm/v1", {"m": 8}, environment)

    assert key.environment_json == environment.canonical


def test_typed_environment_identity_detects_mismatch() -> None:
    environment = _environment()
    other_environment = MeasurementEnvironment(
        system="h200_sxm",
        backend="trtllm",
        backend_version="1.2.0",
        gpu_class="h200-sxm-141gb",
        runtime_versions={"cuda": "12.8", "torch": "2.8"},
    )
    key = PerfKey.build("trtllm/gemm/v1", {"m": 8}, environment)

    assert key.digest != PerfKey.build("trtllm/gemm/v1", {"m": 8}, other_environment).digest
    with pytest.raises(ValueError, match="environment"):
        MeasurementRequest(
            op_id="gemm",
            key=key,
            query={"m": 8},
            environment=other_environment,
            semantic_descriptor={},
            protocol=_protocol(),
        )


@pytest.mark.parametrize(
    ("key", "query", "environment", "semantic", "match"),
    [
        (_key({"m": 16}), {"m": 8}, _environment(), {}, "query"),
        (
            PerfKey.build("trtllm/gemm/v1", {"m": 8}, {**_environment_dict(), "system": "h200_sxm"}),
            {"m": 8},
            _environment(),
            {},
            "environment",
        ),
    ],
)
def test_request_rejects_identity_mismatch(
    key: PerfKey,
    query: dict[str, object],
    environment: MeasurementEnvironment,
    semantic: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MeasurementRequest(
            op_id="gemm",
            key=key,
            query=query,
            environment=environment,
            semantic_descriptor=semantic,
            protocol=_protocol(),
        )


def test_request_copies_nested_query_and_semantic_descriptor() -> None:
    query = {"m": 8, "shape": [8, 4096]}
    semantic = {"lengths": [128, 256]}
    request = MeasurementRequest(
        op_id="gemm",
        key=PerfKey.build("trtllm/gemm/v1", query, _environment_dict()),
        query=query,
        environment=_environment(),
        semantic_descriptor=semantic,
        protocol=_protocol(),
    )

    query["shape"].append(8192)
    semantic["lengths"].append(512)

    assert tuple(request.query["shape"]) == (8, 4096)
    assert tuple(request.semantic_descriptor["lengths"]) == (128, 256)
    with pytest.raises(TypeError):
        request.query["m"] = 16


def test_request_validates_and_freezes_the_same_query_snapshot() -> None:
    query = _ChangingValueMapping()
    request = MeasurementRequest(
        op_id="gemm",
        key=_key(),
        query=query,
        environment=_environment(),
        semantic_descriptor={},
        protocol=_protocol(),
    )

    assert request.query["m"] == 8


def test_request_is_pickle_safe_for_spawn_workers() -> None:
    query = {"m": 8, "shape": [8, 4096]}
    semantic = {"lengths": [128, 256]}
    request = MeasurementRequest(
        op_id="gemm",
        key=PerfKey.build("trtllm/gemm/v1", query, _environment_dict()),
        query=query,
        environment=_environment(),
        semantic_descriptor=semantic,
        protocol=_protocol(),
    )

    assert pickle.loads(pickle.dumps(request)) == request


def test_record_converts_to_overlay_performance_result() -> None:
    record = MeasurementRecord.valid(
        key=_key(),
        latency_ms=0.125,
        energy_wms=0.5,
        samples_ms=(0.126, 0.124, 0.125),
        protocol=_protocol(),
        perf_row={"m": 8, "latency": 0.125},
        provenance={"collector_revision": "abc123"},
    )

    result = record.performance_result(scale_factor=2.0)

    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(1.0)
    assert result.source == "overlay"


def test_record_copies_perf_row_and_provenance() -> None:
    perf_row = {"m": 8, "metadata": {"winner": "kernel_a"}}
    provenance = {"collector_revision": "abc123", "devices": [0]}
    record = MeasurementRecord.valid(
        key=_key(),
        latency_ms=0.125,
        energy_wms=0.5,
        samples_ms=(0.126, 0.124, 0.125),
        protocol=_protocol(),
        perf_row=perf_row,
        provenance=provenance,
    )

    perf_row["metadata"]["winner"] = "kernel_b"
    provenance["devices"].append(1)

    assert record.perf_row["metadata"]["winner"] == "kernel_a"
    assert tuple(record.provenance["devices"]) == (0,)
    with pytest.raises(TypeError):
        record.perf_row["m"] = 16


def test_record_is_pickle_safe_for_spawn_workers() -> None:
    record = MeasurementRecord.valid(
        key=_key(),
        latency_ms=0.125,
        energy_wms=0.5,
        samples_ms=(0.126, 0.124, 0.125),
        protocol=_protocol(),
        perf_row={"m": 8, "metadata": {"winner": "kernel_a"}},
        provenance={"collector_revision": "abc123", "devices": [0]},
    )

    assert pickle.loads(pickle.dumps(record)) == record


@pytest.mark.parametrize("latency_ms", [-1.0, math.nan, math.inf, -math.inf])
def test_valid_record_rejects_invalid_latency(latency_ms: float) -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        MeasurementRecord.valid(
            key=_key(),
            latency_ms=latency_ms,
            energy_wms=0.0,
            samples_ms=(0.1, 0.1, 0.1),
            protocol=_protocol(),
            perf_row={},
            provenance={},
        )


def test_valid_record_rejects_sample_count_mismatch() -> None:
    with pytest.raises(ValueError, match="sample count"):
        MeasurementRecord.valid(
            key=_key(),
            latency_ms=0.1,
            energy_wms=0.0,
            samples_ms=(0.1,),
            protocol=_protocol(),
            perf_row={},
            provenance={},
        )


@pytest.mark.parametrize("sample", [-1.0, math.nan, math.inf, -math.inf])
def test_valid_record_rejects_invalid_samples(sample: float) -> None:
    with pytest.raises(ValueError, match="samples"):
        MeasurementRecord.valid(
            key=_key(),
            latency_ms=0.1,
            energy_wms=0.0,
            samples_ms=(0.1, 0.1, sample),
            protocol=_protocol(),
            perf_row={},
            provenance={},
        )


@pytest.mark.parametrize("energy_wms", [-1.0, math.nan, math.inf, -math.inf])
def test_record_rejects_invalid_energy(energy_wms: float) -> None:
    with pytest.raises(ValueError, match="energy_wms"):
        MeasurementRecord.valid(
            key=_key(),
            latency_ms=0.1,
            energy_wms=energy_wms,
            samples_ms=(0.1, 0.1, 0.1),
            protocol=_protocol(),
            perf_row={},
            provenance={},
        )


@pytest.mark.parametrize("status", list(RecordStatus))
@pytest.mark.parametrize("field", ["perf_row", "provenance"])
def test_record_rejects_non_json_safe_evidence_mappings(field: str, status: RecordStatus) -> None:
    evidence = {"metric": math.nan}
    kwargs = {"perf_row": {}, "provenance": {}, field: evidence}

    with pytest.raises(ValueError, match=rf"{field} must contain JSON-safe finite values"):
        MeasurementRecord(
            key=_key(),
            status=status,
            latency_ms=0.1 if status is RecordStatus.VALID else None,
            energy_wms=0.0,
            samples_ms=(0.1, 0.1, 0.1) if status is RecordStatus.VALID else (),
            protocol=_protocol(),
            failure_code=(None if status is RecordStatus.VALID else UnresolvedCode.INVALID_MEASUREMENT),
            failure_reason=None if status is RecordStatus.VALID else "non-finite diagnostics",
            **kwargs,
        )


def test_failed_record_rejects_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="samples must be finite"):
        MeasurementRecord(
            key=_key(),
            status=RecordStatus.FAILED,
            latency_ms=None,
            energy_wms=0.0,
            samples_ms=(math.nan,),
            protocol=_protocol(),
            perf_row={},
            provenance={},
            failure_code=UnresolvedCode.COLLECTOR_FAILED,
            failure_reason="kernel produced non-finite output",
        )


def test_valid_record_requires_latency() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        MeasurementRecord(
            key=_key(),
            status=RecordStatus.VALID,
            latency_ms=None,
            energy_wms=0.0,
            samples_ms=(0.1, 0.1, 0.1),
            protocol=_protocol(),
            perf_row={},
            provenance={},
        )


def test_non_valid_record_rejects_latency_and_cannot_produce_result() -> None:
    with pytest.raises(ValueError, match="non-valid"):
        MeasurementRecord(
            key=_key(),
            status=RecordStatus.REJECTED,
            latency_ms=0.1,
            energy_wms=0.0,
            samples_ms=(),
            protocol=_protocol(),
            perf_row={},
            provenance={},
            failure_code=UnresolvedCode.INVALID_MEASUREMENT,
            failure_reason="outlier spread exceeded policy",
        )

    failed = MeasurementRecord(
        key=_key(),
        status=RecordStatus.FAILED,
        latency_ms=None,
        energy_wms=0.0,
        samples_ms=(),
        protocol=_protocol(),
        perf_row={},
        provenance={},
        failure_code=UnresolvedCode.COLLECTOR_FAILED,
        failure_reason="worker exited",
    )
    with pytest.raises(ValueError, match="only valid"):
        failed.performance_result()
