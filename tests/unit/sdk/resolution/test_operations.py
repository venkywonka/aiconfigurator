# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiconfigurator.collector.adapters import ProtocolMismatchError
from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.operations.overlap import FallbackOp, OverlapOp
from aiconfigurator.sdk.perf_database import _cached_configured_database_view
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.coordinator import OnlineResolutionCoordinator
from aiconfigurator.sdk.resolution.fallback import FallbackStore
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionFailed, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementFailureKind,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
    UnresolvedReason,
)

pytestmark = pytest.mark.unit


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="operation-resolution-v1",
        warmups=2,
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


def _request(name: str, x: int, protocol: MeasurementProtocol) -> MeasurementRequest:
    query = {"x": x}
    environment = _environment()
    semantic = {"operation": name}
    return MeasurementRequest(
        op_id=name,
        key=PerfKey.build(f"test_operation/{name}/v1", query, environment),
        query=query,
        environment=environment,
        semantic_descriptor=semantic,
        protocol=protocol,
    )


def _record(
    request: MeasurementRequest,
    latency_ms: float,
    energy_wms: float = 0.0,
) -> MeasurementRecord:
    return MeasurementRecord.valid(
        key=request.key,
        latency_ms=latency_ms,
        energy_wms=energy_wms,
        samples_ms=(latency_ms,),
        protocol=request.protocol,
        perf_row={"latency": latency_ms, "energy": energy_wms},
        provenance={"collector_revision": "test-v1"},
    )


def _reopened_lookup(path, key: PerfKey, protocol: MeasurementProtocol):
    overlay = OverlayStore(path)
    try:
        return overlay.lookup(key, protocol)
    finally:
        overlay.close()


class _RootDatabase:
    def __init__(self) -> None:
        self._default_database_mode = common.DatabaseMode.HYBRID
        self._shared_layer_mode = True
        self._transfer_policy = common.ALL_TRANSFERS
        self._extracted_metrics_cache = {"root": object()}
        self.supported_quant_mode = {"gemm": ["bfloat16"]}

    @property
    def transfer_policy(self):
        return self._transfer_policy

    @property
    def enable_shared_layer(self) -> bool:
        return self._shared_layer_mode


class _Executor:
    def __init__(self, values: dict[str, tuple[float, float]] | None = None) -> None:
        self.values = values or {}
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []

    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: object,
    ) -> Sequence[MeasurementRecord]:
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        return tuple(_record(request, *self.values.get(request.op_id, (1.0, 0.0))) for request in batch)


class _BindingExecutor(_Executor):
    def __init__(
        self,
        bound_protocol: MeasurementProtocol,
        values: dict[str, tuple[float, float]] | None = None,
        *,
        binding_error: Exception | None = None,
    ) -> None:
        super().__init__(values)
        self.bound_protocol = bound_protocol
        self.binding_error = binding_error
        self.binding_inputs: list[MeasurementRequest] = []

    def bind_request(self, request: MeasurementRequest) -> MeasurementRequest:
        self.binding_inputs.append(request)
        if self.binding_error is not None:
            raise self.binding_error
        return replace(request, protocol=self.bound_protocol)


class _TypedFailureExecutor(_Executor):
    def __init__(
        self,
        failed_operation: str,
        code: UnresolvedCode,
        *,
        kind: MeasurementFailureKind = MeasurementFailureKind.OPERATIONAL,
    ) -> None:
        super().__init__()
        self.failed_operation = failed_operation
        self.code = code
        self.kind = kind

    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: object,
    ) -> Sequence[MeasurementRecord]:
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        records = []
        for request in batch:
            if request.op_id == self.failed_operation:
                records.append(
                    MeasurementRecord(
                        key=request.key,
                        status=RecordStatus.FAILED,
                        latency_ms=None,
                        energy_wms=0.0,
                        samples_ms=(),
                        protocol=request.protocol,
                        perf_row={},
                        provenance={"collector_revision": "test-v1"},
                        failure_code=self.code,
                        failure_reason=f"injected {self.code.value}",
                        failure_kind=self.kind,
                    )
                )
            else:
                records.append(_record(request, 1.25))
        return tuple(records)


class _BlockingExactExecutor(_Executor):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: object,
    ) -> Sequence[MeasurementRecord]:
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        self.entered.set()
        assert self.release.wait(timeout=5.0)
        return tuple(_record(request, 1.25) for request in batch)


class _DuplicateRecordExecutor(_Executor):
    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: object,
    ) -> Sequence[MeasurementRecord]:
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.request_batches.append(batch)
        record = _record(batch[0], 1.25)
        return (record, record)


class _TableOp(Operation):
    def __init__(
        self,
        name: str,
        *,
        scale_factor: float = 1.0,
        adapter: bool = True,
        curated: dict[int, PerformanceResult] | None = None,
        ordinary: PerformanceResult | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self.adapter = adapter
        self.curated = curated or {}
        self.ordinary = ordinary or PerformanceResult(9.0, energy=90.0, source="silicon")
        self.query_calls: list[tuple[object, dict[str, object]]] = []
        self.measurement_calls: list[tuple[object, int]] = []
        self.curated_calls: list[tuple[object, int]] = []

    def query(self, database, **kwargs) -> PerformanceResult:
        self.query_calls.append((database, dict(kwargs)))
        return self.ordinary

    def get_weights(self, **kwargs) -> float:
        del kwargs
        return 0.0

    def measurement_request(self, database, protocol, **kwargs):
        x = kwargs["x"]
        self.measurement_calls.append((database, x))
        return _request(self._name, x, protocol) if self.adapter else None

    def curated_exact_result(self, database, **kwargs):
        x = kwargs["x"]
        self.curated_calls.append((database, x))
        return self.curated.get(x)


class _HybridTrackingTableOp(_TableOp):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.hybrid_resolver_calls = 0

    def hybrid_fallback_value(self, database, *, normalized_query, **kwargs):
        self.hybrid_resolver_calls += 1
        return super().hybrid_fallback_value(
            database,
            normalized_query=normalized_query,
            **kwargs,
        )


class _RaisingHybridOp(_HybridTrackingTableOp):
    def hybrid_fallback_value(self, database, *, normalized_query, **kwargs):
        self.hybrid_resolver_calls += 1
        raise RuntimeError("injected HYBRID query failure")


class _NormalizedTableOp(_TableOp):
    """Measured fake whose three physical lookup paths share one normalizer."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.normalization_calls: list[dict[str, object]] = []
        self.ordinary_normalized: list[dict[str, object]] = []
        self.provisional_normalized: list[dict[str, object]] = []
        self.request_normalized: list[dict[str, object]] = []
        self.curated_normalized: list[dict[str, object]] = []

    def normalize_perf_query(self, **kwargs) -> dict[str, object]:
        self.normalization_calls.append(dict(kwargs))
        return {
            "x": int(kwargs["tokens"]) * int(kwargs["token_multiplier"]),
        }

    def query(self, database, **kwargs) -> PerformanceResult:
        del database
        normalized = self.normalize_perf_query(**kwargs)
        self.ordinary_normalized.append(normalized)
        return self.ordinary

    def provisional_result(self, database, *, normalized_query, **kwargs) -> PerformanceResult:
        del database, kwargs
        self.provisional_normalized.append(normalized_query)
        return self.ordinary

    def _measurement_request_from_normalized(self, database, protocol, *, normalized_query, **kwargs):
        del database, kwargs
        self.request_normalized.append(normalized_query)
        return _request(self._name, int(normalized_query["x"]), protocol)

    def _curated_exact_result_from_normalized(self, database, *, normalized_query, **kwargs):
        del database, kwargs
        self.curated_normalized.append(normalized_query)
        return self.curated.get(int(normalized_query["x"]))


class _InvalidNormalizerOp(_TableOp):
    def normalize_perf_query(self, **kwargs):
        del kwargs
        return ["not", "a", "mapping"]


class _MissingSiliconTableOp(_TableOp):
    def query(self, database, **kwargs) -> PerformanceResult:
        self.query_calls.append((database, kwargs))
        raise PerfDataNotAvailableError(f"{self._name} has no silicon point")


class _MissingSiliconOverriddenProvisionalTableOp(_TableOp):
    def provisional_result(self, database, *, normalized_query, **kwargs) -> PerformanceResult:
        del database, normalized_query, kwargs
        raise PerfDataNotAvailableError(f"{self._name} has no provisional silicon point")


class _MissingSiliconNoAdapterOp(Operation):
    def query(self, database, **kwargs) -> PerformanceResult:
        del database, kwargs
        raise PerfDataNotAvailableError("no curated data and no lazy adapter")


class _LegacyKwargsOp(_TableOp):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.request_kwargs: list[dict[str, object]] = []
        self.exact_kwargs: list[dict[str, object]] = []

    def measurement_request(self, database, protocol, **kwargs):
        del database
        self.request_kwargs.append(dict(kwargs))
        return _request(self._name, int(kwargs["x"]), protocol)

    def curated_exact_result(self, database, **kwargs):
        del database
        self.exact_kwargs.append(dict(kwargs))
        return None


class _NarrowLegacyOp(_TableOp):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.request_x: list[int] = []
        self.exact_x: list[int] = []

    def measurement_request(self, database, protocol, *, x):
        del database
        self.request_x.append(x)
        return _request(self._name, x, protocol)

    def curated_exact_result(self, database, *, x):
        del database
        self.exact_x.append(x)
        return None


class _BareOp(Operation):
    def query(self, database, **kwargs) -> PerformanceResult:
        del database, kwargs
        return PerformanceResult(3.0, energy=4.0, source="empirical")

    def get_weights(self, **kwargs) -> float:
        del kwargs
        return 0.0


class _InstrumentedBareOp(_BareOp):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1.0)
        self.resolution_calls = 0

    def query_with_resolution(self, database, *, session=None, **kwargs) -> PerformanceResult:
        self.resolution_calls += 1
        return super().query_with_resolution(database, session=session, **kwargs)


@pytest.fixture(autouse=True)
def _clear_database_views():
    _cached_configured_database_view.cache_clear()
    yield
    _cached_configured_database_view.cache_clear()


@pytest.fixture
def session_factory(tmp_path: Path):
    stores: list[OverlayStore] = []

    def make(executor: _Executor, protocol: MeasurementProtocol | None = None) -> ResolutionSession:
        overlay = OverlayStore(tmp_path / f"overlay-{len(stores)}.sqlite")
        stores.append(overlay)
        return ResolutionSession(
            overlay,
            executor,
            ResolutionBudget(max_new_keys=16, max_wall_seconds=30.0),
            protocol or _protocol(),
        )

    yield make
    for store in stores:
        store.close()


def test_base_no_session_delegates_to_ordinary_query_byte_for_byte() -> None:
    database = _RootDatabase()
    expected = PerformanceResult(7.0, energy=8.0, source="empirical")
    op = _TableOp("plain", ordinary=expected)

    result = op.query_with_resolution(database, session=None, x=11, marker="unchanged")

    assert result is expected
    assert op.query_calls == [(database, {"x": 11, "marker": "unchanged"})]


def test_default_perf_query_normalization_is_identity() -> None:
    raw_query = {"x": 11, "batch_size": 2}

    normalized = _BareOp("plain", 1.0).normalize_perf_query(**raw_query)

    assert normalized == raw_query


def test_base_resolution_reuses_one_normalized_mapping_for_request_and_exact_probe(session_factory) -> None:
    database = _RootDatabase()
    op = _NormalizedTableOp("normalized")
    raw_query = {"tokens": 3, "token_multiplier": 4}

    op.query(database, **raw_query)
    ordinary_normalized = op.ordinary_normalized[0]
    calls_before_resolution = len(op.normalization_calls)

    result = op.query_with_resolution(
        database,
        session=session_factory(_Executor()),
        **raw_query,
    )

    assert result is op.ordinary
    assert len(op.normalization_calls) == calls_before_resolution + 1
    resolving_normalized = op.request_normalized[0]
    assert resolving_normalized is op.curated_normalized[0]
    assert resolving_normalized is op.provisional_normalized[0]
    assert resolving_normalized == ordinary_normalized == {"x": 12}


@pytest.mark.parametrize("wrap", (False, True))
def test_resolution_rejects_non_mapping_normalization_before_lookup(session_factory, wrap: bool) -> None:
    invalid = _InvalidNormalizerOp("invalid")
    op = FallbackOp("wrapper", invalid, []) if wrap else invalid

    with pytest.raises(TypeError, match=r"normalize_perf_query.*Mapping"):
        op.query_with_resolution(
            _RootDatabase(),
            session=session_factory(_Executor()),
            x=8,
        )

    assert invalid.measurement_calls == []
    assert invalid.curated_calls == []
    assert invalid.query_calls == []


@pytest.mark.parametrize("wrap", (False, True))
def test_normalized_dispatch_does_not_contaminate_legacy_kwargs_hooks(session_factory, wrap: bool) -> None:
    legacy = _LegacyKwargsOp("legacy")
    op = FallbackOp("wrapper", legacy, []) if wrap else legacy
    raw_query = {"x": 8, "marker": "unchanged"}

    op.query_with_resolution(
        _RootDatabase(),
        session=session_factory(_Executor()),
        **raw_query,
    )

    assert legacy.request_kwargs == [raw_query]
    assert legacy.exact_kwargs == [raw_query]
    assert all("normalized_query" not in kwargs for kwargs in legacy.request_kwargs + legacy.exact_kwargs)


@pytest.mark.parametrize("wrap", (False, True))
def test_normalized_dispatch_preserves_narrow_legacy_hook_signatures(session_factory, wrap: bool) -> None:
    legacy = _NarrowLegacyOp("narrow")
    op = FallbackOp("wrapper", legacy, []) if wrap else legacy

    result = op.query_with_resolution(
        _RootDatabase(),
        session=session_factory(_Executor()),
        x=8,
    )

    assert result is legacy.ordinary
    assert legacy.request_x == [8]
    assert legacy.exact_x == [8]


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda: FallbackOp("fallback", _BareOp("primary", 1.0), [_BareOp("child", 1.0)]),
        lambda: OverlapOp("overlap", [_BareOp("left", 1.0)], [_BareOp("right", 1.0)]),
    ],
)
def test_composite_no_session_delegates_to_ordinary_query_byte_for_byte(wrapper) -> None:
    database = _RootDatabase()
    expected = PerformanceResult(5.0, energy=6.0, source="empirical")
    op = wrapper()
    op.query = MagicMock(return_value=expected)

    result = op.query_with_resolution(database, session=None, x=13)

    assert result is expected
    op.query.assert_called_once_with(database, x=13)


def test_overlay_precedes_curated_and_scales_the_measurement(session_factory) -> None:
    protocol = _protocol()
    executor = _Executor()
    session = session_factory(executor, protocol)
    op = _TableOp(
        "scaled",
        scale_factor=2.5,
        curated={8: PerformanceResult(99.0, energy=99.0, source="curated_exact")},
    )
    request = _request("scaled", 8, protocol)
    session.overlay.append(_record(request, 0.1, 0.2))

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(0.5)
    assert result.source == "overlay"
    assert executor.request_batches == []
    assert op.curated_calls == []
    assert op.query_calls == []
    report = session.report.to_dict()
    assert report["consumer_counts"] == {request.key.digest: 1}
    assert report["callbacks"][0]["final_exact_sources"] == {request.key.digest: "overlay"}


def test_operation_binds_template_request_before_overlay_lookup_and_miss(session_factory) -> None:
    template = _protocol()
    bound_protocol = MeasurementProtocol(
        revision="route-owned-v1",
        warmups=template.warmups,
        samples=template.samples,
        statistic=template.statistic,
        timer="route-owned-timer",
        tuning_revision="route-owned-tuning-v1",
    )
    executor = _BindingExecutor(bound_protocol, {"bound": (0.25, 0.5)})
    session = session_factory(executor, template)
    op = _TableOp("bound")

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(0.5)
    assert result.source == "overlay"
    assert [request.protocol for request in executor.binding_inputs] == [template, template]
    assert len(executor.request_batches) == 1
    dispatched = executor.request_batches[0][0]
    assert dispatched.protocol == bound_protocol
    assert session.overlay.lookup(dispatched.key, bound_protocol) is not None
    assert session.overlay.lookup(dispatched.key, template) is None


def test_literal_curated_hit_survives_route_binding_failure(session_factory) -> None:
    template = _protocol()
    expected = PerformanceResult(0.4, energy=0.7, source="curated_exact")
    executor = _BindingExecutor(
        template,
        binding_error=RuntimeError("no lazy route for request"),
    )
    session = session_factory(executor, template)
    op = _TableOp("literal", curated={4: expected})

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=4))

    assert result is expected
    assert len(executor.binding_inputs) == 1
    assert executor.binding_inputs[0].protocol == template
    assert executor.request_batches == []
    assert len(op.curated_calls) == 1


def test_operation_classifies_route_protocol_binding_failure_as_identity_mismatch(session_factory) -> None:
    template = _protocol()
    executor = _BindingExecutor(
        template,
        binding_error=ProtocolMismatchError("request protocol does not match resolved lazy route"),
    )
    session = session_factory(executor, template)
    op = _TableOp("protocol_mismatch")

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.IDENTITY_MISMATCH]
    assert [reason.operation for reason in failure.value.reasons] == ["protocol_mismatch"]
    assert executor.request_batches == []


def test_literal_curated_result_is_final_and_never_rescaled(session_factory) -> None:
    expected = PerformanceResult(
        0.4,
        energy=0.7,
        source="curated_exact",
        provenance={
            "dataset": "gemm_perf.txt/v1",
            "row": 7,
            "source_revision": "curated-r3",
        },
    )
    executor = _Executor()
    session = session_factory(executor)
    op = _TableOp("literal", scale_factor=9.0, curated={4: expected})
    request = _request("literal", 4, session.protocol)

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=4))

    assert result is expected
    assert result.source == "curated_exact"
    assert executor.request_batches == []
    assert op.query_calls == []
    assert len(op.curated_calls) == 1
    report = session.report.to_dict()
    assert report["consumer_counts"] == {request.key.digest: 1}
    assert report["exact_source_counts"]["curated_exact"] == 1
    assert report["callbacks"][0]["final_exact_sources"] == {request.key.digest: "curated_exact"}
    expected_evidence = {
        "key_digest": request.key.digest,
        "source": "curated_exact",
        "latency_ms": 0.4,
        "energy_wms": 0.7,
        "provenance": {
            "dataset": "gemm_perf.txt/v1",
            "row": 7,
            "source_revision": "curated-r3",
        },
    }
    assert report["evidence_links"] == [expected_evidence]
    assert report["callbacks"][0]["final_evidence"] == [expected_evidence]


def test_literal_curated_result_ignores_non_json_provenance_without_changing_query(session_factory) -> None:
    expected = PerformanceResult(
        0.4,
        energy=0.7,
        source="curated_exact",
        provenance={"opaque": object()},
    )
    session = session_factory(_Executor())
    op = _TableOp("literal", curated={4: expected})

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=4))

    assert result is expected
    assert session.report.to_dict()["callbacks"][0]["final_evidence"][0]["provenance"] == {
        "unavailable": "not_json_safe"
    }


def test_off_grid_shape_collects_exact_evidence_instead_of_interpolating(session_factory) -> None:
    executor = _Executor({"off_grid": (0.25, 0.5)})
    session = session_factory(executor)
    op = _TableOp(
        "off_grid",
        ordinary=PerformanceResult(8.0, energy=80.0, source="silicon"),
        curated={4: PerformanceResult(0.1, energy=0.2, source="curated_exact")},
    )
    database = _RootDatabase()

    result = session.execute_callback(lambda: op.query_with_resolution(database, session=session, x=8))

    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(0.5)
    assert len(op.query_calls) == 1
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["off_grid"]]
    assert all(db._default_database_mode is common.DatabaseMode.SILICON for db, _ in op.measurement_calls)


def test_default_hooks_report_missing_adapter_without_collection(session_factory) -> None:
    executor = _Executor()
    session = session_factory(executor)
    op = _BareOp("bare", 1.0)

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=3))

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.MISSING_ADAPTER]
    assert [reason.operation for reason in failure.value.reasons] == ["bare"]
    assert executor.request_batches == []


def test_direct_miss_returns_nonzero_ordinary_result_as_tainted_discovery_surrogate(session_factory) -> None:
    executor = _Executor()
    session = session_factory(executor)
    provisional = PerformanceResult(9.0, energy=90.0, source="empirical")
    op = _TableOp("pending", ordinary=provisional)
    database = _RootDatabase()
    checkpoint = session.checkpoint()

    result = op.query_with_resolution(database, session=session, x=17)

    assert result is provisional
    assert session.changed_since(checkpoint)
    assert op.query_calls == [(database, {"x": 17})]
    assert executor.request_batches == []


def test_missing_silicon_provisional_does_not_abort_collection_before_replay(session_factory) -> None:
    executor = _Executor({"measured_only": (0.75, 7.5)})
    session = session_factory(executor)
    op = _MissingSiliconTableOp("measured_only")

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=17))

    assert float(result) == pytest.approx(0.75)
    assert result.source == "overlay"
    assert len(op.query_calls) == 1
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["measured_only"]]


def test_overridden_missing_silicon_provisional_does_not_abort_collection_before_replay(
    session_factory,
) -> None:
    executor = _Executor({"measured_only_override": (0.75, 7.5)})
    session = session_factory(executor)
    op = _MissingSiliconOverriddenProvisionalTableOp("measured_only_override")

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=17))

    assert float(result) == pytest.approx(0.75)
    assert result.source == "overlay"
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["measured_only_override"]]


def test_missing_silicon_without_adapter_fails_structured_instead_of_leaking_query_error(session_factory) -> None:
    executor = _Executor()
    session = session_factory(executor)
    op = _MissingSiliconNoAdapterOp("missing", 1.0)

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=17))

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.MISSING_ADAPTER]
    assert executor.request_batches == []


def test_fallback_adapter_capable_primary_blocks_fallback_discovery(session_factory) -> None:
    executor = _Executor({"primary": (0.3, 3.0), "fallback_child": (7.0, 70.0)})
    session = session_factory(executor)
    primary = _TableOp("primary")
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.3)
    assert result.energy == pytest.approx(3.0)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["primary"]]
    assert fallback_child.measurement_calls == []
    assert len(primary.query_calls) == 1


def test_fallback_measured_primary_reuses_one_normalized_mapping_without_recursing(session_factory) -> None:
    primary = _NormalizedTableOp("primary")
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = op.query_with_resolution(
        _RootDatabase(),
        session=session_factory(_Executor()),
        tokens=3,
        token_multiplier=4,
    )

    assert result is primary.ordinary
    assert len(primary.normalization_calls) == 1
    assert primary.request_normalized[0] is primary.curated_normalized[0]
    assert primary.request_normalized[0] is primary.provisional_normalized[0]
    assert primary.request_normalized[0] == {"x": 12}
    assert fallback_child.measurement_calls == []
    assert fallback_child.query_calls == []


def test_fallback_measured_primary_missing_silicon_collects_boundary_without_entering_fallback(session_factory) -> None:
    executor = _Executor({"primary": (0.75, 7.5), "fallback_child": (9.0, 90.0)})
    session = session_factory(executor)
    primary = _MissingSiliconTableOp("primary")
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=17))

    assert float(result) == pytest.approx(0.75)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["primary"]]
    assert len(primary.query_calls) == 1
    assert fallback_child.measurement_calls == []
    assert fallback_child.query_calls == []


def test_fallback_resolution_aware_composite_primary_discovers_its_children(session_factory) -> None:
    executor = _Executor({"left": (1.0, 10.0), "right": (2.0, 20.0), "fallback_child": (7.0, 70.0)})
    session = session_factory(executor)
    left = _TableOp("left")
    right = _TableOp("right")
    primary = OverlapOp("primary_overlap", [left], [right])
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(2.0)
    assert result.energy == pytest.approx(30.0)
    assert result.source == "overlay"
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["left", "right"]]
    assert fallback_child.measurement_calls == []


def test_measured_compound_emits_only_its_boundary_key_and_never_traverses_children(session_factory) -> None:
    executor = _Executor({"module_boundary": (0.75, 7.5), "implementation_child": (9.0, 90.0)})
    session = session_factory(executor)
    implementation_child = _TableOp("implementation_child")
    measured_compound = _TableOp("module_boundary", ordinary=PerformanceResult(4.0, source="empirical"))
    measured_compound.implementation_children = (implementation_child,)

    result = session.execute_callback(
        lambda: measured_compound.query_with_resolution(_RootDatabase(), session=session, x=8)
    )

    assert float(result) == pytest.approx(0.75)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["module_boundary"]]
    assert implementation_child.measurement_calls == []
    assert implementation_child.query_calls == []


@pytest.mark.parametrize(("shape", "selected_name"), ((8, "small_shape"), (32, "large_shape")))
def test_compound_branch_selection_is_shape_stable_across_provisional_replay(
    session_factory,
    shape: int,
    selected_name: str,
) -> None:
    executor = _Executor({selected_name: (0.5, 5.0)})
    session = session_factory(executor)
    children = {
        "small_shape": _TableOp("small_shape", ordinary=PerformanceResult(100.0, source="empirical")),
        "large_shape": _TableOp("large_shape", ordinary=PerformanceResult(0.001, source="empirical")),
    }
    selections: list[str] = []

    def query_selected_child() -> PerformanceResult:
        name = "small_shape" if shape <= 8 else "large_shape"
        selections.append(name)
        return children[name].query_with_resolution(_RootDatabase(), session=session, x=shape)

    result = session.execute_callback(query_selected_child)

    assert float(result) == pytest.approx(0.5)
    assert selections == [selected_name, selected_name]
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [[selected_name]]


def test_fallback_does_not_treat_instrumented_leaf_override_as_composite_owner(session_factory) -> None:
    executor = _Executor({"fallback_child": (0.6, 6.0)})
    session = session_factory(executor)
    primary = _InstrumentedBareOp("instrumented_leaf")
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.6)
    assert result.energy == pytest.approx(6.0)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["fallback_child"]]
    assert primary.resolution_calls == 0
    assert session.report.unresolved == []


def test_fallback_composite_primary_missing_descendant_fails_without_switching_implementation(session_factory) -> None:
    executor = _Executor({"left": (1.0, 10.0), "fallback_child": (7.0, 70.0)})
    session = session_factory(executor)
    left = _TableOp("left")
    missing = _BareOp("missing_descendant", 1.0)
    primary = OverlapOp("primary_overlap", [left], [missing])
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    with pytest.raises(ResolutionFailed) as failure:
        session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.MISSING_ADAPTER]
    assert [reason.operation for reason in failure.value.reasons] == ["missing_descendant"]
    assert executor.request_batches == []
    assert fallback_child.measurement_calls == []


def test_fallback_nested_fallback_primary_owns_resolution(session_factory) -> None:
    executor = _Executor({"inner_primary": (0.4, 4.0)})
    session = session_factory(executor)
    inner_primary = _TableOp("inner_primary")
    inner_fallback = _TableOp("inner_fallback")
    primary = FallbackOp("nested", inner_primary, [inner_fallback])
    outer_fallback = _TableOp("outer_fallback")
    op = FallbackOp("wrapper", primary, [outer_fallback])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.4)
    assert result.energy == pytest.approx(4.0)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["inner_primary"]]
    assert inner_fallback.measurement_calls == []
    assert outer_fallback.measurement_calls == []


def test_fallback_primary_overlay_precedes_curated_and_applies_primary_scale(session_factory) -> None:
    protocol = _protocol()
    executor = _Executor()
    session = session_factory(executor, protocol)
    primary = _TableOp(
        "primary",
        scale_factor=2.0,
        curated={8: PerformanceResult(99.0, energy=99.0, source="curated_exact")},
    )
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])
    session.overlay.append(_record(_request("primary", 8, protocol), 0.2, 0.3))

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.4)
    assert result.energy == pytest.approx(0.6)
    assert result.source == "overlay"
    assert executor.request_batches == []
    assert primary.curated_calls == []
    assert fallback_child.measurement_calls == []


def test_fallback_adapter_capable_primary_literal_precedes_collection_and_children(session_factory) -> None:
    expected = PerformanceResult(0.2, energy=2.0, source="curated_exact")
    executor = _Executor()
    session = session_factory(executor)
    primary = _TableOp("primary", curated={8: expected})
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert result is expected
    assert executor.request_batches == []
    assert fallback_child.measurement_calls == []


def test_fallback_primary_without_adapter_recurses_without_missing_adapter(session_factory) -> None:
    executor = _Executor({"fallback_child": (0.6, 6.0)})
    session = session_factory(executor)
    primary = _TableOp("primary", adapter=False)
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.6)
    assert result.energy == pytest.approx(6.0)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["fallback_child"]]
    assert session.report.unresolved == []
    assert primary.query_calls == []
    assert not hasattr(op, "_primary_unavailable")


def test_fallback_primary_literal_curated_result_precedes_children(session_factory) -> None:
    expected = PerformanceResult(0.2, energy=2.0, source="curated_exact")
    executor = _Executor()
    session = session_factory(executor)
    primary = _TableOp("primary", adapter=False, curated={8: expected})
    fallback_child = _TableOp("fallback_child")
    op = FallbackOp("wrapper", primary, [fallback_child])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert result is expected
    assert executor.request_batches == []
    assert fallback_child.measurement_calls == []


def test_overlap_discovers_both_groups_in_one_callback_and_preserves_math(session_factory) -> None:
    executor = _Executor({"left": (1.0, 10.0), "right": (2.0, 20.0)})
    session = session_factory(executor)
    left = _TableOp("left")
    right = _TableOp("right")
    op = OverlapOp("parallel", [left], [right])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(2.0)
    assert result.energy == pytest.approx(30.0)
    assert result.source == "overlay"
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["left", "right"]]


def test_overlap_replays_duplicate_child_key_at_every_consumer_position(session_factory) -> None:
    executor = _Executor({"shared": (1.5, 2.0)})
    session = session_factory(executor)
    first = _TableOp("shared")
    second = _TableOp("shared")
    op = OverlapOp("parallel", [first, second], [])

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(3.0)
    assert result.energy == pytest.approx(4.0)
    assert [[request.op_id for request in batch] for batch in executor.request_batches] == [["shared"]]
    assert session.report.unique_misses == 1
    assert session.report.consumer_misses == 2


def test_typed_timeout_publishes_only_failed_keys_unscaled_hybrid_and_replays_once(
    tmp_path,
    caplog,
) -> None:
    caplog.set_level(logging.INFO, logger="aiconfigurator.sdk.resolution.session")
    protocol = _protocol()
    executor = _TypedFailureExecutor("b", UnresolvedCode.TIMEOUT)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    exact = _HybridTrackingTableOp("a", ordinary=PerformanceResult(100.0, source="empirical"))
    degraded = _HybridTrackingTableOp(
        "b",
        scale_factor=2.0,
        ordinary=PerformanceResult(8.0, source="empirical"),
    )
    database = _RootDatabase()
    calls = 0

    def query() -> PerformanceResult:
        nonlocal calls
        calls += 1
        return exact.query_with_resolution(database, session=session, x=8) + degraded.query_with_resolution(
            database,
            session=session,
            x=16,
        )

    try:
        result = coordinator.execute_callback(query)
    finally:
        coordinator.close()

    request_a = _request("a", 8, protocol)
    request_b = _request("b", 16, protocol)
    persisted_fallback = fallback_store.lookup(
        request_b.key,
        prediction_revision="prediction-r1",
    )
    assert float(result) == pytest.approx(9.25)
    assert result.source == "mixed"
    assert calls == 2
    assert exact.hybrid_resolver_calls == 0
    assert degraded.hybrid_resolver_calls == 1
    assert _reopened_lookup(overlay.path, request_a.key, protocol) is not None
    assert _reopened_lookup(overlay.path, request_b.key, protocol) is None
    assert persisted_fallback is not None
    assert persisted_fallback.latency_ms == pytest.approx(4.0)
    assert persisted_fallback.hybrid_source == "empirical"
    assert persisted_fallback.measurement_failure.code is UnresolvedCode.TIMEOUT
    assert session.report.to_dict()["hybrid_fallbacks"] == [
        {
            "key_digest": request_b.key.digest,
            "latency_ms": 4.0,
            "latency_units": "ms",
            "path": str(persisted_fallback.path),
            "hybrid_provenance": {
                "source": "empirical",
                "prediction_revision": "prediction-r1",
                "metadata": {},
            },
            "measurement_failure": {
                "code": "timeout",
                "operation": "b",
                "detail": "injected timeout",
            },
        }
    ]
    assert len(tuple(fallback_store.directory.glob("*.json"))) == 1
    assert executor.request_batches == [(_request("a", 8, protocol), _request("b", 16, protocol))]
    publications = [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    assert len(publications) == 1
    assert publications[0].levelno == logging.WARNING
    assert publications[0].key_digest == request_b.key.digest
    assert publications[0].failure_code == "timeout"
    assert publications[0].hybrid_source == "empirical"
    assert publications[0].sidecar_path == str(persisted_fallback.path)


def test_sidecar_warm_reopen_exact_precedence_and_force_remeasure(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="aiconfigurator.sdk.resolution.session")
    protocol = _protocol()
    request = _request("warm", 8, protocol)
    fallback_directory = tmp_path / "fallback"
    store = FallbackStore(fallback_directory)
    sidecar = store.publish(
        request.key,
        prediction_revision="prediction-r1",
        latency_ms=4.0,
        hybrid_source="empirical",
        measurement_failure=UnresolvedReason(
            UnresolvedCode.TIMEOUT,
            request.op_id,
            "injected timeout",
            key=request.key,
            failure_kind=MeasurementFailureKind.OPERATIONAL,
        ),
    )

    def run(
        *,
        overlay: OverlayStore,
        executor: _Executor,
        operation: _TableOp,
        reopened_store: FallbackStore,
        force_remeasure: bool = False,
    ):
        session = ResolutionSession(
            overlay,
            executor,
            ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
            protocol,
        )
        coordinator = OnlineResolutionCoordinator(
            session,
            fallback_store=reopened_store,
            prediction_revision="prediction-r1",
            on_measurement_failure="hybrid",
            force_remeasure=force_remeasure,
        )
        try:
            result = coordinator.execute_callback(
                lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8)
            )
        finally:
            coordinator.close()
        return result, session

    warm_executor = _Executor()
    warm_result, warm_session = run(
        overlay=OverlayStore(tmp_path / "warm-overlay.sqlite"),
        executor=warm_executor,
        operation=_TableOp("warm", scale_factor=2.0),
        reopened_store=store,
    )
    assert float(warm_result) == pytest.approx(8.0)
    assert warm_executor.request_batches == []
    assert warm_session.report.fallback_hits == 1

    reopen_executor = _Executor()
    reopen_result, reopen_session = run(
        overlay=OverlayStore(tmp_path / "reopen-overlay.sqlite"),
        executor=reopen_executor,
        operation=_TableOp("warm", scale_factor=2.0),
        reopened_store=FallbackStore(fallback_directory),
    )
    assert float(reopen_result) == pytest.approx(8.0)
    assert reopen_executor.request_batches == []
    assert reopen_session.report.fallback_hits == 1

    exact_overlay = OverlayStore(tmp_path / "exact-overlay.sqlite")
    exact_overlay.append(_record(request, 1.5))
    exact_executor = _Executor()
    exact_result, exact_session = run(
        overlay=exact_overlay,
        executor=exact_executor,
        operation=_TableOp("warm", scale_factor=2.0),
        reopened_store=FallbackStore(fallback_directory),
    )
    assert float(exact_result) == pytest.approx(3.0)
    assert exact_executor.request_batches == []
    assert exact_session.report.fallback_hits == 0

    curated_executor = _Executor()
    curated_result, curated_session = run(
        overlay=OverlayStore(tmp_path / "curated-overlay.sqlite"),
        executor=curated_executor,
        operation=_TableOp(
            "warm",
            scale_factor=2.0,
            curated={8: PerformanceResult(3.0, source="curated_exact")},
        ),
        reopened_store=FallbackStore(fallback_directory),
    )
    assert float(curated_result) == pytest.approx(3.0)
    assert curated_executor.request_batches == []
    assert curated_session.report.fallback_hits == 0

    force_executor = _Executor({"warm": (1.25, 0.0)})
    force_overlay = OverlayStore(tmp_path / "force-overlay.sqlite")
    force_result, force_session = run(
        overlay=force_overlay,
        executor=force_executor,
        operation=_TableOp("warm", scale_factor=2.0),
        reopened_store=FallbackStore(fallback_directory),
        force_remeasure=True,
    )
    assert float(force_result) == pytest.approx(2.5)
    assert force_executor.request_batches == [(request,)]
    assert force_session.report.fallback_hits == 0
    assert _reopened_lookup(force_overlay.path, request.key, protocol) is not None
    assert sidecar.path.exists()
    hits = [record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_hit"]
    assert len(hits) == 2
    assert all(record.levelno == logging.INFO for record in hits)
    assert all(record.key_digest == request.key.digest for record in hits)


def test_existing_first_writer_does_not_emit_hybrid_publication_warning(tmp_path, caplog) -> None:
    protocol = _protocol()
    request = _request("race", 8, protocol)
    fallback_store = FallbackStore(tmp_path / "fallback")
    winner = fallback_store.publish(
        request.key,
        prediction_revision="prediction-r1",
        latency_ms=4.0,
        hybrid_source="empirical",
        measurement_failure=UnresolvedReason(
            UnresolvedCode.TIMEOUT,
            request.op_id,
            "first writer",
            key=request.key,
            failure_kind=MeasurementFailureKind.OPERATIONAL,
        ),
    )
    caplog.set_level(logging.INFO, logger="aiconfigurator.sdk.resolution.session")
    executor = _TypedFailureExecutor("race", UnresolvedCode.TIMEOUT)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
        force_remeasure=True,
    )
    operation = _HybridTrackingTableOp(
        "race",
        ordinary=PerformanceResult(6.0, source="empirical"),
    )

    try:
        result = coordinator.execute_callback(
            lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8)
        )
    finally:
        coordinator.close()

    assert float(result) == pytest.approx(4.0)
    persisted_winner = fallback_store.lookup(request.key, prediction_revision="prediction-r1")
    assert persisted_winner is not None
    assert persisted_winner.path == winner.path
    assert persisted_winner.latency_ms == pytest.approx(winner.latency_ms)
    payload = session.report.to_dict()
    assert payload["hybrid_publications"] == 0
    assert payload["fallback_hits"] == 1
    assert payload["hybrid_fallbacks"][0]["latency_ms"] == pytest.approx(4.0)
    assert payload["hybrid_fallbacks"][0]["measurement_failure"]["detail"] == "first writer"
    assert not [
        record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_published"
    ]
    reuses = [record for record in caplog.records if getattr(record, "event", None) == "aic_hybrid_fallback_reused"]
    assert len(reuses) == 1
    assert reuses[0].levelno == logging.INFO
    assert reuses[0].sidecar_path == str(winner.path)


def test_callback_block_timeout_is_operational_and_activates_hybrid(tmp_path) -> None:
    protocol = _protocol()
    entered = threading.Event()
    release = threading.Event()
    executor = _BlockingExactExecutor(entered, release)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")

    def expire_callback_deadline(futures, *, deadline_monotonic):
        del deadline_monotonic
        assert futures
        assert entered.wait(timeout=5.0)
        return set()

    coordinator = OnlineResolutionCoordinator(
        session,
        max_block_seconds=1.0,
        wait_for_futures=expire_callback_deadline,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    operation = _HybridTrackingTableOp(
        "deadline",
        ordinary=PerformanceResult(4.0, source="empirical"),
    )

    try:
        result = coordinator.execute_callback(
            lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8)
        )
    finally:
        release.set()
        coordinator.close()

    request = _request("deadline", 8, protocol)
    persisted = fallback_store.lookup(request.key, prediction_revision="prediction-r1")
    assert float(result) == pytest.approx(4.0)
    assert operation.hybrid_resolver_calls == 1
    assert persisted is not None
    assert persisted.measurement_failure.code is UnresolvedCode.TIMEOUT
    assert session.report.callbacks[0]["collection"]["deadline_source"] == "callback_block"


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (UnresolvedCode.MISSING_ADAPTER, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.UNSUPPORTED_SHAPE, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.RESOURCE_UNAVAILABLE, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.COLLECTOR_FAILED, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.TIMEOUT, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.INVALID_MEASUREMENT, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.BUDGET_EXHAUSTED, MeasurementFailureKind.BUDGET),
        (UnresolvedCode.RETRY_EXHAUSTED, MeasurementFailureKind.OPERATIONAL),
        (UnresolvedCode.REQUERY_STILL_MISSING, MeasurementFailureKind.OPERATIONAL),
    ],
)
def test_typed_eligible_measurement_failures_activate_hybrid(tmp_path, code, kind) -> None:
    protocol = _protocol()
    executor = _TypedFailureExecutor("eligible", code, kind=kind)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    operation = _HybridTrackingTableOp(
        "eligible",
        ordinary=PerformanceResult(4.0, source="empirical"),
    )

    try:
        result = coordinator.execute_callback(
            lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8)
        )
    finally:
        coordinator.close()

    request = _request("eligible", 8, protocol)
    persisted = fallback_store.lookup(request.key, prediction_revision="prediction-r1")
    assert float(result) == pytest.approx(4.0)
    assert operation.hybrid_resolver_calls == 1
    assert persisted is not None
    assert persisted.measurement_failure.code is code


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (UnresolvedCode.IDENTITY_MISMATCH, MeasurementFailureKind.INVARIANT),
        (UnresolvedCode.TOPOLOGY_MISMATCH, MeasurementFailureKind.INVARIANT),
        (UnresolvedCode.CANCELLED, MeasurementFailureKind.CANCELLATION),
        (UnresolvedCode.OBSERVE_ONLY, MeasurementFailureKind.OBSERVATION),
        (UnresolvedCode.COLLECTOR_FAILED, MeasurementFailureKind.INVARIANT),
        (UnresolvedCode.INVALID_MEASUREMENT, MeasurementFailureKind.INVARIANT),
        (UnresolvedCode.TIMEOUT, MeasurementFailureKind.CANCELLATION),
        (UnresolvedCode.MISSING_ADAPTER, MeasurementFailureKind.INVARIANT),
    ],
)
def test_structural_and_non_measurement_failures_never_activate_hybrid(tmp_path, code, kind) -> None:
    protocol = _protocol()
    executor = _TypedFailureExecutor("ineligible", code, kind=kind)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    operation = _HybridTrackingTableOp(
        "ineligible",
        ordinary=PerformanceResult(4.0, source="empirical"),
    )

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8))
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [code]
    assert operation.hybrid_resolver_calls == 0
    assert not fallback_store.directory.exists()


def test_missing_adapter_without_validated_key_never_activates_hybrid(tmp_path) -> None:
    protocol = _protocol()
    executor = _Executor()
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    operation = _BareOp("missing", 4.0)

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8))
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.MISSING_ADAPTER]
    assert failure.value.reasons[0].key is None
    assert executor.request_batches == []
    assert not fallback_store.directory.exists()


@pytest.mark.parametrize("failure_mode", ["raises", "non_finite", "missing_source"])
def test_hybrid_query_failure_is_structured_and_never_publishes_sidecar(
    tmp_path,
    failure_mode,
) -> None:
    protocol = _protocol()
    executor = _TypedFailureExecutor("broken", UnresolvedCode.TIMEOUT)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    if failure_mode == "raises":
        operation = _RaisingHybridOp("broken")
    else:
        ordinary = PerformanceResult(
            float("nan") if failure_mode == "non_finite" else 4.0,
            source="empirical" if failure_mode == "non_finite" else "",
        )
        operation = _HybridTrackingTableOp("broken", ordinary=ordinary)

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8))
    finally:
        coordinator.close()

    request = _request("broken", 8, protocol)
    assert [reason.code for reason in failure.value.reasons] == [
        UnresolvedCode.TIMEOUT,
        UnresolvedCode.INVALID_MEASUREMENT,
    ]
    assert [reason.key for reason in failure.value.reasons] == [request.key, request.key]
    assert [reason.failure_kind for reason in failure.value.reasons] == [
        MeasurementFailureKind.OPERATIONAL,
        MeasurementFailureKind.INVARIANT,
    ]
    assert "HYBRID fallback failed after timeout" in failure.value.reasons[1].detail
    assert operation.hybrid_resolver_calls == 1
    assert not fallback_store.directory.exists()


def test_same_operation_two_shapes_route_hybrid_failures_by_physical_key(tmp_path) -> None:
    protocol = _protocol()
    executor = _TypedFailureExecutor("shared", UnresolvedCode.TIMEOUT)
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    first = _HybridTrackingTableOp("shared", ordinary=PerformanceResult(4.0, source="empirical"))
    second = _HybridTrackingTableOp("shared", ordinary=PerformanceResult(6.0, source="empirical"))
    database = _RootDatabase()

    try:
        result = coordinator.execute_callback(
            lambda: (
                first.query_with_resolution(database, session=session, x=8)
                + second.query_with_resolution(database, session=session, x=16)
            )
        )
    finally:
        coordinator.close()

    first_request = _request("shared", 8, protocol)
    second_request = _request("shared", 16, protocol)
    assert float(result) == pytest.approx(10.0)
    assert first.hybrid_resolver_calls == second.hybrid_resolver_calls == 1
    assert fallback_store.lookup(first_request.key, prediction_revision="prediction-r1") is not None
    assert fallback_store.lookup(second_request.key, prediction_revision="prediction-r1") is not None
    assert len(tuple(fallback_store.directory.glob("*.json"))) == 2


def test_duplicate_executor_records_are_invariant_failure_not_hybrid(tmp_path) -> None:
    protocol = _protocol()
    executor = _DuplicateRecordExecutor()
    session = ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        protocol,
    )
    fallback_store = FallbackStore(tmp_path / "fallback")
    coordinator = OnlineResolutionCoordinator(
        session,
        fallback_store=fallback_store,
        prediction_revision="prediction-r1",
        on_measurement_failure="hybrid",
    )
    operation = _HybridTrackingTableOp("malformed", ordinary=PerformanceResult(4.0, source="empirical"))

    try:
        with pytest.raises(ResolutionFailed) as failure:
            coordinator.execute_callback(lambda: operation.query_with_resolution(_RootDatabase(), session=session, x=8))
    finally:
        coordinator.close()

    assert [reason.code for reason in failure.value.reasons] == [UnresolvedCode.INVALID_MEASUREMENT]
    assert operation.hybrid_resolver_calls == 0
    assert not fallback_store.directory.exists()


def test_overlap_direct_first_walk_composes_nonzero_provisional_descendant(session_factory) -> None:
    executor = _Executor()
    session = session_factory(executor)
    left = _TableOp(
        "left",
        curated={8: PerformanceResult(0.4, energy=4.0, source="curated_exact")},
    )
    right = _TableOp("right")
    op = OverlapOp("parallel", [left], [right])
    checkpoint = session.checkpoint()

    result = op.query_with_resolution(_RootDatabase(), session=session, x=8)

    assert float(result) == pytest.approx(9.0)
    assert result.energy == pytest.approx(94.0)
    assert result.source == "mixed"
    assert session.changed_since(checkpoint)
    assert executor.request_batches == []
    assert left.query_calls == []
    assert len(right.query_calls) == 1


def test_hybrid_resolution_uses_shared_silicon_view_without_mutating_root(session_factory) -> None:
    expected = PerformanceResult(0.8, energy=1.2, source="curated_exact")
    executor = _Executor()
    root = _RootDatabase()
    root_cache = root._extracted_metrics_cache
    root_cache_contents = dict(root_cache)
    root_support = root.supported_quant_mode
    root_transfer_policy = root.transfer_policy
    op = _TableOp("literal", curated={5: expected})
    sessions = [session_factory(executor), session_factory(executor)]

    def run(session: ResolutionSession) -> PerformanceResult:
        return session.execute_callback(lambda: op.query_with_resolution(root, session=session, x=5))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(run, sessions))

    configured = [database for database, _ in op.measurement_calls]
    assert results == (expected, expected)
    assert len(configured) == 2
    assert all(database is not root for database in configured)
    assert all(database._root_database_template is root for database in configured)
    assert all(database._default_database_mode is common.DatabaseMode.SILICON for database in configured)
    assert all(database._extracted_metrics_cache == {} for database in configured)
    assert all(database._extracted_metrics_cache is not root_cache for database in configured)
    assert root._default_database_mode is common.DatabaseMode.HYBRID
    assert root._extracted_metrics_cache is root_cache
    assert root._extracted_metrics_cache == root_cache_contents
    assert root.supported_quant_mode is root_support
    assert root.transfer_policy is root_transfer_policy
    assert executor.request_batches == []
