# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.operations.overlap import FallbackOp, OverlapOp
from aiconfigurator.sdk.perf_database import _cached_configured_database_view
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionFailed, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    UnresolvedCode,
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
        key=PerfKey.build("test_operation/v1", query, environment, semantic),
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
    session.overlay.append(_record(_request("scaled", 8, protocol), 0.1, 0.2))

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=8))

    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(0.5)
    assert result.source == "overlay"
    assert executor.request_batches == []
    assert op.curated_calls == []
    assert op.query_calls == []


def test_literal_curated_result_is_final_and_never_rescaled(session_factory) -> None:
    expected = PerformanceResult(0.4, energy=0.7, source="curated_exact")
    executor = _Executor()
    session = session_factory(executor)
    op = _TableOp("literal", scale_factor=9.0, curated={4: expected})

    result = session.execute_callback(lambda: op.query_with_resolution(_RootDatabase(), session=session, x=4))

    assert result is expected
    assert result.source == "curated_exact"
    assert executor.request_batches == []
    assert op.query_calls == []
    assert len(op.curated_calls) == 1


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
    assert op.query_calls == []
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


def test_direct_miss_returns_an_explicit_unresolved_sentinel(session_factory) -> None:
    executor = _Executor()
    session = session_factory(executor)
    op = _TableOp("pending")
    checkpoint = session.checkpoint()

    result = op.query_with_resolution(_RootDatabase(), session=session, x=17)

    assert float(result) == 0.0
    assert result.energy == 0.0
    assert result.source == "unresolved"
    assert session.changed_since(checkpoint)
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
    assert primary.query_calls == []


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


def test_overlap_direct_first_walk_preserves_unresolved_descendant_source(session_factory) -> None:
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

    assert float(result) == pytest.approx(0.4)
    assert result.energy == pytest.approx(4.0)
    assert result.source == "unresolved"
    assert session.changed_since(checkpoint)
    assert executor.request_batches == []
    assert left.query_calls == []
    assert right.query_calls == []


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
