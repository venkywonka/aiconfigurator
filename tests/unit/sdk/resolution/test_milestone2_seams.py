from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import dsv4 as dsv4_ops
from aiconfigurator.sdk.operations.communication import P2P, CustomAllReduce
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)
from aiconfigurator.sdk.operations.elementwise import ElementWise
from aiconfigurator.sdk.operations.embedding import Embedding
from aiconfigurator.sdk.operations.moe import MoEDispatch
from aiconfigurator.sdk.perf_database import PerformanceResult
from aiconfigurator.sdk.resolution import MeasurementEnvironment, MeasurementProtocol, PerfKey
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import (
    ResolutionBudget,
    ResolutionFailed,
    ResolutionSession,
)
from aiconfigurator.sdk.resolution.types import UnresolvedCode

pytestmark = pytest.mark.unit


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="milestone2-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="exact-point-v1",
    )


def _environment(**overrides: object) -> MeasurementEnvironment:
    values: dict[str, object] = {
        "system": "gb200",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "gpu_class": "gb200",
        "runtime_versions": {"cuda": "12.9", "sglang": "0.5.10"},
    }
    values.update(overrides)
    return MeasurementEnvironment(**values)


def test_profile_compatibility_is_optional_physical_environment_identity() -> None:
    generic = _environment()
    flash = _environment(
        profile_compatibility={
            "model": "sgl-project/DeepSeek-V4-Flash-FP8",
            "module_schema": "dsv4-attention-v1",
        }
    )
    pro = _environment(
        profile_compatibility={
            "model": "sgl-project/DeepSeek-V4-Pro-FP8",
            "module_schema": "dsv4-attention-v1",
        }
    )

    assert generic.profile_compatibility is None
    assert flash.profile_compatibility == {
        "model": "sgl-project/DeepSeek-V4-Flash-FP8",
        "module_schema": "dsv4-attention-v1",
    }
    assert PerfKey.build("gemm_perf.txt/v1", {"m": 8}, generic) != PerfKey.build(
        "dsv4_csa_context_module_perf.txt/v1", {"b": 1}, flash
    )
    assert PerfKey.build("dsv4_csa_context_module_perf.txt/v1", {"b": 1}, flash) != PerfKey.build(
        "dsv4_csa_context_module_perf.txt/v1", {"b": 1}, pro
    )


def test_profile_compatibility_is_snapshotted_and_immutable() -> None:
    compatibility = {
        "model": "sgl-project/DeepSeek-V4-Flash-FP8",
        "artifacts": {"config": "sha256:abc"},
    }
    environment = _environment(profile_compatibility=compatibility)
    canonical = environment.canonical

    compatibility["model"] = "mutated"
    compatibility["artifacts"]["config"] = "sha256:def"

    assert environment.canonical == canonical
    assert isinstance(environment.profile_compatibility, Mapping)
    with pytest.raises(TypeError):
        environment.profile_compatibility["model"] = "mutated"


class _DeterministicDatabase:
    def __init__(self) -> None:
        self.mem_queries: list[int] = []

    def query_mem_op(self, size: int) -> PerformanceResult:
        self.mem_queries.append(size)
        return PerformanceResult(2.5, energy=5.0, source="empirical")


class _FailIfResolutionTouched:
    protocol = _protocol()

    def __getattr__(self, name: str):
        raise AssertionError(f"deterministic operation touched resolution session via {name}")


@pytest.mark.parametrize(
    ("operation", "kwargs", "expected_latency", "expected_mem_queries"),
    [
        (Embedding("embedding", 1.0, row_size=128, column_size=16), {"x": 4}, 2.5, [128]),
        (ElementWise("elementwise", 1.0, dim_in=16, dim_out=32), {"x": 4}, 2.5, [384]),
        (P2P("pp1", 1.0, h=16, pp_size=1), {"x": 4}, 0.0, []),
    ],
)
def test_reviewed_deterministic_operations_bypass_resolution_session(
    monkeypatch,
    operation,
    kwargs: dict[str, int],
    expected_latency: float,
    expected_mem_queries: list[int],
) -> None:
    from aiconfigurator.sdk import perf_database

    database = _DeterministicDatabase()
    monkeypatch.setattr(perf_database, "_get_configured_database_view", lambda database, *_args: database)

    assert operation.is_resolution_deterministic(**kwargs)
    result = operation.query_with_resolution(
        database,
        session=_FailIfResolutionTouched(),
        **kwargs,
    )

    assert float(result) == expected_latency
    assert database.mem_queries == expected_mem_queries


def test_nontrivial_p2p_is_not_silently_classified_as_deterministic() -> None:
    assert not P2P("pp2", 1.0, h=16, pp_size=2).is_resolution_deterministic(x=4)


class _MoEDatabase:
    backend = common.BackendName.sglang.value
    system_spec: ClassVar[dict[str, dict[str, int]]] = {
        "gpu": {"sm_version": 100},
        "node": {"num_gpus_per_node": 4},
    }

    def __init__(self) -> None:
        self.custom_allreduce_queries: list[tuple[common.CommQuantMode, int, int]] = []

    def query_custom_allreduce(
        self,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,
    ) -> PerformanceResult:
        self.custom_allreduce_queries.append((quant_mode, tp_size, size))
        return PerformanceResult(1.25, energy=2.5, source="silicon")


def _moe_dispatch(name: str, *, pre_dispatch: bool) -> MoEDispatch:
    return MoEDispatch(
        name=name,
        scale_factor=1.0,
        hidden_size=4096,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=4,
        attention_dp_size=1,
        pre_dispatch=pre_dispatch,
        moe_backend=None,
        is_context=True,
        quant_mode=common.MoEQuantMode.fp8_block,
    )


def test_moe_dispatch_traces_one_custom_allreduce_physical_point_for_two_consumers(monkeypatch) -> None:
    calls: list[tuple[str, int, int, int, object]] = []
    session = _FailIfResolutionTouched()

    def query_with_resolution(self, database, *, session=None, **kwargs):
        del database
        calls.append((self._name, self._tp_size, self._h, kwargs["x"], session))
        return PerformanceResult(1.25, energy=2.5, source="silicon")

    monkeypatch.setattr(CustomAllReduce, "query_with_resolution", query_with_resolution)

    pre = _moe_dispatch("pre_dispatch", pre_dispatch=True)
    post = _moe_dispatch("post_dispatch", pre_dispatch=False)
    assert pre._OWNS_RESOLUTION_WALK
    assert post._OWNS_RESOLUTION_WALK

    pre_result = pre.query_with_resolution(_MoEDatabase(), session=session, x=8)
    post_result = post.query_with_resolution(_MoEDatabase(), session=session, x=8)

    assert float(pre_result) == float(post_result) == 1.25
    assert {(tp_size, h, x) for _, tp_size, h, x, _ in calls} == {(4, 1, 32768)}
    assert {name for name, *_ in calls} == {
        "pre_dispatch.custom_allreduce",
        "post_dispatch.custom_allreduce",
    }
    assert all(call_session is session for *_, call_session in calls)


def test_moe_dispatch_without_session_preserves_direct_database_query() -> None:
    database = _MoEDatabase()

    result = _moe_dispatch("pure_dispatch", pre_dispatch=True).query_with_resolution(
        database,
        session=None,
        x=8,
    )

    assert float(result) == 1.25
    assert database.custom_allreduce_queries == [(common.CommQuantMode.half, 4, 32768)]


class _UnsupportedMoEDatabase(_MoEDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.nccl_queries: list[tuple[common.CommQuantMode, int, str, int]] = []

    def query_nccl(
        self,
        quant_mode: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        size: int,
    ) -> PerformanceResult:
        self.nccl_queries.append((quant_mode, num_gpus, operation, size))
        return PerformanceResult(2.0, energy=0.0, source="silicon")


class _NoMeasurementExecutor:
    def execute(self, *args, **kwargs):
        raise AssertionError(f"unsupported composition reached measurement executor: {args}, {kwargs}")


def test_moe_dispatch_unsupported_selected_child_is_structured(tmp_path) -> None:
    database = _UnsupportedMoEDatabase()
    operation = MoEDispatch(
        name="outside_frozen_profile",
        scale_factor=1.0,
        hidden_size=4096,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=4,
        attention_dp_size=2,
        pre_dispatch=True,
        moe_backend=None,
        is_context=True,
        quant_mode=common.MoEQuantMode.fp8_block,
    )
    overlay = OverlayStore(tmp_path / "unsupported.sqlite")
    session = ResolutionSession(
        overlay,
        _NoMeasurementExecutor(),
        ResolutionBudget(max_new_keys=4, max_wall_seconds=5.0),
        _protocol(),
    )

    try:
        with pytest.raises(ResolutionFailed) as raised:
            session.execute_callback(lambda: operation.query_with_resolution(database, session=session, x=8))
    finally:
        overlay.close()

    assert {reason.code for reason in raised.value.reasons} == {UnresolvedCode.MISSING_ADAPTER}
    assert database.nccl_queries == [
        (common.CommQuantMode.half, 2, "reduce_scatter", 32768),
        (common.CommQuantMode.half, 4, "all_gather", 65536),
    ]


class _DSv4Database:
    system = "gb200"
    backend = "sglang"
    version = "0.5.10"
    _default_database_mode = common.DatabaseMode.SILICON
    _extracted_metrics_cache: ClassVar[dict[str, object]] = {}

    def _query_silicon_or_hybrid(self, *, get_silicon, **_kwargs):
        return get_silicon()

    def _interp_pr(self, latency: float, *, energy: float = 0.0) -> PerformanceResult:
        return PerformanceResult(latency, energy=energy, source="silicon")


def _dsv4_query_kwargs() -> dict[str, object]:
    return {
        "b": 1,
        "s": 54,
        "num_heads": 16,
        "native_heads": 128,
        "tp_size": 8,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
        "window_size": 128,
        "compress_ratio": 4,
        "o_groups": 2,
        "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
        "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
        "gemm_quant_mode": common.GEMMQuantMode.fp8_block,
        "database_mode": common.DatabaseMode.SILICON,
    }


def test_csa_curated_query_uses_shared_raw_measurement_calibration_boundary(monkeypatch) -> None:
    apply_calibration = dsv4_ops._apply_dsv4_topk_calibration
    calls: list[tuple[float, float, float]] = []

    def tracked(latency_ms: float, energy_wms: float, delta_ms: float) -> tuple[float, float]:
        calls.append((latency_ms, energy_wms, delta_ms))
        return apply_calibration(latency_ms, energy_wms, delta_ms)

    monkeypatch.setattr(dsv4_ops, "_apply_dsv4_topk_calibration", tracked)
    monkeypatch.setattr(ContextDeepSeekV4AttentionModule, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(GenerationDeepSeekV4AttentionModule, "load_data", classmethod(lambda cls, database: None))

    context_db = _DSv4Database()
    context_db._context_deepseek_v4_attention_module_data = {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.fp8: {
                common.GEMMQuantMode.fp8_block: {16: {4: {8192: {54: {1: {"latency": 5.0, "energy": 50.0}}}}}}
            }
        }
    }
    context_db._dsv4_csa_topk_calib = {"exact": {(8192, 54, 1): 2.0}}
    context = ContextDeepSeekV4AttentionModule._query_context_attn_table(
        context_db,
        **_dsv4_query_kwargs(),
        prefix=8192,
    )

    generation_db = _DSv4Database()
    generation_db._generation_deepseek_v4_attention_module_data = {
        common.KVCacheQuantMode.fp8: {
            common.GEMMQuantMode.fp8_block: {16: {4: {1: {8193: {"latency": 7.0, "energy": 70.0}}}}}
        }
    }
    generation_db._dsv4_csa_topk_calib = {"exact": {(8192, 1, 1): 3.0}}
    generation_kwargs = _dsv4_query_kwargs()
    generation_kwargs["s"] = 8193
    generation = GenerationDeepSeekV4AttentionModule._query_generation_attn_table(
        generation_db,
        **generation_kwargs,
    )

    assert calls == [(5.0, 50.0, 2.0), (7.0, 70.0, 3.0)]
    assert (float(context), context.energy) == pytest.approx((3.0, 30.0))
    assert (float(generation), generation.energy) == pytest.approx((4.0, 40.0))


def test_shared_csa_calibration_clamps_latency_and_energy_together() -> None:
    apply_calibration = dsv4_ops._apply_dsv4_topk_calibration

    assert apply_calibration(5.0, 50.0, 2.0) == pytest.approx((3.0, 30.0))
    assert apply_calibration(1.0, 10.0, 2.0) == pytest.approx((0.0, 0.0))
    assert apply_calibration(0.0, 0.0, 2.0) == pytest.approx((0.0, 0.0))
