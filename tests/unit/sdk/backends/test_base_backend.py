# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
)
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionSession

pytestmark = pytest.mark.unit


class _LatencyResult:
    def __init__(self, latency_ms: float, energy_wms: float) -> None:
        self._latency_ms = latency_ms
        self.energy = energy_wms

    def __float__(self) -> float:
        return self._latency_ms


class _StaticOp:
    def __init__(self, name: str, latency_ms: float, energy_wms: float) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._energy_wms = energy_wms

    def query(self, *args, **kwargs) -> _LatencyResult:
        return _LatencyResult(self._latency_ms, self._energy_wms)


def _resolution_protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="backend-resolution-v1",
        warmups=2,
        samples=1,
        statistic="median",
        timer="cuda_event",
        tuning_revision="none",
    )


def _resolution_environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="h100_sxm",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="h100",
        runtime_versions={"cuda": "12.9"},
    )


class _ResolutionDatabase:
    def __init__(self) -> None:
        self.backend = "test-backend"
        self.version = "test-version"
        self.system = "test-system"
        self.system_spec = {"gpu": {"mem_capacity": 80 * (1 << 30)}}
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


class _ResolvingStaticOp(Operation):
    def __init__(self, name: str) -> None:
        super().__init__(name, 1.0)
        self.query_calls: list[dict[str, object]] = []
        self.measurement_calls: list[dict[str, object]] = []

    def query(self, database, **kwargs) -> PerformanceResult:
        del database
        self.query_calls.append(dict(kwargs))
        return PerformanceResult(9.0, energy=90.0, source="silicon")

    def get_weights(self, **kwargs) -> float:
        del kwargs
        return 0.0

    def measurement_request(self, database, protocol, **kwargs) -> MeasurementRequest:
        del database
        query = dict(kwargs)
        self.measurement_calls.append(query)
        environment = _resolution_environment()
        semantic = {"operation": "shared_context_point"}
        return MeasurementRequest(
            op_id=self._name,
            key=PerfKey.build("test_backend_static/v1", query, environment, semantic),
            query=query,
            environment=environment,
            semantic_descriptor=semantic,
            protocol=protocol,
        )


class _ResolutionExecutor:
    def __init__(self, latency_ms: float = 0.4, energy_wms: float = 4.0) -> None:
        self.latency_ms = latency_ms
        self.energy_wms = energy_wms
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
        return tuple(
            MeasurementRecord.valid(
                key=request.key,
                latency_ms=self.latency_ms,
                energy_wms=self.energy_wms,
                samples_ms=(self.latency_ms,),
                protocol=request.protocol,
                perf_row={"latency": self.latency_ms, "energy": self.energy_wms},
                provenance={"collector_revision": "backend-test-v1"},
            )
            for request in batch
        )


class _TestBackend(BaseBackend):
    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError

    def _get_memory_usage(
        self,
        model,
        database,
        batch_size,
        beam_width,
        isl,
        osl,
        num_tokens=0,
        prefix=0,
        encoder_memory=None,
    ) -> dict[str, float]:
        return {"total": 1.0}


@pytest.fixture
def backend() -> BaseBackend:
    return _TestBackend()


@pytest.fixture
def database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


@pytest.fixture
def model():
    model = MagicMock()
    model.model_path = "test-model"
    model.model_name = "test-model"
    model._nextn = 0
    model.encoder_ops = []
    model.context_ops = [
        _StaticOp("context_attention", latency_ms=11.0, energy_wms=110.0),
        _StaticOp("logits_gemm", latency_ms=3.0, energy_wms=30.0),
    ]
    model.generation_ops = [
        _StaticOp("generation_attention", latency_ms=2.0, energy_wms=20.0),
        _StaticOp("generation_mlp", latency_ms=1.0, energy_wms=10.0),
    ]
    model.config = ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
    )
    return model


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2)


@pytest.fixture
def resolution_session_factory(tmp_path: Path):
    stores: list[OverlayStore] = []

    def make(executor: _ResolutionExecutor) -> ResolutionSession:
        overlay = OverlayStore(tmp_path / f"backend-overlay-{len(stores)}.sqlite")
        stores.append(overlay)
        return ResolutionSession(
            overlay,
            executor,
            ResolutionBudget(max_new_keys=16, max_wall_seconds=30.0),
            _resolution_protocol(),
        )

    yield make
    for store in stores:
        store.close()


def _resolving_context_model(model):
    ops = [_ResolvingStaticOp("consumer_a"), _ResolvingStaticOp("consumer_b")]
    model.context_ops = ops
    model.generation_ops = []
    return model, ops


@pytest.mark.parametrize("mode", ["static", "static_ctx", "static_gen"])
@pytest.mark.parametrize("latency_correction_scale", [1.0, 1.25])
def test_run_static_latency_only_matches_run_static_latency(
    backend: BaseBackend,
    model,
    database,
    runtime_config: RuntimeConfig,
    mode: str,
    latency_correction_scale: float,
) -> None:
    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )
    latency_only = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode=mode,
        stride=2,
        latency_correction_scale=latency_correction_scale,
    )

    summary_latency = sum(summary.get_context_latency_dict().values()) + sum(
        summary.get_generation_latency_dict().values()
    )
    request_latency = float(summary.get_summary_df().iloc[0]["request_latency"])

    assert latency_only == pytest.approx(summary_latency)
    assert latency_only == pytest.approx(request_latency, abs=1e-3)


def test_run_static_can_route_to_rust_engine_step_backend(
    monkeypatch,
    backend: BaseBackend,
    model,
    database,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    calls = []

    def _fake_rust_breakdown(model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg):
        calls.append((model_arg, database_arg, runtime_config_arg, mode_arg, stride_arg, scale_arg))
        return (
            {"rust_engine_step_context": 7.0},
            {"rust_engine_step_generation": 3.0},
            {"rust_engine_step_context": "rust"},
            {"rust_engine_step_generation": "rust"},
        )

    monkeypatch.setattr(
        base_backend_module,
        "estimate_static_latency_breakdown_with_rust",
        _fake_rust_breakdown,
    )

    summary = backend.run_static(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=5, prefix=2, engine_step_backend="rust"),
        mode="static",
        stride=2,
        latency_correction_scale=1.25,
    )

    assert len(calls) == 1
    assert calls[0][3:] == ("static", 2, 1.25)
    assert summary.get_context_latency_dict() == {"rust_engine_step_context": 7.0}
    assert summary.get_generation_latency_dict() == {"rust_engine_step_generation": 3.0}
    assert summary.get_context_energy_wms_dict() == {"rust_engine_step_context": 0.0}
    assert summary.get_generation_energy_wms_dict() == {"rust_engine_step_generation": 0.0}
    assert summary.get_context_source_dict() == {"rust_engine_step_context": "rust"}
    assert summary.get_generation_source_dict() == {"rust_engine_step_generation": "rust"}


def test_run_static_resolution_batches_one_key_and_replays_both_consumers(
    backend: BaseBackend,
    model,
    runtime_config: RuntimeConfig,
    resolution_session_factory,
) -> None:
    model, ops = _resolving_context_model(model)
    database = _ResolutionDatabase()
    executor = _ResolutionExecutor()
    session = resolution_session_factory(executor)

    summary = backend.run_static(
        model,
        database,
        runtime_config,
        mode="static_ctx",
        resolution_session=session,
    )

    assert summary.get_context_latency_dict() == {"consumer_a": 0.4, "consumer_b": 0.4}
    assert summary.get_context_energy_wms_dict() == {"consumer_a": 4.0, "consumer_b": 4.0}
    assert summary.get_context_source_dict() == {"consumer_a": "overlay", "consumer_b": "overlay"}
    assert [len(batch) for batch in executor.request_batches] == [1]
    assert session.report.unique_misses == 1
    assert session.report.consumer_misses == 2
    assert [len(op.measurement_calls) for op in ops] == [2, 2]
    assert [len(op.query_calls) for op in ops] == [0, 0]

    warm = backend.run_static(
        model,
        database,
        runtime_config,
        mode="static_ctx",
        resolution_session=session,
    )

    assert warm.get_context_latency_dict() == {"consumer_a": 0.4, "consumer_b": 0.4}
    assert [len(batch) for batch in executor.request_batches] == [1]
    assert session.report.unique_misses == 1
    assert session.report.consumer_misses == 2
    assert session.report.overlay_hits == 4
    assert [len(op.measurement_calls) for op in ops] == [3, 3]

    ordinary = backend.run_static(model, database, runtime_config, mode="static_ctx")

    assert ordinary.get_context_latency_dict() == {"consumer_a": 9.0, "consumer_b": 9.0}
    assert [len(op.query_calls) for op in ops] == [1, 1]
    assert [len(op.measurement_calls) for op in ops] == [3, 3]
    assert [len(batch) for batch in executor.request_batches] == [1]


def test_run_static_latency_only_owns_one_resolution_callback(
    backend: BaseBackend,
    model,
    runtime_config: RuntimeConfig,
    resolution_session_factory,
) -> None:
    model, ops = _resolving_context_model(model)
    database = _ResolutionDatabase()
    executor = _ResolutionExecutor()
    session = resolution_session_factory(executor)

    latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static_ctx",
        resolution_session=session,
    )

    assert latency == pytest.approx(0.8)
    assert [len(batch) for batch in executor.request_batches] == [1]
    assert [len(op.measurement_calls) for op in ops] == [2, 2]
    assert [len(op.query_calls) for op in ops] == [0, 0]


def test_run_static_latency_only_batches_encoder_context_and_generation_together(
    backend: BaseBackend,
    model,
    runtime_config: RuntimeConfig,
    resolution_session_factory,
) -> None:
    model.encoder_config = common.VisionEncoderConfig(
        depth=1,
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        patch_size=14,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=128,
    )
    phase_ops = {
        "encoder": [_ResolvingStaticOp("encoder_a"), _ResolvingStaticOp("encoder_b")],
        "context": [_ResolvingStaticOp("context_a"), _ResolvingStaticOp("context_b")],
        "generation": [_ResolvingStaticOp("generation_a"), _ResolvingStaticOp("generation_b")],
    }
    model.encoder_ops = phase_ops["encoder"]
    model.context_ops = phase_ops["context"]
    model.generation_ops = phase_ops["generation"]
    model._nextn = 2
    runtime_config.num_image_tokens = 4
    runtime_config.beam_width = 3
    runtime_config.seq_imbalance_correction_scale = 1.25
    runtime_config.gen_seq_imbalance_correction_scale = 1.5
    database = _ResolutionDatabase()
    executor = _ResolutionExecutor()
    session = resolution_session_factory(executor)

    latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static",
        resolution_session=session,
    )

    assert latency == pytest.approx(4.8)
    assert [len(batch) for batch in executor.request_batches] == [3]
    assert session.report.unique_misses == 3
    assert session.report.consumer_misses == 6
    assert {phase: [len(op.measurement_calls) for op in ops] for phase, ops in phase_ops.items()} == {
        "encoder": [2, 2],
        "context": [2, 2],
        "generation": [2, 2],
    }
    expected_queries = {
        "encoder": {
            "x": 8,
            "batch_size": 2,
            "beam_width": 1,
            "s": 4,
            "prefix": 0,
            "model_name": "test-model",
        },
        "context": {
            "x": 20,
            "batch_size": 2,
            "beam_width": 1,
            "s": 10,
            "prefix": 2,
            "seq_imbalance_correction_scale": 1.25,
        },
        "generation": {
            "x": 18,
            "batch_size": 6,
            "beam_width": 3,
            "s": 13,
            "gen_seq_imbalance_correction_scale": 1.5,
        },
    }
    for phase, ops in phase_ops.items():
        assert [op.measurement_calls for op in ops] == [
            [expected_queries[phase], expected_queries[phase]],
            [expected_queries[phase], expected_queries[phase]],
        ]


def test_resolution_session_bypasses_rust_without_changing_later_fast_path(
    monkeypatch,
    backend: BaseBackend,
    model,
    resolution_session_factory,
) -> None:
    from aiconfigurator.sdk.backends import base_backend as base_backend_module

    model, ops = _resolving_context_model(model)
    database = _ResolutionDatabase()
    runtime_config = RuntimeConfig(
        batch_size=2,
        beam_width=1,
        isl=8,
        osl=5,
        prefix=2,
        engine_step_backend="rust",
    )
    rust_calls = []

    def fake_rust(*args):
        rust_calls.append(args)
        return ({"rust_context": 7.0}, {}, {"rust_context": "rust"}, {})

    monkeypatch.setattr(base_backend_module, "estimate_static_latency_breakdown_with_rust", fake_rust)

    first_fast = backend.run_static(model, database, runtime_config, mode="static_ctx")
    assert first_fast.get_context_latency_dict() == {"rust_context": 7.0}
    assert len(rust_calls) == 1
    assert [len(op.measurement_calls) for op in ops] == [0, 0]

    executor = _ResolutionExecutor()
    session = resolution_session_factory(executor)
    resolved = backend.run_static(
        model,
        database,
        runtime_config,
        mode="static_ctx",
        resolution_session=session,
    )

    assert resolved.get_context_latency_dict() == {"consumer_a": 0.4, "consumer_b": 0.4}
    assert len(rust_calls) == 1
    assert [len(batch) for batch in executor.request_batches] == [1]
    assert [len(op.measurement_calls) for op in ops] == [2, 2]

    second_fast = backend.run_static(model, database, runtime_config, mode="static_ctx")
    assert second_fast.get_context_latency_dict() == {"rust_context": 7.0}
    assert len(rust_calls) == 2
    assert [len(op.measurement_calls) for op in ops] == [2, 2]

    first_latency_fast = backend.run_static_latency_only(model, database, runtime_config, mode="static_ctx")
    assert first_latency_fast == pytest.approx(7.0)
    assert len(rust_calls) == 3

    latency_executor = _ResolutionExecutor()
    latency_session = resolution_session_factory(latency_executor)
    resolved_latency = backend.run_static_latency_only(
        model,
        database,
        runtime_config,
        mode="static_ctx",
        resolution_session=latency_session,
    )
    assert resolved_latency == pytest.approx(0.8)
    assert len(rust_calls) == 3
    assert [len(batch) for batch in latency_executor.request_batches] == [1]
    assert [len(op.measurement_calls) for op in ops] == [4, 4]

    second_latency_fast = backend.run_static_latency_only(model, database, runtime_config, mode="static_ctx")
    assert second_latency_fast == pytest.approx(7.0)
    assert len(rust_calls) == 4
    assert [len(op.measurement_calls) for op in ops] == [4, 4]


@pytest.mark.parametrize(
    "database_mode",
    [common.DatabaseMode.EMPIRICAL, common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL],
)
def test_resolution_session_rejects_formula_only_database_roots(
    backend: BaseBackend,
    model,
    runtime_config: RuntimeConfig,
    resolution_session_factory,
    database_mode: common.DatabaseMode,
) -> None:
    model, ops = _resolving_context_model(model)
    database = _ResolutionDatabase()
    database._default_database_mode = database_mode
    database._shared_layer_mode = False
    executor = _ResolutionExecutor()
    session = resolution_session_factory(executor)

    with pytest.raises(ValueError, match="Cannot create a SILICON query view"):
        backend.run_static(
            model,
            database,
            runtime_config,
            mode="static_ctx",
            resolution_session=session,
        )

    assert executor.request_batches == []
    assert [len(op.measurement_calls) for op in ops] == [0, 0]
    assert [len(op.query_calls) for op in ops] == [0, 0]


def test_run_agg_with_osl_one_does_not_divide_by_zero(
    backend: BaseBackend,
    model,
    database,
    monkeypatch,
) -> None:
    """Regression: osl=1 (no-decode) must not raise and tokens/s/user must be 0.0."""
    monkeypatch.setattr(
        backend,
        "_get_mix_step_latency",
        lambda *args, **kwargs: (1.0, 1.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_genonly_step_latency",
        lambda *args, **kwargs: (0.0, 0.0, {}, {}),
    )
    monkeypatch.setattr(
        backend,
        "_get_memory_usage",
        lambda *args, **kwargs: {"total": 1.0},
    )

    summary = backend.run_agg(
        model,
        database,
        RuntimeConfig(batch_size=2, beam_width=1, isl=8, osl=1, prefix=2),
        ctx_tokens=8,
    )

    row = summary.get_summary_df().iloc[0]
    assert row["tpot"] > 0.0
    assert row["tokens/s/user"] == 0.0


def test_mix_step_efficiency_base_default_is_one(backend: BaseBackend) -> None:
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=16) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0
    assert backend._mix_step_efficiency(ctx_tokens=0, gen_tokens=0) == 1.0
