# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 attention-module normalization and request contracts."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations import dsv4 as dsv4_operations
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
    _dsv4_resolve_module_slice,
    load_context_dsv4_kind_module_data,
    load_generation_dsv4_kind_module_data,
)
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_PROFILE_COMPATIBILITY = {
    "model_artifact": _MODEL_ARTIFACT,
    "serving_mode": "aggregated",
    "tp_size": 4,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
_SEMANTIC_DESCRIPTOR = {
    "full_module": True,
    "tensor_generator": "normal-v1",
    "seed": 0,
    "tp_simulation": "single-gpu-tp4",
    "canonical_num_heads": 16,
    "padded_num_heads": 64,
}
_CASES = [
    pytest.param(
        "context",
        4,
        f"{PerfFile.DSV4_CSA_CONTEXT_MODULE}/v1",
        id="context-csa",
    ),
    pytest.param(
        "context",
        128,
        f"{PerfFile.DSV4_HCA_CONTEXT_MODULE}/v1",
        id="context-hca",
    ),
    pytest.param(
        "generation",
        4,
        f"{PerfFile.DSV4_CSA_GENERATION_MODULE}/v1",
        id="generation-csa",
    ),
    pytest.param(
        "generation",
        128,
        f"{PerfFile.DSV4_HCA_GENERATION_MODULE}/v1",
        id="generation-hca",
    ),
]


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions=_RUNTIME_VERSIONS,
        topology_schema="nvidia-smi-v1",
        topology_fingerprint="gb200-nvlink4",
        profile_compatibility=_PROFILE_COMPATIBILITY,
    )


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )


class _ProfileDatabase:
    def __init__(self) -> None:
        self.system = "gb200"
        self.backend = "sglang"
        self.version = "0.5.10"
        self.measurement_environment = _environment()
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def query_context_deepseek_v4_attention_module(self, **query: Any) -> PerformanceResult:
        self.queries.append(("context", query))
        return PerformanceResult(1.0, energy=2.0, source="silicon")

    def query_generation_deepseek_v4_attention_module(self, **query: Any) -> PerformanceResult:
        self.queries.append(("generation", query))
        return PerformanceResult(1.0, energy=2.0, source="silicon")


@pytest.fixture(scope="module")
def dsv4_profile_model():
    model_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        attention_dp_size=1,
        cp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        nextn=0,
        workload_distribution="power_law",
        moe_backend=None,
    )
    return get_model(_MODEL_ARTIFACT, model_config, backend_name="sglang")


def _operation(model, phase: str, compress_ratio: int):
    operation_type = ContextDeepSeekV4AttentionModule if phase == "context" else GenerationDeepSeekV4AttentionModule
    operations = model.context_ops if phase == "context" else model.generation_ops
    return next(
        operation
        for operation in operations
        if isinstance(operation, operation_type) and operation._compress_ratio == compress_ratio
    )


def _runtime_inputs(phase: str) -> dict[str, object]:
    if phase == "context":
        return {
            "x": 3 * 257,
            "batch_size": 3,
            "beam_width": 1,
            "s": 257,
            "prefix": 91,
            "model_name": _MODEL_ARTIFACT,
            "seq_imbalance_correction_scale": 1.0,
        }

    persisted_isl = 2048
    step = 17
    return {
        "x": 3,
        "batch_size": 3,
        "beam_width": 1,
        # Generation rows persist absolute KV length as isl + step.
        "s": persisted_isl + step,
        "prefix": 91,
        "model_name": _MODEL_ARTIFACT,
        "gen_seq_imbalance_correction_scale": 1.0,
    }


def _expected_query(phase: str, compress_ratio: int) -> dict[str, object]:
    common = {
        "tp_size": 4,
        "num_heads": 16,
        "compress_ratio": compress_ratio,
        "batch_size": 3,
    }
    if phase == "context":
        return {
            **common,
            "sequence_length": 257,
            "prefix_length": 91,
            "mla_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "gemm_type": "fp8_block",
        }
    return {
        **common,
        "sequence_length": 2048 + 17,
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
    }


def _canonicalize_ordinary_query(phase: str, query: dict[str, Any]) -> dict[str, object]:
    common = {
        "tp_size": query["tp_size"],
        "num_heads": query["num_heads"],
        "compress_ratio": query["compress_ratio"],
        "batch_size": query["b"],
        "sequence_length": query["s"],
    }
    if phase == "context":
        return {
            **common,
            "prefix_length": query["prefix"],
            "mla_dtype": query["fmha_quant_mode"].name,
            "kv_cache_dtype": query["kvcache_quant_mode"].name,
            "gemm_type": query["gemm_quant_mode"].name,
        }
    return {
        **common,
        "kv_cache_dtype": query["kvcache_quant_mode"].name,
        "gemm_type": query["gemm_quant_mode"].name,
    }


def _inject_literal_attention_row(
    database: _ProfileDatabase,
    *,
    phase: str,
    compress_ratio: int,
    tp_size: int = 4,
    padded_num_heads: int = 64,
    latency_ms: float = 2.0,
    energy_wms: float = 20.0,
) -> None:
    """Install one literal persisted row without enabling interpolation.

    The V1.2 files persist the runner's padded 64-head axis.  TP remains a
    separate physical dimension: only ``padded_num_heads / tp_size`` becomes
    the canonical rank-local head count used by the SDK request.
    """

    leaf = {"latency": latency_ms, "energy": energy_wms}
    if phase == "context":
        database._context_deepseek_v4_attention_module_data = {
            common.FMHAQuantMode.bfloat16: {
                common.KVCacheQuantMode.fp8: {
                    common.GEMMQuantMode.fp8_block: {
                        padded_num_heads: {
                            tp_size: {
                                compress_ratio: {
                                    91: {
                                        257: {
                                            3: leaf,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    else:
        database._generation_deepseek_v4_attention_module_data = {
            common.KVCacheQuantMode.fp8: {
                common.GEMMQuantMode.fp8_block: {
                    padded_num_heads: {
                        tp_size: {
                            compress_ratio: {
                                3: {
                                    2048 + 17: leaf,
                                }
                            }
                        }
                    }
                }
            }
        }

    prefix = 91 if phase == "context" else 2048 + 17 - 1
    query_length = 257 if phase == "context" else 1
    database._dsv4_csa_topk_calib = {
        "exact": {(prefix, query_length, 3): 0.25},
        "by_pi": {(prefix, query_length): [(3, 0.25)]},
    }


def _disable_attention_loading(monkeypatch: pytest.MonkeyPatch, operation) -> None:
    monkeypatch.setattr(type(operation), "load_data", classmethod(lambda cls, database: None))


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_actual_attention_runtime_inputs_share_one_canonical_normalization(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
) -> None:
    del namespace
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    runtime_inputs = _runtime_inputs(phase)
    expected = _expected_query(phase, compress_ratio)
    database = _ProfileDatabase()

    operation.query(database, **runtime_inputs)

    assert len(database.queries) == 1
    recorded_phase, ordinary_query = database.queries[0]
    assert recorded_phase == phase
    assert _canonicalize_ordinary_query(phase, ordinary_query) == expected
    assert operation.normalize_perf_query(**runtime_inputs) == expected


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_frozen_attention_request_has_exact_namespace_profile_and_key(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
) -> None:
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    database = _ProfileDatabase()
    expected = _expected_query(phase, compress_ratio)

    assert operation._tp_size == 4
    assert operation._cp_size == 1
    assert operation._num_heads == 16
    request = operation.measurement_request(database, _protocol(), **_runtime_inputs(phase))

    assert request is not None
    assert request.query == expected
    assert request.key.namespace == namespace
    assert request.environment is database.measurement_environment
    assert request.environment.profile_compatibility == _PROFILE_COMPATIBILITY
    assert request.key == PerfKey.build(namespace, expected, database.measurement_environment)
    assert request.semantic_descriptor == _SEMANTIC_DESCRIPTOR


def test_coherent_release_candidate_request_keeps_exact_identity_over_stable_curated_profile(
    dsv4_profile_model,
) -> None:
    operation = _operation(dsv4_profile_model, "context", 4)
    database = _ProfileDatabase()
    stable_request = operation.measurement_request(
        database,
        _protocol(),
        **_runtime_inputs("context"),
    )
    database.measurement_environment = replace(
        database.measurement_environment,
        backend_version="0.5.10rc0",
        runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.10rc0"},
    )

    request = operation.measurement_request(
        database,
        _protocol(),
        **_runtime_inputs("context"),
    )

    assert request is not None
    assert stable_request is not None
    assert database.version == "0.5.10"
    assert request.environment.backend_version == "0.5.10rc0"
    assert request.environment.runtime_versions["sglang"] == "0.5.10rc0"
    assert request.key != stable_request.key


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_attention_scale_and_consumer_identity_do_not_change_physical_key(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
) -> None:
    del namespace
    profile_operation = _operation(dsv4_profile_model, phase, compress_ratio)
    single_consumer = copy(profile_operation)
    single_consumer._name = f"another_{phase}_consumer"
    single_consumer._scale_factor = 1.0
    database = _ProfileDatabase()
    protocol = _protocol()
    runtime_inputs = _runtime_inputs(phase)

    profile_request = profile_operation.measurement_request(database, protocol, **runtime_inputs)
    single_request = single_consumer.measurement_request(database, protocol, **runtime_inputs)

    assert profile_operation._scale_factor > single_consumer._scale_factor
    assert profile_request is not None and single_request is not None
    assert profile_request.op_id != single_request.op_id
    assert profile_request.key == single_request.key
    assert profile_request.query == single_request.query == _expected_query(phase, compress_ratio)
    assert "scale_factor" not in profile_request.query
    assert "num_layers" not in profile_request.query


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_literal_curated_attention_row_maps_padded_heads_and_applies_only_the_legacy_csa_delta_once(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del namespace
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    database = _ProfileDatabase()
    _inject_literal_attention_row(database, phase=phase, compress_ratio=compress_ratio)
    _disable_attention_loading(monkeypatch, operation)

    result = operation.curated_exact_result(database, **_runtime_inputs(phase))

    assert operation._tp_size == 4
    assert operation._num_heads == 64 // operation._tp_size == 16
    assert result is not None
    # Persisted heads=64 at TP4 reconstructs the physical rank-local identity
    # heads=16.  Curated CSA rows came from the historical degenerate-top-k
    # sweep, so subtract its delta exactly once.  HCA has no such correction.
    corrected_latency = 1.75 if compress_ratio == 4 else 2.0
    corrected_energy = 17.5 if compress_ratio == 4 else 20.0
    assert float(result) == pytest.approx(corrected_latency * operation._scale_factor)
    assert result.energy == pytest.approx(corrected_energy * operation._scale_factor)
    assert result.source == "curated_exact"


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_literal_curated_attention_lookup_never_interpolates_or_crosses_tp_rows(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del namespace
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    database = _ProfileDatabase()
    _disable_attention_loading(monkeypatch, operation)

    _inject_literal_attention_row(
        database,
        phase=phase,
        compress_ratio=compress_ratio,
        tp_size=2,
        padded_num_heads=64,
    )
    assert operation.curated_exact_result(database, **_runtime_inputs(phase)) is None

    _inject_literal_attention_row(database, phase=phase, compress_ratio=compress_ratio)
    off_grid = dict(_runtime_inputs(phase))
    off_grid["s"] = int(off_grid["s"]) + 1
    if phase == "context":
        by_sequence = database._context_deepseek_v4_attention_module_data[common.FMHAQuantMode.bfloat16][
            common.KVCacheQuantMode.fp8
        ][common.GEMMQuantMode.fp8_block][64][4][compress_ratio][91]
        by_sequence[259] = {3: {"latency": 4.0, "energy": 40.0}}
    else:
        by_sequence = database._generation_deepseek_v4_attention_module_data[common.KVCacheQuantMode.fp8][
            common.GEMMQuantMode.fp8_block
        ][64][4][compress_ratio][3]
        by_sequence[2048 + 19] = {"latency": 4.0, "energy": 40.0}
    # The requested point is bracketed by two literal rows.  Curated exact
    # lookup must still miss rather than silently reusing the interpolator.
    assert operation.curated_exact_result(database, **off_grid) is None


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
@pytest.mark.parametrize("mismatch", ["tp", "local-heads", "model-artifact", "profile-revision"])
def test_literal_curated_attention_rejects_operation_model_and_profile_mismatches(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
    mismatch: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del namespace
    operation = copy(_operation(dsv4_profile_model, phase, compress_ratio))
    database = _ProfileDatabase()
    _inject_literal_attention_row(database, phase=phase, compress_ratio=compress_ratio)
    _disable_attention_loading(monkeypatch, operation)

    if mismatch == "tp":
        operation._tp_size = 2
    elif mismatch == "local-heads":
        operation._num_heads = 8
    elif mismatch == "model-artifact":
        database.measurement_environment = replace(
            database.measurement_environment,
            profile_compatibility={**_PROFILE_COMPATIBILITY, "model_artifact": "other/model"},
        )
    else:
        database.measurement_environment = replace(
            database.measurement_environment,
            runtime_versions={**_RUNTIME_VERSIONS, "model_profile": "dsv4-v1.3"},
        )

    assert operation.curated_exact_result(database, **_runtime_inputs(phase)) is None


def test_literal_curated_attention_rejects_cuda_runtime_mismatch(
    dsv4_profile_model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = "context"
    compress_ratio = 4
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    database = _ProfileDatabase()
    _inject_literal_attention_row(database, phase=phase, compress_ratio=compress_ratio)
    _disable_attention_loading(monkeypatch, operation)
    database.measurement_environment = replace(
        database.measurement_environment,
        runtime_versions={**_RUNTIME_VERSIONS, "cuda": "12.9"},
    )

    assert operation.curated_exact_result(database, **_runtime_inputs(phase)) is None


@pytest.mark.parametrize(("phase", "compress_ratio", "namespace"), _CASES)
def test_representative_full_module_overlay_is_scaled_once_without_legacy_curated_delta(
    dsv4_profile_model,
    phase: str,
    compress_ratio: int,
    namespace: str,
) -> None:
    operation = _operation(dsv4_profile_model, phase, compress_ratio)
    database = _ProfileDatabase()
    runtime_inputs = _runtime_inputs(phase)
    request = operation.measurement_request(database, _protocol(), **runtime_inputs)
    assert request is not None

    perf_row = {
        "model": _MODEL_ARTIFACT,
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
        "num_heads": 64,
        "batch_size": 3,
        "isl": 257 if phase == "context" else 1,
        "tp_size": 4,
        "step": 91 if phase == "context" else 2048 + 17 - 1,
        "compress_ratio": compress_ratio,
        "latency": 2.0,
    }
    record = MeasurementRecord.valid(
        key=PerfKey.build(namespace, _expected_query(phase, compress_ratio), database.measurement_environment),
        latency_ms=2.0,
        energy_wms=20.0,
        samples_ms=(1.9, 2.0, 2.1),
        protocol=_protocol(),
        perf_row=perf_row,
        provenance={
            "full_module": True,
            "tensor_generator": "normal-v1",
            "initialization": "proper-init",
            "tp_simulation": "single-gpu-tp4",
            "canonical_num_heads": 16,
            "padded_num_heads": 64,
        },
    )

    result = operation.performance_from_record(record, **runtime_inputs)

    # Unlike historical curated CSA files, normal-v1/proper-init overlay
    # evidence already measures the representative full module.  Applying the
    # legacy 0.25-ms curated-data delta here would double-correct it.
    assert float(result) == pytest.approx(2.0 * operation._scale_factor)
    assert result.energy == pytest.approx(20.0 * operation._scale_factor)
    assert float(result) != pytest.approx((2.0 - 0.25) * operation._scale_factor)
    assert result.source == "overlay"


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_empirical_attention_fallback_uses_padded_head_and_tp_axes(
    dsv4_profile_model,
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = _operation(dsv4_profile_model, phase, compress_ratio=4)
    database = _ProfileDatabase()
    _inject_literal_attention_row(database, phase=phase, compress_ratio=4)
    _disable_attention_loading(monkeypatch, operation)
    monkeypatch.setattr(dsv4_operations, "_deepseek_v4_attention_sol", lambda *args, **kwargs: (1.0, 1.0, 1.0))

    runtime_inputs = dict(_runtime_inputs(phase))
    if phase == "context":
        runtime_inputs["prefix"] = 0
        prefix_rows = database._context_deepseek_v4_attention_module_data[common.FMHAQuantMode.bfloat16][
            common.KVCacheQuantMode.fp8
        ][common.GEMMQuantMode.fp8_block][64][4][4]
        prefix_rows[0] = prefix_rows.pop(91)

        def query_empirical(**query: Any) -> PerformanceResult:
            return ContextDeepSeekV4AttentionModule._query_context_attn_table(
                database,
                **query,
                database_mode=common.DatabaseMode.EMPIRICAL,
            )

        database.query_context_deepseek_v4_attention_module = query_empirical
    else:

        def query_empirical(**query: Any) -> PerformanceResult:
            return GenerationDeepSeekV4AttentionModule._query_generation_attn_table(
                database,
                **query,
                database_mode=common.DatabaseMode.EMPIRICAL,
            )

        database.query_generation_deepseek_v4_attention_module = query_empirical

    result = operation.query(database, **runtime_inputs)

    # The empirical grid must calibrate from the same padded64/TP4 physical
    # slice as silicon and literal-exact lookup.  Falling back to the legacy
    # {local_heads -> compress_ratio} traversal loses the bundled rows.
    assert float(result) == pytest.approx(2.0 * operation._scale_factor)
    assert result.source == "empirical"


def test_legacy_context_slice_does_not_treat_prefix_equal_to_tp_as_a_tp_axis() -> None:
    expected_prefix_rows = {
        0: {257: {3: {"latency": 1.0}}},
        4: {257: {3: {"latency": 2.0}}},
    }
    legacy_quant_data = {16: {4: expected_prefix_rows}}

    head_axis, module_slice = _dsv4_resolve_module_slice(
        legacy_quant_data,
        num_heads=16,
        tp_size=4,
        compress_ratio=4,
        phase="context",
    )

    assert head_axis == 16
    assert module_slice is expected_prefix_rows


def test_legacy_generation_slice_does_not_treat_batch_equal_to_tp_as_a_tp_axis() -> None:
    expected_batch_rows = {
        4: {2065: {"latency": 1.0}},
        8: {2065: {"latency": 2.0}},
    }
    legacy_quant_data = {16: {4: expected_batch_rows}}

    head_axis, module_slice = _dsv4_resolve_module_slice(
        legacy_quant_data,
        num_heads=16,
        tp_size=4,
        compress_ratio=4,
        phase="generation",
    )

    assert head_axis == 16
    assert module_slice is expected_batch_rows


def test_tp_preserving_slice_missing_compression_does_not_fall_back_to_tp_axis() -> None:
    hca_only_tp4_slice = {128: {0: {257: {3: {"latency": 2.0}}}}}
    quant_data = {
        64: {
            2: {4: {0: {257: {3: {"latency": 9.0}}}}},
            4: hca_only_tp4_slice,
        }
    }

    head_axis, module_slice = _dsv4_resolve_module_slice(
        quant_data,
        num_heads=16,
        tp_size=4,
        compress_ratio=4,
        phase="context",
    )

    assert head_axis == 64
    assert module_slice is None


def test_legacy_tp1_slice_with_exact_head_match_remains_resolvable() -> None:
    expected_prefix_rows = {0: {257: {3: {"latency": 1.0}}}}
    legacy_quant_data = {16: {4: expected_prefix_rows}}

    head_axis, module_slice = _dsv4_resolve_module_slice(
        legacy_quant_data,
        num_heads=16,
        tp_size=1,
        compress_ratio=4,
        phase="context",
    )

    assert head_axis == 16
    assert module_slice is expected_prefix_rows


def test_legacy_generation_slice_with_padded_head_collision_remains_resolvable() -> None:
    expected_batch_rows = {1: {2065: {"latency": 1.0}}}
    legacy_quant_data = {64: {4: expected_batch_rows}}

    head_axis, module_slice = _dsv4_resolve_module_slice(
        legacy_quant_data,
        num_heads=16,
        tp_size=4,
        compress_ratio=4,
        phase="generation",
    )

    assert head_axis == 64
    assert module_slice is expected_batch_rows


def test_raw_context_loader_retains_tp_axis_for_equal_padded_head_rows(tmp_path: Path) -> None:
    perf_file = tmp_path / "dsv4_csa_context_module_perf.txt"
    perf_file.write_text(
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,compress_ratio,latency,power\n"
        "bfloat16,fp8,fp8_block,64,3,257,2,91,4,9.0,100.0\n"
        "bfloat16,fp8,fp8_block,64,3,257,4,91,4,2.0,100.0\n"
    )

    data = load_context_dsv4_kind_module_data(str(perf_file))

    assert data is not None
    by_persisted_heads = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][
        common.GEMMQuantMode.fp8_block
    ][64]
    # The persisted head field is padded64 for both rows.  TP is therefore a
    # required literal identity axis; TP4 later reconstructs canonical16.
    assert by_persisted_heads[2][4][91][257][3]["latency"] == pytest.approx(9.0)
    assert by_persisted_heads[4][4][91][257][3]["latency"] == pytest.approx(2.0)


def test_raw_generation_loader_retains_tp_axis_for_equal_padded_head_rows(tmp_path: Path) -> None:
    perf_file = tmp_path / "dsv4_csa_generation_module_perf.txt"
    perf_file.write_text(
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,compress_ratio,latency,power\n"
        "bfloat16,fp8,fp8_block,64,3,1,2,2064,4,9.0,100.0\n"
        "bfloat16,fp8,fp8_block,64,3,1,4,2064,4,2.0,100.0\n"
    )

    data = load_generation_dsv4_kind_module_data(str(perf_file))

    assert data is not None
    by_persisted_heads = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][64]
    # Decode persists s_total = isl + step. Equal padded-head rows must retain
    # their TP identity rather than last-writer-wins colliding at s_total=2065.
    assert by_persisted_heads[2][4][3][2065]["latency"] == pytest.approx(9.0)
    assert by_persisted_heads[4][4][3][2065]["latency"] == pytest.approx(2.0)
