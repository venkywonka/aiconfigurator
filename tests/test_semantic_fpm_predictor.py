#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import types
from dataclasses import replace
from pathlib import Path

import pytest

from collector.layerwise.diagnostics import semantic_fpm_aic as semantic_aic
from collector.layerwise.diagnostics.semantic_fpm_aic import (
    RepositoryAicConfig,
    RepositoryAicStepRunner,
    _ExactSchedulerDatabase,
    _verify_import_root,
    bounded_operation_lookup,
    build_repository_predictor,
    build_repository_surface,
)
from collector.layerwise.diagnostics.semantic_fpm_insights import (
    AicPredictionRecord,
    AicQueryShape,
    AxisLookup,
    OperationLookup,
    SemanticBinQuery,
    SemanticShape,
    build_semantic_query,
)
from collector.layerwise.diagnostics.semantic_fpm_predictor import (
    AiconfiguratorSemanticBinPredictor,
    AicSurfaceUnavailableError,
    ConservativeSurfaceIndex,
    LayerwiseSurfacePoint,
    RawAicStep,
    predict_clean_bins,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    ContractError,
    FpmSample,
    SemanticPopulation,
    SemanticPopulationBin,
)


def _sample(*, lane: str, sample_id: str, shape: SemanticShape) -> FpmSample:
    return FpmSample(
        lane=lane,
        concurrency=16,
        sample_id=sample_id,
        phase=shape.validate(),
        workload_segment="real",
        counter_id=int(sample_id.rsplit("-", 1)[-1]),
        worker_id="worker-0",
        dp_rank=0,
        shape=shape,
        wall_ms=1.0,
        measured=True,
        analytic_eligible=True,
    )


def _bin(
    *,
    bin_id: str,
    shape: SemanticShape,
    clean: bool,
    profiled: bool,
) -> SemanticPopulationBin:
    phase = shape.validate()
    return SemanticPopulationBin(
        bin_id=bin_id,
        configuration_fingerprint="config-sha",
        concurrency=16,
        phase=phase,
        semantic_key=shape.semantic_key,
        clean_samples=(_sample(lane="clean", sample_id=f"clean-{bin_id[-1]}", shape=shape),) if clean else (),
        profiled_samples=(_sample(lane="profiled", sample_id=f"profiled-{bin_id[-1]}", shape=shape),)
        if profiled
        else (),
    )


def _ok_record(query: SemanticBinQuery, *, total_ms: float = 3.0) -> AicPredictionRecord:
    requested = AicQueryShape.from_query(query)
    operation_inventory = (
        ("generation_layerwise", "compute"),
        ("generation_tp_allreduce", "communication"),
    )
    config_axis = "max_num_seqs" if query.phase == "decode" else "max_num_batched_tokens"
    exact_axes = (
        AxisLookup(
            axis=config_axis,
            requested=256,
            evaluated=256,
            lower=256,
            upper=256,
            weight=0.0,
            delta=0,
            mode="exact",
        ),
        *tuple(
            AxisLookup(
                axis=axis,
                requested=value,
                evaluated=value,
                lower=value,
                upper=value,
                weight=0.0,
                delta=0,
                mode="exact",
            )
            for axis, value in requested.items()
        ),
    )
    return AicPredictionRecord(
        configuration_fingerprint=query.configuration_fingerprint,
        concurrency=query.concurrency,
        phase=query.phase,
        semantic_key=query.semantic_key,
        status="ok",
        reason="predicted",
        requested_shape=requested,
        evaluated_shape=requested,
        lookup_policy="conservative-v1",
        lookup_surface_id="surface-id",
        scheduler_surface_content_hash="surface-sha",
        axis_lookups=exact_axes,
        total_ms=total_ms,
        total_basis="operation_sum",
        compute_ms=2.0,
        communication_ms=1.0,
        other_ms=0.0,
        component_sum_ms=3.0,
        source="silicon",
        match_type="exact",
        predictor_version="test-v1",
        api_version="test-api-v1",
        component_classifier_version="exact-registry-v1",
        classified_operation_count=2,
        unclassified_operation_count=0,
        operation_inventory=operation_inventory,
        operation_inventory_hash=hashlib.sha256(
            json.dumps(operation_inventory, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        operation_values=(
            ("generation_layerwise", 2.0, "silicon"),
            ("generation_tp_allreduce", 1.0, "silicon"),
        ),
        operation_lookups=(_operation_lookup(),),
        configuration_provenance='{"test":true}',
    )


class _SpyPredictor:
    def __init__(self):
        self.queries: list[SemanticBinQuery] = []

    def predict(self, query: SemanticBinQuery) -> AicPredictionRecord:
        self.queries.append(query)
        return _ok_record(query)


def test_predictor_is_called_once_for_every_clean_bin_before_intersection():
    shared_shape = SemanticShape(0, 4, 0, 0, 400)
    clean_only_shape = SemanticShape(0, 2, 0, 0, 80)
    profiled_only_shape = SemanticShape(0, 8, 0, 0, 1600)
    bins = (
        _bin(bin_id="bin-1", shape=shared_shape, clean=True, profiled=True),
        _bin(bin_id="bin-2", shape=clean_only_shape, clean=True, profiled=False),
        _bin(bin_id="bin-3", shape=profiled_only_shape, clean=False, profiled=True),
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=tuple(sample for bin_ in bins for sample in (*bin_.clean_samples, *bin_.profiled_samples)),
        bins=bins,
        excluded_reason_counts={},
    )
    predictor = _SpyPredictor()

    predictions = predict_clean_bins(population, predictor)

    assert [prediction.bin_id for prediction in predictions] == ["bin-1", "bin-2"]
    assert len(predictor.queries) == 2
    assert predictor.queries[1].decode_requests == 2
    assert predictor.queries[1].query_decode_kv == 40


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda record: replace(record, semantic_key="[0,99,null,null,1]"), "predictor_call_identity_mismatch"),
        (lambda record: replace(record, total_ms=math.nan), "predictor_error"),
        (lambda record: replace(record, component_sum_ms=2.5), "predictor_error"),
        (lambda record: replace(record, total_basis="direct", total_ms=100.0), "predictor_error"),
        (lambda record: replace(record, axis_lookups=()), "predictor_error"),
        (lambda record: replace(record, operation_inventory_hash="0" * 64), "predictor_error"),
        (lambda record: replace(record, component_classifier_version="unknown-v1"), "predictor_error"),
        (lambda record: replace(record, source=None), "predictor_error"),
        (lambda record: replace(record, match_type=None), "predictor_error"),
        (lambda record: replace(record, match_type="nearest"), "predictor_error"),
    ],
)
def test_prediction_contract_fails_closed(mutation, expected_code):
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 4, 0, 0, 400),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )

    class BrokenPredictor:
        def predict(self, query):
            return mutation(_ok_record(query))

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, BrokenPredictor())
    assert exc_info.value.process_code == expected_code


def test_unexpected_predictor_exception_is_a_process_failure():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 4, 0, 0, 400),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )

    class ExplodingPredictor:
        def predict(self, query):
            raise RuntimeError("database corruption")

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, ExplodingPredictor())
    assert exc_info.value.process_code == "predictor_error"
    assert "database corruption" in str(exc_info.value)


def test_unavailable_prediction_still_requires_complete_provenance():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 6, 0, 0, 6 * 4096),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )
    base = AiconfiguratorSemanticBinPredictor(
        surface_index=ConservativeSurfaceIndex(
            points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
            surface_provenance='{"fixture":"unit"}',
            phase_axis_lookups=(
                (
                    "decode",
                    (AxisLookup("max_num_seqs", 256, None, None, None, None, None, "missing"),),
                ),
            ),
        ),
        runner=_FakeRunner(
            RawAicStep(
                operations=(("generation_layerwise", 1.0),),
                sources=(("generation_layerwise", "silicon"),),
                operation_lookups=(),
            )
        ),
        configuration_provenance='{"repo":"fixture"}',
    )

    class BrokenUnavailablePredictor:
        def predict(self, query):
            return replace(base.predict(query), api_version="")

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, BrokenUnavailablePredictor())
    assert exc_info.value.process_code == "predictor_error"

    class NearestConfigUnavailablePredictor:
        def predict(self, query):
            record = base.predict(query)
            forged = tuple(
                AxisLookup("max_num_seqs", 256, 300, 256, 300, 0.0, 44, "nearest")
                if lookup.axis == "max_num_seqs"
                else lookup
                for lookup in record.axis_lookups
            )
            return replace(record, axis_lookups=forged)

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, NearestConfigUnavailablePredictor())
    assert exc_info.value.process_code == "predictor_error"

    class ForgedUnavailablePayloadPredictor:
        def __init__(self, changes):
            self.changes = changes

        def predict(self, query):
            return replace(base.predict(query), **self.changes)

    for changes in (
        {"source": "silicon"},
        {"total_basis": "operation_sum"},
        {"match_type": "exact"},
        {"lookup_surface_id": "bogus"},
        {"scheduler_surface_content_hash": "bogus"},
    ):
        with pytest.raises(ContractError) as exc_info:
            predict_clean_bins(population, ForgedUnavailablePayloadPredictor(changes))
        assert exc_info.value.process_code == "predictor_error"


def test_unavailable_prediction_requires_complete_requested_axis_inventory():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 6, 0, 0, 6 * 4096),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )
    base = AiconfiguratorSemanticBinPredictor(
        surface_index=ConservativeSurfaceIndex(
            points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
            surface_provenance='{"fixture":"unit"}',
        ),
        runner=_FakeRunner(
            RawAicStep(
                operations=(("generation_layerwise", 1.0),),
                sources=(("generation_layerwise", "silicon"),),
                operation_lookups=(),
            )
        ),
        configuration_provenance='{"repo":"fixture"}',
    )

    class BrokenUnavailablePredictor:
        def predict(self, query):
            return replace(base.predict(query), axis_lookups=())

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, BrokenUnavailablePredictor())
    assert exc_info.value.process_code == "predictor_error"


def _point(*, phase: str, batch: int, new_tokens: int, past_kv: int) -> LayerwiseSurfacePoint:
    return LayerwiseSurfacePoint(
        phase=phase,
        batch_size=batch,
        new_tokens=new_tokens,
        past_kv=past_kv,
        row_content_hash=f"row-{phase}-{batch}-{new_tokens}-{past_kv}",
        latency_ms=1.0,
        detail_json='{"latency":1.0}',
    )


def _query(shape: SemanticShape) -> SemanticBinQuery:
    return build_semantic_query(
        configuration_fingerprint="config-sha",
        concurrency=16,
        shape=shape,
    )


def test_surface_selection_never_crosses_batch_or_new_token_axes():
    index = ConservativeSurfaceIndex(
        points=(
            _point(phase="decode", batch=4, new_tokens=1, past_kv=4096),
            _point(phase="decode", batch=8, new_tokens=1, past_kv=4096),
            _point(phase="context", batch=1, new_tokens=128, past_kv=0),
        ),
        surface_provenance='{"fixture":"unit"}',
    )

    missing_batch = index.select(_query(SemanticShape(0, 6, 0, 0, 6 * 4096)))
    missing_ctx_new = index.select(_query(SemanticShape(1, 0, 64, 0, 0)))

    assert (missing_batch.status, missing_batch.reason) == ("unavailable", "missing_surface")
    assert (missing_ctx_new.status, missing_ctx_new.reason) == ("unavailable", "missing_surface")


def test_multi_request_context_matches_exact_total_without_batch_one_fallback():
    index = ConservativeSurfaceIndex(
        points=(_point(phase="context", batch=2, new_tokens=128, past_kv=0),),
        surface_provenance='{"fixture":"unit"}',
    )

    selection = index.select(_query(SemanticShape(2, 0, 256, 0, 0)))

    assert selection.status == "ok"
    assert selection.point is not None
    assert (selection.point.batch_size, selection.point.new_tokens) == (2, 128)
    assert selection.evaluated_shape is not None
    assert selection.evaluated_shape.ctx_new_total == 256


def test_zero_prefix_context_requires_an_exact_zero_surface():
    index = ConservativeSurfaceIndex(
        points=(_point(phase="context", batch=1, new_tokens=128, past_kv=16),),
        surface_provenance='{"fixture":"unit"}',
    )

    selection = index.select(_query(SemanticShape(1, 0, 128, 0, 0)))

    assert (selection.status, selection.reason) == ("unavailable", "missing_surface")


def test_nearest_kv_uses_lower_tie_break_and_records_bounded_wiggle_room():
    index = ConservativeSurfaceIndex(
        points=(
            _point(phase="decode", batch=4, new_tokens=1, past_kv=4096),
            _point(phase="decode", batch=4, new_tokens=1, past_kv=8192),
        ),
        surface_provenance='{"fixture":"unit"}',
    )

    selection = index.select(_query(SemanticShape(0, 4, 0, 0, 4 * 6144)))

    assert selection.status == "ok"
    assert selection.point is not None
    assert selection.point.past_kv == 4096
    kv_lookup = next(lookup for lookup in selection.axis_lookups if lookup.axis == "decode_kv")
    assert (kv_lookup.lower, kv_lookup.upper, kv_lookup.evaluated) == (4096, 8192, 4096)
    assert (kv_lookup.delta, kv_lookup.mode) == (-2048, "nearest")


def test_nearest_kv_outside_cap_is_unavailable():
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
    )

    selection = index.select(_query(SemanticShape(0, 4, 0, 0, 4 * 1000)))

    assert (selection.status, selection.reason) == ("unavailable", "out_of_cap")


def test_bounded_one_sided_kv_snap_passes_contract_validation():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 4, 0, 0, 4 * 3000),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
        phase_axis_lookups=(
            (
                "decode",
                (AxisLookup("max_num_seqs", 256, 256, 256, 256, 0.0, 0, "exact"),),
            ),
        ),
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=_FakeRunner(
            RawAicStep(
                operations=(("generation_layerwise", 2.0),),
                sources=(("generation_layerwise", "silicon"),),
                operation_lookups=(),
            )
        ),
        configuration_provenance='{"repo":"fixture"}',
    )

    records = predict_clean_bins(population, predictor)

    assert records[0].record.evaluated_shape is not None
    assert records[0].record.evaluated_shape.decode_kv == 4096


def test_mixed_surface_is_explicitly_unsupported_in_v1():
    index = ConservativeSurfaceIndex(points=(), surface_provenance='{"fixture":"unit"}')

    selection = index.select(_query(SemanticShape(1, 2, 128, 0, 2048)))

    assert (selection.status, selection.reason) == ("unavailable", "unsupported_phase")


class _FakeRunner:
    api_version = "fake-api-v1"

    def __init__(self, raw_step: RawAicStep):
        self.raw_step = raw_step
        self.calls = []

    def predict(self, *, phase, shape, point):
        self.calls.append((phase, shape, point))
        return self.raw_step


def _operation_lookup() -> OperationLookup:
    return OperationLookup(
        operation="custom_allreduce",
        topology='{"quant":"half","tp":8}',
        requested=20480,
        lower=16384,
        upper=32768,
        weight=0.25,
        mode="interpolate",
        surface_content_hash="a" * 64,
        consumer_operations=("generation_tp_allreduce",),
    )


def test_reference_adapter_classifies_exact_operation_registry_and_preserves_snap_provenance():
    index = ConservativeSurfaceIndex(
        points=(
            _point(phase="decode", batch=4, new_tokens=1, past_kv=4096),
            _point(phase="decode", batch=4, new_tokens=1, past_kv=8192),
        ),
        surface_provenance='{"fixture":"unit"}',
    )
    runner = _FakeRunner(
        RawAicStep(
            operations=(("generation_layerwise", 2.0), ("generation_tp_allreduce", 1.0)),
            sources=(("generation_layerwise", "silicon"), ("generation_tp_allreduce", "silicon")),
            operation_lookups=(_operation_lookup(),),
        )
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=runner,
        configuration_provenance='{"repo":"fixture"}',
    )
    query = _query(SemanticShape(0, 4, 0, 0, 4 * 6144))

    record = predictor.predict(query)

    assert len(runner.calls) == 1
    assert runner.calls[0][1].decode_kv == 4096
    assert (record.total_ms, record.compute_ms, record.communication_ms, record.other_ms) == (3.0, 2.0, 1.0, 0.0)
    assert record.match_type == "nearest"
    assert record.operation_lookups == (_operation_lookup(),)


def test_unknown_operation_keeps_total_but_withholds_lossless_components():
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
    )
    runner = _FakeRunner(
        RawAicStep(
            operations=(("generation_layerwise", 2.0), ("future_fused_op", 0.5)),
            sources=(("generation_layerwise", "silicon"), ("future_fused_op", "silicon")),
            operation_lookups=(),
        )
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=runner,
        configuration_provenance='{"repo":"fixture"}',
    )

    record = predictor.predict(_query(SemanticShape(0, 4, 0, 0, 4 * 4096)))

    assert record.total_ms == 2.5
    assert (record.compute_ms, record.communication_ms, record.other_ms, record.component_sum_ms) == (
        None,
        None,
        None,
        None,
    )
    assert (record.classified_operation_count, record.unclassified_operation_count) == (1, 1)


def test_reference_adapter_rejects_blank_source_for_active_operation():
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=ConservativeSurfaceIndex(
            points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
            surface_provenance='{"fixture":"unit"}',
        ),
        runner=_FakeRunner(
            RawAicStep(
                operations=(("generation_layerwise", 2.0),),
                sources=(("generation_layerwise", ""),),
                operation_lookups=(),
            )
        ),
        configuration_provenance='{"repo":"fixture"}',
    )

    with pytest.raises(RuntimeError, match="lack source provenance"):
        predictor.predict(_query(SemanticShape(0, 4, 0, 0, 4 * 4096)))


def test_reference_adapter_rejects_configuration_fingerprint_mismatch_before_runner_call():
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
    )
    runner = _FakeRunner(
        RawAicStep(
            operations=(("generation_layerwise", 2.0),),
            sources=(("generation_layerwise", "silicon"),),
            operation_lookups=(),
        )
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=runner,
        configuration_provenance='{"repo":"fixture"}',
        expected_configuration_fingerprint="different-config-sha",
    )

    with pytest.raises(ContractError) as exc_info:
        predictor.predict(_query(SemanticShape(0, 4, 0, 0, 4 * 4096)))

    assert exc_info.value.process_code == "configuration_mismatch"
    assert runner.calls == []


def test_production_reducer_rejects_artifact_proxy_unless_explicitly_requested():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 4, 0, 0, 4 * 4096),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
        lookup_policy="artifact-proxy-v1",
        phase_axis_lookups=(
            (
                "decode",
                (
                    AxisLookup(
                        axis="max_num_seqs",
                        requested=256,
                        evaluated=64,
                        lower=64,
                        upper=64,
                        weight=0.0,
                        delta=-192,
                        mode="proxy",
                    ),
                ),
            ),
        ),
    )
    runner = _FakeRunner(
        RawAicStep(
            operations=(("generation_layerwise", 2.0),),
            sources=(("generation_layerwise", "silicon"),),
            operation_lookups=(),
        )
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=runner,
        configuration_provenance='{"repo":"fixture"}',
    )

    with pytest.raises(ContractError) as exc_info:
        predict_clean_bins(population, predictor)
    assert exc_info.value.process_code == "predictor_error"

    records = predict_clean_bins(population, predictor, required_lookup_policy="artifact-proxy-v1")
    assert len(records) == 1


def test_missing_lower_level_comm_provenance_returns_expected_unavailable_record():
    bin_ = _bin(
        bin_id="bin-1",
        shape=SemanticShape(0, 4, 0, 0, 4 * 4096),
        clean=True,
        profiled=False,
    )
    population = SemanticPopulation(
        schema_version="fpm-semantic-insights/v1",
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
        samples=bin_.clean_samples,
        bins=(bin_,),
        excluded_reason_counts={},
    )
    index = ConservativeSurfaceIndex(
        points=(_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),),
        surface_provenance='{"fixture":"unit"}',
        phase_axis_lookups=(
            (
                "decode",
                (AxisLookup("max_num_seqs", 256, 256, 256, 256, 0.0, 0, "exact"),),
            ),
        ),
    )
    runner = _FakeRunner(
        RawAicStep(
            operations=(("generation_layerwise", 2.0), ("generation_tp_allreduce", 1.0)),
            sources=(("generation_layerwise", "silicon"), ("generation_tp_allreduce", "silicon")),
            operation_lookups=(),
        )
    )
    predictor = AiconfiguratorSemanticBinPredictor(
        surface_index=index,
        runner=runner,
        configuration_provenance='{"repo":"fixture"}',
    )

    records = predict_clean_bins(population, predictor)
    record = records[0].record

    assert (record.status, record.reason, record.total_ms) == ("unavailable", "missing_surface", None)


_LAYERWISE_FIELDS = (
    "framework",
    "framework_version",
    "system",
    "model",
    "attn_tp",
    "moe_tp",
    "ep",
    "num_slots",
    "gemm_quant",
    "moe_quant",
    "attn_quant",
    "kv_quant",
    "phase",
    "batch_size",
    "new_tokens",
    "past_kv",
    "layer_type",
    "layer_index",
    "measured_layer_count",
    "layer_multiplier",
    "latency_ms",
    "rms_latency_ms",
    "rms_kernel_count",
    "includes_moe",
    "moe_weight_mode",
    "latency_source",
    "physical_gpus",
    "max_num_seqs",
    "max_num_batched_tokens",
    "vllm_config_hash",
)


def _layerwise_row(*, phase: str, max_num_seqs: str, max_num_batched_tokens: str):
    return {
        "framework": "vLLM",
        "framework_version": "0.20.1",
        "system": "h100_sxm",
        "model": "Qwen/Qwen3-32B",
        "attn_tp": "8",
        "moe_tp": "1",
        "ep": "1",
        "num_slots": "",
        "gemm_quant": "bf16",
        "moe_quant": "bf16",
        "attn_quant": "bf16",
        "kv_quant": "bf16",
        "phase": phase,
        "batch_size": "4" if phase == "gen" else "1",
        "new_tokens": "1" if phase == "gen" else "128",
        "past_kv": "4096" if phase == "gen" else "0",
        "layer_type": "dense",
        "layer_index": "0",
        "measured_layer_count": "64",
        "layer_multiplier": "64",
        "latency_ms": "2.5",
        "rms_latency_ms": "0",
        "rms_kernel_count": "0",
        "includes_moe": "False",
        "moe_weight_mode": "dense",
        "latency_source": "execute_model_gpu" if phase == "gen" else "schedule_to_update",
        "physical_gpus": "1",
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "vllm_config_hash": "gen-hash" if phase == "gen" else "ctx-hash",
    }


def _write_layerwise_csv(path, *, gen_batch_size="4"):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_LAYERWISE_FIELDS)
        writer.writeheader()
        writer.writerow(_layerwise_row(phase="ctx", max_num_seqs="", max_num_batched_tokens="40960"))
        gen_row = _layerwise_row(phase="gen", max_num_seqs="64", max_num_batched_tokens="")
        gen_row["batch_size"] = gen_batch_size
        writer.writerow(gen_row)


def _parity_record(**overrides):
    values = {
        "attention_dp_size": 1,
        "attention_quant": "bf16",
        "backend": "vllm",
        "backend_version": "0.20.1",
        "chunked_prefill": True,
        "dp_size": 1,
        "ep_size": 1,
        "gemm_quant": "bf16",
        "gpu_count": 8,
        "kv_cache_dtype": "bf16",
        "kv_cache_quant": "bf16",
        "max_num_batched_tokens": 40960,
        "max_num_seqs": 256,
        "model": "Qwen/Qwen3-32B",
        "model_revision": "fixture",
        "moe_quant": "bf16",
        "numerical_dtype": "bf16",
        "pp_size": 1,
        "prefix_caching": False,
        "runtime_flags": {},
        "schema_version": "aic-runtime-parity/v1",
        "system": "h100_sxm",
        "tp_size": 8,
    }
    values.update(overrides)
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    )


def _repository_config(tmp_path, layerwise_csv, **overrides):
    parity_record = _parity_record()
    values = {
        "repo_root": tmp_path,
        "repo_commit": "repo-commit",
        "configuration_fingerprint": hashlib.sha256(parity_record.encode()).hexdigest(),
        "parity_record": parity_record,
        "layerwise_csv": layerwise_csv,
        "comm_version": "0.19.0",
        "context_vllm_config_hash": "ctx-hash",
        "decode_vllm_config_hash": "gen-hash",
    }
    values.update(overrides)
    return RepositoryAicConfig(**values)


def test_repository_surface_fails_closed_on_scheduler_config_mismatch(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)

    surface = build_repository_surface(_repository_config(tmp_path, layerwise_csv))
    selection = surface.index.select(_query(SemanticShape(0, 4, 0, 0, 4 * 4096)))

    assert surface.index.lookup_policy == "conservative-v1"
    assert (selection.status, selection.reason) == ("unavailable", "missing_surface")
    mns = next(lookup for lookup in selection.axis_lookups if lookup.axis == "max_num_seqs")
    assert (mns.requested, mns.evaluated, mns.mode) == (256, None, "missing")


def test_repository_config_rejects_incoherent_topology_parity(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    config = _repository_config(tmp_path, layerwise_csv)
    bad_parity = _parity_record(gpu_count=4)

    with pytest.raises(ValueError, match="gpu_count == tp_size"):
        replace(
            config,
            parity_record=bad_parity,
            configuration_fingerprint=hashlib.sha256(bad_parity.encode()).hexdigest(),
        )


def test_repository_config_rejects_unexpected_parity_fields(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    parity_record = _parity_record(unexpected_effective_flag=True)

    with pytest.raises(ValueError, match="unexpected fields"):
        _repository_config(
            tmp_path,
            layerwise_csv,
            parity_record=parity_record,
            configuration_fingerprint=hashlib.sha256(parity_record.encode()).hexdigest(),
        )


@pytest.mark.parametrize("proxy_max_num_seqs", [-1, 0, True, 1.5])
def test_repository_config_rejects_invalid_artifact_proxy_axis(
    tmp_path,
    proxy_max_num_seqs,
):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)

    with pytest.raises(ValueError, match="positive integer"):
        _repository_config(
            tmp_path,
            layerwise_csv,
            artifact_proxy_decode_max_num_seqs=proxy_max_num_seqs,
            verify_repository=False,
        )


def test_repository_surface_labels_explicit_cached_artifact_proxy(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)

    surface = build_repository_surface(
        _repository_config(tmp_path, layerwise_csv, artifact_proxy_decode_max_num_seqs=64)
    )
    selection = surface.index.select(_query(SemanticShape(0, 4, 0, 0, 4 * 4096)))

    assert surface.index.lookup_policy == "artifact-proxy-v1"
    assert selection.status == "ok"
    mns = next(lookup for lookup in selection.axis_lookups if lookup.axis == "max_num_seqs")
    assert (mns.requested, mns.evaluated, mns.delta, mns.mode) == (256, 64, -192, "proxy")
    provenance = json.loads(surface.configuration_provenance)
    assert "layerwise_csv_sha256" in provenance
    assert provenance["parity_field_binding"]["provenance_only_unverified"] == [
        "chunked_prefill",
        "model_revision",
        "prefix_caching",
        "runtime_flags",
    ]


def test_unverified_artifact_proxy_excludes_context_surface(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    surface = build_repository_surface(
        _repository_config(
            tmp_path,
            layerwise_csv,
            artifact_proxy_decode_max_num_seqs=64,
            verify_repository=False,
        )
    )

    selection = surface.index.select(_query(SemanticShape(1, 0, 128, 0, 0)))

    assert (selection.status, selection.reason) == ("unavailable", "missing_surface")


@pytest.mark.parametrize(
    ("latency_ms", "detail_json", "error"),
    [
        (float("nan"), '{"latency":1.0}', "finite and non-negative"),
        (1.0, "{}", "non-empty JSON object"),
        (1.0, '{"latency":2.0}', "disagrees"),
    ],
)
def test_layerwise_surface_point_requires_executable_latency_detail(
    latency_ms,
    detail_json,
    error,
):
    with pytest.raises(ValueError, match=error):
        LayerwiseSurfacePoint(
            phase="decode",
            batch_size=4,
            new_tokens=1,
            past_kv=4096,
            row_content_hash="row-sha",
            latency_ms=latency_ms,
            detail_json=detail_json,
        )


def test_repository_surface_rejects_fractional_integer_axis(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv, gen_batch_size="4.5")

    with pytest.raises(ValueError, match="batch_size must be an integer"):
        build_repository_surface(_repository_config(tmp_path, layerwise_csv, artifact_proxy_decode_max_num_seqs=64))


def test_repository_predictor_rejects_unverified_commit(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)

    with pytest.raises(ContractError) as exc_info:
        build_repository_predictor(_repository_config(tmp_path, layerwise_csv))

    assert exc_info.value.process_code == "repository_commit_mismatch"


def test_repository_verification_rejects_execution_module_absent_from_commit(
    tmp_path,
    monkeypatch,
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Semantic FPM Test",
            "-c",
            "user.email=semantic-fpm@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    untracked_module = tmp_path / "semantic_runtime.py"
    untracked_module.write_text("VALUE = 1\n")
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    monkeypatch.setattr(
        semantic_aic,
        "_SEMANTIC_EXECUTION_PATHS",
        ("semantic_runtime.py",),
    )
    config = _repository_config(
        tmp_path,
        layerwise_csv,
        repo_commit=head,
    )

    with pytest.raises(ContractError) as exc_info:
        semantic_aic._verify_repository_checkout(config)

    assert exc_info.value.process_code == "repository_commit_mismatch"
    assert "absent from declared commit" in str(exc_info.value)


def test_import_root_verification_rejects_untracked_module_inside_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Semantic FPM Test",
            "-c",
            "user.email=semantic-fpm@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gap_path = tmp_path / "untracked_gap.py"
    backend_path = tmp_path / "untracked_backend.py"
    gap_path.write_text("VALUE = 1\n")
    backend_path.write_text("VALUE = 2\n")
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    config = _repository_config(tmp_path, layerwise_csv, repo_commit=head)
    gap = types.SimpleNamespace(__file__=str(gap_path))
    backend = types.SimpleNamespace(__file__=str(backend_path))

    with pytest.raises(ContractError) as exc_info:
        _verify_import_root(config, {"vllm_backend": backend}, gap)

    assert exc_info.value.process_code == "repository_commit_mismatch"
    assert "absent from declared commit" in str(exc_info.value)


def test_lower_level_operation_interpolation_is_bounded_and_introspectable():
    surface = {
        128: {"latency": 1.0, "energy": 0.0},
        256: {"latency": 2.0, "energy": 0.0},
    }

    lookup = bounded_operation_lookup(
        operation="custom_allreduce",
        topology={"tp": 8, "strategy": "AUTO"},
        requested=192,
        surface=surface,
    )

    assert (lookup.lower, lookup.upper, lookup.weight, lookup.mode) == (128, 256, 0.5, "interpolate")
    assert len(lookup.surface_content_hash) == 64

    with pytest.raises(AicSurfaceUnavailableError):
        bounded_operation_lookup(
            operation="custom_allreduce",
            topology={"tp": 8, "strategy": "AUTO"},
            requested=512,
            surface=surface,
        )


class _FakeBackendModule:
    _USE_LAYERWISE = False
    _DECODE_COMPUTE_BATCH_CAL = 7.0
    _LAYERWISE_USE_FUSED_ALLREDUCE_RMS = True


class _FakeExactDatabase:
    def __init__(self):
        self.aborted = False

    def begin(self, **kwargs):
        self.aborted = False

    def finish(self):
        return (_operation_lookup(),)

    def abort(self):
        self.aborted = True


class _FakeRepositoryBackend:
    def _get_decode_step_latency(self, *args, **kwargs):
        return (
            {"generation_layerwise": 2.0, "generation_tp_allreduce": 1.0},
            {},
            {"generation_layerwise": "silicon", "generation_tp_allreduce": "silicon"},
        )


def test_repository_runner_restores_backend_flags_and_maps_lookup_consumer(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    config = _repository_config(
        tmp_path,
        layerwise_csv,
        artifact_proxy_decode_max_num_seqs=64,
    )
    module = _FakeBackendModule()
    database = _FakeExactDatabase()
    runner = RepositoryAicStepRunner(
        config=config,
        backend=_FakeRepositoryBackend(),
        model=object(),
        database=database,
        runtime_config=object(),
        vllm_backend_module=module,
        use_fused_allreduce_rms=False,
    )
    previous = (
        module._USE_LAYERWISE,
        module._DECODE_COMPUTE_BATCH_CAL,
        module._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
    )

    raw = runner.predict(
        phase="decode",
        shape=AicQueryShape(0, 4, 0, 0, 4096),
        point=_point(phase="decode", batch=4, new_tokens=1, past_kv=4096),
    )

    assert raw.operation_lookups[0].consumer_operations == ("generation_tp_allreduce",)
    assert previous == (
        module._USE_LAYERWISE,
        module._DECODE_COMPUTE_BATCH_CAL,
        module._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
    )


def test_exact_scheduler_database_rejects_any_fallback_query(tmp_path):
    layerwise_csv = tmp_path / "layerwise.csv"
    _write_layerwise_csv(layerwise_csv)
    config = _repository_config(
        tmp_path,
        layerwise_csv,
        artifact_proxy_decode_max_num_seqs=64,
    )

    class Inner:
        backend = "vllm"
        system = "h100_sxm"
        version = "0.19.0"

        def __init__(self):
            self.layerwise = {}
            self.system_spec = {}

    database = _ExactSchedulerDatabase(Inner(), config)
    point = LayerwiseSurfacePoint(
        phase="decode",
        batch_size=4,
        new_tokens=1,
        past_kv=4096,
        row_content_hash="row",
        latency_ms=2.5,
        detail_json='{"latency":2.5}',
    )
    database.begin(phase="decode", shape=AicQueryShape(0, 4, 0, 0, 4096), point=point)

    detail = database.query_layerwise_detail(
        "Qwen/Qwen3-32B",
        "GEN",
        8,
        4,
        4096,
        max_num_seqs=256,
        moe_tp_size=1,
        moe_ep_size=1,
    )
    assert detail["latency"] == 2.5
    with pytest.raises(RuntimeError, match=r"more than one|fallback"):
        database.query_layerwise_detail(
            "Qwen/Qwen3-32B",
            "GEN",
            8,
            2,
            4096,
            max_num_seqs=256,
            moe_tp_size=1,
            moe_ep_size=1,
        )
    database.abort()


def test_repository_predictor_executes_exact_leaf_and_restores_backend_globals():
    pytest.importorskip("pyarrow", reason="repository parquet runtime is optional in the unit-test environment")
    repo_root = Path(__file__).resolve().parents[1]
    parity_record = _parity_record(max_num_batched_tokens=2048)
    config = RepositoryAicConfig(
        repo_root=repo_root,
        repo_commit="unverified-artifact-proxy",
        configuration_fingerprint=hashlib.sha256(parity_record.encode()).hexdigest(),
        parity_record=parity_record,
        layerwise_csv=repo_root / "src/aiconfigurator/systems/data/h100_sxm/vllm/0.20.1/layerwise_perf.csv",
        comm_version="0.19.0",
        context_vllm_config_hash="6952ce31bf653b1a",
        decode_vllm_config_hash="4166d029102c3ecd",
        artifact_proxy_decode_max_num_seqs=128,
        verify_repository=False,
    )
    from aiconfigurator.sdk.backends import vllm_backend
    from collector.layerwise.diagnostics import aic_fpm_gap

    previous = (
        vllm_backend._USE_LAYERWISE,
        vllm_backend._DECODE_COMPUTE_BATCH_CAL,
        vllm_backend._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
        aic_fpm_gap.MODEL_NAME,
    )
    predictor = build_repository_predictor(config)
    assert previous[3] == aic_fpm_gap.MODEL_NAME

    record = predictor.predict(
        build_semantic_query(
            configuration_fingerprint=config.configuration_fingerprint,
            concurrency=16,
            shape=SemanticShape(0, 16, 0, 0, 16 * 8192),
        )
    )

    assert record.status == "ok"
    assert record.total_ms is not None and record.total_ms > 0.0
    assert record.operation_lookups[0].consumer_operations == ("generation_tp_allreduce",)
    assert previous == (
        vllm_backend._USE_LAYERWISE,
        vllm_backend._DECODE_COMPUTE_BATCH_CAL,
        vllm_backend._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
        aic_fpm_gap.MODEL_NAME,
    )
