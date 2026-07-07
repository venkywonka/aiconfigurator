# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from collector.layerwise.diagnostics import semantic_fpm_contract as contract
from collector.layerwise.diagnostics.semantic_fpm_aic import RepositoryAicConfig
from collector.layerwise.diagnostics.semantic_fpm_contract import (
    AlignmentMarkerIdentity,
    AlignmentSegmentProvenance,
    CohortAlignmentProvenance,
    ContractSourceMetadata,
    ProfiledStepComposition,
    RankStepComposition,
    reduce_and_write_semantic_contract,
)

AIC_COMMIT = "a" * 40
AUTO_COLLECTOR_COMMIT = "b" * 40
MAPPING_HASH = "c" * 64
RANK_KEYS = tuple(str(rank) for rank in range(8))
EXPECTED_SAMPLE_FIELDS = (
    "schema_version",
    "configuration_fingerprint",
    "concurrency",
    "phase",
    "lane",
    "sample_id",
    "workload_segment",
    "counter_id",
    "worker_id",
    "dp_rank",
    "ctx_requests",
    "decode_requests",
    "ctx_new_tokens",
    "ctx_kv_tokens",
    "decode_kv_tokens",
    "ctx_new_per_request",
    "ctx_kv_per_request",
    "decode_kv_per_request",
    "semantic_key",
    "bin_id",
    "wall_ms",
    "gpu_compute_ms",
    "gpu_comm_ms",
    "gpu_busy_ms",
    "rank_key",
    "rank_identity_kind",
    "rank_mapping_provenance",
    "captured_rank_key_count",
    "captured_rank_keys_json",
    "rank_compositions_json",
    "rank_busy_min_ms",
    "rank_busy_max_ms",
    "imbalance_ratio",
    "alignment_mapping_hash",
    "kernel_classifier_version",
    "unknown_kernel_count",
    "unknown_kernel_duration_ms",
    "unknown_kernel_name_hash",
    "same_run_mapping_status",
    "analytic_eligible",
    "sample_eligibility_reason",
    "semantic_support_status",
    "overall_gap_eligible",
    "decomposition_eligible",
    "aic_status",
    "eligibility_reason",
)
EXPECTED_BIN_FIELDS = (
    "schema_version",
    "bin_id",
    "configuration_fingerprint",
    "concurrency",
    "phase",
    "semantic_key",
    "ctx_requests",
    "decode_requests",
    "ctx_new_per_request",
    "ctx_kv_per_request",
    "decode_kv_per_request",
    "n_clean",
    "n_profiled",
    "structural_status",
    "clean_row_mass_weight",
    "profiled_row_mass_weight",
    "all_clean",
    "overall_gap_eligible",
    "nsight_shared",
    "decomposition_eligible",
    "eligibility_reason",
    "decomposition_support_count",
    "decomposition_support_class",
    "semantic_query_json",
    "aic_prediction_json",
    "aic_status",
    "aic_reason",
    "aic_total_ms",
    "aic_compute_ms",
    "aic_comm_ms",
    "aic_other_ms",
    "clean_wall_ms_count",
    "clean_wall_ms_support_class",
    "clean_wall_ms_band_kind",
    "clean_wall_ms_median",
    "clean_wall_ms_minimum",
    "clean_wall_ms_q1",
    "clean_wall_ms_q3",
    "clean_wall_ms_maximum",
    "profiled_wall_ms_count",
    "profiled_wall_ms_support_class",
    "profiled_wall_ms_band_kind",
    "profiled_wall_ms_median",
    "profiled_wall_ms_minimum",
    "profiled_wall_ms_q1",
    "profiled_wall_ms_q3",
    "profiled_wall_ms_maximum",
    "gpu_compute_ms_count",
    "gpu_compute_ms_support_class",
    "gpu_compute_ms_band_kind",
    "gpu_compute_ms_median",
    "gpu_compute_ms_minimum",
    "gpu_compute_ms_q1",
    "gpu_compute_ms_q3",
    "gpu_compute_ms_maximum",
    "gpu_comm_ms_count",
    "gpu_comm_ms_support_class",
    "gpu_comm_ms_band_kind",
    "gpu_comm_ms_median",
    "gpu_comm_ms_minimum",
    "gpu_comm_ms_q1",
    "gpu_comm_ms_q3",
    "gpu_comm_ms_maximum",
    "gpu_busy_ms_count",
    "gpu_busy_ms_support_class",
    "gpu_busy_ms_band_kind",
    "gpu_busy_ms_median",
    "gpu_busy_ms_minimum",
    "gpu_busy_ms_q1",
    "gpu_busy_ms_q3",
    "gpu_busy_ms_maximum",
    "clean_raw_shape_cardinality",
    "clean_ctx_requests_raw_min",
    "clean_ctx_requests_raw_max",
    "clean_decode_requests_raw_min",
    "clean_decode_requests_raw_max",
    "clean_ctx_new_tokens_raw_min",
    "clean_ctx_new_tokens_raw_max",
    "clean_ctx_kv_tokens_raw_min",
    "clean_ctx_kv_tokens_raw_max",
    "clean_decode_kv_tokens_raw_min",
    "clean_decode_kv_tokens_raw_max",
    "clean_ctx_new_tokens_query_delta_min",
    "clean_ctx_new_tokens_query_delta_max",
    "clean_ctx_kv_tokens_query_delta_min",
    "clean_ctx_kv_tokens_query_delta_max",
    "clean_decode_kv_tokens_query_delta_min",
    "clean_decode_kv_tokens_query_delta_max",
    "profiled_raw_shape_cardinality",
    "profiled_ctx_requests_raw_min",
    "profiled_ctx_requests_raw_max",
    "profiled_decode_requests_raw_min",
    "profiled_decode_requests_raw_max",
    "profiled_ctx_new_tokens_raw_min",
    "profiled_ctx_new_tokens_raw_max",
    "profiled_ctx_kv_tokens_raw_min",
    "profiled_ctx_kv_tokens_raw_max",
    "profiled_decode_kv_tokens_raw_min",
    "profiled_decode_kv_tokens_raw_max",
    "profiled_ctx_new_tokens_query_delta_min",
    "profiled_ctx_new_tokens_query_delta_max",
    "profiled_ctx_kv_tokens_query_delta_min",
    "profiled_ctx_kv_tokens_query_delta_max",
    "profiled_decode_kv_tokens_query_delta_min",
    "profiled_decode_kv_tokens_query_delta_max",
    "overall_gap_signed_ms",
    "overall_gap_abs_ms",
    "relative_error_signed",
    "relative_error_abs",
    "gpu_concurrency_ms",
    "profiled_overhead_ms",
    "profile_wall_delta_ms",
    "term_compute_err_ms",
    "term_comm_err_ms",
    "term_aic_other_ms",
    "term_gpu_concurrency_ms",
    "term_neg_profiled_overhead_ms",
    "term_profile_wall_delta_ms",
    "decomposition_closure_error_ms",
)
from collector.layerwise.diagnostics.semantic_fpm_insights import (
    AicPredictionRecord,
    AicQueryShape,
    AxisLookup,
    OperationLookup,
    SemanticShape,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    FpmSample,
    build_semantic_population,
)


def _sample(
    *,
    lane: str,
    sample_id: str,
    shape: SemanticShape,
    wall_ms: float,
) -> FpmSample:
    return FpmSample(
        lane=lane,
        concurrency=16,
        sample_id=sample_id,
        phase=shape.validate(),
        workload_segment="real",
        counter_id=int(sample_id.rsplit("-", 1)[-1]),
        worker_id=f"{lane}-worker",
        dp_rank=0,
        shape=shape,
        wall_ms=wall_ms,
    )


def _population():
    shared = SemanticShape(0, 2, 0, 0, 200)
    clean_only = SemanticShape(0, 1, 0, 0, 300)
    profiled_only = SemanticShape(0, 4, 0, 0, 1600)
    return build_semantic_population(
        (
            _sample(lane="clean", sample_id="clean-1", shape=shared, wall_ms=10.0),
            _sample(lane="profiled", sample_id="profiled-2", shape=shared, wall_ms=12.0),
            _sample(lane="clean", sample_id="clean-3", shape=clean_only, wall_ms=7.0),
            _sample(lane="profiled", sample_id="profiled-4", shape=profiled_only, wall_ms=14.0),
        ),
        configuration_fingerprint="config-sha",
        measured_segments=frozenset({"real"}),
    )


def _inventory_hash(inventory) -> str:
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _record(query, *, available: bool) -> AicPredictionRecord:
    shape = AicQueryShape.from_query(query)
    config_lookup = AxisLookup("max_num_seqs", 256, 256, 256, 256, 0.0, 0, "exact")
    if not available:
        axes = (config_lookup,) + tuple(
            AxisLookup(axis, requested, None, None, None, None, None, "missing") for axis, requested in shape.items()
        )
        return AicPredictionRecord(
            configuration_fingerprint=query.configuration_fingerprint,
            concurrency=query.concurrency,
            phase=query.phase,
            semantic_key=query.semantic_key,
            status="unavailable",
            reason="missing_surface",
            requested_shape=shape,
            evaluated_shape=None,
            lookup_policy="conservative-v1",
            lookup_surface_id=None,
            scheduler_surface_content_hash=None,
            axis_lookups=axes,
            total_ms=None,
            total_basis=None,
            compute_ms=None,
            communication_ms=None,
            other_ms=None,
            component_sum_ms=None,
            source=None,
            match_type=None,
            predictor_version="fixture-v1",
            api_version="fixture-api-v1",
            component_classifier_version="exact-registry-v1",
            classified_operation_count=0,
            unclassified_operation_count=0,
            operation_inventory=(),
            operation_inventory_hash=_inventory_hash([]),
            operation_values=(),
            operation_lookups=(),
            configuration_provenance='{"fixture":true}',
        )

    axes = (config_lookup,) + tuple(
        AxisLookup(axis, requested, requested, requested, requested, 0.0, 0, "exact")
        for axis, requested in shape.items()
    )
    inventory = (
        ("generation_layerwise", "compute"),
        ("generation_tp_allreduce", "communication"),
    )
    return AicPredictionRecord(
        configuration_fingerprint=query.configuration_fingerprint,
        concurrency=query.concurrency,
        phase=query.phase,
        semantic_key=query.semantic_key,
        status="ok",
        reason="predicted",
        requested_shape=shape,
        evaluated_shape=shape,
        lookup_policy="conservative-v1",
        lookup_surface_id="surface-id",
        scheduler_surface_content_hash="surface-content-sha",
        axis_lookups=axes,
        total_ms=8.0,
        total_basis="operation_sum",
        compute_ms=6.0,
        communication_ms=2.0,
        other_ms=0.0,
        component_sum_ms=8.0,
        source="silicon",
        match_type="exact",
        predictor_version="fixture-v1",
        api_version="fixture-api-v1",
        component_classifier_version="exact-registry-v1",
        classified_operation_count=2,
        unclassified_operation_count=0,
        operation_inventory=inventory,
        operation_inventory_hash=_inventory_hash(inventory),
        operation_values=(
            ("generation_layerwise", 6.0, "silicon"),
            ("generation_tp_allreduce", 2.0, "silicon"),
        ),
        operation_lookups=(
            OperationLookup(
                operation="custom_allreduce",
                topology='{"tp":8}',
                requested=100,
                lower=100,
                upper=100,
                weight=0.0,
                mode="exact",
                surface_content_hash="a" * 64,
                consumer_operations=("generation_tp_allreduce",),
            ),
        ),
        configuration_provenance='{"fixture":true}',
    )


class _SpyPredictor:
    def __init__(self):
        self.queries = []

    def predict(self, query):
        self.queries.append(query)
        return _record(query, available=query.decode_kv_per_request == 100)


def _repository_config(tmp_path: Path) -> RepositoryAicConfig:
    parity = json.dumps(
        {
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
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return RepositoryAicConfig(
        repo_root=tmp_path,
        repo_commit=AIC_COMMIT,
        configuration_fingerprint=hashlib.sha256(parity.encode()).hexdigest(),
        parity_record=parity,
        layerwise_csv=tmp_path / "layerwise.csv",
        comm_version="0.19.0",
        context_vllm_config_hash="ctx-hash",
        decode_vllm_config_hash="decode-hash",
    )


def _with_config_fingerprint(population, fingerprint):
    samples = tuple(population.samples)
    return build_semantic_population(
        samples,
        configuration_fingerprint=fingerprint,
        measured_segments=population.measured_segments,
    )


def _compositions():
    busy_by_rank = tuple((rank, 6.0 if rank == "0" else 5.5) for rank in RANK_KEYS)
    common = {
        "gpu_compute_ms": 5.0,
        "gpu_comm_ms": 1.5,
        "gpu_busy_ms": 6.0,
        "rank_key": "rank-0",
        "rank_identity_kind": "logical_rank",
        "rank_mapping_provenance": "launcher-rank-map-v1",
        "captured_rank_keys": RANK_KEYS,
        "busy_by_rank_key_ms": busy_by_rank,
        "rank_compositions": tuple(
            RankStepComposition(
                rank_key=rank,
                gpu_compute_ms=5.0,
                gpu_comm_ms=1.5,
                gpu_busy_ms=busy,
            )
            for rank, busy in busy_by_rank
        ),
        "alignment_mapping_hash": MAPPING_HASH,
        "kernel_classifier_version": "nsys-exact-registry-v1",
    }
    common["rank_key"] = "0"
    return {
        "profiled-2": ProfiledStepComposition(sample_id="profiled-2", **common),
        "profiled-4": ProfiledStepComposition(sample_id="profiled-4", **common),
    }


def _source_metadata() -> ContractSourceMetadata:
    return ContractSourceMetadata(
        job_id=354281415,
        pipeline_id=56729332,
        model="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend="vllm",
        backend_version="0.20.1",
        nsight_kernel_classifier_version="nsys-exact-registry-v1",
        alignments=(
            CohortAlignmentProvenance(
                concurrency=16,
                input_profiled_rows=2,
                mapped_profiled_rows=2,
                raw_marker_count=2,
                canonical_marker_count=2,
                mapping_hash=MAPPING_HASH,
                expected_rank_keys=RANK_KEYS,
                captured_rank_keys=RANK_KEYS,
                rank_identity_kind="logical_rank",
                rank_mapping_provenance="launcher-rank-map-v1",
                nonmonotonic_marker_identities=(),
                unmatched_prefix_marker_identities=(),
                unmatched_suffix_marker_identities=(),
                internal_skipped_marker_identities=(),
                mapped_segments=(
                    AlignmentSegmentProvenance(
                        measure_run=0,
                        start_marker_canonical_index=0,
                        end_marker_canonical_index=1,
                        start_marker_step=10,
                        end_marker_step=11,
                        mapped_rows=2,
                    ),
                ),
            ),
        ),
    )


def _read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_shipping_reducer_calls_predictor_once_per_clean_bin_and_emits_atomic_contract(
    tmp_path,
    monkeypatch,
):
    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    spy = _SpyPredictor()
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: spy)

    first = reduce_and_write_semantic_contract(
        population=population,
        repository_config=repository_config,
        compositions=_compositions(),
        output_dir=tmp_path / "semantic_insights",
        aiconfigurator_commit=AIC_COMMIT,
        auto_collector_commit=AUTO_COLLECTOR_COMMIT,
        source_metadata=_source_metadata(),
    )

    assert len(spy.queries) == 2
    assert {path.name for path in first.root.iterdir()} == {"samples.csv", "bins.csv", "manifest.json"}
    samples = _read_csv(first.samples_csv)
    bins = _read_csv(first.bins_csv)
    manifest = json.loads(first.manifest_json.read_text())
    assert len(samples) == 4
    assert len(bins) == 3
    assert tuple(samples[0]) == EXPECTED_SAMPLE_FIELDS
    assert tuple(bins[0]) == EXPECTED_BIN_FIELDS
    assert set(manifest) == {
        "aiconfigurator_commit",
        "alignment",
        "auto_collector_commit",
        "classifiers",
        "concurrencies",
        "configuration_fingerprint",
        "counts",
        "coverage_by_concurrency_phase",
        "files",
        "lookup_policy",
        "measured_segments",
        "nsight_provenance",
        "parity_record",
        "predictor_provenance",
        "schema_version",
        "shape_audit",
        "source_metadata",
        "validation",
        "writer_version",
    }
    assert manifest["counts"]["predictor_calls"] == 2
    assert manifest["counts"]["orthogonal_status_counts"] == {
        "aic_components_unavailable": 0,
        "aic_total_unavailable": 1,
        "clean_only_bin": 1,
        "nsight_components_unavailable": 0,
        "profiled_only_bin": 1,
    }
    assert manifest["source_metadata"]["query_reconstruction_version"] == (contract.QUERY_RECONSTRUCTION_VERSION)
    assert manifest["alignment"]["c16"]["mapped_segments"] == [
        {
            "end_marker_canonical_index": 1,
            "end_marker_step": 11,
            "mapped_rows": 2,
            "measure_run": 0,
            "start_marker_canonical_index": 0,
            "start_marker_step": 10,
        }
    ]
    assert manifest["files"]["samples.csv"]["sha256"] == hashlib.sha256(first.samples_csv.read_bytes()).hexdigest()
    assert manifest["files"]["bins.csv"]["sha256"] == hashlib.sha256(first.bins_csv.read_bytes()).hexdigest()

    shared = next(row for row in bins if row["n_clean"] == "1" and row["n_profiled"] == "1")
    assert shared["decomposition_eligible"] == "True"
    assert float(shared["decomposition_closure_error_ms"]) == pytest.approx(0.0)
    clean_only = next(row for row in bins if row["eligibility_reason"] == "clean_only_bin")
    assert clean_only["aic_status"] == "unavailable"
    assert clean_only["aic_total_ms"] == ""
    profiled_only = next(row for row in bins if row["eligibility_reason"] == "profiled_only_bin")
    assert profiled_only["aic_status"] == ""

    spy.queries.clear()
    second = reduce_and_write_semantic_contract(
        population=population,
        repository_config=repository_config,
        compositions=_compositions(),
        output_dir=tmp_path / "semantic_insights_second",
        aiconfigurator_commit=AIC_COMMIT,
        auto_collector_commit=AUTO_COLLECTOR_COMMIT,
        source_metadata=_source_metadata(),
    )
    assert first.samples_csv.read_bytes() == second.samples_csv.read_bytes()
    assert first.bins_csv.read_bytes() == second.bins_csv.read_bytes()
    assert first.manifest_json.read_bytes() == second.manifest_json.read_bytes()


def test_shipping_reducer_fails_before_publication_on_incomplete_profiled_mapping(
    tmp_path,
    monkeypatch,
):
    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    spy = _SpyPredictor()
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: spy)
    output = tmp_path / "semantic_insights"

    with pytest.raises(contract.ContractError) as exc_info:
        reduce_and_write_semantic_contract(
            population=population,
            repository_config=repository_config,
            compositions={"profiled-2": _compositions()["profiled-2"]},
            output_dir=output,
            aiconfigurator_commit=AIC_COMMIT,
            auto_collector_commit=AUTO_COLLECTOR_COMMIT,
            source_metadata=_source_metadata(),
        )

    assert exc_info.value.process_code == "same_run_alignment_missing"
    assert not output.exists()
    assert not tuple(tmp_path.glob(".semantic_insights.tmp-*"))
    assert spy.queries == []


def test_atomic_writer_removes_temporary_tree_when_rename_fails(tmp_path, monkeypatch):
    output = tmp_path / "semantic_insights"

    def fail_replace(source, target):
        raise OSError("injected rename failure")

    monkeypatch.setattr(contract, "_rename_noreplace", fail_replace)
    with pytest.raises(OSError, match="injected rename failure"):
        contract._write_atomic_bundle(
            output,
            samples_payload=b"samples\n",
            bins_payload=b"bins\n",
            manifest_payload=b"{}\n",
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".semantic_insights.tmp-*"))


def test_atomic_writer_does_not_clobber_a_concurrent_winner(tmp_path, monkeypatch):
    output = tmp_path / "semantic_insights"
    original = contract._rename_noreplace

    def publish_winner_then_rename(source, target):
        target.mkdir()
        (target / "sentinel").write_text("winner")
        original(source, target)

    monkeypatch.setattr(contract, "_rename_noreplace", publish_winner_then_rename)
    with pytest.raises(FileExistsError):
        contract._write_atomic_bundle(
            output,
            samples_payload=b"samples\n",
            bins_payload=b"bins\n",
            manifest_payload=b"{}\n",
        )

    assert (output / "sentinel").read_text() == "winner"
    assert not tuple(tmp_path.glob(".semantic_insights.tmp-*"))


@pytest.mark.parametrize(
    ("population_schema", "aic_commit", "expected_code"),
    (
        ("semantic-fpm/v0", AIC_COMMIT, "schema_incompatible"),
        (contract.SCHEMA_VERSION, "f" * 40, "repository_commit_mismatch"),
    ),
)
def test_shipping_reducer_fails_before_prediction_on_identity_mismatch(
    tmp_path,
    monkeypatch,
    population_schema,
    aic_commit,
    expected_code,
):
    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    population = replace(population, schema_version=population_schema)
    spy = _SpyPredictor()
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: spy)
    output = tmp_path / "semantic_insights"

    with pytest.raises(contract.ContractError) as exc_info:
        reduce_and_write_semantic_contract(
            population=population,
            repository_config=repository_config,
            compositions=_compositions(),
            output_dir=output,
            aiconfigurator_commit=aic_commit,
            auto_collector_commit=AUTO_COLLECTOR_COMMIT,
            source_metadata=_source_metadata(),
        )

    assert exc_info.value.process_code == expected_code
    assert spy.queries == []
    assert not output.exists()


def test_profiled_composition_enforces_rank_classifier_and_component_invariants(tmp_path):
    composition = _compositions()["profiled-2"]
    with pytest.raises(contract.ContractError) as exc_info:
        replace(composition, rank_key="1")
    assert exc_info.value.process_code == "rank_tuple_identity_mismatch"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(composition, gpu_compute_ms=3.0, gpu_comm_ms=1.0)
    assert exc_info.value.process_code == "rank_tuple_identity_mismatch"

    forged_selected = replace(
        composition.rank_compositions[0],
        gpu_compute_ms=4.0,
        gpu_comm_ms=2.0,
    )
    with pytest.raises(contract.ContractError) as exc_info:
        replace(
            composition,
            rank_compositions=(forged_selected, *composition.rank_compositions[1:]),
        )
    assert exc_info.value.process_code == "rank_tuple_identity_mismatch"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(
            composition,
            unknown_kernel_count=1,
            unknown_kernel_duration_ms=0.5,
            unknown_kernel_name_hash="d" * 64,
        )
    assert exc_info.value.process_code == "rank_tuple_identity_mismatch"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(composition, gpu_busy_ms=True)
    assert exc_info.value.process_code == "invalid_measurement"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(
            composition,
            gpu_compute_ms=None,
            gpu_comm_ms=None,
            unknown_kernel_count=1.5,
            unknown_kernel_duration_ms=0.5,
            unknown_kernel_name_hash="d" * 64,
        )
    assert exc_info.value.process_code == "invalid_measurement"

    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    compositions = _compositions()
    compositions["profiled-2"] = replace(
        composition,
        kernel_classifier_version="wrong-classifier-v1",
    )
    with pytest.raises(contract.ContractError) as exc_info:
        reduce_and_write_semantic_contract(
            population=population,
            repository_config=repository_config,
            compositions=compositions,
            output_dir=tmp_path / "semantic_insights",
            aiconfigurator_commit=AIC_COMMIT,
            auto_collector_commit=AUTO_COLLECTOR_COMMIT,
            source_metadata=_source_metadata(),
        )
    assert exc_info.value.process_code == "schema_incompatible"


def test_every_measured_profiled_row_including_idle_requires_same_run_mapping(
    tmp_path,
    monkeypatch,
):
    base = _population()
    idle = _sample(
        lane="profiled",
        sample_id="profiled-5",
        shape=SemanticShape(0, 0, 0, 0, 0),
        wall_ms=1.0,
    )
    population = build_semantic_population(
        (*base.samples, idle),
        configuration_fingerprint=_repository_config(tmp_path).configuration_fingerprint,
        measured_segments=frozenset({"real"}),
    )
    source = _source_metadata()
    source = replace(
        source,
        alignments=(
            replace(
                source.alignments[0],
                input_profiled_rows=3,
                mapped_profiled_rows=3,
                raw_marker_count=3,
                canonical_marker_count=3,
                mapped_segments=(
                    AlignmentSegmentProvenance(
                        measure_run=0,
                        start_marker_canonical_index=0,
                        end_marker_canonical_index=2,
                        start_marker_step=10,
                        end_marker_step=12,
                        mapped_rows=3,
                    ),
                ),
            ),
        ),
    )
    spy = _SpyPredictor()
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: spy)

    with pytest.raises(contract.ContractError) as exc_info:
        reduce_and_write_semantic_contract(
            population=population,
            repository_config=_repository_config(tmp_path),
            compositions=_compositions(),
            output_dir=tmp_path / "semantic_insights",
            aiconfigurator_commit=AIC_COMMIT,
            auto_collector_commit=AUTO_COLLECTOR_COMMIT,
            source_metadata=source,
        )

    assert exc_info.value.process_code == "same_run_alignment_missing"
    assert spy.queries == []


def test_direct_total_basis_is_a_supported_predictor_contract(tmp_path, monkeypatch):
    class DirectPredictor(_SpyPredictor):
        def predict(self, query):
            self.queries.append(query)
            return replace(_record(query, available=True), total_basis="direct")

    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    predictor = DirectPredictor()
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: predictor)
    bundle = reduce_and_write_semantic_contract(
        population=population,
        repository_config=repository_config,
        compositions=_compositions(),
        output_dir=tmp_path / "semantic_insights",
        aiconfigurator_commit=AIC_COMMIT,
        auto_collector_commit=AUTO_COLLECTOR_COMMIT,
        source_metadata=_source_metadata(),
    )

    manifest = json.loads(bundle.manifest_json.read_text())
    assert manifest["predictor_provenance"]["total_bases"] == ["direct"]


def test_alignment_detail_must_reconcile_mapped_rows():
    with pytest.raises(contract.ContractError) as exc_info:
        replace(
            _source_metadata().alignments[0],
            mapped_segments=(replace(_source_metadata().alignments[0].mapped_segments[0], mapped_rows=1),),
        )
    assert exc_info.value.process_code == "same_run_alignment_missing"

    with pytest.raises(contract.ContractError) as exc_info:
        AlignmentSegmentProvenance(
            measure_run=0,
            start_marker_canonical_index=5,
            end_marker_canonical_index=4,
            start_marker_step=10,
            end_marker_step=9,
            mapped_rows=2,
        )
    assert exc_info.value.process_code == "same_run_alignment_missing"

    first = replace(
        _source_metadata().alignments[0].mapped_segments[0],
        end_marker_canonical_index=0,
        end_marker_step=10,
        mapped_rows=1,
    )
    second = AlignmentSegmentProvenance(
        measure_run=0,
        start_marker_canonical_index=2,
        end_marker_canonical_index=2,
        start_marker_step=12,
        end_marker_step=12,
        mapped_rows=1,
    )
    with pytest.raises(contract.ContractError) as exc_info:
        replace(_source_metadata().alignments[0], mapped_segments=(first, second))
    assert exc_info.value.process_code == "same_run_alignment_missing"

    shifted = replace(
        _source_metadata().alignments[0].mapped_segments[0],
        start_marker_canonical_index=5,
        end_marker_canonical_index=6,
    )
    with pytest.raises(contract.ContractError) as exc_info:
        replace(
            _source_metadata().alignments[0],
            raw_marker_count=7,
            canonical_marker_count=7,
            unmatched_suffix_marker_identities=tuple(AlignmentMarkerIdentity(index, 0, 0, 0) for index in range(5)),
            mapped_segments=(shifted,),
        )
    assert exc_info.value.process_code == "same_run_alignment_missing"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(_source_metadata().alignments[0], concurrency=16.5)
    assert exc_info.value.process_code == "invalid_measurement"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(_source_metadata().alignments[0], input_profiled_rows=2.0)
    assert exc_info.value.process_code == "invalid_measurement"

    with pytest.raises(contract.ContractError) as exc_info:
        AlignmentMarkerIdentity(
            marker_step=1,
            measure_run=2,
            decode_batch=3,
            mean_decode_kv=4.5,
        )
    assert exc_info.value.process_code == "same_run_alignment_ambiguous"

    with pytest.raises(contract.ContractError) as exc_info:
        replace(_source_metadata(), job_id=1.5)
    assert exc_info.value.process_code == "invalid_measurement"


def test_named_marker_identity_has_unambiguous_manifest_serialization(tmp_path, monkeypatch):
    repository_config = _repository_config(tmp_path)
    population = _with_config_fingerprint(_population(), repository_config.configuration_fingerprint)
    source = _source_metadata()
    source = replace(
        source,
        alignments=(
            replace(
                source.alignments[0],
                raw_marker_count=3,
                nonmonotonic_marker_identities=(
                    AlignmentMarkerIdentity(
                        marker_step=12,
                        measure_run=3,
                        decode_batch=7,
                        mean_decode_kv=4096,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(contract, "build_repository_predictor", lambda config: _SpyPredictor())
    bundle = reduce_and_write_semantic_contract(
        population=population,
        repository_config=repository_config,
        compositions=_compositions(),
        output_dir=tmp_path / "semantic_insights",
        aiconfigurator_commit=AIC_COMMIT,
        auto_collector_commit=AUTO_COLLECTOR_COMMIT,
        source_metadata=source,
    )

    marker = json.loads(bundle.manifest_json.read_text())["alignment"]["c16"]["nonmonotonic_marker_identities"][0]
    assert marker == {
        "decode_batch": 7,
        "marker_step": 12,
        "mean_decode_kv": 4096,
        "measure_run": 3,
    }
