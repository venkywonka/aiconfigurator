# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionFailed, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
)

pytestmark = pytest.mark.unit


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="microbench-v1",
        warmups=3,
        samples=1,
        statistic="median",
        timer="cuda_event",
        tuning_revision="none",
    )


def _request(*, op_id: str, m: int = 8) -> MeasurementRequest:
    environment = MeasurementEnvironment(
        system="h100_sxm",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="h100",
        runtime_versions={"cuda": "12.9"},
    )
    query = {"m": m, "n": 16, "k": 32}
    return MeasurementRequest(
        op_id=op_id,
        key=PerfKey.build("gemm/v1", query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"dtype": "bf16"},
        protocol=_protocol(),
    )


def _record(
    request: MeasurementRequest,
    *,
    latency_ms: float | None,
    status: RecordStatus = RecordStatus.VALID,
    failure_code: UnresolvedCode | None = None,
) -> MeasurementRecord:
    return MeasurementRecord(
        key=request.key,
        status=status,
        latency_ms=latency_ms,
        energy_wms=0.25 if status is RecordStatus.VALID else 0.0,
        samples_ms=(latency_ms,) if latency_ms is not None else (),
        protocol=request.protocol,
        perf_row={"latency": latency_ms} if latency_ms is not None else {},
        provenance={"collector_revision": "r1", "gpu_uuid": "GPU-0"},
        failure_code=failure_code,
        failure_reason=f"injected {failure_code.value}" if failure_code is not None else None,
    )


class _Executor:
    def __init__(self, records: Sequence[MeasurementRecord]) -> None:
        self.records = tuple(records)
        self.request_batches: list[tuple[MeasurementRequest, ...]] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        del deadline_monotonic, cancellation
        self.request_batches.append(tuple(requests))
        return self.records


def _session(tmp_path, executor: _Executor) -> ResolutionSession:
    return ResolutionSession(
        OverlayStore(tmp_path / "overlay.sqlite"),
        executor,
        ResolutionBudget(max_new_keys=8, max_wall_seconds=30.0),
        _protocol(),
    )


def test_report_captures_deduplicated_key_consumers_sources_record_and_replay(tmp_path) -> None:
    first = _request(op_id="layer.0.qkv")
    second = _request(op_id="layer.1.qkv")
    measured = _record(first, latency_ms=1.25)
    executor = _Executor((measured,))
    session = _session(tmp_path, executor)

    def query() -> float:
        total = 0.0
        for request in (first, second):
            record = session.lookup(request.key, request.protocol, consumer=request.op_id)
            if record is None:
                session.record_miss(request, request.op_id)
            else:
                assert record.latency_ms is not None
                total += record.latency_ms
        return total

    assert session.execute_callback(query) == pytest.approx(2.5)

    payload = session.report.to_dict()
    digest = first.key.digest
    overlay_link = {
        "key_digest": digest,
        "source": "overlay",
        "overlay_path": str(session.overlay.path),
        "sequence": 1,
    }
    assert payload["consumer_counts"] == {digest: 2}
    assert payload["exact_source_counts"] == {
        "overlay": 2,
        "curated_exact": 0,
        "miss": 2,
    }
    assert payload["final_exact_source_counts"] == {
        "overlay": 1,
        "curated_exact": 0,
    }
    assert payload["perf_keys"] == [
        {
            "digest": digest,
            "namespace": first.key.namespace,
            "query": dict(first.query),
            "environment": json.loads(first.key.environment_json),
            "consumers": [first.op_id, second.op_id],
            "consumer_count": 2,
            "source_counts": {
                "overlay": 2,
                "curated_exact": 0,
                "miss": 2,
            },
            "final_source_counts": {
                "overlay": 1,
                "curated_exact": 0,
            },
        }
    ]
    assert payload["records"] == [
        {
            "key_digest": digest,
            "status": "valid",
            "latency_ms": 1.25,
            "latency_units": "ms",
            "energy_wms": 0.25,
            "energy_units": "watt_milliseconds",
            "samples_ms": [1.25],
            "protocol": json.loads(first.protocol.canonical),
            "perf_row": {"latency": 1.25},
            "provenance": {"collector_revision": "r1", "gpu_uuid": "GPU-0"},
            "failure": None,
            "persisted": True,
            "sequence": 1,
            "rejection_reason": None,
        }
    ]
    assert payload["evidence_links"] == [overlay_link]
    assert payload["callbacks"] == [
        {
            "sequence": 1,
            "context": {},
            "outcome": "resolved",
            "collection": {
                "attempted": True,
                "accepted_records": 1,
                "rejected_records": 0,
                "wall_seconds": payload["callbacks"][0]["collection"]["wall_seconds"],
            },
            "replay": {"attempted": True, "outcome": "succeeded"},
            "result_source": None,
            "consumer_counts": {digest: 2},
            "source_counts": {
                digest: {
                    "overlay": 2,
                    "curated_exact": 0,
                    "miss": 2,
                }
            },
            "final_exact_sources": {digest: "overlay"},
            "final_evidence": [overlay_link],
            "failures": [],
            "error": None,
        }
    ]
    for key in (
        "registry_routes",
        "resources",
        "waves",
        "assignments",
        "workers",
        "invocations",
        "inventories",
        "leases",
    ):
        assert payload[key] == []
    assert json.loads(json.dumps(payload, allow_nan=False, sort_keys=True)) == payload
    assert json.dumps(session.report.to_dict(), allow_nan=False, sort_keys=True) == json.dumps(
        payload,
        allow_nan=False,
        sort_keys=True,
    )


def test_report_distinguishes_curated_final_evidence_without_replay(tmp_path) -> None:
    request = _request(op_id="gemm")
    executor = _Executor(())
    session = _session(tmp_path, executor)
    curated_link = {"dataset": "gemm_perf.txt/v1", "row": 7}

    def query() -> PerformanceResult:
        session.record_curated_hit(request.key, request.op_id, evidence_link=curated_link)
        return PerformanceResult(0.5, energy=0.0, source="curated_exact")

    assert session.execute_callback(query) == PerformanceResult(0.5, energy=0.0, source="curated_exact")

    payload = session.report.to_dict()
    digest = request.key.digest
    expected_link = {
        "dataset": "gemm_perf.txt/v1",
        "row": 7,
        "key_digest": digest,
        "source": "curated_exact",
    }
    assert payload["consumer_counts"] == {digest: 1}
    assert payload["exact_source_counts"]["curated_exact"] == 1
    assert payload["final_exact_source_counts"]["curated_exact"] == 1
    assert payload["evidence_links"] == [expected_link]
    assert payload["callbacks"][0]["replay"] == {"attempted": False, "outcome": "not_attempted"}
    assert payload["callbacks"][0]["result_source"] == "curated_exact"
    assert payload["callbacks"][0]["final_exact_sources"] == {digest: "curated_exact"}
    assert payload["callbacks"][0]["final_evidence"] == [expected_link]
    assert executor.request_batches == []


def test_warm_callback_explicitly_reports_zero_collection_work(tmp_path) -> None:
    request = _request(op_id="gemm")
    executor = _Executor((_record(request, latency_ms=1.25),))
    session = _session(tmp_path, executor)

    def query() -> float:
        record = session.lookup(request.key, request.protocol, consumer=request.op_id)
        if record is None:
            session.record_miss(request, request.op_id)
            return 0.0
        assert record.latency_ms is not None
        return record.latency_ms

    assert session.execute_callback(query) == pytest.approx(1.25)
    assert session.execute_callback(query) == pytest.approx(1.25)

    cold, warm = session.report.to_dict()["callbacks"]
    assert cold["collection"]["attempted"] is True
    assert cold["collection"]["accepted_records"] == 1
    assert cold["collection"]["rejected_records"] == 0
    assert cold["collection"]["wall_seconds"] >= 0.0
    assert warm["collection"] == {
        "attempted": False,
        "accepted_records": 0,
        "rejected_records": 0,
        "wall_seconds": 0.0,
    }
    assert warm["replay"] == {"attempted": False, "outcome": "not_attempted"}
    assert warm["final_exact_sources"] == {request.key.digest: "overlay"}
    assert executor.request_batches == [(request,)]


def test_callback_context_is_json_snapshotted_for_fpm_identity_and_selected_path(tmp_path) -> None:
    request = _request(op_id="gemm")
    session = _session(tmp_path, _Executor(()))
    context = {
        "fpm": {
            "rank": 0,
            "counter": 17,
            "num_prefill_requests": 2,
            "sum_prefill_tokens": 64,
            "sum_prefill_kv_tokens": 128,
            "num_decode_requests": 3,
            "sum_decode_kv_tokens": 192,
        },
        "selected_path": "mixed",
    }

    result = session.execute_callback(
        lambda: PerformanceResult(0.5, energy=0.0, source="curated_exact"),
        context=context,
    )
    context["fpm"]["counter"] = 99

    assert result.source == "curated_exact"
    assert session.report.to_dict()["callbacks"][0]["context"] == {
        "fpm": {
            "rank": 0,
            "counter": 17,
            "num_prefill_requests": 2,
            "sum_prefill_tokens": 64,
            "sum_prefill_kv_tokens": 128,
            "num_decode_requests": 3,
            "sum_decode_kv_tokens": 192,
        },
        "selected_path": "mixed",
    }
    with pytest.raises(ValueError, match="JSON-safe finite"):
        session.execute_callback(lambda: 0.0, context={"counter": float("nan")})
    assert len(session.report.to_dict()["callbacks"]) == 1
    assert request.key.digest not in session.report.to_dict()["consumer_counts"]


def test_report_preserves_partial_records_and_failure_without_final_evidence(tmp_path) -> None:
    valid_request = _request(op_id="first", m=8)
    failed_request = _request(op_id="second", m=16)
    executor = _Executor(
        (
            _record(valid_request, latency_ms=1.25),
            _record(
                failed_request,
                latency_ms=None,
                status=RecordStatus.FAILED,
                failure_code=UnresolvedCode.TIMEOUT,
            ),
        )
    )
    session = _session(tmp_path, executor)

    def query() -> float:
        for request in (valid_request, failed_request):
            if session.lookup(request.key, request.protocol, consumer=request.op_id) is None:
                session.record_miss(request, request.op_id)
        return 0.0

    with pytest.raises(ResolutionFailed):
        session.execute_callback(query)

    payload = session.report.to_dict()
    assert [record["status"] for record in payload["records"]] == ["valid", "failed"]
    assert [record["persisted"] for record in payload["records"]] == [True, True]
    assert [link["sequence"] for link in payload["evidence_links"]] == [1, 2]
    assert payload["accepted_records"] == 1
    assert payload["rejected_records"] == 1
    assert payload["callbacks"][0]["outcome"] == "failed"
    assert payload["callbacks"][0]["replay"] == {"attempted": False, "outcome": "not_attempted"}
    assert payload["callbacks"][0]["final_exact_sources"] == {}
    assert payload["callbacks"][0]["final_evidence"] == []
    assert payload["callbacks"][0]["failures"] == [
        {
            "code": "timeout",
            "operation": failed_request.op_id,
            "detail": "injected timeout",
        }
    ]
    assert payload["callbacks"][0]["error"] == {
        "type": "ResolutionFailed",
        "message": "timeout: second: injected timeout",
    }
    assert payload["unresolved"] == [
        {
            "code": "timeout",
            "operation": failed_request.op_id,
            "detail": "injected timeout",
            "key_digest": failed_request.key.digest,
            "failure_kind": "invariant",
        }
    ]
    assert session.overlay.lookup(valid_request.key, valid_request.protocol) is not None
    assert session.overlay.lookup(failed_request.key, failed_request.protocol) is None
