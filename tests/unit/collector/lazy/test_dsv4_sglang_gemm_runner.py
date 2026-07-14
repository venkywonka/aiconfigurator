# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pytest

from aiconfigurator.collector.sglang import gemm
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

pytestmark = pytest.mark.unit


def _protocol(**overrides) -> MeasurementProtocol:
    values = {
        "revision": "cuda-event-samples-v1",
        "warmups": 2,
        "samples": 3,
        "statistic": "median",
        "timer": "cuda_event",
        "tuning_revision": "sglang-gemm-v1",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)


def test_exact_sglang_gemm_runner_returns_raw_samples_and_cleans_up(monkeypatch) -> None:
    calls: list[object] = []

    def _prepare(gemm_type, m, n, k, device):
        calls.append(("prepare", gemm_type, m, n, k, device))
        return gemm.PreparedGemmCase(
            kernel_func=lambda: calls.append(("kernel",)),
            cleanup_func=lambda: calls.append(("cleanup",)),
            outside_loop_count=2,
            kernel_source="deepgemm",
            framework_version="0.5.10rc0",
            device_name="NVIDIA GB200",
            device=object(),
        )

    @contextmanager
    def _benchmark(**kwargs):
        calls.append(("benchmark", kwargs))
        yield {
            "latency_ms": 8.0,
            "samples_ms": (6.0, 8.0, 10.0),
            "power_stats": {"power": 100.0, "power_limit": 1200.0},
            "throttled": False,
            "used_cuda_graph": True,
            "num_runs_executed": 3,
        }

    monkeypatch.setattr(gemm, "_prepare_gemm_case", _prepare)
    monkeypatch.setattr(gemm, "benchmark_with_power", _benchmark)
    protocol = _protocol()

    result = gemm.run_gemm_case("fp8_block", 8, 16, 32, protocol=protocol)

    assert isinstance(result, RawMeasurement)
    assert result.latency_ms == 4.0
    assert result.samples_ms == (3.0, 4.0, 5.0)
    assert result.energy_wms == 400.0
    assert result.protocol_digest == protocol.digest
    assert result.perf_row == {
        "gemm_dtype": "fp8_block",
        "m": 8,
        "n": 16,
        "k": 32,
        "latency": 4.0,
    }
    assert result.provenance == {
        "framework": "SGLang",
        "framework_version": "0.5.10rc0",
        "kernel_source": "deepgemm",
        "device": "NVIDIA GB200",
        "used_cuda_graph": True,
        "throttled": False,
        "tensor_generator": "normal-v1",
        "seed": 0,
    }
    benchmark_kwargs = next(call[1] for call in calls if isinstance(call, tuple) and call[0] == "benchmark")
    assert benchmark_kwargs["num_warmups"] == 2
    assert benchmark_kwargs["num_runs"] == 3
    assert benchmark_kwargs["repeat_n"] == 1
    assert benchmark_kwargs["return_samples"] is True
    assert calls[-1] == ("cleanup",)


def test_sglang_gemm_runner_rejects_protocol_before_gpu_preparation(monkeypatch) -> None:
    prepare_calls: list[object] = []
    monkeypatch.setattr(gemm, "_prepare_gemm_case", lambda *args: prepare_calls.append(args))

    with pytest.raises(ValueError, match="protocol"):
        gemm.run_gemm_case(
            "bfloat16",
            8,
            16,
            32,
            protocol=_protocol(tuning_revision="trtllm-linear-v1"),
        )

    assert prepare_calls == []


def test_sglang_gemm_runner_cleans_up_after_measurement_failure(monkeypatch) -> None:
    cleanup_calls: list[object] = []
    monkeypatch.setattr(
        gemm,
        "_prepare_gemm_case",
        lambda *args: gemm.PreparedGemmCase(
            kernel_func=lambda: None,
            cleanup_func=lambda: cleanup_calls.append("cleanup"),
            outside_loop_count=1,
            kernel_source="torch_flow",
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
        ),
    )

    @contextmanager
    def _benchmark(**kwargs):
        del kwargs
        raise RuntimeError("benchmark failed")
        yield  # pragma: no cover

    monkeypatch.setattr(gemm, "benchmark_with_power", _benchmark)

    with pytest.raises(RuntimeError, match="benchmark failed"):
        gemm.run_gemm_case("bfloat16", 8, 16, 32, protocol=_protocol())

    assert cleanup_calls == ["cleanup"]


def test_sglang_gemm_runner_preserves_measurement_failure_when_cleanup_also_fails(monkeypatch) -> None:
    def _cleanup() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        gemm,
        "_prepare_gemm_case",
        lambda *args: gemm.PreparedGemmCase(
            kernel_func=lambda: None,
            cleanup_func=_cleanup,
            outside_loop_count=1,
            kernel_source="torch_flow",
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
        ),
    )

    @contextmanager
    def _benchmark(**kwargs):
        del kwargs
        raise RuntimeError("benchmark failed")
        yield  # pragma: no cover

    monkeypatch.setattr(gemm, "benchmark_with_power", _benchmark)

    with pytest.raises(RuntimeError, match="benchmark failed"):
        gemm.run_gemm_case("bfloat16", 8, 16, 32, protocol=_protocol())
