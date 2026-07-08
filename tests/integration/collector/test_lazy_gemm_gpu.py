# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-GPU coverage for the packaged TensorRT-LLM GEMM runner."""

from __future__ import annotations

import math
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

_GPU_OPT_IN = "AICONFIGURATOR_RUN_GPU_TESTS"


def _require_cuda_device():
    if os.environ.get(_GPU_OPT_IN) != "1":
        pytest.skip(f"set {_GPU_OPT_IN}=1 to run real-GPU collector tests")
    torch = pytest.importorskip("torch", reason="real-GPU GEMM test requires PyTorch")
    pytest.importorskip("tensorrt_llm", reason="real-GPU GEMM test requires TensorRT-LLM")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("real-GPU GEMM test requires one visible CUDA device")
    return torch


def test_packaged_bf16_gemm_returns_positive_replay_samples_and_provenance() -> None:
    torch = _require_cuda_device()
    from aiconfigurator.collector.trtllm.gemm import run_gemm_case
    from aiconfigurator.sdk.resolution.types import MeasurementProtocol

    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=1,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )

    measured = run_gemm_case(
        "bfloat16",
        64,
        64,
        64,
        protocol=protocol,
        device="cuda:0",
    )

    assert measured.protocol_digest == protocol.digest
    assert measured.statistic == "median"
    assert math.isfinite(measured.latency_ms) and measured.latency_ms > 0
    assert len(measured.samples_ms) == protocol.samples
    assert all(math.isfinite(sample) and sample > 0 for sample in measured.samples_ms)
    assert measured.perf_row == {
        "gemm_dtype": "bfloat16",
        "m": 64,
        "n": 64,
        "k": 64,
        "latency": measured.latency_ms,
    }
    assert measured.provenance["framework"] == "TRTLLM"
    assert measured.provenance["framework_version"]
    assert measured.provenance["kernel_source"]
    assert measured.provenance["device"] == torch.cuda.get_device_name(0)
    assert isinstance(measured.provenance["used_cuda_graph"], bool)
    assert isinstance(measured.provenance["throttled"], bool)
    assert measured.provenance["tensor_generator"] == "normal-v1"
    assert measured.provenance["seed"] == 0
