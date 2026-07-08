# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-GPU coverage for the packaged persistent NCCL runner."""

from __future__ import annotations

import math
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

_GPU_OPT_IN = "AICONFIGURATOR_RUN_GPU_TESTS"


def _require_two_cuda_devices():
    if os.environ.get(_GPU_OPT_IN) != "1":
        pytest.skip(f"set {_GPU_OPT_IN}=1 to run real-GPU collector tests")
    torch = pytest.importorskip("torch", reason="real-GPU NCCL test requires PyTorch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("real-GPU NCCL test requires at least two visible CUDA devices")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("real-GPU NCCL test requires a PyTorch build with NCCL support")
    return torch


def _visible_device_tokens() -> tuple[str, str]:
    configured = tuple(
        token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()
    )
    return (configured[0], configured[1]) if len(configured) >= 2 else ("0", "1")


def test_packaged_nccl_reuses_rank_group_and_returns_positive_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_two_cuda_devices()
    from aiconfigurator.collector.network import nccl as nccl_runner
    from aiconfigurator.sdk.resolution.types import MeasurementProtocol

    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=1,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )
    visible_devices = _visible_device_tokens()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(visible_devices))
    nccl_runner.close_nccl_worker()

    try:
        first = nccl_runner.run_nccl_case(
            "half",
            "all_reduce",
            1024,
            len(visible_devices),
            protocol=protocol,
        )
        first_group = nccl_runner._PERSISTENT_RANK_GROUP
        first_rank_pids = tuple(first.provenance["rank_pids"])

        second = nccl_runner.run_nccl_case(
            "half",
            "all_reduce",
            1024,
            len(visible_devices),
            protocol=protocol,
        )
        second_group = nccl_runner._PERSISTENT_RANK_GROUP
        second_rank_pids = tuple(second.provenance["rank_pids"])

        assert first_group is not None
        assert second_group is first_group
        assert len(first_rank_pids) == len(visible_devices)
        assert first_rank_pids == second_rank_pids
        assert all(isinstance(pid, int) and pid > 0 for pid in first_rank_pids)
        for measured in (first, second):
            assert measured.protocol_digest == protocol.digest
            assert measured.statistic == "median"
            assert math.isfinite(measured.latency_ms) and measured.latency_ms > 0
            assert len(measured.samples_ms) == protocol.samples
            assert all(math.isfinite(sample) and sample > 0 for sample in measured.samples_ms)
            assert measured.provenance["runtime"] == "persistent_torch_distributed"
            assert tuple(measured.provenance["device_uuids"]) == visible_devices
            assert tuple(measured.provenance["rank_pids"]) == first_rank_pids
    finally:
        nccl_runner.close_nccl_worker()
