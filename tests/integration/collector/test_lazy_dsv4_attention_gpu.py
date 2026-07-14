# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in one-GB200 cold/warm/reopen gate for all four DSv4 attention routes."""

from __future__ import annotations

import os

import pytest

from tools.run_dsv4_attention_lifecycle import ATTENTION_ROUTE_MATRIX, run_local_gpu_gate

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.timeout(1800)]

_GPU_OPT_IN = "AICONFIGURATOR_RUN_GPU_TESTS"


def test_dsv4_attention_routes_measure_once_then_reuse_after_reopen(tmp_path) -> None:
    if os.environ.get(_GPU_OPT_IN) != "1":
        pytest.skip(f"set {_GPU_OPT_IN}=1 to run the real-GB200 attention lifecycle gate")

    report = run_local_gpu_gate(
        overlay_path=tmp_path / "dsv4-attention-lifecycle.sqlite",
        max_wall_seconds=1500.0,
    )
    lifecycle = report.lifecycle

    assert lifecycle.route_ids == tuple(route.route_id for route in ATTENTION_ROUTE_MATRIX)
    assert lifecycle.pure.command_count == 0
    assert lifecycle.pure.additional_commands == 0
    assert lifecycle.cold.command_count == 4
    assert lifecycle.cold.additional_commands == 4
    assert lifecycle.warm.command_count == 4
    assert lifecycle.warm.additional_commands == 0
    assert lifecycle.reopened.command_count == 4
    assert lifecycle.reopened.additional_commands == 0
    assert lifecycle.cold_unique_misses == 4
    assert lifecycle.cold_accepted_records == 4
    assert lifecycle.reopened_unique_misses == 0
    assert lifecycle.cold.results == lifecycle.warm.results == lifecycle.reopened.results
    assert all(record.status == "valid" for record in lifecycle.records)
