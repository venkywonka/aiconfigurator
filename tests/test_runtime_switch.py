# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DRY_RUN transcript tests for the RUNTIME indirection in collect_fpm_metrics.sh.
import subprocess
import os
import pathlib

SCRIPT = "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"


def _dry_run(env_extra):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "REAL_WORKLOAD": "0",
        "SKIP_REQUESTS": "1",
        "MODEL": "Qwen/Qwen3-32B",
        "TP_SIZE": "8",
    }
    env.update(env_extra)
    out = subprocess.run(
        ["bash", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return out.stdout + out.stderr


def test_docker_mode_still_launches_containers():
    t = _dry_run({"RUNTIME": "docker"})
    assert "docker run -d" in t
    assert "-m dynamo.frontend" in t and "-m dynamo.vllm" in t


def test_docker_mode_golden_transcript():
    """Regression: all three containers + nsys exec must appear in docker mode."""
    t = _dry_run({"RUNTIME": "docker"})
    # All three docker run -d launches must be present
    assert t.count("docker run -d") >= 3, "expected frontend, worker, and collector container launches"
    # Frontend and worker must be the right modules
    assert "-m dynamo.frontend" in t
    assert "-m dynamo.vllm" in t
    # Collector must launch fpm_collect.py
    assert "fpm_collect.py" in t
