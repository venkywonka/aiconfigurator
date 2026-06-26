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


def test_runtime_indirection_was_wired():
    """Fail-first guard: pre-Task-1, RUNTIME was not sourced; docker run -d
    appeared regardless of RUNTIME=process.  Post-Task-1, runtime.sh routes
    the process branch through die(), so no docker run -d is emitted and the
    die message appears in the transcript.
    """
    t = _dry_run({"RUNTIME": "process"})
    assert "docker run -d" not in t, (
        "RUNTIME=process must NOT emit docker run -d "
        "(runtime.sh indirection was not wired)"
    )
    assert "process mode not implemented" in t, (
        "RUNTIME=process must print the 'process mode not implemented' die message"
    )


def test_docker_mode_still_launches_containers():
    t = _dry_run({"RUNTIME": "docker"})
    assert "docker run -d" in t
    assert "-m dynamo.frontend" in t and "-m dynamo.vllm" in t


def test_docker_mode_golden_transcript():
    """Regression: all three containers must appear in docker mode."""
    t = _dry_run({"RUNTIME": "docker"})
    # All three docker run -d launches must be present
    assert t.count("docker run -d") >= 3, "expected frontend, worker, and collector container launches"
    # Frontend and worker must be the right modules
    assert "-m dynamo.frontend" in t
    assert "-m dynamo.vllm" in t
    # Collector must launch fpm_collect.py
    assert "fpm_collect.py" in t


def test_docker_mode_nsys_exec_in_transcript():
    """Step 6 guard: when NSYS_PROFILE_WORKER=1, the runtime_exec_worker path
    must emit 'docker exec ... nsys' in docker mode.  This verifies that the
    nsys start/stop call-sites (start_nsys_worker_collection /
    stop_nsys_worker_collection) were wired through runtime_exec_worker and
    that runtime_exec_worker emits the correct docker exec command.
    """
    t = _dry_run({"RUNTIME": "docker", "NSYS_PROFILE_WORKER": "1"})
    assert "docker exec" in t, (
        "NSYS_PROFILE_WORKER=1 must emit 'docker exec' via runtime_exec_worker"
    )
    assert "nsys" in t, (
        "NSYS_PROFILE_WORKER=1 must include 'nsys' in the exec command"
    )
