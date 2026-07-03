# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DRY_RUN transcript tests for the RUNTIME indirection in collect_fpm_metrics.sh.
import os
import pathlib
import re
import subprocess

SCRIPT = "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"
RUNTIME_SCRIPT = "collector/layerwise/fpm_ground_truth/runtime.sh"
OUTER_SCRIPT = "collector/layerwise/reproduce_layerwise_fpm.sh"


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


def _outer_dry_run(tmp_path, env_extra):
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "STAGES": "layerwise",
        "MODEL": "Qwen/Qwen3-0.6B",
        "OUT_ROOT": str(tmp_path / "outer-output"),
    }
    env.pop("AIC_MODEL_MODE", None)
    env.update(env_extra)
    result = subprocess.run(
        ["bash", OUTER_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, result.stdout + result.stderr


def test_runtime_indirection_was_wired():
    """Fail-first guard: pre-Task-1, RUNTIME was not sourced; docker run -d
    appeared regardless of RUNTIME=process.  Post-Task-1, runtime.sh routes
    the process branch through die(), so no docker run -d is emitted and the
    die message appears in the transcript.
    """
    t = _dry_run({"RUNTIME": "process"})
    assert "docker run -d" not in t, (
        "RUNTIME=process must NOT emit docker run -d (runtime.sh indirection was not wired)"
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
    assert "docker exec" in t, "NSYS_PROFILE_WORKER=1 must emit 'docker exec' via runtime_exec_worker"
    assert "nsys" in t, "NSYS_PROFILE_WORKER=1 must include 'nsys' in the exec command"


def test_full_worker_nsys_waits_for_primary_application_only():
    """Nsight must not wait forever on multiprocessing children re-parented at shutdown."""
    transcript = _dry_run(
        {
            "RUNTIME": "docker",
            "NSYS_PROFILE_WORKER": "1",
            "NSYS_PROFILE_TRAFFIC_ONLY": "0",
        }
    )
    profile_command = next(
        line for line in transcript.splitlines() if "nsys profile" in line
    )

    assert "--wait=primary" in profile_command


def test_full_worker_shutdown_interrupts_profiler_then_waits_for_report_flush():
    """A service-shaped worker never exits on its own.

    Sending Docker's default SIGTERM to the nsys PID and exhausting the stop
    timeout leads to SIGKILL, which loses the report.  Full-worker capture must
    instead use the profiler's interactive SIGINT path and wait for nsys to
    finish writing the report.
    """
    shell = f"""
set -euo pipefail
RUN_DIR=/tmp/aic-fpm-test
WORKER_NAME=test-worker
DRY_RUN=1
RUNTIME=docker
run() {{
  printf '+'
  printf ' %q' "$@"
  printf '\\n'
}}
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf '%s\\n' "$*" >&2; return 1; }}
source {RUNTIME_SCRIPT}
runtime_flush_profiled_worker
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        timeout=30,
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert "docker kill --signal=SIGINT test-worker" in transcript
    assert "timeout 180 docker wait test-worker" in transcript


def test_full_worker_flush_timeout_dumps_diagnostics_before_fallback_stop():
    """A failed graceful flush must leave enough evidence to diagnose nsys."""
    shell = f"""
set -euo pipefail
RUN_DIR=/tmp/aic-fpm-test
WORKER_NAME=test-worker
DRY_RUN=0
RUNTIME=docker
run() {{ "$@"; }}
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf '%s\\n' "$*" >&2; return 1; }}
docker() {{ printf 'docker'; printf ' %s' "$@"; printf '\\n'; return 0; }}
timeout() {{ printf 'timeout'; printf ' %s' "$@"; printf '\\n'; return 124; }}
source {RUNTIME_SCRIPT}
runtime_flush_profiled_worker
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        timeout=30,
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert "docker top test-worker" in transcript
    assert "docker logs test-worker" in transcript
    assert "docker stop -t 10 test-worker" in transcript


def test_profile_diagnostics_preserve_nsys_and_worker_evidence():
    shell = f"""
set -euo pipefail
RUN_DIR=/tmp/aic-fpm-test
WORKER_NAME=test-worker
DRY_RUN=0
RUNTIME=docker
run() {{ "$@"; }}
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf '%s\\n' "$*" >&2; return 1; }}
docker() {{ printf 'docker'; printf ' %s' "$@"; printf '\\n'; return 0; }}
source {RUNTIME_SCRIPT}
runtime_dump_profile_diagnostics
"""
    result = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        timeout=30,
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert "docker inspect" in transcript
    assert "test-worker" in transcript
    assert "docker logs test-worker" in transcript


def test_missing_report_path_invokes_profile_diagnostics():
    script = pathlib.Path(SCRIPT).read_text()
    missing_report_branch = script.split(
        'log "WARNING: no Nsight worker report found under ${RUN_DIR}/nsys"', maxsplit=1
    )[1].split("\n        fi", maxsplit=1)[0]
    assert "runtime_dump_profile_diagnostics" in missing_report_branch


def test_layerwise_driver_passes_hf_token_without_rendering_or_materializing_it(tmp_path):
    token = "sentinel-hf-token-must-stay-secret"
    env = {
        **os.environ,
        "AIC_REPO": str(pathlib.Path.cwd()),
        "DRY_RUN": "1",
        "STAGES": "layerwise",
        "OUT_ROOT": str(tmp_path / "out"),
        "HF_HOME": str(tmp_path / "hf-cache"),
        "VLLM_CACHE_HOST": str(tmp_path / "vllm-cache"),
        "HF_TOKEN": token,
    }
    result = subprocess.run(
        ["bash", "collector/layerwise/reproduce_layerwise_fpm.sh"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert "-e HF_TOKEN" in transcript
    assert token not in transcript
    assert not list(tmp_path.rglob("hf.token"))


def test_fpm_driver_passes_inherited_hf_token_to_docker_by_name_only():
    token = "sentinel-fpm-hf-token-must-stay-secret"
    transcript = _dry_run(
        {
            "RUNTIME": "docker",
            "HF_TOKEN": token,
            "HF_TOKEN_FILE": "/definitely/missing/hf.token",
        }
    )

    assert "-e HF_TOKEN" in transcript
    assert token not in transcript
    assert "/run/secrets/hf.token" not in transcript


def test_smoke_scheduler_budget_is_not_below_vllm_model_length():
    driver = pathlib.Path("collector/layerwise/reproduce_layerwise_fpm.sh").read_text()
    smoke_block = driver.split('if [[ "$SMOKE" == "1" ]]', maxsplit=1)[1].split("fi", maxsplit=1)[0]

    assert 'FPM_MAX_NUM_BATCHED_TOKENS="40960"' in smoke_block


def test_enroot_mode_nsys_dry_run_previews_exec_without_live_pid():
    t = _dry_run({"RUNTIME": "enroot", "NSYS_PROFILE_WORKER": "1"})

    assert "Enroot worker" not in t
    assert "not running" not in t
    assert "enroot exec" in t
    assert "nsys" in t


def test_enroot_dry_run_needs_no_runtime_binary_image_or_state(tmp_path):
    run_dir = tmp_path / "enroot-preview"
    t = _dry_run(
        {
            "RUNTIME": "enroot",
            "RUN_DIR": str(run_dir),
            "NAME_PREFIX": "enroot-preview",
            "ENROOT_BIN": str(tmp_path / "missing-enroot"),
            "SETSID_BIN": str(tmp_path / "missing-setsid"),
            "ENROOT_IMAGE_PATH": str(tmp_path / "missing-image.sqsh"),
        }
    )

    assert "runtime error" not in t.lower()
    assert "enroot start" in t
    assert not (run_dir / ".runtime/enroot").exists()


def test_runtime_dumps_frontend_and_worker_logs_when_model_registration_fails():
    """If the frontend is alive but /v1/models never contains the target model,
    the failure branch must dump frontend/worker logs so CI postmortems can see
    whether the worker crashed, is downloading config/tokenizer, or registered a
    different model name. This is a source-level guard because dry-run does not
    execute the model-registration wait/failure branch.
    """
    source = pathlib.Path(SCRIPT).read_text()

    assert "model registration timed out" in source
    assert 'runtime_logs "${FRONTEND_NAME}"' in source
    assert 'runtime_logs "${WORKER_NAME}"' in source


def test_collect_driver_routes_every_docker_dependency_through_runtime():
    """Explicit Enroot mode must not require any collector-side Docker binary."""
    source = pathlib.Path(SCRIPT).read_text()
    raw_docker = re.compile(r"\bdocker\b")
    offending_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("log "):
            continue
        if raw_docker.search(stripped):
            offending_lines.append(stripped)

    assert offending_lines == []
    for runtime_api in (
        "runtime_prepare",
        "runtime_container_exists",
        "runtime_run_oneshot",
        "runtime_logs",
        "runtime_stop",
    ):
        assert runtime_api in source


def test_version_probe_keeps_docker_network_isolation_and_uses_host_for_enroot():
    source = pathlib.Path(SCRIPT).read_text()
    block = re.search(
        r"VERSION_CHECK_RUNTIME_OPTS=.*?log \"Verifying image",
        source,
        flags=re.DOTALL,
    )

    assert block is not None
    assert "--network none" in block.group(0)
    assert '"${RUNTIME}" == "enroot"' in block.group(0)
    assert "--network host" in block.group(0)


def test_outer_version_probe_keeps_docker_network_isolation_and_uses_host_for_enroot():
    source = pathlib.Path(OUTER_SCRIPT).read_text()
    block = re.search(
        r"runtime_image_version\(\).*?runtime_prepare",
        source,
        flags=re.DOTALL,
    )

    assert block is not None
    assert "--network none" in block.group(0)
    assert '"${RUNTIME}" == "enroot"' in block.group(0)
    assert "--network host" in block.group(0)


def test_cleanup_is_armed_before_every_runtime_prepare():
    inner = pathlib.Path(SCRIPT).read_text()
    outer = pathlib.Path(OUTER_SCRIPT).read_text()

    assert inner.index("CLEANUP_ENABLED=1") < inner.index("runtime_prepare")
    for function_name in ("runtime_image_version()", "stage_layerwise()"):
        function_start = outer.index(function_name)
        prepare = outer.index("runtime_prepare", function_start)
        trap = outer.index("trap 'runtime_teardown", function_start)
        assert trap < prepare


def test_outer_driver_routes_every_docker_dependency_through_runtime():
    """Layerwise must use the same native backend as the nested FPM driver."""
    source = pathlib.Path(OUTER_SCRIPT).read_text()
    raw_docker = re.compile(r"\bdocker\b")
    offending_lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith(("log ", "warn ")):
            continue
        executable = stripped.split("#", 1)[0]
        if raw_docker.search(executable):
            offending_lines.append(executable.rstrip())

    assert offending_lines == []
    assert "runtime_prepare" in source
    assert "runtime_run_oneshot" in source


def test_outer_layerwise_enroot_dry_run_needs_no_docker_or_runtime_binary(tmp_path):
    result, transcript = _outer_dry_run(
        tmp_path,
        {
            "RUNTIME": "enroot",
            "ENROOT_BIN": str(tmp_path / "missing-enroot"),
            "SETSID_BIN": str(tmp_path / "missing-setsid"),
            "ENROOT_IMAGE_PATH": str(tmp_path / "missing-image.sqsh"),
            "NSYS_TARGET_HOST_DIR": str(tmp_path / "missing-nsys-target"),
            "NSYS_IMPORTER_HOST_DIR": str(tmp_path / "missing-nsys-importer"),
        },
    )

    assert result.returncode == 0, transcript
    assert "docker run" not in transcript
    assert "enroot start" in transcript


def test_outer_layerwise_docker_dry_run_does_not_execute_runtime_binary(tmp_path):
    docker_marker = tmp_path / "docker-executed"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(f"#!/usr/bin/env bash\nprintf executed > {docker_marker!s}\nexit 97\n")
    fake_docker.chmod(0o755)

    result, transcript = _outer_dry_run(
        tmp_path,
        {
            "RUNTIME": "docker",
            "DOCKER_BIN": str(fake_docker),
        },
    )

    assert result.returncode == 0, transcript
    assert str(fake_docker) in transcript
    assert not docker_marker.exists()


def test_docker_mode_worker_command_uses_clean_weightless_defaults():
    """FPM timing uses scheduler shapes and must not materialize full model weights.
    The final worker command (not just helper unit defaults) must include dummy
    loading and the clean-prefill flags.
    """
    t = _dry_run({"RUNTIME": "docker"})

    assert "--load-format dummy" in t or "--load-format=dummy" in t
    assert "--no-enable-prefix-caching" in t
    assert "--no-enable-chunked-prefill" in t


def test_effective_config_snapshot_container_has_worker_gpu_visibility():
    """Resolving vLLM's effective config requires CUDA device inference.

    The snapshot helper runs in a short-lived container before the worker.  It
    must receive the same Docker GPU selection as the worker or vLLM 0.20.1
    fails device inference and writes metadata with ``effective_config=null``.
    """
    source = pathlib.Path(SCRIPT).read_text()
    snapshot = source.split("snapshot_effective_vllm_config()", maxsplit=1)[1].split(
        "resolved_max_model_len()", maxsplit=1
    )[0]

    assert '--gpus "${GPUS}"' in snapshot
