#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runtime lifecycle contract for collect_fpm_metrics.sh.
# Sourced after run(), log(), and die() are defined.
# Profiler shutdown remains backend-aware: Docker uses Nsight's SIGINT flush
# path, while native runtimes use their normal bounded stop implementation.

: "${RUNTIME:=docker}"
: "${DOCKER_BIN:=docker}"

# Callers may inject a logging/tee wrapper without changing the backend API.
# The FPM driver keeps its historical run <cmd...> helper; the outer layerwise
# driver binds this hook to its run <logfile> <cmd...> helper.
if ! declare -F runtime_invoke >/dev/null 2>&1; then
    runtime_invoke() { run "$@"; }
fi

runtime_error() {
    printf 'runtime error: %s\n' "$*" >&2
    return 1
}

runtime_dispatch() {
    local operation="${1:-}"
    if [[ -z "${operation}" ]]; then
        runtime_error "unknown runtime operation: <empty>"
        return 64
    fi
    shift
    local implementation="runtime_${RUNTIME}_${operation}"
    if ! declare -F "${implementation}" >/dev/null 2>&1; then
        runtime_error "unknown runtime operation '${operation}' for backend '${RUNTIME}'"
        return 64
    fi
    "${implementation}" "$@"
}

runtime_prepare() { runtime_dispatch prepare "$@"; }
runtime_container_exists() { runtime_dispatch container_exists "$@"; }
runtime_status() { runtime_dispatch status "$@"; }
runtime_launch_detached() { runtime_dispatch launch_detached "$@"; }
runtime_run_oneshot() { runtime_dispatch run_oneshot "$@"; }
runtime_exec_worker() { runtime_dispatch exec_worker "$@"; }
runtime_copy_in() { runtime_dispatch copy_in "$@"; }
runtime_copy_out() { runtime_dispatch copy_out "$@"; }
runtime_logs() { runtime_dispatch logs "$@"; }
runtime_stop() { runtime_dispatch stop "$@"; }
runtime_teardown() { runtime_dispatch teardown "$@"; }

runtime_docker_prepare() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    command -v "${DOCKER_BIN}" >/dev/null 2>&1 || {
        runtime_error "docker executable '${DOCKER_BIN}' was not found"
        return 1
    }
    "${DOCKER_BIN}" image inspect "${IMAGE}" >/dev/null 2>&1 || {
        runtime_error "Docker image '${IMAGE}' was not found locally"
        return 1
    }
}

runtime_docker_container_exists() {
    "${DOCKER_BIN}" ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

runtime_docker_status() {
    runtime_docker_container_exists "$1"
}

runtime_docker_launch_detached() {
    local role="$1" gpus="$2" optsvar="$3"
    shift 3
    if [[ "${1:-}" != "--" ]]; then
        runtime_error "runtime_launch_detached requires '--' before the command"
        return 64
    fi
    shift
    local name="${NAME_PREFIX}-${role}"
    local -n docker_opts="${optsvar}"
    local gpu_args=()
    [[ -n "${gpus}" ]] && gpu_args=(--gpus "${gpus}")
    runtime_invoke "${DOCKER_BIN}" run -d \
        --name "${name}" \
        --network host \
        "${gpu_args[@]}" \
        -v "${RUN_DIR}:/work" \
        "${docker_opts[@]}" \
        "${IMAGE}" \
        "$@"
}

runtime_docker_run_oneshot() {
    local role="$1" gpus="$2" optsvar="$3"
    shift 3
    if [[ "${1:-}" != "--" ]]; then
        runtime_error "runtime_run_oneshot requires '--' before the command"
        return 64
    fi
    shift
    local -n docker_opts="${optsvar}"
    local name_args=() gpu_args=()
    [[ -n "${role}" ]] && name_args=(--name "${NAME_PREFIX}-${role}")
    [[ -n "${gpus}" ]] && gpu_args=(--gpus "${gpus}")
    runtime_invoke "${DOCKER_BIN}" run --rm \
        "${name_args[@]}" \
        "${gpu_args[@]}" \
        "${docker_opts[@]}" \
        "${IMAGE}" \
        "$@"
}

runtime_docker_exec_worker() {
    runtime_invoke "${DOCKER_BIN}" exec "${WORKER_NAME}" "$@"
}

# Keep profiler failure evidence without dumping the container configuration
# (which may contain credentials).  Docker logs include nsys stderr, while the
# state and file listing distinguish a profiler crash from an incomplete export.
runtime_dump_profile_diagnostics() {
    log "Nsight worker diagnostic state/logs/files follow"
    if [[ "${RUNTIME}" == "docker" ]]; then
        "${DOCKER_BIN}" inspect --format '{{json .State}}' "${WORKER_NAME}" || true
        "${DOCKER_BIN}" top "${WORKER_NAME}" || true
    fi
    runtime_logs "${WORKER_NAME}" || true
    if [[ -d "${RUN_DIR}/nsys" ]]; then
        find "${RUN_DIR}/nsys" -maxdepth 1 -type f -print || true
    fi
}

# ---------------------------------------------------------------------------
# runtime_flush_profiled_worker
#
# Full-lifetime `nsys profile` wraps a service that otherwise never exits.  A
# regular `docker stop` sends SIGTERM to the nsys PID, waits for the timeout,
# then SIGKILLs the container; that can discard the report.  SIGINT mirrors the
# interactive Ctrl-C path supported by nsys, and `docker wait` gives it time to
# finalize the report.  If finalization times out, retain actionable container
# diagnostics before the bounded fallback stop.
# ---------------------------------------------------------------------------
runtime_flush_profiled_worker() {
    local flush_timeout="${NSYS_FLUSH_TIMEOUT_SECONDS:-180}"
    if [[ "${RUNTIME}" != "docker" ]]; then
        runtime_stop "${WORKER_NAME}" 60 >/dev/null || true
        return
    fi

    log "Interrupting profiled worker so Nsight can finalize its report"
    if [[ "${DRY_RUN}" == "1" ]]; then
        runtime_invoke "${DOCKER_BIN}" kill --signal=SIGINT "${WORKER_NAME}"
        runtime_invoke timeout "${flush_timeout}" "${DOCKER_BIN}" wait "${WORKER_NAME}"
        return
    fi

    "${DOCKER_BIN}" kill --signal=SIGINT "${WORKER_NAME}" >/dev/null || \
        log "WARNING: failed to send SIGINT to profiled worker ${WORKER_NAME}"
    if ! timeout "${flush_timeout}" "${DOCKER_BIN}" wait "${WORKER_NAME}" >/dev/null; then
        log "WARNING: Nsight worker did not exit within ${flush_timeout}s; dumping diagnostics"
        runtime_dump_profile_diagnostics
        runtime_stop "${WORKER_NAME}" 10 || true
    fi
}

runtime_docker_copy_in() {
    local name="$1" source="$2" destination="$3"
    runtime_invoke "${DOCKER_BIN}" cp "${source}" "${name}:${destination}"
}

runtime_docker_copy_out() {
    local name="$1" source="$2" destination="$3"
    runtime_invoke "${DOCKER_BIN}" cp "${name}:${source}" "${destination}"
}

runtime_docker_logs() {
    local name="$1"
    shift
    "${DOCKER_BIN}" logs "$@" "${name}"
}

runtime_docker_stop() {
    local name="$1" timeout="${2:-10}"
    "${DOCKER_BIN}" stop -t "${timeout}" "${name}"
}

runtime_docker_teardown() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    local name
    for name in \
        "${NAME_PREFIX}-collector" \
        "${NAME_PREFIX}-worker" \
        "${NAME_PREFIX}-frontend"; do
        if runtime_docker_container_exists "${name}"; then
            "${DOCKER_BIN}" rm -f "${name}" >/dev/null 2>&1 || true
        fi
    done
    return 0
}

runtime_process_unimplemented() {
    runtime_error "process mode not implemented"
}

for runtime_process_operation in \
    prepare container_exists status launch_detached run_oneshot exec_worker \
    copy_in copy_out logs stop teardown; do
    eval "runtime_process_${runtime_process_operation}() { runtime_process_unimplemented; }"
done
unset runtime_process_operation

if [[ "${RUNTIME}" == "enroot" ]]; then
    # shellcheck source=runtime_enroot.sh
    runtime_script_dir="${BASH_SOURCE[0]%/*}"
    source "${runtime_script_dir}/runtime_enroot.sh"
    unset runtime_script_dir
fi
