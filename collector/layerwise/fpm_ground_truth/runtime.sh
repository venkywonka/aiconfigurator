#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# RUNTIME indirection for collect_fpm_metrics.sh.
# Sourced AFTER run()/log()/die() are defined in collect_fpm_metrics.sh.
#
# Env:
#   RUNTIME        docker (default) | process
#   PROC_PIDDIR    where process-mode PID/PGID files live (default: ${RUN_DIR}/.pids)
#
# Functions:
#   runtime_launch_detached <role> <gpus|""> <docker_opt_array_name> -- <cmd...>
#   runtime_exec_worker     <cmd...>
#   runtime_teardown

: "${RUNTIME:=docker}"
: "${PROC_PIDDIR:=${RUN_DIR}/.pids}"

# ---------------------------------------------------------------------------
# runtime_launch_detached <role> <gpus|""> <docker_opt_array_name> -- <cmd...>
#
# In docker mode: runs `docker run -d --name <NAME_PREFIX>-<role> --network host
#   [--gpus <gpus>] -v <RUN_DIR>:/work <docker_opts[@]> <IMAGE> <cmd...>`
#   (identical to the inline calls that were here before refactor).
# In process mode: (Task 2) bg-process under setsid; not yet implemented.
# ---------------------------------------------------------------------------
runtime_launch_detached() {
    local role="$1" gpus="$2" optsvar="$3"; shift 3
    [[ "$1" == "--" ]] && shift
    local name="${NAME_PREFIX}-${role}"
    if [[ "${RUNTIME}" == "docker" ]]; then
        local -n _opts="${optsvar}"
        local gpuargs=()
        [[ -n "${gpus}" ]] && gpuargs=(--gpus "${gpus}")
        run docker run -d \
            --name "${name}" \
            --network host \
            "${gpuargs[@]}" \
            -v "${RUN_DIR}:/work" \
            "${_opts[@]}" \
            "${IMAGE}" \
            "$@"
    else
        die "runtime_launch_detached: process mode not implemented (Task 2)"
    fi
}

# ---------------------------------------------------------------------------
# runtime_exec_worker <cmd...>
#
# In docker mode: `docker exec <WORKER_NAME> <cmd...>` (identical to before).
# In process mode: (Task 2) runs cmd directly.
# ---------------------------------------------------------------------------
runtime_exec_worker() {
    if [[ "${RUNTIME}" == "docker" ]]; then
        run docker exec "${WORKER_NAME}" "$@"
    else
        die "runtime_exec_worker: process mode not implemented (Task 2)"
    fi
}

# ---------------------------------------------------------------------------
# runtime_teardown
#
# In docker mode: docker rm -f for collector/worker/frontend (identical to before).
# In process mode: (Task 2) SIGTERM process groups.
# ---------------------------------------------------------------------------
runtime_teardown() {
    if [[ "${RUNTIME}" == "docker" ]]; then
        local name
        for name in \
            "${NAME_PREFIX}-collector" \
            "${NAME_PREFIX}-worker" \
            "${NAME_PREFIX}-frontend"; do
            container_exists "${name}" && \
                docker rm -f "${name}" >/dev/null 2>&1 || true
        done
    else
        die "runtime_teardown: process mode not implemented (Task 2)"
    fi
}
