#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Native Enroot implementation of the runtime.sh lifecycle contract.
# This file is normally sourced by runtime.sh when RUNTIME=enroot.

: "${ENROOT_BIN:=enroot}"
: "${SETSID_BIN:=setsid}"
: "${ENROOT_LAUNCH_TIMEOUT_SECONDS:=60}"

_enroot_error() {
    if declare -F runtime_error >/dev/null 2>&1; then
        runtime_error "$*"
    else
        printf 'runtime error: %s\n' "$*" >&2
        return 1
    fi
}

_enroot_log() {
    if declare -F log >/dev/null 2>&1; then
        log "$*"
    else
        printf '%s\n' "$*" >&2
    fi
}

_enroot_validate_name() {
    local label="$1" value="$2"
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        _enroot_error "invalid Enroot ${label} name"
        return 64
    }
}

_enroot_command_available() {
    local command_name="$1"
    if [[ "${command_name}" == */* ]]; then
        [[ -x "${command_name}" ]]
    else
        command -v "${command_name}" >/dev/null 2>&1
    fi
}

_enroot_realpath_m() {
    realpath -m -- "$1"
}

_enroot_path_is_under_run_dir() {
    local candidate run_root
    candidate="$(_enroot_realpath_m "$1")" || return 1
    run_root="$(_enroot_realpath_m "${RUN_DIR}")" || return 1
    [[ "${candidate}" == "${run_root}" || "${candidate}" == "${run_root}/"* ]]
}

_enroot_validate_state_path() {
    local label="$1" path="$2"
    if [[ -L "${path}" ]]; then
        _enroot_error "Enroot ${label} directory must not be a symlink: ${path}"
        return 1
    fi
    if ! _enroot_path_is_under_run_dir "${path}"; then
        _enroot_error "Enroot ${label} directory must be under RUN_DIR: ${path}"
        return 1
    fi
}

_enroot_make_private_dir() {
    local path="$1"
    mkdir -p -- "${path}" || return 1
    chmod 0700 -- "${path}" || return 1
    [[ -d "${path}" && -O "${path}" && ! -L "${path}" ]]
}

_enroot_configure_paths() {
    : "${ENROOT_STATE_DIR:=${RUN_DIR}/.runtime/enroot}"
    : "${ENROOT_PIDDIR:=${ENROOT_STATE_DIR}/pids}"
    : "${ENROOT_DATA_PATH:=${ENROOT_STATE_DIR}/data}"
    : "${ENROOT_CACHE_PATH:=${ENROOT_STATE_DIR}/cache}"
    : "${ENROOT_TEMP_PATH:=${ENROOT_STATE_DIR}/tmp}"
    : "${ENROOT_RUNTIME_PATH:=${ENROOT_STATE_DIR}/runtime}"
    : "${ENROOT_IMAGE_PATH:=${ENROOT_IMAGE:-${IMAGE:-}}}"
}

_enroot_image_name() {
    if [[ -n "${ENROOT_CONTAINER_NAME:-}" ]]; then
        printf '%s\n' "${ENROOT_CONTAINER_NAME}"
        return
    fi
    local raw="${NAME_PREFIX:-aic}-image"
    raw="${raw//\//_}"
    raw="${raw//:/_}"
    raw="${raw//./_}"
    printf '%s\n' "${raw}"
}

_enroot_mapped_image_name() {
    local image="$1" pair key value
    local -a pairs=()
    IFS=';' read -r -a pairs <<< "${ENROOT_IMAGE_MAP:-}"
    for pair in "${pairs[@]}"; do
        [[ "${pair}" == *=* ]] || {
            _enroot_error "invalid ENROOT_IMAGE_MAP entry '${pair}'"
            return 64
        }
        key="${pair%%=*}"
        value="${pair#*=}"
        if [[ "${key}" == "${image}" ]]; then
            [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
                _enroot_error "invalid mapped Enroot image name '${value}'"
                return 64
            }
            printf '%s\n' "${value}"
            return 0
        fi
    done
    _enroot_error "no Enroot image mapping exists for '${image}'"
    return 1
}

_enroot_image_exists() {
    local wanted="$1" candidate
    while IFS= read -r candidate; do
        [[ "${candidate}" == "${wanted}" ]] && return 0
    done < <("${ENROOT_BIN}" list 2>/dev/null)
    return 1
}

runtime_enroot_prepare() {
    _enroot_configure_paths
    local mapped_image=""
    _ENROOT_SHARED_CONTAINER=0
    if [[ -n "${ENROOT_IMAGE_MAP:-}" ]]; then
        mapped_image="$(_enroot_mapped_image_name "${IMAGE}")" || return $?
        _ENROOT_SHARED_CONTAINER=1
        ENROOT_CONTAINER_NAME="${mapped_image}"
    else
        ENROOT_CONTAINER_NAME="$(_enroot_image_name)"
    fi
    _enroot_validate_name "container" "${ENROOT_CONTAINER_NAME}" || return $?
    export ENROOT_CONTAINER_NAME _ENROOT_SHARED_CONTAINER

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    if ! _enroot_command_available "${ENROOT_BIN}"; then
        _enroot_error "Enroot executable '${ENROOT_BIN}' was not found"
        return 1
    fi
    if ! _enroot_command_available "${SETSID_BIN}"; then
        _enroot_error "setsid executable '${SETSID_BIN}' required by Enroot was not found"
        return 1
    fi
    if [[ "${_ENROOT_SHARED_CONTAINER}" != "1" && \
          ( -z "${ENROOT_IMAGE_PATH}" || ! -f "${ENROOT_IMAGE_PATH}" || -L "${ENROOT_IMAGE_PATH}" ) ]]; then
        _enroot_error "Enroot image '${ENROOT_IMAGE_PATH:-<unset>}' was not found as a regular local file"
        return 1
    fi

    _enroot_validate_state_path state "${ENROOT_STATE_DIR}" || return 1
    _enroot_validate_state_path pid "${ENROOT_PIDDIR}" || return 1
    if [[ "${_ENROOT_SHARED_CONTAINER}" == "1" ]]; then
        if [[ ! -d "${ENROOT_DATA_PATH}" || -L "${ENROOT_DATA_PATH}" || ! -O "${ENROOT_DATA_PATH}" ]]; then
            _enroot_error "shared Enroot data directory is unavailable or not job-user-owned: ${ENROOT_DATA_PATH}"
            return 1
        fi
    else
        _enroot_validate_state_path data "${ENROOT_DATA_PATH}" || return 1
    fi
    _enroot_validate_state_path cache "${ENROOT_CACHE_PATH}" || return 1
    _enroot_validate_state_path temporary "${ENROOT_TEMP_PATH}" || return 1
    _enroot_validate_state_path runtime "${ENROOT_RUNTIME_PATH}" || return 1

    local old_umask
    old_umask="$(umask)"
    umask 077
    local path
    local -a private_paths=(
        "${ENROOT_STATE_DIR}" \
        "${ENROOT_PIDDIR}" \
        "${ENROOT_CACHE_PATH}" \
        "${ENROOT_TEMP_PATH}" \
        "${ENROOT_RUNTIME_PATH}"
    )
    [[ "${_ENROOT_SHARED_CONTAINER}" == "1" ]] || private_paths+=("${ENROOT_DATA_PATH}")
    for path in "${private_paths[@]}"; do
        if ! _enroot_make_private_dir "${path}"; then
            umask "${old_umask}"
            _enroot_error "failed to create private Enroot state directory: ${path}"
            return 1
        fi
    done
    umask "${old_umask}"

    export ENROOT_DATA_PATH ENROOT_CACHE_PATH ENROOT_TEMP_PATH ENROOT_RUNTIME_PATH
    "${ENROOT_BIN}" version >/dev/null 2>&1 || {
        _enroot_error "Enroot executable '${ENROOT_BIN}' is unavailable"
        return 1
    }
    if [[ "${_ENROOT_SHARED_CONTAINER}" == "1" ]]; then
        if ! _enroot_image_exists "${ENROOT_CONTAINER_NAME}"; then
            _enroot_error "mapped Enroot image '${ENROOT_CONTAINER_NAME}' is unavailable"
            return 1
        fi
    elif ! _enroot_image_exists "${ENROOT_CONTAINER_NAME}"; then
        "${ENROOT_BIN}" create --name "${ENROOT_CONTAINER_NAME}" "${ENROOT_IMAGE_PATH}" || {
            local rc=$?
            _enroot_error "failed to prepare Enroot image '${ENROOT_IMAGE_PATH}'"
            return "${rc}"
        }
    fi
    printf '%s\n' "${ENROOT_IMAGE_PATH:-mapped:${ENROOT_CONTAINER_NAME}}" > "${ENROOT_STATE_DIR}/prepared-image"
    chmod 0600 "${ENROOT_STATE_DIR}/prepared-image"
}

_enroot_require_prepared() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        ENROOT_CONTAINER_NAME="${ENROOT_CONTAINER_NAME:-$(_enroot_image_name)}"
        export ENROOT_CONTAINER_NAME
        return 0
    fi
    ENROOT_CONTAINER_NAME="${ENROOT_CONTAINER_NAME:-$(_enroot_image_name)}"
    export ENROOT_CONTAINER_NAME ENROOT_DATA_PATH ENROOT_CACHE_PATH ENROOT_TEMP_PATH ENROOT_RUNTIME_PATH
    if ! _enroot_image_exists "${ENROOT_CONTAINER_NAME}"; then
        _enroot_error "Enroot image '${ENROOT_CONTAINER_NAME}' is not prepared"
        return 1
    fi
}

_enroot_mount_spec() {
    local volume="$1" host rest destination options create_type
    host="${volume%%:*}"
    rest="${volume#*:}"
    if [[ "${rest}" == "${volume}" || -z "${host}" || -z "${rest}" ]]; then
        _enroot_error "unsupported Enroot volume option '${volume}'"
        return 64
    fi
    destination="${rest%%:*}"
    options=""
    [[ "${rest}" == *:* ]] && options="${rest#*:}"
    if [[ ! -e "${host}" && "${DRY_RUN:-0}" != "1" ]]; then
        _enroot_error "Enroot mount source does not exist: ${host}"
        return 1
    fi
    if [[ "${host}" == *$'\n'* || "${host}" == *$'\t'* || "${destination}" == *$'\n'* || "${destination}" == *$'\t'* ]]; then
        _enroot_error "unsupported Enroot mount path containing control characters"
        return 64
    fi
    create_type=dir
    [[ -f "${host}" ]] && create_type=file
    case "${options}" in
        ""|rw)
            if [[ "${create_type}" == file ]]; then
                printf '%s:%s:rbind,x-create=file\n' "${host}" "${destination}"
            else
                printf '%s:%s\n' "${host}" "${destination}"
            fi
            ;;
        ro)
            printf '%s:%s:rbind,ro,x-create=%s\n' "${host}" "${destination}" "${create_type}"
            ;;
        *)
            _enroot_error "unsupported Enroot volume mode '${options}'"
            return 64
            ;;
    esac
}

_enroot_translate_options() {
    local optsvar="$1"
    if ! declare -p "${optsvar}" >/dev/null 2>&1; then
        _enroot_error "Enroot option array '${optsvar}' does not exist"
        return 64
    fi
    local -n source_options="${optsvar}"
    _ENROOT_MOUNTS=()
    _ENROOT_ENVS=()
    _ENROOT_WORKDIR=""
    _ENROOT_ENTRYPOINT=""
    local index=0 option value spec
    while (( index < ${#source_options[@]} )); do
        option="${source_options[index]}"
        case "${option}" in
            -v|--volume)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                value="${source_options[index + 1]}"
                spec="$(_enroot_mount_spec "${value}")" || return $?
                _ENROOT_MOUNTS+=(--mount "${spec}")
                index=$((index + 2))
                ;;
            -e|--env)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                value="${source_options[index + 1]}"
                [[ "${value}" == *=* || "${value}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || {
                    _enroot_error "unsupported Enroot environment option '${value}'"
                    return 64
                }
                # A bare variable name deliberately preserves Docker's secure
                # `-e NAME` contract: Enroot inherits it from the launcher
                # environment without rendering the value in argv or logs.
                _ENROOT_ENVS+=(--env "${value}")
                index=$((index + 2))
                ;;
            -w|--workdir)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                _ENROOT_WORKDIR="${source_options[index + 1]}"
                index=$((index + 2))
                ;;
            --entrypoint)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                value="${source_options[index + 1]}"
                [[ -n "${value}" && "${value}" != *$'\n'* && "${value}" != *$'\t'* ]] || {
                    _enroot_error "unsupported Enroot entrypoint option"
                    return 64
                }
                _ENROOT_ENTRYPOINT="${value}"
                index=$((index + 2))
                ;;
            --entrypoint=*)
                value="${option#--entrypoint=}"
                [[ -n "${value}" && "${value}" != *$'\n'* && "${value}" != *$'\t'* ]] || {
                    _enroot_error "unsupported Enroot entrypoint option"
                    return 64
                }
                _ENROOT_ENTRYPOINT="${value}"
                index=$((index + 1))
                ;;
            --ipc)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                value="${source_options[index + 1]}"
                [[ "${value}" == host ]] || {
                    _enroot_error "unsupported Enroot IPC option '${value}'"
                    return 64
                }
                index=$((index + 2))
                ;;
            --ipc=host)
                index=$((index + 1))
                ;;
            --ipc=*)
                _enroot_error "unsupported Enroot IPC option '${option#--ipc=}'"
                return 64
                ;;
            --network|--net)
                (( index + 1 < ${#source_options[@]} )) || {
                    _enroot_error "unsupported Enroot option '${option}' without a value"
                    return 64
                }
                value="${source_options[index + 1]}"
                [[ "${value}" == host ]] || {
                    _enroot_error "unsupported Enroot network option '${value}'"
                    return 64
                }
                index=$((index + 2))
                ;;
            --network=host|--net=host)
                index=$((index + 1))
                ;;
            --network=*|--net=*)
                _enroot_error "unsupported Enroot network option '${option#*=}'"
                return 64
                ;;
            *)
                _enroot_error "unsupported Enroot option '${option}'"
                return 64
                ;;
        esac
    done
}

_enroot_command_with_workdir() {
    local -n output_command="$1"
    shift
    local -a requested_command=("$@")
    if [[ -n "${_ENROOT_ENTRYPOINT}" ]]; then
        requested_command=("${_ENROOT_ENTRYPOINT}" "${requested_command[@]}")
    fi
    output_command=("${requested_command[@]}")
    if [[ -n "${_ENROOT_WORKDIR}" ]]; then
        output_command=(
            bash -c 'cd "$0" || exit 127; exec "$@"' \
            "${_ENROOT_WORKDIR}" "${requested_command[@]}"
        )
    fi
}

_enroot_gpu_env() {
    local gpus="$1" explicit_devices=0
    gpus="${gpus//\"/}"
    if [[ "${gpus}" == device=* ]]; then
        explicit_devices=1
        gpus="${gpus#device=}"
    fi
    if [[ "${explicit_devices}" == "0" && "${gpus}" =~ ^[0-9]+$ ]]; then
        if (( gpus < 1 )); then
            _enroot_error "unsupported Enroot GPU count '${gpus}'"
            return 64
        fi
        local index
        local -a devices=()
        for ((index = 0; index < gpus; index++)); do
            devices+=("${index}")
        done
        gpus="$(IFS=,; printf '%s' "${devices[*]}")"
    fi
    if [[ -n "${gpus}" ]]; then
        printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${gpus}"
    else
        printf 'NVIDIA_VISIBLE_DEVICES=void\n'
    fi
}

_enroot_process_starttime() {
    local pid="$1" stat_line rest
    [[ -r "/proc/${pid}/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/${pid}/stat" || return 1
    rest="${stat_line##*) }"
    local -a fields=()
    read -r -a fields <<< "${rest}"
    [[ ${#fields[@]} -gt 19 ]] || return 1
    printf '%s\n' "${fields[19]}"
}

_enroot_process_state() {
    local pid="$1" stat_line rest
    [[ -r "/proc/${pid}/stat" ]] || return 1
    IFS= read -r stat_line < "/proc/${pid}/stat" || return 1
    rest="${stat_line##*) }"
    local -a fields=()
    read -r -a fields <<< "${rest}"
    [[ ${#fields[@]} -gt 0 ]] || return 1
    printf '%s\n' "${fields[0]}"
}

_enroot_own_pgid() {
    local pgid
    pgid="$(ps -o pgid= -p "$$" 2>/dev/null)" || return 1
    pgid="${pgid//[[:space:]]/}"
    printf '%s\n' "${pgid}"
}

_enroot_group_has_runtime_token() {
    local pgid="$1" token="$2" pid candidate_pgid session_id entry
    while read -r pid candidate_pgid session_id; do
        [[ "${candidate_pgid}" == "${pgid}" && "${session_id}" == "${pgid}" ]] || continue
        [[ -r "/proc/${pid}/environ" ]] || continue
        while IFS= read -r -d '' entry; do
            [[ "${entry}" == "AIC_ENROOT_RUNTIME_TOKEN=${token}" ]] && return 0
        done < "/proc/${pid}/environ"
    done < <(ps -eo pid=,pgid=,sid= 2>/dev/null)
    return 1
}

_enroot_owned_group_from_state() {
    local name="$1" pgid token own_pgid
    [[ -f "${ENROOT_PIDDIR}/${name}.pgid" && -f "${ENROOT_PIDDIR}/${name}.token" ]] || return 1
    pgid="$(<"${ENROOT_PIDDIR}/${name}.pgid")"
    token="$(<"${ENROOT_PIDDIR}/${name}.token")"
    own_pgid="$(_enroot_own_pgid)" || return 1
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" -le 1 || "${pgid}" == "${own_pgid}" ]]; then
        return 1
    fi
    kill -0 -- "-${pgid}" 2>/dev/null || return 1
    _enroot_group_has_runtime_token "${pgid}" "${token}" || return 1
    printf '%s\n' "${pgid}"
}

_enroot_remove_process_state() {
    local name="$1"
    rm -f -- \
        "${ENROOT_PIDDIR}/${name}.pid" \
        "${ENROOT_PIDDIR}/${name}.pgid" \
        "${ENROOT_PIDDIR}/${name}.starttime" \
        "${ENROOT_PIDDIR}/${name}.token" \
        "${ENROOT_PIDDIR}/${name}.launch" \
        "${ENROOT_PIDDIR}/${name}.ready"
}

_enroot_persist_process_state() {
    local name="$1" pid="$2" pgid="$3" starttime="$4" token="$5"
    local suffix="tmp.${BASHPID}.${RANDOM}"
    local pid_tmp="${ENROOT_PIDDIR}/${name}.pid.${suffix}"
    local pgid_tmp="${ENROOT_PIDDIR}/${name}.pgid.${suffix}"
    local starttime_tmp="${ENROOT_PIDDIR}/${name}.starttime.${suffix}"
    local token_tmp="${ENROOT_PIDDIR}/${name}.token.${suffix}"
    local rc=0 old_umask
    old_umask="$(umask)"
    umask 077

    printf '%s\n' "${pid}" > "${pid_tmp}" || rc=$?
    if [[ "${rc}" == "0" ]]; then
        printf '%s\n' "${pgid}" > "${pgid_tmp}" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        printf '%s\n' "${starttime}" > "${starttime_tmp}" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        printf '%s\n' "${token}" > "${token_tmp}" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        chmod 0600 "${pid_tmp}" "${pgid_tmp}" "${starttime_tmp}" "${token_tmp}" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        mv -f -- "${pid_tmp}" "${ENROOT_PIDDIR}/${name}.pid" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        mv -f -- "${pgid_tmp}" "${ENROOT_PIDDIR}/${name}.pgid" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        mv -f -- "${starttime_tmp}" "${ENROOT_PIDDIR}/${name}.starttime" || rc=$?
    fi
    if [[ "${rc}" == "0" ]]; then
        mv -f -- "${token_tmp}" "${ENROOT_PIDDIR}/${name}.token" || rc=$?
    fi
    umask "${old_umask}"

    if [[ "${rc}" != "0" ]]; then
        rm -f -- "${pid_tmp}" "${pgid_tmp}" "${starttime_tmp}" "${token_tmp}" || true
        _enroot_remove_process_state "${name}"
        return "${rc}"
    fi
}

_enroot_terminate_pending_launch() {
    local pid="$1" pgid="${2:-}" own_pgid="" safe_private_group=0
    own_pgid="$(_enroot_own_pgid)" || own_pgid=""
    if [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 && \
          "${pgid}" =~ ^[0-9]+$ && "${pgid}" == "${pid}" && \
          -n "${own_pgid}" && "${pgid}" != "${own_pgid}" ]]; then
        safe_private_group=1
    fi
    if [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
    fi
    if [[ "${safe_private_group}" == "1" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
    fi
    sleep 0.05
    if [[ "${safe_private_group}" == "1" ]]; then
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    elif [[ "${pid}" =~ ^[0-9]+$ && "${pid}" -gt 1 ]]; then
        kill -KILL "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
}

_enroot_terminate_owned_group() {
    local name="$1" pgid="" remaining_pgid=""
    pgid="$(_enroot_owned_group_from_state "${name}")" || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    sleep 0.05
    remaining_pgid="$(_enroot_owned_group_from_state "${name}")" || remaining_pgid=""
    if [[ "${remaining_pgid}" == "${pgid}" ]]; then
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
    return 0
}

_enroot_active_pid() {
    local name="$1" pid pgid expected_start actual_start actual_pgid own_pgid process_state
    if [[ ! -f "${ENROOT_PIDDIR}/${name}.pid" || \
          ! -f "${ENROOT_PIDDIR}/${name}.pgid" || \
          ! -f "${ENROOT_PIDDIR}/${name}.starttime" || \
          ! -f "${ENROOT_PIDDIR}/${name}.token" ]]; then
        _enroot_remove_process_state "${name}"
        return 1
    fi
    pid="$(<"${ENROOT_PIDDIR}/${name}.pid")"
    pgid="$(<"${ENROOT_PIDDIR}/${name}.pgid")"
    expected_start="$(<"${ENROOT_PIDDIR}/${name}.starttime")"
    if [[ ! "${pid}" =~ ^[0-9]+$ || ! "${pgid}" =~ ^[0-9]+$ || "${pid}" -le 1 || "${pgid}" -le 1 ]]; then
        _enroot_remove_process_state "${name}"
        return 1
    fi
    kill -0 "${pid}" 2>/dev/null || return 1
    actual_start="$(_enroot_process_starttime "${pid}")" || return 1
    [[ "${actual_start}" == "${expected_start}" ]] || {
        return 1
    }
    process_state="$(_enroot_process_state "${pid}")" || return 1
    [[ "${process_state}" != "Z" && "${process_state}" != "X" ]] || return 1
    actual_pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null)" || return 1
    actual_pgid="${actual_pgid//[[:space:]]/}"
    own_pgid="$(_enroot_own_pgid)" || return 1
    if [[ "${actual_pgid}" != "${pgid}" || "${pgid}" == "${own_pgid}" ]]; then
        _enroot_error "refusing unsafe Enroot process group '${pgid}' for '${name}'"
        return 1
    fi
    printf '%s\n' "${pid}"
}

runtime_enroot_container_exists() {
    _enroot_validate_name "workload" "$1" || return $?
    _enroot_active_pid "$1" >/dev/null || _enroot_owned_group_from_state "$1" >/dev/null
}

runtime_enroot_status() {
    runtime_enroot_container_exists "$1"
}

runtime_enroot_launch_detached() {
    local role="$1" gpus="$2" optsvar="$3"
    shift 3
    if [[ "${1:-}" != "--" ]]; then
        _enroot_error "runtime_launch_detached requires '--' before the command"
        return 64
    fi
    shift
    _enroot_require_prepared || return 1

    local -a merged_options=(-v "${RUN_DIR}:/work")
    local -n caller_options="${optsvar}"
    merged_options+=("${caller_options[@]}")
    _enroot_translate_options merged_options || return $?

    local -a command=()
    _enroot_command_with_workdir command "$@"
    local name="${NAME_PREFIX}-${role}" gpu_env runtime_token
    _enroot_validate_name "workload" "${name}" || return $?
    if [[ ! "${ENROOT_LAUNCH_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
        _enroot_error "invalid Enroot launch timeout '${ENROOT_LAUNCH_TIMEOUT_SECONDS}'"
        return 64
    fi
    local -a rootfs_options=()
    [[ "${_ENROOT_SHARED_CONTAINER:-0}" == "1" ]] || rootfs_options=(--rw)
    gpu_env="$(_enroot_gpu_env "${gpus}")" || return $?
    runtime_token="${NAME_PREFIX}-${role}-${BASHPID}-${RANDOM}"
    local log_path="${ENROOT_PIDDIR}/${name}.log"
    local launch_gate="${ENROOT_PIDDIR}/${name}.launch"
    local launch_ready="${ENROOT_PIDDIR}/${name}.ready"
    local run_prefix="${RUN_DIR%/}/"
    if [[ "${launch_ready}" != "${run_prefix}"* ]]; then
        _enroot_error "Enroot launch readiness path is outside RUN_DIR"
        return 1
    fi
    local launch_ready_container="/work/${launch_ready#"${run_prefix}"}"
    local handoff_script='
import errno
import os
import sys

fifo = sys.argv[1]
command = sys.argv[2:]
if not command:
    raise SystemExit(126)

fd = os.open(fifo, os.O_WRONLY)
os.set_inheritable(fd, False)
os.write(fd, b"ATTEMPT\n")
try:
    os.execvp(command[0], command)
except OSError as error:
    rc = 127 if error.errno == errno.ENOENT else 126
    os.write(fd, f"FAIL:{rc}\n".encode())
    os.close(fd)
    raise SystemExit(rc)
'
    command=(
        python3 -c "${handoff_script}" "${launch_ready_container}" "${command[@]}"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        runtime_invoke "${ENROOT_BIN}" start "${rootfs_options[@]}" \
            "${_ENROOT_MOUNTS[@]}" \
            "${_ENROOT_ENVS[@]}" \
            --env "${gpu_env}" \
            --env "AIC_ENROOT_RUNTIME_TOKEN=${runtime_token}" \
            "${ENROOT_CONTAINER_NAME}" \
            "${command[@]}"
        return 0
    fi
    if runtime_enroot_container_exists "${name}"; then
        _enroot_error "Enroot workload '${name}' is already running"
        return 1
    fi
    _enroot_remove_process_state "${name}"
    if ! mkfifo -m 0600 -- "${launch_ready}"; then
        _enroot_error "failed to create Enroot launch handoff FIFO for '${name}'"
        return 1
    fi
    local launch_bootstrap_fd launch_handoff_fd
    if ! exec {launch_bootstrap_fd}<>"${launch_ready}"; then
        _enroot_remove_process_state "${name}"
        _enroot_error "failed to open Enroot launch handoff FIFO for '${name}'"
        return 1
    fi
    if ! exec {launch_handoff_fd}<"${launch_ready}"; then
        exec {launch_bootstrap_fd}>&-
        _enroot_remove_process_state "${name}"
        _enroot_error "failed to read Enroot launch handoff FIFO for '${name}'"
        return 1
    fi
    exec {launch_bootstrap_fd}>&-
    local parent_pid="${BASHPID}" shell_bin="${BASH:-bash}"
    AIC_ENROOT_RUNTIME_TOKEN="${runtime_token}" \
        "${SETSID_BIN}" "${shell_bin}" -c '
            gate=$1
            parent_pid=$2
            handoff_fd=$3
            shift 3
            exec {handoff_fd}<&-
            while [[ ! -e "${gate}" ]]; do
                kill -0 "${parent_pid}" 2>/dev/null || exit 125
                sleep 0.02
            done
            rm -f -- "${gate}" 2>/dev/null || true
            exec "$@"
        ' _ "${launch_gate}" "${parent_pid}" "${launch_handoff_fd}" \
        "${ENROOT_BIN}" start "${rootfs_options[@]}" \
            "${_ENROOT_MOUNTS[@]}" \
            "${_ENROOT_ENVS[@]}" \
            --env "${gpu_env}" \
            --env "AIC_ENROOT_RUNTIME_TOKEN=${runtime_token}" \
            "${ENROOT_CONTAINER_NAME}" \
            "${command[@]}" >"${log_path}" 2>&1 &
    local pid=$!
    local pgid="" starttime="" own_pgid="" attempt
    for ((attempt = 0; attempt < 200; attempt++)); do
        kill -0 "${pid}" 2>/dev/null || break
        pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null)" || pgid=""
        pgid="${pgid//[[:space:]]/}"
        starttime="$(_enroot_process_starttime "${pid}")" || starttime=""
        [[ "${pgid}" == "${pid}" && -n "${starttime}" ]] && break
        sleep 0.01
    done
    if ! kill -0 "${pid}" 2>/dev/null; then
        local launch_rc=0
        exec {launch_handoff_fd}<&-
        if wait "${pid}"; then
            launch_rc=1
        else
            launch_rc=$?
        fi
        _enroot_remove_process_state "${name}"
        _enroot_log "Enroot workload '${name}' failed during launch (rc=${launch_rc}); log: ${log_path}"
        [[ -f "${log_path}" ]] && cat "${log_path}" >&2
        return "${launch_rc}"
    fi

    own_pgid="$(_enroot_own_pgid)" || own_pgid=""
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" != "${pid}" || "${pgid}" == "${own_pgid}" || -z "${starttime}" ]]; then
        _enroot_terminate_pending_launch "${pid}" "${pgid}"
        exec {launch_handoff_fd}<&-
        _enroot_remove_process_state "${name}"
        _enroot_error "Enroot workload '${name}' did not enter a private process group"
        return 1
    fi
    if ! _enroot_persist_process_state "${name}" "${pid}" "${pgid}" "${starttime}" "${runtime_token}"; then
        _enroot_terminate_pending_launch "${pid}" "${pgid}"
        exec {launch_handoff_fd}<&-
        _enroot_remove_process_state "${name}"
        _enroot_error "failed to persist Enroot workload state for '${name}'"
        return 1
    fi
    if ! : > "${launch_gate}"; then
        _enroot_terminate_pending_launch "${pid}" "${pgid}"
        exec {launch_handoff_fd}<&-
        _enroot_remove_process_state "${name}"
        _enroot_error "failed to release Enroot workload launch gate for '${name}'"
        return 1
    fi

    local launch_start_seconds="${SECONDS}" launch_rc=0 handoff_line="" handoff_read_rc=0
    local handoff_attempted=0 handoff_complete=0
    while [[ "${handoff_complete}" != "1" ]]; do
        handoff_line=""
        handoff_read_rc=0
        if IFS= read -r -t 0.02 -u "${launch_handoff_fd}" handoff_line; then
            if [[ "${handoff_attempted}" == "0" && "${handoff_line}" == "ATTEMPT" ]]; then
                handoff_attempted=1
            elif [[ "${handoff_attempted}" == "1" && "${handoff_line}" =~ ^FAIL:([0-9]+)$ ]]; then
                launch_rc="${BASH_REMATCH[1]}"
                handoff_complete=1
            else
                exec {launch_handoff_fd}<&-
                _enroot_terminate_owned_group "${name}"
                wait "${pid}" 2>/dev/null || true
                _enroot_remove_process_state "${name}"
                _enroot_error "Enroot workload '${name}' produced an invalid command handoff status"
                return 125
            fi
        else
            handoff_read_rc=$?
            if [[ "${handoff_attempted}" == "1" && "${handoff_read_rc}" == "1" ]]; then
                handoff_complete=1
            fi
        fi
        [[ "${handoff_complete}" == "1" ]] && break
        if ! _enroot_active_pid "${name}" >/dev/null; then
            exec {launch_handoff_fd}<&-
            _enroot_terminate_owned_group "${name}"
            if wait "${pid}"; then
                launch_rc=1
            else
                launch_rc=$?
            fi
            _enroot_remove_process_state "${name}"
            _enroot_log "Enroot workload '${name}' failed during launch (rc=${launch_rc}); log: ${log_path}"
            [[ -f "${log_path}" ]] && cat "${log_path}" >&2
            return "${launch_rc}"
        fi
        if (( SECONDS - launch_start_seconds >= ENROOT_LAUNCH_TIMEOUT_SECONDS )); then
            exec {launch_handoff_fd}<&-
            _enroot_terminate_pending_launch "${pid}" "${pgid}"
            _enroot_remove_process_state "${name}"
            _enroot_error "Enroot workload '${name}' timed out before command handoff" || true
            return 124
        fi
        [[ "${handoff_read_rc}" == "1" ]] && sleep 0.02
    done
    exec {launch_handoff_fd}<&-
    if [[ "${launch_rc}" != "0" ]]; then
        _enroot_terminate_owned_group "${name}"
        wait "${pid}" 2>/dev/null || true
        _enroot_remove_process_state "${name}"
        _enroot_log "Enroot workload '${name}' failed command handoff (rc=${launch_rc}); log: ${log_path}"
        [[ -f "${log_path}" ]] && cat "${log_path}" >&2
        return "${launch_rc}"
    fi
    if ! _enroot_active_pid "${name}" >/dev/null; then
        _enroot_terminate_owned_group "${name}"
        if wait "${pid}"; then
            launch_rc=1
        else
            launch_rc=$?
        fi
        _enroot_remove_process_state "${name}"
        _enroot_log "Enroot workload '${name}' exited during command handoff (rc=${launch_rc}); log: ${log_path}"
        [[ -f "${log_path}" ]] && cat "${log_path}" >&2
        return "${launch_rc}"
    fi
    rm -f -- "${launch_ready}"
    _enroot_log "Enroot workload '${name}' started (pid=${pid}, log=${log_path})"
}

runtime_enroot_run_oneshot() {
    local role="$1" gpus="$2" optsvar="$3"
    shift 3
    if [[ "${1:-}" != "--" ]]; then
        _enroot_error "runtime_run_oneshot requires '--' before the command"
        return 64
    fi
    shift
    _enroot_require_prepared || return 1
    _enroot_translate_options "${optsvar}" || return $?
    local -a command=()
    _enroot_command_with_workdir command "$@"
    local gpu_env
    local -a rootfs_options=()
    [[ "${_ENROOT_SHARED_CONTAINER:-0}" == "1" ]] || rootfs_options=(--rw)
    gpu_env="$(_enroot_gpu_env "${gpus}")" || return $?
    runtime_invoke "${ENROOT_BIN}" start "${rootfs_options[@]}" \
        "${_ENROOT_MOUNTS[@]}" \
        "${_ENROOT_ENVS[@]}" \
        --env "${gpu_env}" \
        "${ENROOT_CONTAINER_NAME}" \
        "${command[@]}"
}

runtime_enroot_exec_worker() {
    local pid
    _enroot_validate_name "workload" "${WORKER_NAME}" || return $?
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        runtime_invoke "${ENROOT_BIN}" exec "<pid:${WORKER_NAME}>" "$@"
        return 0
    fi
    pid="$(_enroot_active_pid "${WORKER_NAME}")" || {
        _enroot_error "Enroot worker '${WORKER_NAME}' is not running"
        return 1
    }
    runtime_invoke "${ENROOT_BIN}" exec "${pid}" "$@"
}

_enroot_create_copy_stage() {
    local identifier="copy-${BASHPID}-${RANDOM}"
    _ENROOT_COPY_STAGE_HOST="${ENROOT_STATE_DIR}/${identifier}"
    local relative="${_ENROOT_COPY_STAGE_HOST#"${RUN_DIR%/}/"}"
    if [[ "${relative}" == "${_ENROOT_COPY_STAGE_HOST}" ]]; then
        _enroot_error "Enroot copy staging directory is outside RUN_DIR"
        return 1
    fi
    _ENROOT_COPY_STAGE_CONTAINER="/work/${relative}"
    _enroot_make_private_dir "${_ENROOT_COPY_STAGE_HOST}" || {
        _enroot_error "failed to create private Enroot copy staging directory"
        return 1
    }
}

_enroot_remove_copy_stage() {
    if [[ -n "${_ENROOT_COPY_STAGE_HOST:-}" ]]; then
        rm -rf -- "${_ENROOT_COPY_STAGE_HOST}" || true
    fi
}

runtime_enroot_copy_in() {
    local name="$1" source="$2" destination="$3" pid copy_rc=0
    _enroot_validate_name "workload" "${name}" || return $?
    [[ -e "${source}" ]] || {
        _enroot_error "copy-in source does not exist: ${source}"
        return 1
    }
    pid="$(_enroot_active_pid "${name}")" || {
        _enroot_error "Enroot workload '${name}' is not running"
        return 1
    }
    _enroot_create_copy_stage || return 1
    runtime_invoke cp -a -- "${source}" "${_ENROOT_COPY_STAGE_HOST}/payload" || copy_rc=$?
    if [[ "${copy_rc}" == "0" ]]; then
        runtime_invoke "${ENROOT_BIN}" exec "${pid}" \
            cp -a -- "${_ENROOT_COPY_STAGE_CONTAINER}/payload" "${destination}" || copy_rc=$?
    fi
    _enroot_remove_copy_stage
    return "${copy_rc}"
}

runtime_enroot_copy_out() {
    local name="$1" source="$2" destination="$3" pid copy_rc=0
    _enroot_validate_name "workload" "${name}" || return $?
    pid="$(_enroot_active_pid "${name}")" || {
        _enroot_error "Enroot workload '${name}' is not running"
        return 1
    }
    _enroot_create_copy_stage || return 1
    runtime_invoke "${ENROOT_BIN}" exec "${pid}" \
        cp -a -- "${source}" "${_ENROOT_COPY_STAGE_CONTAINER}/payload" || copy_rc=$?
    if [[ "${copy_rc}" == "0" ]]; then
        runtime_invoke cp -a -- "${_ENROOT_COPY_STAGE_HOST}/payload" "${destination}" || copy_rc=$?
    fi
    _enroot_remove_copy_stage
    return "${copy_rc}"
}

runtime_enroot_logs() {
    local name="$1"
    shift
    _enroot_validate_name "workload" "${name}" || return $?
    local tail_count="" option
    while (($#)); do
        option="$1"
        case "${option}" in
            --tail)
                (($# >= 2)) || {
                    _enroot_error "unsupported Enroot logs option '--tail' without a value"
                    return 64
                }
                tail_count="$2"
                shift 2
                ;;
            --tail=*)
                tail_count="${option#--tail=}"
                shift
                ;;
            *)
                _enroot_error "unsupported Enroot logs option '${option}'"
                return 64
                ;;
        esac
    done
    local log_path="${ENROOT_PIDDIR}/${name}.log"
    [[ -f "${log_path}" ]] || {
        _enroot_error "Enroot log for '${name}' was not found"
        return 1
    }
    if [[ -n "${tail_count}" ]]; then
        [[ "${tail_count}" =~ ^[0-9]+$ ]] || {
            _enroot_error "unsupported Enroot log tail '${tail_count}'"
            return 64
        }
        tail -n "${tail_count}" "${log_path}"
    else
        cat "${log_path}"
    fi
}

runtime_enroot_stop() {
    local name="$1" timeout="${2:-10}" pid="" pgid="" own_pgid start_seconds
    _enroot_validate_name "workload" "${name}" || return $?
    [[ "${timeout}" =~ ^[0-9]+$ ]] || {
        _enroot_error "unsupported Enroot stop timeout '${timeout}'"
        return 64
    }
    pid="$(_enroot_active_pid "${name}")" || pid=""
    pgid="$(_enroot_owned_group_from_state "${name}")" || pgid=""
    if [[ -z "${pid}" && -z "${pgid}" ]]; then
        _enroot_remove_process_state "${name}"
        return 0
    fi
    own_pgid="$(_enroot_own_pgid)" || own_pgid=""
    if [[ -n "${pgid}" && ( ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" -le 1 || "${pgid}" == "${own_pgid}" ) ]]; then
        _enroot_error "refusing unsafe Enroot process group '${pgid}' for '${name}'"
        return 1
    fi

    if [[ -n "${pid}" ]]; then
        kill -TERM "${pid}" 2>/dev/null || true
        start_seconds="${SECONDS}"
        while kill -0 "${pid}" 2>/dev/null; do
            (( SECONDS - start_seconds >= timeout )) && break
            sleep 0.05
        done
        if kill -0 "${pid}" 2>/dev/null && _enroot_active_pid "${name}" >/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        wait "${pid}" 2>/dev/null || true
    fi

    pgid="$(_enroot_owned_group_from_state "${name}")" || pgid=""
    if [[ -n "${pgid}" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        start_seconds="${SECONDS}"
        while kill -0 -- "-${pgid}" 2>/dev/null; do
            (( SECONDS - start_seconds >= timeout )) && break
            sleep 0.05
        done
        if kill -0 -- "-${pgid}" 2>/dev/null && _enroot_owned_group_from_state "${name}" >/dev/null; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        fi
    fi
    _enroot_remove_process_state "${name}"
    return 0
}

runtime_enroot_teardown() {
    local role name timeout
    for role in collector worker frontend; do
        name="${NAME_PREFIX}-${role}"
        timeout=5
        [[ "${role}" == worker ]] && timeout=60
        runtime_enroot_stop "${name}" "${timeout}" || true
    done
    if [[ "${DRY_RUN:-0}" != "1" && "${_ENROOT_SHARED_CONTAINER:-0}" != "1" && -n "${ENROOT_CONTAINER_NAME:-}" ]]; then
        "${ENROOT_BIN}" remove -f "${ENROOT_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    operation="${1:-}"
    case "${operation}" in
        "")
            _enroot_error "unknown Enroot operation: <empty>"
            exit 64
            ;;
        *)
            _enroot_error "unknown Enroot operation '${operation}'"
            exit 64
            ;;
    esac
fi
