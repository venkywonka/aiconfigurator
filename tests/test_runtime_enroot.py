from __future__ import annotations

import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPT = ROOT / "collector/layerwise/fpm_ground_truth/runtime.sh"
ENROOT_BACKEND = ROOT / "collector/layerwise/fpm_ground_truth/runtime_enroot.sh"
COLLECT_SCRIPT = ROOT / "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"


@dataclass(frozen=True)
class RuntimeHarness:
    env: dict[str, str]
    run_dir: Path
    state_dir: Path
    enroot_log: Path
    docker_log: Path
    child_pid_dir: Path

    def run(
        self,
        *args: str,
        env_updates: dict[str, str] | None = None,
        timeout: int = 15,
    ) -> subprocess.CompletedProcess[str]:
        env = {**self.env, **(env_updates or {})}
        return subprocess.run(
            ["/bin/bash", str(self.run_dir / "harness.sh"), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def enroot_calls(self) -> list[list[str]]:
        if not self.enroot_log.exists():
            return []
        return [json.loads(line) for line in self.enroot_log.read_text().splitlines()]

    def docker_calls(self) -> list[list[str]]:
        if not self.docker_log.exists():
            return []
        return [json.loads(line) for line in self.docker_log.read_text().splitlines()]

    def child_pids(self) -> list[int]:
        if not self.child_pid_dir.exists():
            return []
        return [int(path.read_text()) for path in self.child_pid_dir.glob("*.pid")]


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


@pytest.fixture
def runtime_harness(tmp_path: Path) -> RuntimeHarness:
    run_dir = tmp_path / "job-run"
    state_dir = run_dir / ".runtime/enroot"
    fake_bin = tmp_path / "fake-bin"
    child_pid_dir = tmp_path / "fake-enroot-children"
    enroot_log = tmp_path / "enroot.jsonl"
    docker_log = tmp_path / "docker.jsonl"
    signal_log = tmp_path / "signals.log"
    image = tmp_path / "runtime.sqsh"
    home = tmp_path / "home"
    container_root = tmp_path / "container-root"
    for directory in (run_dir, fake_bin, child_pid_dir, home, container_root):
        directory.mkdir(parents=True)
    image.write_text("fake squashfs image")

    fake_enroot = fake_bin / "enroot"
    _write_executable(
        fake_enroot,
        f"""#!{sys.executable}
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def hold_workload():
    print(os.environ.get("FAKE_ENROOT_WORKLOAD_LOG", "contract workload log"), flush=True)
    pid_path = Path(os.environ["FAKE_ENROOT_CHILD_PID_DIR"]) / f"{{os.getpid()}}.pid"
    pid_path.write_text(str(os.getpid()))

    ready_file = os.environ.get("FAKE_FPM_READY_FILE", "")
    if ready_file:
        ready_path = Path(ready_file)
        ready_token = os.environ["FAKE_FPM_READY_TOKEN"]
        ready_mode = os.environ.get("FAKE_FPM_READY_MODE", "ready")
        if ready_mode == "delayed":
            time.sleep(float(os.environ.get("FAKE_FPM_READY_DELAY", "0.2")))
            ready_mode = "ready"
        if ready_mode == "early-death":
            time.sleep(float(os.environ.get("FAKE_FPM_READY_DELAY", "0.4")))
            print("fake collector exited before readiness", flush=True)
            raise SystemExit(42)
        if ready_mode == "stale-check":
            if ready_path.exists() or ready_path.is_symlink():
                print("stale readiness marker survived collector launch", flush=True)
                raise SystemExit(96)
            ready_mode = "ready"
        if ready_mode == "ready":
            temporary = ready_path.with_name(f".{{ready_path.name}}.fake.tmp")
            temporary.write_text(ready_token + "\\n")
            os.replace(temporary, ready_path)
        elif ready_mode == "wrong":
            ready_path.write_text("wrong-run-token\\n")
        elif ready_mode == "symlink":
            target = ready_path.with_name("fake-ready-target")
            target.write_text(ready_token + "\\n")
            ready_path.symlink_to(target)
        elif ready_mode != "never":
            raise SystemExit(f"unknown fake readiness mode: {{ready_mode}}")

    data_ready_file = os.environ.get("FAKE_FPM_DATA_READY_FILE", "")
    data_ready_published = False
    data_ready_started = None
    if data_ready_file and os.environ.get("FAKE_FPM_DATA_READY_MODE") == "stale-check":
        data_ready_path = Path(data_ready_file)
        if data_ready_path.exists() or data_ready_path.is_symlink():
            print("stale data-readiness marker survived collector launch", flush=True)
            raise SystemExit(96)

    def maybe_publish_data_ready():
        nonlocal data_ready_published, data_ready_started
        if not data_ready_file or data_ready_published:
            return
        trigger = Path(os.environ["RUN_DIR"]) / "readiness-probe.trigger"
        if not trigger.exists():
            return
        mode = os.environ.get("FAKE_FPM_DATA_READY_MODE", "ready")
        if mode == "early-death":
            print("fake collector exited before data readiness", flush=True)
            raise SystemExit(43)
        if mode == "never":
            return
        if mode == "delayed":
            if data_ready_started is None:
                data_ready_started = time.monotonic()
            delay = float(os.environ.get("FAKE_FPM_DATA_READY_DELAY", "0.2"))
            if time.monotonic() - data_ready_started < delay:
                return
            mode = "ready"
        if mode == "stale-check":
            mode = "ready"
        path = Path(data_ready_file)
        token = os.environ["FAKE_FPM_DATA_READY_TOKEN"]
        if mode == "ready":
            temporary = path.with_name(f".{{path.name}}.fake.tmp")
            temporary.write_text(token + "\\n")
            os.replace(temporary, path)
        elif mode == "wrong":
            path.write_text("wrong-data-ready-token\\n")
        elif mode == "symlink":
            target = path.with_name("fake-data-ready-target")
            target.write_text(token + "\\n")
            path.symlink_to(target)
        else:
            raise SystemExit(f"unknown fake data-readiness mode: {{mode}}")
        data_ready_published = True

    def stop(signum, _frame):
        with open(os.environ["FAKE_ENROOT_SIGNAL_LOG"], "a") as stream:
            stream.write(f"{{os.getpid()}}:{{signal.Signals(signum).name}}\\n")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        maybe_publish_data_ready()
        time.sleep(0.05)


args = sys.argv[1:]
if args == ["__fake_hold__"]:
    hold_workload()

with open(os.environ["FAKE_ENROOT_LOG"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")

operation = args[0] if args else ""
if operation == "version":
    print("enroot 3.4.1")
    raise SystemExit(0)
if operation == "--version":
    raise SystemExit(64)
if operation == "import":
    if "-o" in args:
        output = Path(args[args.index("-o") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
    raise SystemExit(0)
if operation == "create":
    name = args[args.index("--name") + 1] if "--name" in args else "contract-image"
    (Path(os.environ["ENROOT_DATA_PATH"]) / name).mkdir(parents=True, exist_ok=True)
    raise SystemExit(int(os.environ.get("FAKE_ENROOT_CREATE_RC", "0")))
if operation == "list":
    data = Path(os.environ["ENROOT_DATA_PATH"])
    if data.is_dir():
        for entry in sorted(data.iterdir()):
            if entry.is_dir():
                print(entry.name)
    raise SystemExit(0)
if operation == "remove":
    for name in (item for item in args[1:] if not item.startswith("-")):
        shutil.rmtree(Path(os.environ["ENROOT_DATA_PATH"]) / name, ignore_errors=True)
    raise SystemExit(0)
if operation == "start":
    start_args = args[1:]
    child_env = os.environ.copy()
    index = 0
    while index < len(start_args) and start_args[index].startswith("-"):
        option = start_args[index]
        if option in {"--env", "--mount"}:
            value = start_args[index + 1]
            if option == "--env":
                key, _, env_value = value.partition("=")
                child_env[key] = env_value
            index += 2
        elif option.startswith("--env="):
            key, _, env_value = option.removeprefix("--env=").partition("=")
            child_env[key] = env_value
            index += 1
        elif option.startswith("--mount=") or option in {"--rw", "--root"}:
            index += 1
        else:
            raise SystemExit(64)
    if index >= len(start_args):
        raise SystemExit(64)
    command = start_args[index + 1 :]
    handoff_contract = (
        len(command) >= 4
        and command[0] == "python3"
        and command[1] == "-c"
        and "os.set_inheritable(fd, False)" in command[2]
    )
    if os.environ.get("FAKE_ENROOT_REQUIRE_READY_CONTRACT") == "1" and not handoff_contract:
        parent_pid = os.getppid()
        while os.getppid() == parent_pid:
            time.sleep(0.01)
    if os.environ.get("FAKE_ENROOT_BLOCK_BEFORE_HANDOFF") == "1":
        while True:
            time.sleep(0.05)
    launch_rc = int(os.environ.get("FAKE_ENROOT_LAUNCH_RC", "0"))
    if launch_rc:
        raise SystemExit(launch_rc)
    fork_survivor_rc = int(os.environ.get("FAKE_ENROOT_FORK_SURVIVOR_RC", "0"))
    if fork_survivor_rc:
        survivor_code = '''
import os
from pathlib import Path
import signal
import time

pid_path = Path(os.environ["FAKE_ENROOT_CHILD_PID_DIR"]) / (str(os.getpid()) + ".pid")
pid_path.write_text(str(os.getpid()))

def stop(_signum, _frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.05)
'''
        survivor = subprocess.Popen([sys.executable, "-c", survivor_code], env=child_env)
        survivor_pid_path = Path(os.environ["FAKE_ENROOT_CHILD_PID_DIR"]) / (
            str(survivor.pid) + ".pid"
        )
        for _ in range(100):
            if survivor_pid_path.is_file():
                break
            time.sleep(0.01)
        raise SystemExit(fork_survivor_rc)
    mapped = [item.replace("/work", os.environ["RUN_DIR"]) for item in command]
    long_running = os.environ.get("FAKE_ENROOT_HOLD") == "1"
    if os.environ.get("FAKE_ENROOT_DRIVER_MODE") == "1":
        joined = " ".join(command)
        if "vllm.__version__" in joined:
            print("0.20.1")
            raise SystemExit(0)
        if "fpm_collect.py" in joined:
            output = Path(os.environ["RUN_DIR"]) / "fpm_metrics.csv"
            output.write_text("metric,value\\ncontract,1\\n")
            if "--ready-file" in command:
                ready_file = command[command.index("--ready-file") + 1]
                ready_token = command[command.index("--ready-token") + 1]
                child_env["FAKE_FPM_READY_FILE"] = ready_file.replace("/work", os.environ["RUN_DIR"])
                child_env["FAKE_FPM_READY_TOKEN"] = ready_token
            if "--data-ready-file" in command:
                data_ready_file = command[command.index("--data-ready-file") + 1]
                data_ready_token = command[command.index("--data-ready-token") + 1]
                child_env["FAKE_FPM_DATA_READY_FILE"] = data_ready_file.replace(
                    "/work", os.environ["RUN_DIR"]
                )
                child_env["FAKE_FPM_DATA_READY_TOKEN"] = data_ready_token
        if "send_requests.py" in joined and "--workload-label" in command:
            label = command[command.index("--workload-label") + 1]
            if label == "readiness-probe":
                trigger = Path(os.environ["RUN_DIR"]) / "readiness-probe.trigger"
                trigger.write_text("probe-request-complete\\n")
        module = command[command.index("-m") + 1] if "-m" in command else ""
        long_running = module in {"dynamo.frontend", "dynamo.vllm"} or "fpm_collect.py" in joined
        if not long_running:
            raise SystemExit(0)
    if handoff_contract:
        # Exercise the real injected FIFO/CLOEXEC wrapper. Only replace its payload
        # when the test needs a deterministic long-running workload.
        if long_running:
            mapped = mapped[:4] + [
                sys.executable,
                str(Path(__file__).resolve()),
                "__fake_hold__",
            ]
        os.execvpe(mapped[0], mapped, child_env)
    elif os.environ.get("FAKE_ENROOT_HOLD") != "1":
        if not command:
            raise SystemExit(0)
        result = subprocess.run(mapped, env=child_env, check=False)
        raise SystemExit(result.returncode)
    os.environ.update(child_env)
    hold_workload()
if operation == "exec":
    command = args[2:]
    mapped = [
        item.replace("/work", os.environ["RUN_DIR"]).replace(
            "/container-only", os.environ["FAKE_ENROOT_CONTAINER_ROOT"]
        )
        for item in command
    ]
    result = subprocess.run(mapped, check=False)
    raise SystemExit(result.returncode)
raise SystemExit(64)
""",
    )

    fake_docker = fake_bin / "docker"
    _write_executable(
        fake_docker,
        f"""#!{sys.executable}
import json
import os
import sys

with open(os.environ["FAKE_DOCKER_LOG"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
raise SystemExit(97)
""",
    )

    harness = run_dir / "harness.sh"
    _write_executable(
        harness,
        """#!/usr/bin/env bash
set -Eeuo pipefail

run() { "$@"; }
log() { printf '[harness] %s\n' "$*" >&2; }
die() { printf '[harness ERROR] %s\n' "$*" >&2; exit 1; }
container_exists() { return 1; }

source "$RUNTIME_SCRIPT"

opts=(-v "$RUN_DIR:/work" -e "CONTRACT_ENV=present")

prepare_backend() {
    export FAKE_ENROOT_HOLD=0
    runtime_prepare
}

launch_worker() {
    export FAKE_ENROOT_HOLD=1
    runtime_launch_detached worker "" opts -- sh -c 'while :; do sleep 1; done'
    sleep 0.15
}

operation="${1:?operation required}"
shift
case "$operation" in
    prepare)
        runtime_prepare "$@"
        ;;
    late-run-dir)
        RUN_DIR="$1"
        unset ENROOT_STATE_DIR ENROOT_PIDDIR ENROOT_DATA_PATH
        unset ENROOT_CACHE_PATH ENROOT_TEMP_PATH ENROOT_RUNTIME_PATH
        runtime_prepare
        ;;
    lifecycle)
        prepare_backend
        launch_worker
        runtime_exec_worker sh -c 'printf executed > "$RUN_DIR/exec-marker"'
        runtime_copy_in "$WORKER_NAME" "$1" /work/copied.txt
        runtime_copy_out "$WORKER_NAME" /work/copied.txt "$2"
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        ;;
    copy-unmounted)
        prepare_backend
        launch_worker
        runtime_copy_in "$WORKER_NAME" "$1" /container-only/copied.txt
        runtime_copy_out "$WORKER_NAME" /container-only/copied.txt "$2"
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        ;;
    stop-teardown)
        prepare_backend
        launch_worker
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        runtime_teardown
        ;;
    fast-stop)
        prepare_backend
        launch_worker
        runtime_stop "$WORKER_NAME" 6
        runtime_teardown
        ;;
    gpu-count)
        prepare_backend
        export FAKE_ENROOT_HOLD=1
        runtime_launch_detached worker "$1" opts -- true
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        ;;
    shared-image)
        prepare_backend
        launch_worker
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        ;;
    status)
        prepare_backend
        launch_worker
        runtime_container_exists "$WORKER_NAME"
        runtime_status "$WORKER_NAME"
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        if runtime_container_exists "$WORKER_NAME"; then
            die "runtime_container_exists reported a torn-down worker"
        fi
        if runtime_status "$WORKER_NAME"; then
            die "runtime_status reported a torn-down worker as running"
        fi
        ;;
    oneshot)
        prepare_backend
        runtime_run_oneshot probe "" opts -- \
            sh -c 'printf %s "$CONTRACT_ENV" > /work/oneshot-marker'
        runtime_teardown
        ;;
    oneshot-rc)
        prepare_backend
        set +e
        runtime_run_oneshot probe "" opts -- sh -c "exit $1"
        rc=$?
        set -e
        runtime_teardown
        exit "$rc"
        ;;
    logs)
        prepare_backend
        launch_worker
        runtime_logs "$WORKER_NAME" --tail 1 > "$RUN_DIR/runtime-logs.txt"
        runtime_stop "$WORKER_NAME" 1
        runtime_teardown
        ;;
    launch-failure)
        prepare_backend
        export FAKE_ENROOT_HOLD=1
        runtime_launch_detached worker "" opts -- true
        ;;
    exec-handoff-failure)
        prepare_backend
        export FAKE_ENROOT_HOLD=0
        runtime_launch_detached worker "" opts -- /definitely/missing-aic-command
        ;;
    state-write-failure)
        prepare_backend
        export FAKE_ENROOT_HOLD=1
        name="$NAME_PREFIX-worker"
        : > "$ENROOT_PIDDIR/$name.log"
        chmod 0600 "$ENROOT_PIDDIR/$name.log"
        chmod 0500 "$ENROOT_PIDDIR"
        runtime_launch_detached worker "" opts -- true
        ;;
    exec-rc)
        prepare_backend
        launch_worker
        set +e
        runtime_exec_worker sh -c "exit $1"
        rc=$?
        set -e
        runtime_teardown
        exit "$rc"
        ;;
    signal)
        prepare_backend
        launch_worker
        trap 'runtime_teardown; exit 143' TERM INT
        kill -TERM "$$"
        sleep 5
        ;;
    orphan-stop)
        prepare_backend
        token="orphan-${BASHPID}-${RANDOM}"
        child_file="$RUN_DIR/orphan-child.pid"
        "$SETSID_BIN" bash -c \
            'export AIC_ENROOT_RUNTIME_TOKEN="$1"; sleep 30 & echo $! > "$2"' \
            _ "$token" "$child_file" &
        leader=$!
        for _ in {1..100}; do
            [[ -s "$child_file" ]] && break
            sleep 0.01
        done
        child="$(<"$child_file")"
        for _ in {1..100}; do
            kill -0 "$leader" 2>/dev/null || break
            sleep 0.01
        done
        printf '%s\n' "$leader" > "$ENROOT_PIDDIR/$WORKER_NAME.pid"
        printf '%s\n' "$leader" > "$ENROOT_PIDDIR/$WORKER_NAME.pgid"
        printf '%s\n' 1 > "$ENROOT_PIDDIR/$WORKER_NAME.starttime"
        printf '%s\n' "$token" > "$ENROOT_PIDDIR/$WORKER_NAME.token"
        runtime_stop "$WORKER_NAME" 1
        if kill -0 "$child" 2>/dev/null; then
            kill -KILL -- "-$leader" 2>/dev/null || true
            exit 93
        fi
        ;;
    orphan-status)
        prepare_backend
        token="orphan-status-${BASHPID}-${RANDOM}"
        child_file="$RUN_DIR/orphan-status-child.pid"
        "$SETSID_BIN" bash -c \
            'export AIC_ENROOT_RUNTIME_TOKEN="$1"; sleep 30 & echo $! > "$2"' \
            _ "$token" "$child_file" &
        leader=$!
        for _ in {1..100}; do
            [[ -s "$child_file" ]] && break
            sleep 0.01
        done
        child="$(<"$child_file")"
        for _ in {1..100}; do
            kill -0 "$leader" 2>/dev/null || break
            sleep 0.01
        done
        printf '%s\n' "$leader" > "$ENROOT_PIDDIR/$WORKER_NAME.pid"
        printf '%s\n' "$leader" > "$ENROOT_PIDDIR/$WORKER_NAME.pgid"
        printf '%s\n' 1 > "$ENROOT_PIDDIR/$WORKER_NAME.starttime"
        printf '%s\n' "$token" > "$ENROOT_PIDDIR/$WORKER_NAME.token"
        runtime_container_exists "$WORKER_NAME"
        if runtime_status "$WORKER_NAME"; then
            kill -KILL -- "-$leader" 2>/dev/null || true
            exit 94
        fi
        runtime_stop "$WORKER_NAME" 1
        if kill -0 "$child" 2>/dev/null; then
            kill -KILL -- "-$leader" 2>/dev/null || true
            exit 95
        fi
        ;;
    unknown-option)
        prepare_backend
        case "${2:-unknown}" in
            unknown) bad_opts=(--definitely-unknown option-value) ;;
            ipc) bad_opts=(--ipc=private) ;;
            network) bad_opts=(--network none) ;;
            *) die "unknown option fixture: $2" ;;
        esac
        case "$1" in
            launch)
                runtime_launch_detached worker "" bad_opts -- true
                ;;
            oneshot)
                runtime_run_oneshot probe "" bad_opts -- true
                ;;
            *)
                die "unknown unknown-option target: $1"
                ;;
        esac
        ;;
    supported-outer-options)
        prepare_backend
        outer_opts=(--entrypoint bash --ipc=host --network=host)
        runtime_run_oneshot probe "device=0" outer_opts -- -lc true
        runtime_teardown
        ;;
    unsafe-name)
        NAME_PREFIX="$1"
        prepare_backend
        runtime_launch_detached worker "" opts -- true
        ;;
    unsafe-log-name)
        prepare_backend
        outside="$ENROOT_PIDDIR/../../outside.log"
        mkdir -p "$(dirname "$outside")"
        printf 'outside-sentinel\n' > "$outside"
        runtime_logs "../../outside"
        ;;
    *)
        die "unknown harness operation: $operation"
        ;;
esac
""",
    )

    env = {
        **os.environ,
        "DOCKER_BIN": str(fake_docker),
        "DRY_RUN": "0",
        "ENROOT_BIN": str(fake_enroot),
        "ENROOT_DATA_PATH": str(state_dir / "data"),
        "ENROOT_IMAGE": str(image),
        "ENROOT_IMAGE_PATH": str(image),
        "ENROOT_PIDDIR": str(state_dir / "pids"),
        "ENROOT_STATE_DIR": str(state_dir),
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_ENROOT_CONTAINER_ROOT": str(container_root),
        "FAKE_ENROOT_CHILD_PID_DIR": str(child_pid_dir),
        "FAKE_ENROOT_LOG": str(enroot_log),
        "FAKE_ENROOT_SIGNAL_LOG": str(signal_log),
        "HOME": str(home),
        "IMAGE": str(image),
        "NAME_PREFIX": "contract",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "RUN_DIR": str(run_dir),
        "RUNTIME": "enroot",
        "RUNTIME_SCRIPT": str(RUNTIME_SCRIPT),
        "RUNTIME_STATE_DIR": str(run_dir / ".runtime"),
        "WORKER_NAME": "contract-worker",
    }
    runtime = RuntimeHarness(
        env=env,
        run_dir=run_dir,
        state_dir=state_dir,
        enroot_log=enroot_log,
        docker_log=docker_log,
        child_pid_dir=child_pid_dir,
    )
    yield runtime

    for pid in runtime.child_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _assert_reason(output: str, pattern: str) -> None:
    assert re.search(pattern, output, flags=re.IGNORECASE), output


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _collect_driver_environment(
    runtime_harness: RuntimeHarness,
    case_name: str,
    **updates: str,
) -> tuple[Path, dict[str, str]]:
    driver_run_dir = runtime_harness.run_dir / f"readiness-{case_name}"
    state_dir = driver_run_dir / ".runtime/enroot"
    driver_run_dir.mkdir()
    env = {
        **runtime_harness.env,
        "DRY_RUN": "0",
        "ENROOT_CACHE_PATH": str(state_dir / "cache"),
        "ENROOT_DATA_PATH": str(state_dir / "data"),
        "ENROOT_PIDDIR": str(state_dir / "pids"),
        "ENROOT_RUNTIME_PATH": str(state_dir / "runtime"),
        "ENROOT_STATE_DIR": str(state_dir),
        "ENROOT_TEMP_PATH": str(state_dir / "tmp"),
        "FAKE_ENROOT_DRIVER_MODE": "1",
        "FPM_COLLECTOR_DATA_READY_TIMEOUT_SECONDS": "2",
        "FPM_COLLECTOR_READY_TIMEOUT_SECONDS": "2",
        "FPM_READINESS_PROBE_IN_SKIP_REQUESTS": "1",
        "HF_HOME": str(driver_run_dir / "hf-home"),
        "MODEL": "Qwen/Qwen3-32B",
        "NAME_PREFIX": f"readiness-{case_name}",
        "POST_REQUEST_COLLECT_SECONDS": "0",
        "REAL_WORKLOAD": "0",
        "RUN_DIR": str(driver_run_dir),
        "SKIP_REQUESTS": "1",
        "START_TIMEOUT_SECONDS": "1",
        "TP_SIZE": "1",
        "WARMUP_REQUESTS": "0",
        **updates,
    }
    return driver_run_dir, env


def _run_collect_driver(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(COLLECT_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_enroot_prepare_fails_closed_when_executable_is_missing(
    runtime_harness: RuntimeHarness,
) -> None:
    missing = runtime_harness.run_dir / "missing-enroot"
    result = runtime_harness.run(
        "prepare",
        env_updates={
            "ENROOT_BIN": str(missing),
            "PATH": str(runtime_harness.run_dir),
        },
    )

    assert result.returncode != 0
    _assert_reason(
        _combined_output(result),
        r"enroot (executable|command).*(missing|not found|unavailable)",
    )
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_fails_closed_when_image_is_missing(
    runtime_harness: RuntimeHarness,
) -> None:
    missing = runtime_harness.run_dir / "missing-image.sqsh"
    result = runtime_harness.run(
        "prepare",
        env_updates={
            "ENROOT_IMAGE": str(missing),
            "ENROOT_IMAGE_PATH": str(missing),
            "IMAGE": str(missing),
        },
    )

    assert result.returncode != 0
    _assert_reason(
        _combined_output(result),
        r"enroot image.*(missing|not found|unavailable)|"
        r"(missing|not found|unavailable).*enroot image",
    )
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_uses_only_job_owned_state(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run("prepare")

    assert result.returncode == 0, _combined_output(result)
    assert runtime_harness.state_dir.is_dir()
    assert runtime_harness.state_dir.resolve().is_relative_to(runtime_harness.run_dir.resolve())
    state_paths = (
        runtime_harness.state_dir,
        Path(runtime_harness.env["ENROOT_PIDDIR"]),
        Path(runtime_harness.env["ENROOT_DATA_PATH"]),
    )
    assert all(path.is_dir() for path in state_paths)
    assert {_mode(path) for path in state_paths} == {0o700}
    assert any(call and call[0] in {"import", "create"} for call in runtime_harness.enroot_calls())
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_derives_default_state_after_run_dir_is_finalized(
    runtime_harness: RuntimeHarness,
) -> None:
    final_run_dir = runtime_harness.run_dir / "final-run-dir"
    final_run_dir.mkdir()

    result = runtime_harness.run("late-run-dir", str(final_run_dir))

    assert result.returncode == 0, _combined_output(result)
    assert (final_run_dir / ".runtime/enroot").is_dir()
    assert runtime_harness.docker_calls() == []


def test_enroot_reuses_explicit_precreated_image_without_mutating_shared_data(
    runtime_harness: RuntimeHarness,
) -> None:
    shared_data = runtime_harness.run_dir.parent / "shared-enroot-data"
    shared_container = shared_data / "shared-contract-image"
    shared_container.mkdir(parents=True)
    image = runtime_harness.env["IMAGE"]

    result = runtime_harness.run(
        "shared-image",
        env_updates={
            "ENROOT_DATA_PATH": str(shared_data),
            "ENROOT_IMAGE_MAP": f"{image}=shared-contract-image",
        },
    )

    assert result.returncode == 0, _combined_output(result)
    operations = [call[0] for call in runtime_harness.enroot_calls() if call]
    assert "create" not in operations
    assert "remove" not in operations
    start_call = next(call for call in runtime_harness.enroot_calls() if call[0] == "start")
    assert "shared-contract-image" in start_call
    assert "--rw" not in start_call
    assert shared_container.is_dir()
    assert runtime_harness.docker_calls() == []


def test_enroot_rejects_option_like_mapped_container_name(
    runtime_harness: RuntimeHarness,
) -> None:
    shared_data = runtime_harness.run_dir.parent / "unsafe-shared-enroot-data"
    (shared_data / "-root").mkdir(parents=True)
    image = runtime_harness.env["IMAGE"]

    result = runtime_harness.run(
        "prepare",
        env_updates={
            "ENROOT_DATA_PATH": str(shared_data),
            "ENROOT_IMAGE_MAP": f"{image}=-root",
        },
    )

    assert result.returncode != 0
    _assert_reason(_combined_output(result), r"invalid.*(container|image).*name")
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_rejects_state_directory_outside_run_dir(
    runtime_harness: RuntimeHarness,
) -> None:
    outside = runtime_harness.run_dir.parent / "outside-enroot-state"
    result = runtime_harness.run(
        "prepare",
        env_updates={"ENROOT_STATE_DIR": str(outside)},
    )

    assert result.returncode != 0
    _assert_reason(
        _combined_output(result),
        r"enroot state.*(outside|within|under).*run[_ ]dir|"
        r"run[_ ]dir.*enroot state.*(outside|within|under)",
    )
    assert not outside.exists()
    assert runtime_harness.enroot_calls() == []
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_rejects_pid_directory_outside_run_dir(
    runtime_harness: RuntimeHarness,
) -> None:
    outside = runtime_harness.run_dir.parent / "outside-enroot-pids"
    result = runtime_harness.run(
        "prepare",
        env_updates={"ENROOT_PIDDIR": str(outside)},
    )

    assert result.returncode != 0
    _assert_reason(
        _combined_output(result),
        r"enroot pid.*(outside|within|under).*run[_ ]dir|"
        r"run[_ ]dir.*enroot pid.*(outside|within|under)",
    )
    assert not outside.exists()
    assert runtime_harness.enroot_calls() == []
    assert runtime_harness.docker_calls() == []


def test_enroot_prepare_rejects_symlinked_state_directory(
    runtime_harness: RuntimeHarness,
) -> None:
    outside = runtime_harness.run_dir.parent / "symlink-target"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    runtime_root = runtime_harness.run_dir / ".runtime"
    runtime_root.mkdir(mode=0o700)
    state_link = runtime_root / "linked-enroot-state"
    state_link.symlink_to(outside, target_is_directory=True)

    result = runtime_harness.run(
        "prepare",
        env_updates={
            "ENROOT_STATE_DIR": str(state_link),
            "ENROOT_PIDDIR": str(state_link / "pids"),
            "ENROOT_DATA_PATH": str(state_link / "data"),
        },
    )

    assert result.returncode != 0
    _assert_reason(_combined_output(result), r"enroot state.*(symlink|symbolic link)")
    assert list(outside.iterdir()) == []
    assert _mode(outside) == 0o755
    assert runtime_harness.enroot_calls() == []
    assert runtime_harness.docker_calls() == []


def test_enroot_launch_exec_copy_stop_and_teardown(runtime_harness: RuntimeHarness) -> None:
    source = runtime_harness.run_dir / "copy source.txt"
    copied_out = runtime_harness.run_dir / "copy result.txt"
    source.write_text("copy payload")

    result = runtime_harness.run("lifecycle", str(source), str(copied_out))

    assert result.returncode == 0, _combined_output(result)
    assert (runtime_harness.run_dir / "exec-marker").read_text() == "executed"
    assert copied_out.read_text() == "copy payload"
    operations = [call[0] for call in runtime_harness.enroot_calls() if call]
    assert "start" in operations
    assert "exec" in operations
    assert runtime_harness.docker_calls() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_copy_parity_works_for_unmounted_container_paths(
    runtime_harness: RuntimeHarness,
) -> None:
    source = runtime_harness.run_dir / "unmounted copy source.txt"
    copied_out = runtime_harness.run_dir / "unmounted copy result.txt"
    source.write_text("unmounted copy payload")

    result = runtime_harness.run("copy-unmounted", str(source), str(copied_out))

    assert result.returncode == 0, _combined_output(result)
    assert copied_out.read_text() == "unmounted copy payload"
    assert runtime_harness.docker_calls() == []


def test_enroot_stop_and_teardown_are_signal_safe_and_idempotent(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("stop-teardown")

    assert result.returncode == 0, _combined_output(result)
    signal_log = Path(runtime_harness.env["FAKE_ENROOT_SIGNAL_LOG"])
    assert "SIGTERM" in signal_log.read_text()
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_graceful_stop_does_not_wait_out_timeout_for_exited_child(
    runtime_harness: RuntimeHarness,
) -> None:
    started = time.monotonic()
    result = runtime_harness.run("fast-stop", timeout=12)
    elapsed = time.monotonic() - started

    assert result.returncode == 0, _combined_output(result)
    # Leave enough host-load margin for process startup while still proving
    # that stop did not consume the configured six-second timeout.
    assert elapsed < 4.0, f"graceful child exit took {elapsed:.2f}s"


def test_enroot_docker_style_gpu_count_maps_to_visible_device_range(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("gpu-count", "2")

    assert result.returncode == 0, _combined_output(result)
    start_call = next(call for call in runtime_harness.enroot_calls() if call[0] == "start")
    env_values = [start_call[index + 1] for index, value in enumerate(start_call[:-1]) if value == "--env"]
    assert "NVIDIA_VISIBLE_DEVICES=0,1" in env_values


def test_enroot_container_exists_and_status_track_lifecycle(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("status")

    assert result.returncode == 0, _combined_output(result)
    assert runtime_harness.docker_calls() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_oneshot_runs_with_translated_options(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run("oneshot")

    assert result.returncode == 0, _combined_output(result)
    assert (runtime_harness.run_dir / "oneshot-marker").read_text() == "present"
    assert any(call and call[0] == "start" for call in runtime_harness.enroot_calls())
    assert runtime_harness.docker_calls() == []


def test_enroot_oneshot_preserves_inner_status_after_teardown(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("oneshot-rc", "41")

    assert result.returncode == 41, _combined_output(result)
    assert runtime_harness.docker_calls() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_logs_reads_detached_workload_log(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run(
        "logs",
        env_updates={"FAKE_ENROOT_WORKLOAD_LOG": "enroot-log-contract-token"},
    )

    assert result.returncode == 0, _combined_output(result)
    captured = runtime_harness.run_dir / "runtime-logs.txt"
    assert captured.read_text().strip() == "enroot-log-contract-token"
    assert runtime_harness.docker_calls() == []


def test_collect_driver_non_dry_enroot_path_never_invokes_docker(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir = runtime_harness.run_dir / "real-driver"
    state_dir = driver_run_dir / ".runtime/enroot"
    driver_run_dir.mkdir()
    env = {
        **runtime_harness.env,
        "DRY_RUN": "0",
        "ENROOT_CACHE_PATH": str(state_dir / "cache"),
        "ENROOT_DATA_PATH": str(state_dir / "data"),
        "ENROOT_PIDDIR": str(state_dir / "pids"),
        "ENROOT_RUNTIME_PATH": str(state_dir / "runtime"),
        "ENROOT_STATE_DIR": str(state_dir),
        "ENROOT_TEMP_PATH": str(state_dir / "tmp"),
        "FAKE_ENROOT_DRIVER_MODE": "1",
        "HF_HOME": str(driver_run_dir / "hf-home"),
        "MODEL": "Qwen/Qwen3-32B",
        "NAME_PREFIX": "driver-contract",
        "POST_REQUEST_COLLECT_SECONDS": "0",
        "REAL_WORKLOAD": "0",
        "RUN_DIR": str(driver_run_dir),
        "SKIP_REQUESTS": "1",
        "START_TIMEOUT_SECONDS": "1",
        "TP_SIZE": "1",
        "WARMUP_REQUESTS": "0",
    }

    try:
        result = subprocess.run(
            ["/bin/bash", str(COLLECT_SCRIPT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"collect driver timed out; stdout={error.stdout!r}; stderr={error.stderr!r}")

    assert result.returncode == 0, _combined_output(result)
    assert runtime_harness.docker_calls() == []
    operations = [call[0] for call in runtime_harness.enroot_calls() if call]
    assert "create" in operations
    assert operations.count("start") >= 6
    assert "remove" in operations
    assert (driver_run_dir / "fpm_metrics.csv").is_file()


def test_collect_driver_waits_for_delayed_exact_readiness_marker(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "delayed",
        FAKE_FPM_READY_MODE="delayed",
        FAKE_FPM_READY_DELAY="0.2",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "FPM collector transport handshake is ready" in output
    assert output.index("FPM collector transport handshake is ready") < output.index(
        "Skipping measured sample requests after readiness probe"
    )
    assert not (driver_run_dir / "fpm_collector.ready").exists()
    assert runtime_harness.docker_calls() == []


@pytest.mark.parametrize("ready_mode", ("wrong", "symlink"))
def test_collect_driver_rejects_wrong_or_symlink_readiness_marker(
    runtime_harness: RuntimeHarness,
    ready_mode: str,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        ready_mode,
        FAKE_FPM_READY_MODE=ready_mode,
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"invalid.*FPM collector readiness marker")
    assert "FPM collector transport handshake is ready" not in output
    assert "Skipping sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_removes_stale_marker_before_waiting(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "stale",
        FAKE_FPM_READY_MODE="stale-check",
    )
    marker = driver_run_dir / "fpm_collector.ready"
    marker.write_text("readiness-stale-24831-17342\n")

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "FPM collector transport handshake is ready" in output
    assert "stale readiness marker survived" not in output
    assert not marker.exists()
    assert runtime_harness.docker_calls() == []


def test_collect_driver_fails_when_collector_dies_before_readiness(
    runtime_harness: RuntimeHarness,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "early-death",
        FAKE_FPM_READY_MODE="early-death",
        FAKE_FPM_READY_DELAY="0.4",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"collector.*exited before.*readiness")
    assert "fake collector exited before readiness" in output
    assert "Skipping sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_times_out_without_readiness_marker(
    runtime_harness: RuntimeHarness,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "timeout",
        FAKE_FPM_READY_MODE="never",
        FPM_COLLECTOR_READY_TIMEOUT_SECONDS="1",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"timed out.*FPM collector.*handshake")
    assert "Skipping sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_waits_for_delayed_data_ready_marker_before_skip(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "data-delayed",
        FAKE_FPM_DATA_READY_MODE="delayed",
        FAKE_FPM_DATA_READY_DELAY="0.2",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "FPM collector decoded data path is ready" in output
    assert output.index("FPM collector decoded data path is ready") < output.index(
        "Skipping measured sample requests after readiness probe"
    )
    assert "--workload-label readiness-probe" in " ".join(
        item for call in runtime_harness.enroot_calls() for item in call
    )
    assert not (driver_run_dir / "fpm_collector.data-ready").exists()
    assert runtime_harness.docker_calls() == []


def test_collect_driver_skip_requests_defaults_to_transport_only(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "transport-only",
        FAKE_FPM_DATA_READY_MODE="never",
        FPM_READINESS_PROBE_IN_SKIP_REQUESTS="0",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)
    command = " ".join(item for call in runtime_harness.enroot_calls() for item in call)

    assert result.returncode == 0, output
    assert "FPM collector transport handshake is ready" in output
    assert ("SKIP_REQUESTS manual mode is transport-ready only; decoded data readiness was not claimed") in output
    assert "FPM collector decoded data path is ready" not in output
    assert "--workload-label readiness-probe" not in command
    assert "--data-ready-file" not in command
    assert not (driver_run_dir / "fpm_collector.data-ready").exists()
    assert runtime_harness.docker_calls() == []


@pytest.mark.parametrize("data_ready_mode", ("wrong", "symlink"))
def test_collect_driver_rejects_wrong_or_symlink_data_ready_marker(
    runtime_harness: RuntimeHarness,
    data_ready_mode: str,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        f"data-{data_ready_mode}",
        FAKE_FPM_DATA_READY_MODE=data_ready_mode,
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"invalid.*FPM collector data-readiness marker")
    assert "Skipping measured sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_removes_stale_data_ready_marker_before_probe(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "data-stale",
        FAKE_FPM_DATA_READY_MODE="stale-check",
    )
    marker = driver_run_dir / "fpm_collector.data-ready"
    marker.write_text("data-readiness-stale-24831-17342\n")

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "FPM collector decoded data path is ready" in output
    assert "stale data-readiness marker survived" not in output
    assert not marker.exists()
    assert runtime_harness.docker_calls() == []


def test_collect_driver_fails_when_collector_dies_before_data_readiness(
    runtime_harness: RuntimeHarness,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "data-early-death",
        FAKE_FPM_DATA_READY_MODE="early-death",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"collector.*exited before.*data readiness")
    assert "fake collector exited before data readiness" in output
    assert "Skipping measured sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_times_out_with_live_collector_but_no_data_ack(
    runtime_harness: RuntimeHarness,
) -> None:
    _driver_run_dir, env = _collect_driver_environment(
        runtime_harness,
        "data-timeout",
        FAKE_FPM_DATA_READY_MODE="never",
        FPM_COLLECTOR_DATA_READY_TIMEOUT_SECONDS="1",
    )

    result = _run_collect_driver(env)
    output = _combined_output(result)

    assert result.returncode != 0
    _assert_reason(output, r"timed out.*FPM collector.*data")
    assert "Skipping measured sample requests" not in output
    assert runtime_harness.docker_calls() == []


def test_collect_driver_cleans_partially_created_image_when_prepare_fails(
    runtime_harness: RuntimeHarness,
) -> None:
    driver_run_dir = runtime_harness.run_dir / "prepare-failure-driver"
    state_dir = driver_run_dir / ".runtime/enroot"
    driver_run_dir.mkdir()
    env = {
        **runtime_harness.env,
        "DRY_RUN": "0",
        "ENROOT_CACHE_PATH": str(state_dir / "cache"),
        "ENROOT_DATA_PATH": str(state_dir / "data"),
        "ENROOT_PIDDIR": str(state_dir / "pids"),
        "ENROOT_RUNTIME_PATH": str(state_dir / "runtime"),
        "ENROOT_STATE_DIR": str(state_dir),
        "ENROOT_TEMP_PATH": str(state_dir / "tmp"),
        "FAKE_ENROOT_CREATE_RC": "73",
        "FAKE_ENROOT_DRIVER_MODE": "1",
        "HF_HOME": str(driver_run_dir / "hf-home"),
        "MODEL": "Qwen/Qwen3-32B",
        "NAME_PREFIX": "prepare-failure-contract",
        "REAL_WORKLOAD": "0",
        "RUN_DIR": str(driver_run_dir),
        "SKIP_REQUESTS": "1",
        "TP_SIZE": "1",
    }

    result = subprocess.run(
        ["/bin/bash", str(COLLECT_SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode != 0
    operations = [call[0] for call in runtime_harness.enroot_calls() if call]
    assert "create" in operations
    assert "remove" in operations
    assert runtime_harness.docker_calls() == []


def test_enroot_launch_failure_preserves_status_and_leaves_no_live_process(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run(
        "launch-failure",
        env_updates={
            "FAKE_ENROOT_LAUNCH_RC": "73",
            "FAKE_ENROOT_REQUIRE_READY_CONTRACT": "1",
        },
    )

    assert result.returncode == 73, _combined_output(result)
    assert runtime_harness.docker_calls() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []
    assert runtime_harness.child_pids() == []


def test_enroot_launch_reports_actual_command_exec_failure(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("exec-handoff-failure")

    assert result.returncode == 127, _combined_output(result)
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_launch_handoff_timeout_reaps_pre_exec_process_group(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run(
        "launch-failure",
        env_updates={
            "ENROOT_LAUNCH_TIMEOUT_SECONDS": "1",
            "FAKE_ENROOT_BLOCK_BEFORE_HANDOFF": "1",
        },
        timeout=5,
    )

    assert result.returncode == 124, _combined_output(result)
    _assert_reason(_combined_output(result), r"timed out before command handoff")
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_post_gate_launch_failure_reaps_token_owned_process_group(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run(
        "launch-failure",
        env_updates={
            "FAKE_ENROOT_FORK_SURVIVOR_RC": "73",
            "FAKE_ENROOT_REQUIRE_READY_CONTRACT": "1",
        },
    )

    deadline = time.monotonic() + 2
    alive: list[int] = []
    while time.monotonic() < deadline:
        alive = []
        for pid in runtime_harness.child_pids():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            break
        time.sleep(0.05)

    assert result.returncode == 73, _combined_output(result)
    assert alive == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_non_private_launch_cleanup_never_signals_collectors_process_group(
    runtime_harness: RuntimeHarness,
) -> None:
    passthrough_setsid = runtime_harness.run_dir / "passthrough-setsid"
    _write_executable(passthrough_setsid, '#!/usr/bin/env bash\nexec "$@"\n')

    result = subprocess.run(
        ["/bin/bash", str(runtime_harness.run_dir / "harness.sh"), "launch-failure"],
        cwd=ROOT,
        env={**runtime_harness.env, "SETSID_BIN": str(passthrough_setsid)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        start_new_session=True,
    )

    assert result.returncode == 1, _combined_output(result)
    _assert_reason(_combined_output(result), r"did not enter a private process group")
    assert runtime_harness.child_pids() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_state_persistence_failure_cannot_strand_launched_process(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("state-write-failure")
    pid_dir = Path(runtime_harness.env["ENROOT_PIDDIR"])
    pid_dir.chmod(0o700)

    deadline = time.monotonic() + 2
    alive: list[int] = []
    while time.monotonic() < deadline:
        alive = []
        for pid in runtime_harness.child_pids():
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            break
        time.sleep(0.05)

    assert result.returncode != 0
    assert alive == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []
    assert runtime_harness.docker_calls() == []


def test_enroot_workload_status_survives_teardown(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run("exec-rc", "37")

    assert result.returncode == 37, _combined_output(result)
    assert runtime_harness.docker_calls() == []
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_signal_cleanup_preserves_signal_status(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run("signal")

    assert result.returncode == 143, _combined_output(result)
    signal_log = Path(runtime_harness.env["FAKE_ENROOT_SIGNAL_LOG"])
    assert "SIGTERM" in signal_log.read_text()
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []


def test_enroot_teardown_reaps_owned_descendants_after_leader_exit(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("orphan-stop")

    assert result.returncode == 0, _combined_output(result)
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []
    assert runtime_harness.docker_calls() == []


def test_enroot_status_requires_live_leader_while_exists_preserves_cleanup_group(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("orphan-status")

    assert result.returncode == 0, _combined_output(result)
    assert list(runtime_harness.state_dir.rglob("*.pid")) == []
    assert runtime_harness.docker_calls() == []


def test_enroot_unknown_operation_is_rejected(runtime_harness: RuntimeHarness) -> None:
    result = subprocess.run(
        ["/bin/bash", str(ENROOT_BACKEND), "definitely-unknown-operation"],
        cwd=ROOT,
        env=runtime_harness.env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode != 0
    _assert_reason(_combined_output(result), r"(unknown|unsupported) (enroot )?operation")
    assert runtime_harness.docker_calls() == []


@pytest.mark.parametrize("operation", ["launch", "oneshot"])
@pytest.mark.parametrize("option_fixture", ["unknown", "ipc", "network"])
def test_enroot_unknown_option_is_rejected(
    runtime_harness: RuntimeHarness,
    operation: str,
    option_fixture: str,
) -> None:
    result = runtime_harness.run("unknown-option", operation, option_fixture)

    assert result.returncode != 0
    _assert_reason(_combined_output(result), r"(unknown|unsupported).*option")
    assert not any(call and call[0] == "start" for call in runtime_harness.enroot_calls())
    assert runtime_harness.docker_calls() == []


def test_enroot_supports_outer_layerwise_entrypoint_and_host_ipc(
    runtime_harness: RuntimeHarness,
) -> None:
    result = runtime_harness.run("supported-outer-options")

    assert result.returncode == 0, _combined_output(result)
    start_calls = [call for call in runtime_harness.enroot_calls() if call and call[0] == "start"]
    assert start_calls
    assert start_calls[-1][-3:] == ["bash", "-lc", "true"]
    assert not any(item.startswith("--ipc") for item in start_calls[-1])
    assert runtime_harness.docker_calls() == []


def test_enroot_rejects_path_escaping_workload_name(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run(
        "unsafe-name",
        "../../escape",
        env_updates={"DRY_RUN": "1"},
    )

    assert result.returncode != 0
    _assert_reason(_combined_output(result), r"invalid.*(container|workload|runtime).*name")
    assert runtime_harness.docker_calls() == []


def test_enroot_logs_rejects_path_escaping_name(runtime_harness: RuntimeHarness) -> None:
    result = runtime_harness.run("unsafe-log-name")

    assert result.returncode != 0
    assert "outside-sentinel" not in _combined_output(result)
    _assert_reason(_combined_output(result), r"invalid.*(workload|runtime).*name")
    assert runtime_harness.docker_calls() == []
