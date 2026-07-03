# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "collector/layerwise/fpm_ground_truth/fpm_collect.py"


class FakeAgainError(Exception):
    pass


class FakeMonitor:
    def __init__(self, events: list[int], calls: list[str] | None = None) -> None:
        self.events = list(events)
        self.calls = calls if calls is not None else []
        self.closed = False

    def poll(self, _timeout: int, _flags: int) -> bool:
        self.calls.append("monitor.poll")
        return bool(self.events)

    def close(self, *, linger: int) -> None:
        assert linger == 0
        self.closed = True


class FakeSocket:
    def __init__(
        self,
        monitor: FakeMonitor,
        calls: list[str],
        payloads: list[list[bytes]],
        empty_callback=None,
    ) -> None:
        self.monitor = monitor
        self.calls = calls
        self.payloads = list(payloads)
        self.empty_callback = empty_callback
        self.recv_calls = 0
        self.closed = False

    def setsockopt(self, option: int, _value: bytes | int) -> None:
        self.calls.append(f"setsockopt:{option}")

    def get_monitor_socket(self) -> FakeMonitor:
        self.calls.append("get_monitor_socket")
        return self.monitor

    def connect(self, endpoint: str) -> None:
        self.calls.append(f"connect:{endpoint}")

    def recv_multipart(self):
        self.recv_calls += 1
        if self.payloads:
            return self.payloads.pop(0)
        if self.empty_callback is not None:
            payload = self.empty_callback()
            if payload is not None:
                return payload
        raise FakeAgainError

    def close(self, *, linger: int) -> None:
        assert linger == 0
        self.closed = True


class FakeContext:
    def __init__(self, socket: FakeSocket) -> None:
        self._socket = socket

    def socket(self, socket_type: int) -> FakeSocket:
        assert socket_type == 1
        return self._socket


def _load_collector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[int] | None = None,
    payloads: list[list[bytes]] | None = None,
    empty_callback=None,
):
    calls: list[str] = []
    constants = {
        "SUB": 1,
        "SUBSCRIBE": 2,
        "RCVTIMEO": 3,
        "POLLIN": 4,
        "EVENT_CONNECT_DELAYED": 10,
        "EVENT_CONNECTED": 11,
        "EVENT_CONNECT_RETRIED": 12,
        "EVENT_DISCONNECTED": 13,
        "EVENT_HANDSHAKE_SUCCEEDED": 20,
        "EVENT_HANDSHAKE_FAILED_NO_DETAIL": 21,
        "EVENT_HANDSHAKE_FAILED_PROTOCOL": 22,
        "EVENT_HANDSHAKE_FAILED_AUTH": 23,
        "EVENT_MONITOR_STOPPED": 24,
    }
    monitor = FakeMonitor(events or [], calls)
    socket = FakeSocket(monitor, calls, payloads or [], empty_callback)
    context = FakeContext(socket)

    zmq = ModuleType("zmq")
    for name, value in constants.items():
        setattr(zmq, name, value)
    zmq.Again = FakeAgainError
    zmq.Context = SimpleNamespace(instance=lambda: context)

    zmq_utils = ModuleType("zmq.utils")
    zmq_utils.__path__ = []  # type: ignore[attr-defined]
    zmq_monitor = ModuleType("zmq.utils.monitor")

    def recv_monitor_message(source: FakeMonitor):
        return {"event": source.events.pop(0)}

    zmq_monitor.recv_monitor_message = recv_monitor_message

    dynamo = ModuleType("dynamo")
    dynamo.__path__ = []  # type: ignore[attr-defined]
    dynamo_common = ModuleType("dynamo.common")
    dynamo_common.__path__ = []  # type: ignore[attr-defined]
    fpm = ModuleType("dynamo.common.forward_pass_metrics")
    fpm.decode = lambda _payload: None
    for name, module in {
        "zmq": zmq,
        "zmq.utils": zmq_utils,
        "zmq.utils.monitor": zmq_monitor,
        "dynamo": dynamo,
        "dynamo.common": dynamo_common,
        "dynamo.common.forward_pass_metrics": fpm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = f"test_fpm_collect_{id(monkeypatch)}"
    spec = importlib.util.spec_from_file_location(module_name, COLLECTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, socket, monitor, calls, constants


def test_subscriber_configures_filter_and_monitor_before_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _socket, _monitor, calls, constants = _load_collector(monkeypatch)

    module.create_subscriber("tcp://127.0.0.1:20380")

    subscribe = calls.index(f"setsockopt:{constants['SUBSCRIBE']}")
    monitor = calls.index("get_monitor_socket")
    connect = calls.index("connect:tcp://127.0.0.1:20380")
    assert subscribe < monitor < connect


def test_wait_requires_handshake_success_not_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _socket, monitor, _calls, constants = _load_collector(monkeypatch)
    monitor.events[:] = [constants["EVENT_CONNECT_DELAYED"], constants["EVENT_CONNECTED"]]
    ticks = iter([0.0, 0.0, 0.01, 1.01])

    with pytest.raises(TimeoutError, match="handshake"):
        module.wait_for_transport_ready(monitor, 1.0, clock=lambda: next(ticks))


@pytest.mark.parametrize(
    "failure_name",
    (
        "EVENT_HANDSHAKE_FAILED_NO_DETAIL",
        "EVENT_HANDSHAKE_FAILED_PROTOCOL",
        "EVENT_HANDSHAKE_FAILED_AUTH",
        "EVENT_MONITOR_STOPPED",
    ),
)
def test_wait_fails_closed_on_terminal_monitor_events(
    monkeypatch: pytest.MonkeyPatch,
    failure_name: str,
) -> None:
    module, _socket, monitor, _calls, constants = _load_collector(monkeypatch)
    monitor.events[:] = [constants[failure_name]]

    with pytest.raises(RuntimeError, match=r"handshake|monitor"):
        module.wait_for_transport_ready(monitor, 1.0)


def test_wait_tolerates_delays_until_handshake_success(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _socket, monitor, _calls, constants = _load_collector(monkeypatch)
    monitor.events[:] = [
        constants["EVENT_CONNECT_DELAYED"],
        constants["EVENT_CONNECT_RETRIED"],
        constants["EVENT_CONNECTED"],
        constants["EVENT_HANDSHAKE_SUCCEEDED"],
    ]

    module.wait_for_transport_ready(monitor, 1.0)


def test_ready_marker_is_atomic_and_leaves_no_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _socket, _monitor, _calls, _constants = _load_collector(monkeypatch)
    marker = tmp_path / "collector.ready"
    replacements: list[tuple[Path, Path]] = []
    real_replace = module.os.replace

    def record_replace(source: Path, destination: Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", record_replace)

    module.publish_ready_marker(marker, "atomic-token")

    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert destination == marker
    assert temporary.parent == marker.parent
    assert temporary != marker
    assert marker.read_text() == "atomic-token\n"
    assert marker.stat().st_mode & 0o777 == 0o644
    assert not temporary.exists()


def test_ready_marker_cleans_temporary_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _socket, _monitor, _calls, _constants = _load_collector(monkeypatch)
    marker = tmp_path / "collector.ready"

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        module.publish_ready_marker(marker, "must-not-publish")

    assert not marker.exists()
    assert list(tmp_path.iterdir()) == []


def test_main_marks_ready_after_headers_without_creating_a_data_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, socket, monitor, _calls, constants = _load_collector(
        monkeypatch,
        [20],
    )
    assert constants["EVENT_HANDSHAKE_SUCCEEDED"] == 20
    output = tmp_path / "fpm.csv"
    detail = tmp_path / "fpm-detail.csv"
    marker = tmp_path / "collector.ready"
    original_publish = module.publish_ready_marker

    def publish_after_headers(path: Path, token: str) -> None:
        assert output.read_text().splitlines() == ["num_context_tokens,num_decode_tokens,latency_ms"]
        assert len(detail.read_text().splitlines()) == 1
        original_publish(path, token)
        module.running = False

    monkeypatch.setattr(module, "publish_ready_marker", publish_after_headers)
    module.running = True

    rc = module.main(
        [
            "--port",
            "20380",
            "--output",
            str(output),
            "--detail-output",
            str(detail),
            "--ready-file",
            str(marker),
            "--ready-token",
            "exact-run-token",
            "--ready-timeout",
            "1",
            "--data-ready-file",
            str(tmp_path / "collector.data-ready"),
            "--data-ready-token",
            "exact-data-token",
        ]
    )

    assert rc == 0
    assert marker.read_text() == "exact-run-token\n"
    assert socket.recv_calls == 0
    assert monitor.closed
    assert socket.closed
    assert output.read_text().splitlines() == ["num_context_tokens,num_decode_tokens,latency_ms"]


def test_handshake_failure_never_publishes_ready_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, _socket, _monitor, _calls, constants = _load_collector(
        monkeypatch,
        [21],
    )
    assert constants["EVENT_HANDSHAKE_FAILED_NO_DETAIL"] == 21
    marker = tmp_path / "collector.ready"

    with pytest.raises(RuntimeError, match="handshake"):
        module.main(
            [
                "--port",
                "20380",
                "--output",
                str(tmp_path / "fpm.csv"),
                "--detail-output",
                str(tmp_path / "fpm-detail.csv"),
                "--ready-file",
                str(marker),
                "--ready-token",
                "must-not-appear",
                "--ready-timeout",
                "1",
                "--data-ready-file",
                str(tmp_path / "collector.data-ready"),
                "--data-ready-token",
                "must-not-appear-data",
            ]
        )

    assert not marker.exists()


def test_missing_handshake_success_capability_fails_before_poll_or_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, socket, monitor, calls, _constants = _load_collector(monkeypatch, [20])
    monkeypatch.delattr(module.zmq, "EVENT_HANDSHAKE_SUCCEEDED")
    marker = tmp_path / "collector.ready"

    with pytest.raises(RuntimeError, match=r"EVENT_HANDSHAKE_SUCCEEDED.*required"):
        module.main(
            [
                "--port",
                "20380",
                "--output",
                str(tmp_path / "fpm.csv"),
                "--detail-output",
                str(tmp_path / "fpm-detail.csv"),
                "--ready-file",
                str(marker),
                "--ready-token",
                "must-not-appear",
                "--ready-timeout",
                "1",
                "--data-ready-file",
                str(tmp_path / "collector.data-ready"),
                "--data-ready-token",
                "must-not-appear-data",
            ]
        )

    assert "monitor.poll" not in calls
    assert not marker.exists()
    assert monitor.closed
    assert socket.closed


def _scheduled_metrics(**updates):
    values = {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "var_prefill_length": 0.0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
        "var_decode_kv_tokens": 0.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _queued_metrics(**updates):
    values = {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "var_prefill_length": 0.0,
        "num_decode_requests": 0,
        "sum_decode_kv_tokens": 0,
        "var_decode_kv_tokens": 0.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _metrics(
    counter_id: int,
    *,
    worker_id: str = "worker-0",
    dp_rank: int = 0,
    wall_time: float = 0.001,
    scheduled=None,
    queued=None,
):
    return SimpleNamespace(
        wall_time=wall_time,
        scheduled_requests=scheduled or _scheduled_metrics(),
        queued_requests=queued or _queued_metrics(),
        counter_id=counter_id,
        worker_id=worker_id,
        dp_rank=dp_rank,
    )


def test_active_then_same_identity_newer_idle_ack_discards_probe_before_measured_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    segment = tmp_path / "segment.txt"
    segment.write_text("readiness-probe\n")
    data_ready = tmp_path / "collector.data-ready"
    measured_sent = False

    def emit_measured_after_ack():
        nonlocal measured_sent
        if data_ready.exists() and not measured_sent:
            measured_sent = True
            segment.write_text("measured\n")
            return [b"topic", b"3", b"measured-payload"]
        return None

    module, socket, _monitor, _calls, _constants = _load_collector(
        monkeypatch,
        [20],
        [
            [b"topic", b"1", b"probe-payload-1"],
            [b"topic", b"2", b"probe-payload-2"],
            [b"topic", b"3", b"probe-idle-heartbeat"],
        ],
        empty_callback=emit_measured_after_ack,
    )
    decoded = {
        b"probe-payload-1": _metrics(
            10,
            scheduled=_scheduled_metrics(num_prefill_requests=1, sum_prefill_tokens=1),
        ),
        b"probe-payload-2": _metrics(
            11,
            scheduled=_scheduled_metrics(num_prefill_requests=1, sum_prefill_tokens=1),
        ),
        b"probe-idle-heartbeat": _metrics(12, wall_time=0.0),
        b"measured-payload": _metrics(
            13,
            wall_time=0.001,
            scheduled=_scheduled_metrics(num_prefill_requests=1, sum_prefill_tokens=8),
        ),
    }

    def decode_payload(payload: bytes):
        return decoded[payload]

    monkeypatch.setattr(module, "decode", decode_payload)
    output = tmp_path / "fpm.csv"
    detail = tmp_path / "fpm-detail.csv"

    rc = module.main(
        [
            "--port",
            "20380",
            "--output",
            str(output),
            "--detail-output",
            str(detail),
            "--segment-file",
            str(segment),
            "--idle-timeout",
            "0.01",
            "--ready-file",
            str(tmp_path / "collector.ready"),
            "--ready-token",
            "row-token",
            "--ready-timeout",
            "1",
            "--data-ready-file",
            str(data_ready),
            "--data-ready-token",
            "data-row-token",
        ]
    )

    assert rc == 0
    assert data_ready.read_text() == "data-row-token\n"
    assert output.read_text().splitlines() == [
        "num_context_tokens,num_decode_tokens,latency_ms",
        "8,0,1.000",
    ]
    detail_lines = detail.read_text().splitlines()
    assert len(detail_lines) == 2
    assert "readiness-probe" not in "\n".join(detail_lines)
    assert socket.recv_calls >= 4


def test_undecodable_probe_payload_never_publishes_data_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    segment = tmp_path / "segment.txt"
    segment.write_text("readiness-probe\n")
    module, _socket, _monitor, _calls, _constants = _load_collector(
        monkeypatch,
        [20],
        [[b"topic", b"1", b"invalid-probe"]],
    )
    monkeypatch.setattr(module, "decode", lambda _payload: None)
    data_ready = tmp_path / "collector.data-ready"

    rc = module.main(
        [
            "--port",
            "20380",
            "--output",
            str(tmp_path / "fpm.csv"),
            "--detail-output",
            str(tmp_path / "fpm-detail.csv"),
            "--segment-file",
            str(segment),
            "--idle-timeout",
            "0.01",
            "--ready-file",
            str(tmp_path / "collector.ready"),
            "--ready-token",
            "transport-token",
            "--ready-timeout",
            "1",
            "--data-ready-file",
            str(data_ready),
            "--data-ready-token",
            "must-not-appear",
        ]
    )

    assert rc == 0
    assert not data_ready.exists()


@pytest.mark.parametrize(
    ("frames", "error"),
    (
        (
            [
                _metrics(1, scheduled=_scheduled_metrics(num_prefill_requests=1)),
                _metrics(
                    2,
                    wall_time=0.0,
                    scheduled=_scheduled_metrics(num_prefill_requests=1),
                ),
            ],
            "nonzero pseudo-idle heartbeat",
        ),
        (
            [
                _metrics(1, scheduled=_scheduled_metrics(num_prefill_requests=1)),
                _metrics(2, worker_id="worker-1", wall_time=0.0),
            ],
            "changed worker/dp identity",
        ),
        (
            [
                _metrics(2, scheduled=_scheduled_metrics(num_prefill_requests=1)),
                _metrics(2, wall_time=0.0),
            ],
            "counter was stale",
        ),
        (
            [
                _metrics(2, scheduled=_scheduled_metrics(num_prefill_requests=1)),
                _metrics(1, scheduled=_scheduled_metrics(num_prefill_requests=1)),
            ],
            "counter was not monotonic",
        ),
    ),
)
def test_invalid_active_idle_barrier_fails_closed_without_data_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frames,
    error: str,
) -> None:
    payloads = [[b"topic", str(index).encode(), b"frame"] for index in range(len(frames))]
    module, _socket, _monitor, _calls, _constants = _load_collector(
        monkeypatch,
        [20],
        payloads,
    )
    decoded = iter(frames)
    monkeypatch.setattr(module, "decode", lambda _payload: next(decoded))
    data_ready = tmp_path / "collector.data-ready"

    with pytest.raises(RuntimeError, match=error):
        module.main(
            [
                "--port",
                "20380",
                "--output",
                str(tmp_path / "fpm.csv"),
                "--detail-output",
                str(tmp_path / "fpm-detail.csv"),
                "--ready-file",
                str(tmp_path / "collector.ready"),
                "--ready-token",
                "transport-token",
                "--ready-timeout",
                "1",
                "--data-ready-file",
                str(data_ready),
                "--data-ready-token",
                "must-not-appear",
            ]
        )

    assert not data_ready.exists()


def test_zero_idle_before_probe_active_is_discarded_until_identity_is_known(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frames = [
        _metrics(8, worker_id="unbound-idle", wall_time=0.0),
        _metrics(
            9,
            worker_id="probe-worker",
            scheduled=_scheduled_metrics(num_prefill_requests=1),
        ),
        _metrics(10, worker_id="probe-worker", wall_time=0.0),
    ]
    payloads = [[b"topic", str(index).encode(), b"frame"] for index in range(len(frames))]
    module, _socket, _monitor, _calls, _constants = _load_collector(
        monkeypatch,
        [20],
        payloads,
    )
    decoded = iter(frames)
    monkeypatch.setattr(module, "decode", lambda _payload: next(decoded))
    data_ready = tmp_path / "collector.data-ready"

    rc = module.main(
        [
            "--port",
            "20380",
            "--output",
            str(tmp_path / "fpm.csv"),
            "--detail-output",
            str(tmp_path / "fpm-detail.csv"),
            "--idle-timeout",
            "0.01",
            "--ready-file",
            str(tmp_path / "collector.ready"),
            "--ready-token",
            "transport-token",
            "--ready-timeout",
            "1",
            "--data-ready-file",
            str(data_ready),
            "--data-ready-token",
            "data-token",
        ]
    )

    assert rc == 0
    assert data_ready.read_text() == "data-token\n"


def test_real_pyzmq_data_path_ack_discards_probe_before_measured_row() -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is required for the real pyzmq readiness integration")
    image = os.environ.get("AIC_FPM_PYZMQ_TEST_IMAGE", "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0")
    inspect = subprocess.run(
        [docker, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        pytest.skip(f"real pyzmq readiness image is unavailable: {image}")

    integration = textwrap.dedent(
        r"""
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        import time

        import zmq
        from dynamo.common.forward_pass_metrics import (
            ForwardPassMetrics,
            ScheduledRequestMetrics,
            encode,
        )

        def wait_for(predicate, child, timeout=8.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if predicate():
                    return
                if child.poll() is not None:
                    output = child.stdout.read() if child.stdout is not None else ""
                    raise AssertionError(f"collector exited rc={child.returncode}: {output}")
                time.sleep(0.02)
            raise AssertionError("timed out waiting for real pyzmq data-path acknowledgment")

        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.bind("tcp://127.0.0.1:*")
        endpoint = publisher.getsockopt_string(zmq.LAST_ENDPOINT)
        port = endpoint.rsplit(":", 1)[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "fpm.csv"
            detail = root / "fpm-detail.csv"
            segment = root / "segment.txt"
            transport_ready = root / "transport.ready"
            data_ready = root / "data.ready"
            segment.write_text("readiness-probe\n")
            child = subprocess.Popen(
                [
                    sys.executable,
                    "/work/fpm_collect.py",
                    "--port",
                    port,
                    "--output",
                    str(output),
                    "--detail-output",
                    str(detail),
                    "--segment-file",
                    str(segment),
                    "--ready-file",
                    str(transport_ready),
                    "--ready-token",
                    "transport-token",
                    "--ready-timeout",
                    "4",
                    "--data-ready-file",
                    str(data_ready),
                    "--data-ready-token",
                    "data-token",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                wait_for(transport_ready.exists, child)
                assert transport_ready.read_text() == "transport-token\n"
                time.sleep(0.2)
                assert not data_ready.exists(), "transport handshake must not claim data-path readiness"

                for sequence in range(1, 4):
                    probe = ForwardPassMetrics(
                        worker_id="integration-worker",
                        counter_id=sequence,
                        wall_time=0.001,
                        scheduled_requests=ScheduledRequestMetrics(
                            num_prefill_requests=1,
                            sum_prefill_tokens=1,
                        ),
                    )
                    publisher.send_multipart(
                        [b"fpm", str(sequence).encode(), encode(probe)]
                    )
                    time.sleep(0.05)
                idle = ForwardPassMetrics(
                    worker_id="integration-worker",
                    counter_id=4,
                )
                publisher.send_multipart([b"fpm", b"4", encode(idle)])
                wait_for(data_ready.exists, child)
                assert data_ready.read_text() == "data-token\n"
                assert output.read_text().splitlines() == [
                    "num_context_tokens,num_decode_tokens,latency_ms"
                ]
                assert len(detail.read_text().splitlines()) == 1

                segment.write_text("measured\n")
                measured = ForwardPassMetrics(
                    worker_id="integration-worker",
                    counter_id=5,
                    wall_time=0.002,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_prefill_requests=1,
                        sum_prefill_tokens=8,
                    ),
                )
                publisher.send_multipart([b"fpm", b"5", encode(measured)])
                wait_for(lambda: len(output.read_text().splitlines()) == 2, child)
                assert output.read_text().splitlines() == [
                    "num_context_tokens,num_decode_tokens,latency_ms",
                    "8,0,2.000",
                ]
                detail_lines = detail.read_text().splitlines()
                assert len(detail_lines) == 2
                assert "readiness-probe" not in "\n".join(detail_lines)
            finally:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=3)
        publisher.close(linger=0)
        context.term()
        """
    )
    result = subprocess.run(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{COLLECTOR.resolve()}:/work/fpm_collect.py:ro",
            "--entrypoint",
            "python3",
            image,
            "-c",
            integration,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
