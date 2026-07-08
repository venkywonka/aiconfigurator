# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the concrete spawn worker transport."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

pytestmark = pytest.mark.unit


def _api() -> SimpleNamespace:
    executor = importlib.import_module("aiconfigurator.collector.executor")
    return SimpleNamespace(
        ProcessWorkerFactory=executor.ProcessWorkerFactory,
        WorkerBootstrap=executor.WorkerBootstrap,
        WorkerCommand=executor.WorkerCommand,
        WorkerReply=executor.WorkerReply,
        worker_process_main=executor.worker_process_main,
    )


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=5,
        statistic="median",
        timer="cuda_event",
        tuning_revision="spawn-worker-v1",
    )


def _bootstrap(api: SimpleNamespace):
    return api.WorkerBootstrap(
        run_module="aiconfigurator.collector.testing.fake_runner",
        run_func="run_case",
        adapter_namespace="fake_perf.txt/v1",
        protocol_digest=_protocol().digest,
        device_uuids=("GPU-stable-a", "GPU-stable-b"),
        topology_fingerprint="topology-v1",
    )


class _Queue:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.puts: list[object] = []
        self._reader = object()
        self.close_calls = 0
        self.join_thread_calls = 0
        self.cancel_join_thread_calls = 0

    def get(self) -> object:
        return self.items.pop(0)

    def put(self, item: object) -> None:
        self.puts.append(item)

    def close(self) -> None:
        self.close_calls += 1

    def join_thread(self) -> None:
        self.join_thread_calls += 1

    def cancel_join_thread(self) -> None:
        self.cancel_join_thread_calls += 1


@dataclass
class _Process:
    target: object
    args: tuple[object, ...]
    daemon: bool
    sentinel: object
    start_calls: int = 0
    join_calls: list[float | None] = field(default_factory=list)
    terminate_calls: int = 0
    kill_calls: int = 0
    alive: bool = True
    finish_on_join: bool = True
    finish_on_terminate: bool = True

    def start(self) -> None:
        self.start_calls += 1

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        if self.finish_on_join:
            self.alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.finish_on_terminate:
            self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False


class _SpawnContext:
    def __init__(self) -> None:
        self.queues: list[_Queue] = []
        self.processes: list[_Process] = []

    def Queue(self) -> _Queue:  # noqa: N802 - multiprocessing's API is capitalized
        queue = _Queue()
        self.queues.append(queue)
        return queue

    def Process(  # noqa: N802 - multiprocessing's API is capitalized
        self,
        *,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
    ) -> _Process:
        process = _Process(target=target, args=args, daemon=daemon, sentinel=object())
        self.processes.append(process)
        return process


def test_worker_loop_binds_uuids_before_one_import_and_propagates_json_case_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    protocol = _protocol()
    commands = (
        api.WorkerCommand(
            invocation_id="invocation-1",
            request_digest="digest-1",
            payload=json.dumps({"m": 8, "nested": {"mode": "bf16"}}).encode("utf-8"),
            protocol=protocol,
        ),
        api.WorkerCommand(
            invocation_id="invocation-2",
            request_digest="digest-2",
            payload=json.dumps({"m": 16, "nested": {"mode": "bf16"}}).encode("utf-8"),
            protocol=protocol,
        ),
    )
    command_queue = _Queue(*commands, None)
    reply_queue = _Queue()
    imports: list[tuple[str, str | None]] = []
    calls: list[dict[str, Any]] = []

    def run_case(*, m: int, nested: dict[str, str], protocol: MeasurementProtocol) -> RawMeasurement:
        calls.append({"m": m, "nested": nested, "protocol": protocol})
        return RawMeasurement(
            latency_ms=float(m),
            energy_wms=1.5,
            samples_ms=(float(m), float(m) + 0.25),
            statistic=protocol.statistic,
            perf_row={"m": m, "mode": nested["mode"]},
            provenance={"runner": "fake"},
            protocol_digest=protocol.digest,
        )

    def import_module(name: str) -> object:
        imports.append((name, os.environ.get("CUDA_VISIBLE_DEVICES")))
        return SimpleNamespace(run_case=run_case)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "ordinal-that-must-not-leak")
    api.worker_process_main(
        _bootstrap(api),
        command_queue,
        reply_queue,
        import_module=import_module,
    )

    assert imports == [("aiconfigurator.collector.testing.fake_runner", "GPU-stable-a,GPU-stable-b")]
    assert calls == [
        {"m": 8, "nested": {"mode": "bf16"}, "protocol": protocol},
        {"m": 16, "nested": {"mode": "bf16"}, "protocol": protocol},
    ]
    assert len(reply_queue.puts) == 2
    for index, reply in enumerate(reply_queue.puts, start=1):
        assert isinstance(reply, api.WorkerReply)
        assert reply.invocation_id == f"invocation-{index}"
        assert reply.request_digest == f"digest-{index}"
        assert reply.error is None
        assert type(reply.raw_result) is dict
        assert reply.raw_result["protocol_digest"] == protocol.digest
        assert reply.raw_result["perf_row"] == {"m": index * 8, "mode": "bf16"}


def test_worker_loop_serializes_traceback_with_the_original_correlation() -> None:
    api = _api()
    command = api.WorkerCommand(
        invocation_id="failed-invocation",
        request_digest="failed-digest",
        payload=b'{"m": 8}',
        protocol=_protocol(),
    )
    command_queue = _Queue(command, None)
    reply_queue = _Queue()

    def run_case(**case: object) -> RawMeasurement:
        raise RuntimeError(f"runner exploded for m={case['m']}")

    api.worker_process_main(
        _bootstrap(api),
        command_queue,
        reply_queue,
        import_module=lambda _: SimpleNamespace(run_case=run_case),
    )

    assert len(reply_queue.puts) == 1
    reply = reply_queue.puts[0]
    assert isinstance(reply, api.WorkerReply)
    assert reply.invocation_id == "failed-invocation"
    assert reply.request_digest == "failed-digest"
    assert reply.raw_result is None
    assert "Traceback (most recent call last)" in reply.error
    assert "RuntimeError: runner exploded for m=8" in reply.error


def test_worker_loop_rejects_protocol_digest_mismatch_before_runner_invocation() -> None:
    api = _api()
    protocol = replace(_protocol(), warmups=99)
    command = api.WorkerCommand(
        invocation_id="mismatched-protocol",
        request_digest="mismatched-digest",
        payload=b'{"m": 8}',
        protocol=protocol,
    )
    command_queue = _Queue(command, None)
    reply_queue = _Queue()
    calls: list[dict[str, object]] = []

    def run_case(**case: object) -> dict[str, object]:
        calls.append(case)
        return case

    api.worker_process_main(
        _bootstrap(api),
        command_queue,
        reply_queue,
        import_module=lambda _: SimpleNamespace(run_case=run_case),
    )

    assert calls == []
    reply = reply_queue.puts[0]
    assert reply.invocation_id == "mismatched-protocol"
    assert reply.request_digest == "mismatched-digest"
    assert reply.raw_result is None
    assert "ValueError: worker command protocol does not match its persistent lease" in reply.error


@pytest.mark.parametrize(
    ("payload", "runner_result", "expected_error"),
    [
        (b"[]", {"latency_ms": 1.0}, "TypeError: worker case payload must decode to a JSON object"),
        (b'{"m": 8}', "not-a-mapping", "TypeError: exact runner must return a raw result mapping"),
    ],
)
def test_worker_loop_rejects_non_object_json_and_non_mapping_runner_results(
    payload: bytes,
    runner_result: object,
    expected_error: str,
) -> None:
    api = _api()
    command = api.WorkerCommand(
        invocation_id="invalid-wire",
        request_digest="invalid-wire-digest",
        payload=payload,
        protocol=_protocol(),
    )
    command_queue = _Queue(command, None)
    reply_queue = _Queue()

    api.worker_process_main(
        _bootstrap(api),
        command_queue,
        reply_queue,
        import_module=lambda _: SimpleNamespace(run_case=lambda **_: runner_result),
    )

    reply = reply_queue.puts[0]
    assert reply.invocation_id == "invalid-wire"
    assert reply.request_digest == "invalid-wire-digest"
    assert reply.raw_result is None
    assert expected_error in reply.error


def test_process_factory_uses_spawn_reuses_channel_closes_joins_and_waits_ready() -> None:
    api = _api()
    context = _SpawnContext()
    requested_methods: list[str] = []
    wait_calls: list[tuple[tuple[object, ...], float]] = []

    def get_context(method: str) -> _SpawnContext:
        requested_methods.append(method)
        return context

    def connection_wait(handles: list[object], timeout: float) -> list[object]:
        wait_calls.append((tuple(handles), timeout))
        return [handles[0], handles[-1]]

    target = object()
    factory = api.ProcessWorkerFactory(
        context_getter=get_context,
        worker_target=target,
        connection_wait=connection_wait,
    )
    left = factory(_bootstrap(api))
    right = factory(_bootstrap(api))

    assert requested_methods == ["spawn"]
    assert len(context.processes) == 2
    assert all(process.target is target for process in context.processes)
    assert all(process.start_calls == 1 for process in context.processes)
    assert context.processes[0].args == (_bootstrap(api), context.queues[0], context.queues[1])

    first = api.WorkerCommand("invocation-1", "digest-1", b"{}", _protocol())
    second = api.WorkerCommand("invocation-2", "digest-2", b"{}", _protocol())
    left.send(first)
    left.send(second)
    assert context.queues[0].puts == [first, second]
    assert len(context.processes) == 2

    ready = tuple(factory.wait_ready((left, right), 1.25))
    assert ready == (left, right)
    assert wait_calls == [
        (
            (
                context.queues[1]._reader,
                context.processes[0].sentinel,
                context.queues[3]._reader,
                context.processes[1].sentinel,
            ),
            1.25,
        )
    ]

    left.close()
    left.close()
    left.join()
    assert context.queues[0].puts == [first, second, None]
    assert context.processes[0].join_calls == [5.0]


def test_wait_ready_maps_a_dead_process_sentinel_to_its_channel() -> None:
    api = _api()
    context = _SpawnContext()
    observed_handles: list[tuple[object, ...]] = []

    def connection_wait(handles: list[object], timeout: float) -> list[object]:
        assert timeout == 0.75
        observed_handles.append(tuple(handles))
        return [handles[-1]]

    factory = api.ProcessWorkerFactory(
        context_getter=lambda method: context if method == "spawn" else None,
        worker_target=object(),
        connection_wait=connection_wait,
    )
    left = factory(_bootstrap(api))
    right = factory(_bootstrap(api))
    context.processes[1].alive = False

    assert tuple(factory.wait_ready((left, right), 0.75)) == (right,)
    assert observed_handles == [
        (
            context.queues[1]._reader,
            context.processes[0].sentinel,
            context.queues[3]._reader,
            context.processes[1].sentinel,
        )
    ]


def test_join_terminates_a_hung_worker_with_bounded_waits_and_cleans_queues() -> None:
    api = _api()
    context = _SpawnContext()
    factory = api.ProcessWorkerFactory(
        context_getter=lambda method: context if method == "spawn" else None,
        worker_target=object(),
        connection_wait=lambda handles, timeout: (),
        shutdown_timeout_seconds=0.125,
    )
    channel = factory(_bootstrap(api))
    process = context.processes[0]
    process.finish_on_join = False
    process.finish_on_terminate = False

    channel.close()
    channel.join()
    channel.join()

    assert process.join_calls == [0.125, 0.125, 0.125]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.alive is False
    assert context.queues[0].puts == [None]
    assert [queue.close_calls for queue in context.queues] == [1, 1]
    assert [queue.cancel_join_thread_calls for queue in context.queues] == [1, 1]
    assert [queue.join_thread_calls for queue in context.queues] == [0, 0]
