# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the concrete persistent multi-rank NCCL transport."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from aiconfigurator.sdk.resolution.types import MeasurementProtocol

pytestmark = pytest.mark.unit

_OPERATIONS = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")
_API_NAMES = (
    "NcclRankBootstrap",
    "NcclRankCommand",
    "NcclRankReply",
    "PersistentNcclRankGroup",
    "nccl_rank_process_main",
)


def test_uneven_alltoall_uses_rank_specific_receive_splits() -> None:
    executor = importlib.import_module("aiconfigurator.collector.executor")

    assert executor._nccl_alltoall_splits(5, 2, 0) == ((3, 2), (3, 3))
    assert executor._nccl_alltoall_splits(5, 2, 1) == ((3, 2), (2, 2))


def _api() -> SimpleNamespace:
    executor = importlib.import_module("aiconfigurator.collector.executor")
    missing = tuple(name for name in _API_NAMES if not hasattr(executor, name))
    assert not missing, f"missing concrete NCCL rank-group API: {missing!r}"
    return SimpleNamespace(**{name: getattr(executor, name) for name in _API_NAMES})


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )


class _Queue:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.puts: list[object] = []
        self.get_factory: Callable[[], object] | None = None
        self.get_calls = 0

    def get(self) -> object:
        self.get_calls += 1
        if self.get_factory is not None:
            return self.get_factory()
        return self.items.pop(0)

    def put(self, item: object) -> None:
        self.puts.append(item)


@dataclass
class _Process:
    target: object
    args: tuple[object, ...]
    daemon: bool
    start_calls: int = 0
    join_calls: int = 0
    terminate_calls: int = 0
    alive: bool = True

    def start(self) -> None:
        self.start_calls += 1

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1
        self.alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class _SpawnContext:
    def __init__(self) -> None:
        self.queues: list[_Queue] = []
        self.processes: list[_Process] = []

    def Queue(self) -> _Queue:  # noqa: N802 - mirrors multiprocessing
        queue = _Queue()
        self.queues.append(queue)
        return queue

    def Process(  # noqa: N802 - mirrors multiprocessing
        self,
        *,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
    ) -> _Process:
        process = _Process(target=target, args=args, daemon=daemon)
        self.processes.append(process)
        return process


def _rank_queues(context: _SpawnContext) -> tuple[tuple[_Queue, ...], _Queue]:
    reply_queues = [process.args[2] for process in context.processes]
    assert reply_queues and all(queue is reply_queues[0] for queue in reply_queues)
    return tuple(process.args[1] for process in context.processes), reply_queues[0]


def test_rank_group_spawns_one_process_per_assigned_gpu_and_reuses_one_lease() -> None:
    api = _api()
    context = _SpawnContext()
    requested_methods: list[str] = []
    worker_target = object()

    def get_context(method: str) -> _SpawnContext:
        requested_methods.append(method)
        return context

    group = api.PersistentNcclRankGroup(
        device_uuids=("GPU-a", "GPU-b", "GPU-c"),
        protocol=_protocol(),
        context_getter=get_context,
        worker_target=worker_target,
    )

    assert requested_methods == ["spawn"]
    assert len(context.processes) == 3
    for rank, process in enumerate(context.processes):
        bootstrap = process.args[0]
        assert bootstrap == api.NcclRankBootstrap(
            rank=rank,
            world_size=3,
            device_uuid=("GPU-a", "GPU-b", "GPU-c")[rank],
            protocol=_protocol(),
        )
        assert process.target is worker_target
        assert process.daemon is False
        assert process.start_calls == 1

    group.close()
    group.close()
    command_queues, _ = _rank_queues(context)
    assert [queue.puts for queue in command_queues] == [[None], [None], [None]]
    assert [process.join_calls for process in context.processes] == [1, 1, 1]


@dataclass
class _FakeCollectiveBackend:
    fail_on_operation: str | None = None
    initialize_calls: int = 0
    destroy_calls: int = 0
    abort_calls: int = 0
    barrier_calls: int = 0
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def initialize(self) -> None:
        self.initialize_calls += 1

    def run(self, dtype: str, operation: str, element_count: int) -> float:
        self.calls.append((dtype, operation, element_count))
        if operation == self.fail_on_operation:
            raise RuntimeError(f"{operation} rank failure")
        return float(len(self.calls))

    def destroy(self) -> None:
        self.destroy_calls += 1

    def abort(self) -> None:
        self.abort_calls += 1

    def barrier(self) -> None:
        self.barrier_calls += 1


def test_rank_worker_initializes_once_and_honors_each_full_protocol_case() -> None:
    api = _api()
    protocol = _protocol()
    commands = tuple(
        api.NcclRankCommand(
            invocation_id=f"case-{index}",
            dtype="half",
            operation=operation,
            element_count=128 * index,
            protocol=protocol,
        )
        for index, operation in enumerate(_OPERATIONS, start=1)
    )
    command_queue = _Queue(*commands, None)
    reply_queue = _Queue()
    backend = _FakeCollectiveBackend()
    bootstrap = api.NcclRankBootstrap(
        rank=0,
        world_size=2,
        device_uuid="GPU-a",
        protocol=protocol,
    )

    api.nccl_rank_process_main(
        bootstrap,
        command_queue,
        reply_queue,
        runtime_factory=lambda _: backend,
    )

    assert backend.initialize_calls == 1
    assert backend.destroy_calls == 1
    assert backend.abort_calls == 0
    assert backend.barrier_calls == 0
    assert len(backend.calls) == len(_OPERATIONS) * (protocol.warmups + protocol.samples)
    for index, operation in enumerate(_OPERATIONS):
        start = index * (protocol.warmups + protocol.samples)
        assert backend.calls[start : start + protocol.warmups + protocol.samples] == [
            ("half", operation, 128 * (index + 1))
        ] * (protocol.warmups + protocol.samples)

    assert len(reply_queue.puts) == len(_OPERATIONS)
    for command, reply in zip(commands, reply_queue.puts, strict=True):
        assert reply.invocation_id == command.invocation_id
        assert reply.rank == 0
        assert len(reply.samples_ms) == protocol.samples
        assert reply.error is None


def test_rank_group_returns_only_the_correlated_rank0_samples() -> None:
    api = _api()
    context = _SpawnContext()
    group = api.PersistentNcclRankGroup(
        device_uuids=("GPU-a", "GPU-b"),
        protocol=_protocol(),
        context_getter=lambda method: context if method == "spawn" else None,
        worker_target=object(),
    )
    command_queues, reply_queue = _rank_queues(context)
    replies_seen = 0

    def reply() -> object:
        nonlocal replies_seen
        command = command_queues[0].puts[-1]
        rank = (0, 1)[replies_seen]
        replies_seen += 1
        return api.NcclRankReply(
            invocation_id=command.invocation_id,
            rank=rank,
            samples_ms=() if rank else (1.0, 1.5, 2.0),
        )

    reply_queue.get_factory = reply

    samples = group.measure("half", "all_reduce", 4096)

    assert samples == (1.0, 1.5, 2.0)
    assert reply_queue.get_calls == 2
    assert len(command_queues) == 2
    for queue in command_queues:
        command = queue.puts[0]
        assert isinstance(command, api.NcclRankCommand)
        assert command.dtype == "half"
        assert command.operation == "all_reduce"
        assert command.element_count == 4096
        assert command.protocol == _protocol()


def test_rank_failure_poison_terminates_all_ranks_without_waiting_or_barrier() -> None:
    api = _api()
    context = _SpawnContext()
    group = api.PersistentNcclRankGroup(
        device_uuids=("GPU-a", "GPU-b"),
        protocol=_protocol(),
        context_getter=lambda method: context if method == "spawn" else None,
        worker_target=object(),
    )
    command_queues, reply_queue = _rank_queues(context)

    def failed_reply() -> object:
        command = command_queues[0].puts[-1]
        return api.NcclRankReply(
            invocation_id=command.invocation_id,
            rank=1,
            error="rank 1 crashed",
        )

    reply_queue.get_factory = failed_reply

    with pytest.raises(RuntimeError, match="rank 1 crashed"):
        group.measure("half", "alltoall", 2048)

    assert group.poisoned is True
    assert reply_queue.get_calls == 1
    assert [process.terminate_calls for process in context.processes] == [1, 1]
    with pytest.raises(RuntimeError, match="poisoned"):
        group.measure("half", "all_reduce", 2048)

    backend = _FakeCollectiveBackend(fail_on_operation="alltoall")
    api.nccl_rank_process_main(
        api.NcclRankBootstrap(
            rank=1,
            world_size=2,
            device_uuid="GPU-b",
            protocol=_protocol(),
        ),
        _Queue(
            api.NcclRankCommand(
                invocation_id="failed-case",
                dtype="half",
                operation="alltoall",
                element_count=2048,
                protocol=_protocol(),
            )
        ),
        _Queue(),
        runtime_factory=lambda _: backend,
    )
    assert backend.abort_calls == 1
    assert backend.destroy_calls == 0
    assert backend.barrier_calls == 0
