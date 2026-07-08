# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts joining the packaged NCCL runner to the generic worker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from aiconfigurator.collector import executor
from aiconfigurator.collector.network import nccl as nccl_runner
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

pytestmark = pytest.mark.unit


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )


@dataclass
class _RankGroup:
    device_uuids: tuple[str, ...]
    protocol: MeasurementProtocol
    calls: list[tuple[str, str, int]] = field(default_factory=list)
    close_calls: int = 0

    @property
    def protocol_digest(self) -> str:
        return self.protocol.digest

    @property
    def rank_pids(self) -> tuple[int, ...]:
        return (101, 102)

    def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
        self.calls.append((dtype, operation, element_count))
        return (3.0, 1.0, 2.0)

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _reset_packaged_runtime() -> None:
    nccl_runner.close_nccl_worker()
    yield
    nccl_runner.close_nccl_worker()


def test_packaged_runner_reuses_one_visible_uuid_rank_group_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups: list[_RankGroup] = []

    def rank_group(**kwargs: Any) -> _RankGroup:
        group = _RankGroup(**kwargs)
        groups.append(group)
        return group

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(executor, "PersistentNcclRankGroup", rank_group)
    protocol = _protocol()

    first = nccl_runner.run_nccl_case("half", "all_reduce", 640, 2, protocol=protocol)
    second = nccl_runner.run_nccl_case("half", "all_gather", 1280, 2, protocol=protocol)
    nccl_runner.run_nccl_case.close_worker()
    nccl_runner.run_nccl_case.close_worker()

    assert len(groups) == 1
    assert groups[0].device_uuids == ("GPU-a", "GPU-b")
    assert groups[0].protocol is protocol
    assert groups[0].calls == [
        ("half", "all_reduce", 640),
        ("half", "all_gather", 1280),
    ]
    assert first.samples_ms == second.samples_ms == (3.0, 1.0, 2.0)
    assert first.protocol_digest == second.protocol_digest == protocol.digest
    assert first.statistic == second.statistic == protocol.statistic
    assert groups[0].close_calls == 1


class _Queue:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.puts: list[object] = []

    def get(self) -> object:
        return self.items.pop(0)

    def put(self, value: object) -> None:
        self.puts.append(value)


def test_generic_worker_invokes_packaged_nccl_runner_with_full_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups: list[_RankGroup] = []

    def rank_group(**kwargs: Any) -> _RankGroup:
        group = _RankGroup(**kwargs)
        groups.append(group)
        return group

    monkeypatch.setattr(executor, "PersistentNcclRankGroup", rank_group)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "pre-worker-value")
    protocol = _protocol()
    bootstrap = executor.WorkerBootstrap(
        run_module="aiconfigurator.collector.network.nccl",
        run_func="run_nccl_case",
        adapter_namespace="nccl_perf.txt/v1",
        protocol_digest=protocol.digest,
        device_uuids=("GPU-a", "GPU-b"),
        topology_fingerprint="topology-v1",
    )
    command = executor.WorkerCommand(
        invocation_id="nccl-case-1",
        request_digest="request-digest-1",
        payload=json.dumps(
            {
                "dtype": "half",
                "nccl_op": "alltoall",
                "element_count": 2048,
                "num_gpus": 2,
            }
        ).encode("utf-8"),
        protocol=protocol,
    )
    replies = _Queue()

    executor.worker_process_main(
        bootstrap,
        _Queue(command, None),
        replies,
    )

    assert len(replies.puts) == 1
    reply = replies.puts[0]
    assert reply.invocation_id == command.invocation_id
    assert reply.request_digest == command.request_digest
    assert reply.error is None
    assert reply.raw_result["samples_ms"] == (3.0, 1.0, 2.0)
    assert reply.raw_result["protocol_digest"] == protocol.digest
    assert len(groups) == 1
    assert groups[0].calls == [("half", "alltoall", 2048)]
    assert groups[0].close_calls == 1
