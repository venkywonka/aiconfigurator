# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts joining the packaged NCCL runner to the generic worker."""

from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector import executor
from aiconfigurator.collector.network import nccl as nccl_runner
from aiconfigurator.collector.types import RawMeasurement
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
def _reset_packaged_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                nccl=SimpleNamespace(version=lambda: (2, 27, 7)),
                get_device_name=lambda: "NVIDIA GB200",
            )
        ),
    )
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


def test_canonical_nccl_runner_has_no_offline_subprocess_fallback() -> None:
    source = inspect.getsource(nccl_runner)

    assert "_run_nccl_tests_case" not in source
    assert "subprocess" not in source


def test_offline_nccl_sweep_logs_only_canonical_runner_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from collector.network import collect_nccl

    rows = [
        RawMeasurement(
            latency_ms=1.25,
            energy_wms=125.0,
            samples_ms=(1.25,),
            statistic="median",
            perf_row={
                "nccl_dtype": "half",
                "op_name": "all_reduce",
                "num_gpus": 2,
                "message_size": 640,
                "latency": 1.25,
            },
            provenance={
                "framework": "TRTLLM",
                "framework_version": "2.27.3",
                "device": "NVIDIA GB200",
                "kernel_source": "NCCL",
            },
            protocol_digest=None,
            power_stats={"power": 100.0, "power_limit": 1200.0},
        ),
        RawMeasurement(
            latency_ms=2.5,
            energy_wms=0.0,
            samples_ms=(2.5,),
            statistic="median",
            perf_row={
                "nccl_dtype": "half",
                "op_name": "all_reduce",
                "num_gpus": 2,
                "message_size": 1280,
                "latency": 2.5,
            },
            provenance={
                "framework": "TRTLLM",
                "framework_version": "2.27.3",
                "device": "NVIDIA GB200",
                "kernel_source": "NCCL",
            },
            protocol_digest=None,
            power_stats=None,
        ),
    ]
    case_calls = []
    runner_calls = []
    log_calls = []
    close_calls = []

    def get_cases(*args):
        case_calls.append(args)
        return (
            ("half", "all_reduce", 640, 2),
            ("half", "all_reduce", 1280, 2),
        )

    monkeypatch.setattr(collect_nccl, "get_nccl_test_cases", get_cases)
    monkeypatch.setattr(
        collect_nccl,
        "run_nccl_case",
        lambda *args, **kwargs: runner_calls.append((args, kwargs)) or rows[len(runner_calls) - 1],
    )
    monkeypatch.setattr(collect_nccl, "log_perf", lambda **kwargs: log_calls.append(kwargs))
    monkeypatch.setattr(collect_nccl, "close_nccl_worker", lambda: close_calls.append(True), raising=False)

    result = collect_nccl.nccl_benchmark(
        "half",
        "all_reduce",
        "1280,2561,2",
        2,
    )

    assert result is None
    assert case_calls == [("half", "all_reduce", "1280,2561,2", 2)]
    assert [args for args, _kwargs in runner_calls] == [
        ("half", "all_reduce", 640, 2),
        ("half", "all_reduce", 1280, 2),
    ]
    protocols = [kwargs["protocol"] for _args, kwargs in runner_calls]
    assert all(protocol.tuning_revision == "torch-nccl-persistent-v1" for protocol in protocols)
    assert protocols[0] is protocols[1]
    assert close_calls == [True]
    assert [call["item_list"] for call in log_calls] == [[dict(rows[0].perf_row)], [dict(rows[1].perf_row)]]
    assert all(call["perf_filename"] == "nccl_perf.txt" for call in log_calls)


@pytest.mark.parametrize(
    ("initial_visibility", "num_gpus", "expected_visibility"),
    [
        (None, 2, "0,1"),
        ("GPU-a,GPU-b,GPU-c,GPU-d", 2, "GPU-a,GPU-b"),
        ("0,1,2,3,4,5,6,7", 4, "0,1,2,3"),
        ("0,1,2,3,4,5,6,7", 8, "0,1,2,3,4,5,6,7"),
    ],
)
def test_offline_nccl_sweep_binds_exact_requested_visibility(
    monkeypatch: pytest.MonkeyPatch,
    initial_visibility: str | None,
    num_gpus: int,
    expected_visibility: str,
) -> None:
    from collector.network import collect_nccl

    if initial_visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial_visibility)
    observed_visibility: list[str | None] = []
    monkeypatch.setattr(
        collect_nccl,
        "get_nccl_test_cases",
        lambda *args: (("half", "all_reduce", 640, num_gpus),),
    )
    monkeypatch.setattr(
        collect_nccl,
        "run_nccl_case",
        lambda *args, **kwargs: observed_visibility.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        or RawMeasurement(
            latency_ms=1.0,
            energy_wms=0.0,
            samples_ms=(1.0,),
            statistic="median",
            perf_row={
                "nccl_dtype": "half",
                "op_name": "all_reduce",
                "num_gpus": num_gpus,
                "message_size": 640,
                "latency": 1.0,
            },
            provenance={
                "framework": "NCCL",
                "framework_version": "test",
                "device": "test-gpu",
                "kernel_source": "NCCL",
            },
        ),
    )
    monkeypatch.setattr(collect_nccl, "log_perf", lambda **kwargs: None)
    monkeypatch.setattr(collect_nccl, "close_nccl_worker", lambda: None)

    collect_nccl.nccl_benchmark("half", "all_reduce", "512,1024,2", num_gpus)

    assert observed_visibility == [expected_visibility]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == initial_visibility


@pytest.mark.parametrize("initial_visibility", ["", "GPU-a"])
def test_offline_nccl_sweep_rejects_explicit_undersized_visibility(
    monkeypatch: pytest.MonkeyPatch,
    initial_visibility: str,
) -> None:
    from collector.network import collect_nccl

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial_visibility)

    with pytest.raises(RuntimeError, match="does not have enough visible GPUs"):
        collect_nccl.nccl_benchmark("half", "all_reduce", "512,1024,2", 2)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == initial_visibility


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
