# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent-worker contracts for exact SGLang CustomAllReduce collection."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.sdk.resolution.types import MeasurementProtocol

pytestmark = pytest.mark.unit


def _protocol(**overrides: object) -> MeasurementProtocol:
    values: dict[str, object] = {
        "revision": "cuda-event-samples-v1",
        "warmups": 2,
        "samples": 3,
        "statistic": "median",
        "timer": "cuda_event",
        "tuning_revision": "sglang-custom-allreduce-v1",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)  # type: ignore[arg-type]


def _runner():
    return importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce")


def test_uuid_visibility_resolves_exact_noncontiguous_physical_indices() -> None:
    runner = _runner()
    assigned_uuids = ("GPU-c", "GPU-a", "GPU-d", "GPU-b")
    by_uuid = {"GPU-c": 7, "GPU-a": 2, "GPU-d": 11, "GPU-b": 5}
    by_index = {index: device_uuid for device_uuid, index in by_uuid.items()}

    class _Nvml:
        init_calls = 0
        shutdown_calls = 0

        def nvmlInit(self) -> None:  # noqa: N802 - mirrors pynvml
            self.init_calls += 1

        def nvmlShutdown(self) -> None:  # noqa: N802 - mirrors pynvml
            self.shutdown_calls += 1

        def nvmlDeviceGetHandleByUUID(self, device_uuid: str) -> tuple[str, object]:  # noqa: N802
            if device_uuid not in by_uuid:
                raise RuntimeError(f"foreign UUID: {device_uuid}")
            return ("uuid", device_uuid)

        def nvmlDeviceGetIndex(self, handle: tuple[str, object]) -> int:  # noqa: N802
            return by_uuid[str(handle[1])]

        def nvmlDeviceGetHandleByIndex(self, index: int) -> tuple[str, object]:  # noqa: N802
            return ("index", index)

        def nvmlDeviceGetUUID(self, handle: tuple[str, object]) -> str:  # noqa: N802
            return by_index[int(handle[1])]

    nvml = _Nvml()
    physical_indices = runner._physical_indices_for_device_uuids(assigned_uuids, nvml=nvml)

    assert physical_indices == (7, 2, 11, 5)
    assert physical_indices != tuple(range(len(assigned_uuids)))
    assert (nvml.init_calls, nvml.shutdown_calls) == (1, 1)


def test_uuid_visibility_rejects_duplicate_and_non_roundtripping_identity() -> None:
    runner = _runner()

    with pytest.raises(ValueError, match="unique"):
        runner._physical_indices_for_device_uuids(
            ("GPU-a", "GPU-a", "GPU-c", "GPU-d"),
            nvml=SimpleNamespace(),
        )

    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda device_uuid: ("uuid", device_uuid),
        nvmlDeviceGetIndex=lambda handle: {"GPU-a": 4, "GPU-b": 7, "GPU-c": 9, "GPU-d": 12}[handle[1]],
        nvmlDeviceGetHandleByIndex=lambda index: ("index", index),
        nvmlDeviceGetUUID=lambda handle: "GPU-foreign"
        if handle[1] == 9
        else {4: "GPU-a", 7: "GPU-b", 12: "GPU-d"}[handle[1]],
    )
    with pytest.raises(RuntimeError, match="round-trip"):
        runner._physical_indices_for_device_uuids(
            ("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
            nvml=nvml,
        )

    shutdowns: list[str] = []
    foreign_nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: shutdowns.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda device_uuid: (_ for _ in ()).throw(
            RuntimeError(f"foreign UUID: {device_uuid}")
        ),
    )
    with pytest.raises(RuntimeError, match="foreign UUID"):
        runner._physical_indices_for_device_uuids(
            ("GPU-foreign", "GPU-b", "GPU-c", "GPU-d"),
            nvml=foreign_nvml,
        )
    assert shutdowns == ["shutdown"]


def test_rank_process_binds_exact_physical_indices_for_its_uuid_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    assigned_uuids = ("GPU-c", "GPU-a", "GPU-d", "GPU-b")
    resolved_inputs: list[tuple[str, ...]] = []

    def resolve_uuid_indices(device_uuids: tuple[str, ...]) -> tuple[int, ...]:
        resolved_inputs.append(device_uuids)
        return (7, 2, 11, 5)

    bootstrap = SimpleNamespace(rank=2, world_size=4, device_uuid="GPU-d")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(assigned_uuids))

    runner._bind_rank_process_physical_visibility(
        bootstrap,
        resolve_uuid_indices=resolve_uuid_indices,
    )

    assert resolved_inputs == [assigned_uuids]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7,2,11,5"
    assert os.environ["CUDA_VISIBLE_DEVICES"] != "0,1,2,3"


def test_rank_process_keeps_valid_numeric_visibility_without_uuid_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,2,11,5")

    runner._bind_rank_process_physical_visibility(
        SimpleNamespace(rank=2, world_size=4, device_uuid="11"),
        resolve_uuid_indices=lambda _device_uuids: (_ for _ in ()).throw(
            AssertionError("numeric visibility must not invoke UUID resolution")
        ),
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7,2,11,5"


@pytest.mark.parametrize(
    ("visibility", "bootstrap", "resolver", "message"),
    [
        (
            "GPU-a,GPU-b,GPU-c",
            SimpleNamespace(rank=0, world_size=4, device_uuid="GPU-a"),
            lambda _uuids: (0, 1, 2),
            "exactly four",
        ),
        (
            "GPU-a,GPU-b,GPU-c,GPU-d",
            SimpleNamespace(rank=0, world_size=4, device_uuid="GPU-b"),
            lambda _uuids: (0, 1, 2, 3),
            "rank UUID",
        ),
        (
            "GPU-a,GPU-b,GPU-c,GPU-d",
            SimpleNamespace(rank=0, world_size=4, device_uuid="GPU-a"),
            lambda _uuids: (4, 4, 7, 9),
            "unique physical",
        ),
    ],
)
def test_rank_process_visibility_fails_closed_on_lease_or_mapping_drift(
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
    bootstrap: object,
    resolver: object,
    message: str,
) -> None:
    runner = _runner()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)

    with pytest.raises((RuntimeError, ValueError), match=message):
        runner._bind_rank_process_physical_visibility(
            bootstrap,
            resolve_uuid_indices=resolver,
        )


def test_rank_backend_binds_physical_visibility_before_torch_cuda_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    initializer_calls: list[dict[str, object]] = []

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(set_device=lambda rank: events.append(f"torch:{rank}"))  # type: ignore[attr-defined]
    torch_distributed = ModuleType("torch.distributed")

    def init_process_group(**kwargs: object) -> None:
        return None

    torch_distributed.init_process_group = init_process_group  # type: ignore[attr-defined]
    torch.distributed = torch_distributed  # type: ignore[attr-defined]

    parallel_state = ModuleType("sglang.srt.distributed.parallel_state")

    def initialize_model_parallel(**kwargs: object) -> None:
        initializer_calls.append(kwargs)

    parallel_state.get_tp_group = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        ca_comm=SimpleNamespace(disabled=False)
    )
    parallel_state.graph_capture = object()  # type: ignore[attr-defined]
    parallel_state.init_distributed_environment = lambda **kwargs: None  # type: ignore[attr-defined]
    parallel_state.initialize_model_parallel = initialize_model_parallel  # type: ignore[attr-defined]
    parallel_state.set_custom_all_reduce = lambda enabled: None  # type: ignore[attr-defined]
    parallel_state.set_mscclpp_all_reduce = lambda enabled: None  # type: ignore[attr-defined]
    parallel_state.set_torch_symm_mem_all_reduce = lambda enabled: None  # type: ignore[attr-defined]

    server_args = ModuleType("sglang.srt.server_args")
    server_args.set_global_server_args_for_scheduler = lambda args: None  # type: ignore[attr-defined]

    for package_name in ("sglang", "sglang.srt", "sglang.srt.distributed"):
        package = ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", torch_distributed)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    bootstrap = SimpleNamespace(rank=0, world_size=4, device_uuid="GPU-a")

    def visibility_helper(actual_bootstrap: object) -> None:
        assert actual_bootstrap is bootstrap
        events.append("bind")

    monkeypatch.setattr(
        runner,
        "_bind_rank_process_physical_visibility",
        visibility_helper,
        raising=False,
    )
    monkeypatch.setenv("AICONFIGURATOR_NCCL_INIT_METHOD", "file:///tmp/aic-rendezvous")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")

    runner._SglangCustomAllReduceRankBackend(bootstrap).initialize()

    expected_kwargs = {
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
    }
    assert events[:2] == ["bind", "torch:0"]
    assert initializer_calls == [expected_kwargs]


def test_lazy_custom_allreduce_reuses_and_closes_one_four_rank_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiconfigurator.collector import executor

    runner = _runner()
    protocol = _protocol()
    groups: list[Any] = []

    class _RankGroup:
        def __init__(self, *, device_uuids, protocol, worker_target, collect_all_rank_samples) -> None:
            self.device_uuids = device_uuids
            self.protocol_digest = protocol.digest
            self.worker_target = worker_target
            self.collect_all_rank_samples = collect_all_rank_samples
            self.rank_pids = (101, 102, 103, 104)
            self.calls: list[tuple[str, str, int]] = []
            self.close_calls = 0
            groups.append(self)

        def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
            self.calls.append((dtype, operation, element_count))
            return (1.2, 1.25, 1.3)

        def close(self) -> None:
            self.close_calls += 1

    runner.close_custom_allreduce_worker()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(executor, "PersistentNcclRankGroup", _RankGroup)
    monkeypatch.setattr(runner, "_runtime_metadata", lambda: ("0.5.10", "NVIDIA GB200"))

    first = runner.run_custom_allreduce_case("half", 4, 32768, protocol=protocol)
    second = runner.run_custom_allreduce_case("half", 4, 65536, protocol=protocol)
    runner.run_custom_allreduce_case.close_worker()
    runner.run_custom_allreduce_case.close_worker()

    assert len(groups) == 1
    assert groups[0].device_uuids == ("GPU-a", "GPU-b", "GPU-c", "GPU-d")
    assert groups[0].worker_target is runner.custom_allreduce_rank_process_main
    assert groups[0].collect_all_rank_samples is True
    assert groups[0].calls == [
        ("half", "all_reduce", 32768),
        ("half", "all_reduce", 65536),
    ]
    assert groups[0].close_calls == 1
    assert first.protocol_digest == second.protocol_digest == protocol.digest
    assert first.samples_ms == second.samples_ms == (1.2, 1.25, 1.3)
    assert first.latency_ms == second.latency_ms == pytest.approx(1.25)
    assert first.perf_row == {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA GB200",
        "op_name": "all_reduce",
        "kernel_source": "SGLang_CustomAllReduce_graph",
        "allreduce_dtype": "half",
        "num_gpus": 4,
        "message_size": 32768,
        "backend": "sglang_graph",
        "latency": 1.25,
    }
    assert second.perf_row["message_size"] == 65536
    assert first.provenance == {
        "framework": "SGLang",
        "framework_version": "0.5.10",
        "kernel_source": "SGLang_CustomAllReduce_graph",
        "device": "NVIDIA GB200",
        "runtime": "persistent_sglang_custom_allreduce",
        "used_cuda_graph": True,
        "throttled": False,
        "world_size": 4,
        "rank_pids": [101, 102, 103, 104],
        "model_artifact": "sgl-project/DeepSeek-V4-Flash-FP8",
        "physical_dtype": "bfloat16",
    }


def test_custom_allreduce_protocol_and_visibility_fail_before_rank_group_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiconfigurator.collector import executor

    runner = _runner()
    groups: list[object] = []

    def _rank_group(**kwargs: object) -> object:
        groups.append(kwargs)
        raise AssertionError("invalid request acquired a rank group")

    runner.close_custom_allreduce_worker()
    monkeypatch.setattr(executor, "PersistentNcclRankGroup", _rank_group)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    with pytest.raises(ValueError, match=r"protocol|tuning"):
        runner.run_custom_allreduce_case(
            "half",
            4,
            32768,
            protocol=_protocol(tuning_revision="wrong"),
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    with pytest.raises(RuntimeError, match=r"visibility|visible|four|4"):
        runner.run_custom_allreduce_case("half", 4, 32768, protocol=_protocol())

    assert groups == []


def test_custom_allreduce_rejects_noncanonical_cases_before_runtime_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiconfigurator.collector import executor

    runner = _runner()
    groups: list[object] = []

    def _rank_group(**kwargs: object) -> object:
        groups.append(kwargs)
        raise AssertionError("invalid case acquired a rank group")

    runner.close_custom_allreduce_worker()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(executor, "PersistentNcclRankGroup", _rank_group)

    with pytest.raises(ValueError, match=r"half|dtype"):
        runner.run_custom_allreduce_case("fp8", 4, 32768, protocol=_protocol())
    with pytest.raises(ValueError, match=r"world|four|4"):
        runner.run_custom_allreduce_case("half", 2, 32768, protocol=_protocol())
    with pytest.raises(ValueError, match="element"):
        runner.run_custom_allreduce_case("half", 4, 0, protocol=_protocol())

    assert groups == []


def test_rank_backend_proves_custom_kernel_and_retains_captured_storage() -> None:
    runner = _runner()
    tensors: list[object] = []

    class _Graph:
        pass

    class _Cuda:
        CUDAGraph = _Graph

        @staticmethod
        def synchronize(device: object) -> None:
            assert device == "cuda:0"

        @staticmethod
        def graph(graph: object, *, stream: object):
            assert isinstance(graph, _Graph)
            assert stream == "capture-stream"
            return nullcontext()

    class _Torch:
        bfloat16 = "bfloat16"
        cuda = _Cuda()

        @staticmethod
        def ones(shape: tuple[int, ...], *, dtype: object, device: object) -> object:
            tensor = SimpleNamespace(shape=shape, dtype=dtype, device=device)
            tensors.append(tensor)
            return tensor

    class _CustomCommunicator:
        disabled = False

        def __init__(self) -> None:
            self.should_calls: list[object] = []
            self.calls: list[object] = []

        def should_custom_ar(self, value: object) -> bool:
            self.should_calls.append(value)
            return True

        def custom_all_reduce(self, value: object) -> object:
            self.calls.append(value)
            return SimpleNamespace(source=value)

    communicator = _CustomCommunicator()
    backend = runner._SglangCustomAllReduceRankBackend(SimpleNamespace(rank=0))
    backend._torch = _Torch()
    backend._ca_comm = communicator
    backend._graph_capture = lambda: nullcontext(SimpleNamespace(stream="capture-stream"))
    backend._all_reduce = lambda value: (_ for _ in ()).throw(AssertionError("generic all-reduce fallback used"))

    prepared = backend._prepare_case("half", "all_reduce", 32768)

    assert tuple(prepared.inputs) == tuple(tensors)
    assert len(prepared.outputs) == len(prepared.inputs) == 5
    assert communicator.should_calls == [tensors[0]]
    assert communicator.calls == tensors


def test_rank_backend_rejects_shape_when_sglang_custom_kernel_cannot_run() -> None:
    runner = _runner()
    backend = runner._SglangCustomAllReduceRankBackend(SimpleNamespace(rank=0))
    backend._torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(synchronize=lambda device: None),
        ones=lambda *args, **kwargs: object(),
    )
    backend._ca_comm = SimpleNamespace(disabled=False, should_custom_ar=lambda value: False)

    with pytest.raises(RuntimeError, match=r"custom|fallback|unsupported"):
        backend._prepare_case("half", "all_reduce", 32768)
