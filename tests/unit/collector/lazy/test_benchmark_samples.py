# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for per-replay collector benchmark samples."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_BENCHMARK_MODULE = "aiconfigurator.collector.benchmark"


class _FakeEvent:
    def __init__(self, owner: _FakeDeviceModule, index: int) -> None:
        self._owner = owner
        self.index = index

    def record(self) -> None:
        self._owner.timeline.append(f"record:{self.index}")

    def elapsed_time(self, end_event: _FakeEvent) -> float:
        assert end_event.index == self.index + 1
        assert self.index % 2 == 0
        pair_index = self.index // 2
        self._owner.timeline.append(f"elapsed:{pair_index}")
        return self._owner.elapsed_ms[pair_index]


class _FakeDeviceModule:
    def __init__(self, elapsed_ms: list[float]) -> None:
        self.elapsed_ms = elapsed_ms
        self.events: list[_FakeEvent] = []
        self.timeline: list[str] = []

    def is_available(self) -> bool:
        return True

    def Event(self, *, enable_timing: bool) -> _FakeEvent:  # noqa: N802 - mirrors Torch
        assert enable_timing is True
        event = _FakeEvent(self, len(self.events))
        self.events.append(event)
        return event

    def synchronize(self) -> None:
        self.timeline.append("sync")

    def empty_cache(self) -> None:
        self.timeline.append("empty_cache")


class _UnavailableDeviceModule:
    @staticmethod
    def is_available() -> bool:
        return False


def _fake_torch(device_module: _FakeDeviceModule) -> ModuleType:
    torch = ModuleType("torch")
    torch.cuda = device_module
    torch.xpu = _UnavailableDeviceModule()
    return torch


def _load_benchmark(monkeypatch: pytest.MonkeyPatch, device_module: _FakeDeviceModule):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(device_module))
    module = importlib.import_module(_BENCHMARK_MODULE)
    monkeypatch.setattr(module, "get_device_module", lambda: device_module)
    return module


def _assert_one_sync_after_measured_series(
    device_module: _FakeDeviceModule,
    *,
    first_event_index: int,
    event_count: int,
) -> None:
    measured_records = [f"record:{index}" for index in range(first_event_index, first_event_index + event_count)]
    positions = [device_module.timeline.index(record) for record in measured_records]
    assert positions == sorted(positions)

    first_elapsed = min(
        index
        for index, action in enumerate(device_module.timeline)
        if action.startswith("elapsed:") and index > positions[-1]
    )
    between_records_and_elapsed = device_module.timeline[positions[0] : first_elapsed]
    assert between_records_and_elapsed.count("sync") == 1
    assert between_records_and_elapsed[-1] == "sync"


def test_default_path_keeps_aggregate_two_event_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    device_module = _FakeDeviceModule([60.0])
    benchmark = _load_benchmark(monkeypatch, device_module)

    with benchmark.benchmark_with_power(
        device=SimpleNamespace(index=0),
        kernel_func=lambda: None,
        num_warmups=0,
        num_runs=3,
        repeat_n=2,
        measure_power=False,
        use_cuda_graph=False,
    ) as result:
        assert result["latency_ms"] == pytest.approx(10.0)
        assert "samples_ms" not in result
        assert result["used_cuda_graph"] is False
        assert result["power_stats"] is None

    assert len(device_module.events) == 2


def test_return_samples_times_each_replay_then_synchronizes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_module = _FakeDeviceModule([12.0, 4.0, 8.0])
    benchmark = _load_benchmark(monkeypatch, device_module)

    with benchmark.benchmark_with_power(
        device=SimpleNamespace(index=0),
        kernel_func=lambda: None,
        num_warmups=0,
        num_runs=3,
        repeat_n=2,
        measure_power=False,
        use_cuda_graph=False,
        return_samples=True,
    ) as result:
        assert result["samples_ms"] == pytest.approx((6.0, 2.0, 4.0))
        assert result["latency_ms"] == pytest.approx(4.0)
        assert result["num_runs_executed"] == 3

    assert len(device_module.events) == 6
    _assert_one_sync_after_measured_series(device_module, first_event_index=0, event_count=6)


def test_sample_path_preserves_cuda_graph_and_power_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first pair measures the adaptive power warmup. The remaining pairs are
    # the three replay samples.
    device_module = _FakeDeviceModule([100.0, 12.0, 4.0, 8.0])
    benchmark = _load_benchmark(monkeypatch, device_module)
    expected_power = {"power": 432.5, "power_limit": 700.0}

    class _PowerMonitor:
        def __init__(self, device_id: int) -> None:
            assert device_id == 0

        @staticmethod
        def start_sampling() -> bool:
            return True

        @staticmethod
        def stop_sampling() -> dict[str, float]:
            return expected_power

    monkeypatch.setattr(benchmark, "PowerMonitor", _PowerMonitor)
    monkeypatch.setattr(benchmark, "_NVML_INITIALIZED", False)

    with benchmark.benchmark_with_power(
        device=SimpleNamespace(index=0),
        kernel_func=lambda: None,
        num_warmups=1,
        num_runs=3,
        repeat_n=2,
        measure_power=True,
        power_min_duration=1.0,
        use_cuda_graph=False,
        return_samples=True,
    ) as result:
        assert result["samples_ms"] == pytest.approx((6.0, 2.0, 4.0))
        assert result["latency_ms"] == pytest.approx(4.0)
        assert result["power_stats"] == expected_power
        assert result["used_cuda_graph"] is False
        assert result["throttled"] is False

    assert len(device_module.events) == 8
    _assert_one_sync_after_measured_series(device_module, first_event_index=2, event_count=6)


def test_legacy_helper_delegates_to_namespaced_primitive_with_default_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    delegated_result = {
        "latency_ms": 1.25,
        "power_stats": None,
        "throttled": False,
        "num_runs_executed": 2,
        "used_cuda_graph": False,
    }
    namespaced = ModuleType(_BENCHMARK_MODULE)

    @contextmanager
    def _delegate(*args: Any, **kwargs: Any):
        calls.append((args, kwargs))
        yield delegated_result

    namespaced.benchmark_with_power = _delegate
    device_module = _FakeDeviceModule([2.5])

    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, _BENCHMARK_MODULE, namespaced)
        scoped.setitem(sys.modules, "torch", _fake_torch(device_module))
        legacy = importlib.import_module("collector.helper")
        legacy = importlib.reload(legacy)
        scoped.setattr(legacy, "get_device_module", lambda: device_module)

        with legacy.benchmark_with_power(
            device=SimpleNamespace(index=0),
            kernel_func=lambda: None,
            num_warmups=0,
            num_runs=2,
            repeat_n=1,
            measure_power=False,
            use_cuda_graph=False,
        ) as result:
            assert result is delegated_result

        assert len(calls) == 1
        assert calls[0][1]["return_samples"] is False

    importlib.reload(legacy)


def test_power_monitor_stops_when_measured_kernel_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    device_module = _FakeDeviceModule([1.0, 1.0])
    benchmark = _load_benchmark(monkeypatch, device_module)
    events: list[str] = []

    class _PowerMonitor:
        def __init__(self, device_id: int) -> None:
            assert device_id == 0

        def start_sampling(self) -> bool:
            events.append("start")
            return True

        def stop_sampling(self) -> None:
            events.append("stop")

    calls = 0

    def _kernel() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("kernel failed")

    monkeypatch.setattr(benchmark, "PowerMonitor", _PowerMonitor)
    with (
        pytest.raises(RuntimeError, match="kernel failed"),
        benchmark.benchmark_with_power(
            device=SimpleNamespace(index=0),
            kernel_func=_kernel,
            num_warmups=1,
            num_runs=1,
            repeat_n=1,
            measure_power=True,
            power_min_duration=0.0,
            use_cuda_graph=False,
            return_samples=True,
        ),
    ):
        pass

    assert events == ["start", "stop"]
