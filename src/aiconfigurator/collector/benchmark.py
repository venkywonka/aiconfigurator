# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installable CUDA/XPU benchmark primitive with optional replay samples."""

from __future__ import annotations

import functools
import logging
import os
import statistics
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_NVML_INITIALIZED = False
_NVML_LOCK = threading.Lock()


def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    value = os.environ.get(env_var)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _ensure_nvml_initialized() -> bool:
    global _NVML_INITIALIZED
    with _NVML_LOCK:
        if not _NVML_INITIALIZED:
            try:
                import pynvml as nvml

                nvml.nvmlInit()
                _NVML_INITIALIZED = True
                logging.getLogger(__name__).info("NVML initialized for power monitoring")
            except Exception as error:
                logging.getLogger(__name__).warning("Failed to initialize NVML: %s", error)
                return False
        return _NVML_INITIALIZED


class PowerMonitor:
    """Background NVML sampler retained across benchmark invocations."""

    SAMPLE_INTERVAL_MS = 100

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        self.interval_s = self.SAMPLE_INTERVAL_MS / 1000.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._lock = threading.Lock()
        self._nvml_handle: Any = None
        self._power_limit_mw: float | None = None
        self._is_initialized = False

    def _init_handle(self) -> bool:
        if self._is_initialized:
            return True
        if not _ensure_nvml_initialized():
            return False
        try:
            import pynvml as nvml

            self._nvml_handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
            self._power_limit_mw = nvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle)
            self._is_initialized = True
            return True
        except Exception as error:
            logging.getLogger(__name__).warning("Failed to get NVML handle for device %s: %s", self.device_id, error)
            return False

    def start_sampling(self) -> bool:
        if not self._init_handle():
            return False
        with self._lock:
            self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        return True

    def stop_sampling(self) -> dict[str, float | None] | None:
        if self._thread is None:
            return None
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            if not self._samples:
                return None
            power_values_w = [power_mw / 1000.0 for _, power_mw in self._samples]
        return {
            "power": statistics.fmean(power_values_w),
            "power_limit": self._power_limit_mw / 1000.0 if self._power_limit_mw else None,
        }

    def _monitoring_loop(self) -> None:
        import pynvml as nvml

        while not self._stop_event.is_set():
            try:
                timestamp = time.time()
                power_mw = nvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                with self._lock:
                    self._samples.append((timestamp, power_mw))
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)


@functools.lru_cache(maxsize=1)
def get_device_module():
    import torch

    if torch.cuda.is_available():
        return torch.cuda
    if torch.xpu.is_available():
        return torch.xpu
    raise RuntimeError("No supported device (need CUDA or XPU)")


@contextmanager
def benchmark_with_power(
    device,
    kernel_func,
    num_warmups: int = 3,
    num_runs: int = 6,
    repeat_n: int = 1,
    measure_power: bool | None = None,
    power_min_duration: float | None = None,
    allow_graph_fail: bool = False,
    use_cuda_graph: bool = True,
    return_samples: bool = False,
) -> Iterator[dict[str, Any]]:
    """Benchmark one callable and optionally retain one CUDA-event sample per replay."""

    import torch

    if measure_power is None:
        measure_power = _parse_bool_env("COLLECTOR_MEASURE_POWER", default=False)
    if power_min_duration is None:
        power_min_duration = float(os.environ.get("COLLECTOR_POWER_MIN_DURATION", "1.0"))

    actual_num_runs = num_runs
    if measure_power:
        start_warmup = torch.cuda.Event(enable_timing=True)
        end_warmup = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_warmup.record()
        for _ in range(num_warmups):
            kernel_func()
        end_warmup.record()
        torch.cuda.synchronize()
        single_iter_time = start_warmup.elapsed_time(end_warmup) / num_warmups / 1000.0
        target_duration = power_min_duration
        if single_iter_time < 0.0001:
            target_duration = min(power_min_duration, 0.3)
        if not return_samples:
            actual_num_runs = max(num_runs, int(target_duration / (single_iter_time * repeat_n)) + 1)
            actual_num_runs = min(actual_num_runs, 3000)
            if actual_num_runs > 1000:
                logging.getLogger(__name__).warning(
                    "Kernel is very fast (%.3fms), running %s iterations",
                    single_iter_time * 1000,
                    actual_num_runs,
                )
    else:
        get_device_module().synchronize()
        for _ in range(num_warmups):
            kernel_func()
        get_device_module().synchronize()

    graph = None
    if torch.cuda.is_available() and use_cuda_graph:
        use_graph = True
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                for _ in range(repeat_n):
                    kernel_func()
            torch.cuda.synchronize()
        except Exception as error:
            if not allow_graph_fail:
                raise
            logging.getLogger(__name__).warning(
                "CUDA graph capture failed: %s. Falling back to eager execution.",
                error,
            )
            graph = None
            torch.cuda.empty_cache()
            use_graph = False
    else:
        use_graph = False

    power_monitor = None
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for _ in range(num_warmups):
                if use_graph:
                    graph.replay()
                else:
                    for _ in range(repeat_n):
                        kernel_func()
            torch.cuda.synchronize()

        power_stats = None
        if measure_power:
            power_monitor = PowerMonitor(device.index)
            if not power_monitor.start_sampling():
                power_monitor = None

        initial_clocks = None
        if measure_power and _NVML_INITIALIZED:
            try:
                import pynvml as nvml

                handle = nvml.nvmlDeviceGetHandleByIndex(device.index)
                initial_clocks = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
            except Exception:
                pass

        device_module = get_device_module()
        samples_ms: tuple[float, ...] | None = None
        if return_samples:
            event_pairs = tuple(
                (device_module.Event(enable_timing=True), device_module.Event(enable_timing=True))
                for _ in range(actual_num_runs)
            )
            for start_event, end_event in event_pairs:
                start_event.record()
                if use_graph:
                    graph.replay()
                else:
                    for _ in range(repeat_n):
                        kernel_func()
                end_event.record()
            device_module.synchronize()
            samples_ms = tuple(start.elapsed_time(end) / repeat_n for start, end in event_pairs)
            latency_ms = statistics.median(samples_ms)
        else:
            start_event = device_module.Event(enable_timing=True)
            end_event = device_module.Event(enable_timing=True)
            start_event.record()
            for _ in range(actual_num_runs):
                if use_graph:
                    graph.replay()
                else:
                    for _ in range(repeat_n):
                        kernel_func()
            end_event.record()
            device_module.synchronize()
            latency_ms = start_event.elapsed_time(end_event) / actual_num_runs / repeat_n

        throttled = False
        if initial_clocks is not None:
            try:
                import pynvml as nvml

                handle = nvml.nvmlDeviceGetHandleByIndex(device.index)
                final_clocks = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)
                if final_clocks < initial_clocks * 0.9:
                    throttled = True
                    logging.getLogger(__name__).warning(
                        "Clock throttling detected: %sMHz -> %sMHz",
                        initial_clocks,
                        final_clocks,
                    )
            except Exception:
                pass
        if power_monitor:
            power_stats = power_monitor.stop_sampling()
            power_monitor = None

        result = {
            "latency_ms": latency_ms,
            "power_stats": power_stats,
            "throttled": throttled,
            "num_runs_executed": actual_num_runs,
            "used_cuda_graph": use_graph,
        }
        if samples_ms is not None:
            result["samples_ms"] = samples_ms
        yield result
    finally:
        if power_monitor is not None:
            try:
                power_monitor.stop_sampling()
            except Exception:
                pass
        if graph is not None:
            graph = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


__all__ = ["PowerMonitor", "benchmark_with_power", "get_device_module"]
