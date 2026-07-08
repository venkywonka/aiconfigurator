# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import contextmanager
from math import prod
from types import ModuleType, SimpleNamespace

from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol


def test_cached_gemm_weights_are_independent_of_first_shape_order(monkeypatch) -> None:
    from aiconfigurator.collector.trtllm import gemm

    observed: list[tuple[int, tuple[int, int, tuple[int, ...]]]] = []

    class _Device:
        index = 0

        def __str__(self) -> str:
            return "cuda:0"

    class _Generator:
        def __init__(self, *, device) -> None:
            del device
            self.seed = 0
            self.offset = 0

        def manual_seed(self, seed: int):
            self.seed = seed
            self.offset = 0
            return self

    class _Tensor:
        def __init__(self, shape, signature) -> None:
            self.shape = tuple(shape)
            self.signature = signature

    def _randn(shape, *, dtype, device, generator):
        del dtype, device
        shape = tuple(shape)
        signature = (generator.seed, generator.offset, shape)
        generator.offset += prod(shape)
        return _Tensor(shape, signature)

    class _Linear:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.weight_signature = None

        def load_weights(self, weights) -> None:
            self.weight_signature = weights[0]["weight"].signature

        def to(self, device):
            del device
            return self

        def forward(self, activation) -> None:
            observed.append((activation.shape[0], self.weight_signature))

    torch = ModuleType("torch")
    torch.__path__ = []
    torch.bfloat16 = object()
    torch.device = lambda value: _Device()
    torch.Generator = _Generator
    torch.randn = _randn
    torch.set_default_device = lambda device: None
    torch.cuda = SimpleNamespace(
        set_device=lambda device: None,
        get_device_capability=lambda device: (9, 0),
        get_device_name=lambda device: "NVIDIA H100",
    )
    torch_nn = ModuleType("torch.nn")
    torch_nn.__path__ = []
    torch_functional = ModuleType("torch.nn.functional")
    torch.nn = torch_nn
    torch_nn.functional = torch_functional

    tensorrt_llm = ModuleType("tensorrt_llm")
    tensorrt_llm.__path__ = []
    tensorrt_llm.__version__ = "1.3.0"
    trt_torch = ModuleType("tensorrt_llm._torch")
    trt_torch.__path__ = []
    trt_modules = ModuleType("tensorrt_llm._torch.modules")
    trt_modules.__path__ = []
    trt_linear = ModuleType("tensorrt_llm._torch.modules.linear")
    trt_linear.Linear = _Linear
    trt_models = ModuleType("tensorrt_llm.models")
    trt_models.__path__ = []
    trt_modeling = ModuleType("tensorrt_llm.models.modeling_utils")
    trt_modeling.QuantAlgo = SimpleNamespace(FP8=object(), FP8_BLOCK_SCALES=object(), NVFP4=object())
    trt_modeling.QuantConfig = object

    for module in (
        torch,
        torch_nn,
        torch_functional,
        tensorrt_llm,
        trt_torch,
        trt_modules,
        trt_linear,
        trt_models,
        trt_modeling,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(gemm, "_WEIGHT_CACHE", {})
    monkeypatch.setattr(gemm, "_get_l2_cache_bytes", lambda device_id: 1)

    def _weight_for_m16(order: tuple[int, int]):
        gemm._WEIGHT_CACHE.clear()
        observed.clear()
        for m in order:
            prepared = gemm._prepare_gemm_case("bfloat16", m, 4, 8, "cuda:0")
            assert prepared.outside_loop_count == 1
        return next(weight for m, weight in observed if m == 16)

    assert _weight_for_m16((8, 16)) == _weight_for_m16((16, 8))


def test_exact_gemm_runner_returns_raw_samples_without_logging(monkeypatch, tmp_path) -> None:
    from aiconfigurator.collector.trtllm import gemm

    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )
    calls = []

    def _prepare(gemm_type, m, n, k, device):
        calls.append(("prepare", gemm_type, m, n, k, device))
        return gemm.PreparedGemmCase(
            kernel_func=lambda: calls.append(("kernel",)),
            outside_loop_count=2,
            kernel_source="torch_flow",
            framework_version="1.3.0rc10",
            device_name="NVIDIA GB200",
            device=object(),
        )

    @contextmanager
    def _benchmark(**kwargs):
        calls.append(("benchmark", kwargs))
        yield {
            "latency_ms": 8.0,
            "samples_ms": (6.0, 8.0, 10.0),
            "power_stats": {"power": 100.0, "power_limit": 1200.0},
            "throttled": False,
            "used_cuda_graph": True,
            "num_runs_executed": 3,
        }

    monkeypatch.setattr(gemm, "_prepare_gemm_case", _prepare)
    monkeypatch.setattr(gemm, "benchmark_with_power", _benchmark)
    monkeypatch.chdir(tmp_path)

    result = gemm.run_gemm_case("bfloat16", 8, 16, 32, protocol=protocol)

    assert isinstance(result, RawMeasurement)
    assert result.latency_ms == 4.0
    assert result.samples_ms == (3.0, 4.0, 5.0)
    assert result.energy_wms == 400.0
    assert result.protocol_digest == protocol.digest
    assert result.perf_row == {
        "gemm_dtype": "bfloat16",
        "m": 8,
        "n": 16,
        "k": 32,
        "latency": 4.0,
    }
    assert result.provenance == {
        "framework": "TRTLLM",
        "framework_version": "1.3.0rc10",
        "kernel_source": "torch_flow",
        "device": "NVIDIA GB200",
        "used_cuda_graph": True,
        "throttled": False,
        "tensor_generator": "normal-v1",
        "seed": 0,
    }
    benchmark_kwargs = next(call[1] for call in calls if call[0] == "benchmark")
    assert benchmark_kwargs["num_warmups"] == 2
    assert benchmark_kwargs["num_runs"] == 3
    assert benchmark_kwargs["repeat_n"] == 1
    assert benchmark_kwargs["return_samples"] is True
    assert not tuple(tmp_path.iterdir())


def test_legacy_wrapper_logs_the_exact_runner_row_once(monkeypatch) -> None:
    from collector.trtllm import collect_gemm

    raw = RawMeasurement(
        latency_ms=1.25,
        energy_wms=125.0,
        samples_ms=(1.2, 1.25, 1.3),
        statistic="median",
        perf_row={"gemm_dtype": "bfloat16", "m": 8, "n": 16, "k": 32, "latency": 1.25},
        provenance={
            "framework": "TRTLLM",
            "framework_version": "1.3.0rc10",
            "device": "NVIDIA GB200",
            "kernel_source": "torch_flow",
        },
        protocol_digest="protocol-digest",
        power_stats={"power": 100.0, "power_limit": 1200.0},
    )
    runner_calls = []
    log_calls = []
    monkeypatch.setattr(
        collect_gemm,
        "run_gemm_case",
        lambda *args, **kwargs: runner_calls.append((args, kwargs)) or raw,
    )
    monkeypatch.setattr(collect_gemm, "log_perf", lambda **kwargs: log_calls.append(kwargs))

    result = collect_gemm.run_gemm("bfloat16", 8, 16, 32, perf_filename="gemm_perf.txt", device="cuda:3")

    assert result is None
    assert runner_calls == [(("bfloat16", 8, 16, 32), {"device": "cuda:3"})]
    assert len(log_calls) == 1
    assert log_calls[0]["item_list"] == [dict(raw.perf_row)]
    assert log_calls[0]["perf_filename"] == "gemm_perf.txt"
    assert log_calls[0]["power_stats"] == {"power": 100.0, "power_limit": 1200.0}
