# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact TensorRT-LLM GEMM runner."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiconfigurator.collector.benchmark import benchmark_with_power
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

_WEIGHT_CACHE: dict[tuple[str, int, int, str], dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class PreparedGemmCase:
    kernel_func: Callable[[], Any]
    outside_loop_count: int
    kernel_source: str
    framework_version: str
    device_name: str
    device: Any


def _get_l2_cache_bytes(device_id: int) -> int:
    cuda_dev_attr_l2_cache_size = 38
    libcudart = ctypes.CDLL("libcudart.so")
    value = ctypes.c_int()
    result = libcudart.cudaDeviceGetAttribute(ctypes.byref(value), cuda_dev_attr_l2_cache_size, device_id)
    if result != 0 or value.value <= 0:
        raise RuntimeError(f"Failed to query L2 cache size (cudaError={result}, value={value.value})")
    return value.value


def _prepare_gemm_case(
    gemm_type: str,
    m: int,
    n: int,
    k: int,
    device: str,
) -> PreparedGemmCase:
    """Import the heavy framework only inside the UUID-bound worker."""

    import tensorrt_llm
    import torch
    import torch.nn.functional as functional
    from tensorrt_llm._torch.modules.linear import Linear
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    if gemm_type not in {"bfloat16", "fp8", "fp8_block", "nvfp4"}:
        raise ValueError(f"unsupported TensorRT-LLM GEMM type {gemm_type!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (m, n, k)):
        raise ValueError("GEMM m, n, and k must be positive integers")

    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    torch.set_default_device(torch_device)
    major, minor = torch.cuda.get_device_capability(torch_device)
    sm_version = major * 10 + minor
    dtype = torch.bfloat16
    activation_generator = torch.Generator(device=torch_device)
    activation_generator.manual_seed(0)
    weight_generator = torch.Generator(device=torch_device)
    weight_generator.manual_seed(0)
    activation = torch.randn(
        (m, k),
        dtype=dtype,
        device=torch_device,
        generator=activation_generator,
    )

    if gemm_type == "fp8":
        group_size = None
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    elif gemm_type == "fp8_block":
        group_size = 128
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=group_size)
    elif gemm_type == "nvfp4":
        group_size = 128
        quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=group_size)
    else:
        group_size = None
        quant_config = None

    def _pad_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    def _block_scales(weight, block_size: int):
        n_blocks = math.ceil(n / block_size)
        k_blocks = math.ceil(k / block_size)
        pad_n = n_blocks * block_size - n
        pad_k = k_blocks * block_size - k
        weight_abs = weight.abs()
        if pad_n or pad_k:
            weight_abs = functional.pad(weight_abs, (0, pad_k, 0, pad_n))
        block_max = weight_abs.view(n_blocks, block_size, k_blocks, block_size).amax(dim=(1, 3))
        return (block_max / 448.0).clamp_min(1e-6).to(dtype=torch.float32)

    def _build_weights() -> dict[str, Any]:
        if gemm_type == "fp8":
            return {
                "weight": torch.randn(
                    (n, k),
                    dtype=torch.bfloat16,
                    device=torch_device,
                    generator=weight_generator,
                ).to(torch.float8_e4m3fn),
                "weight_scale": torch.randn(
                    1,
                    dtype=torch.float32,
                    device=torch_device,
                    generator=weight_generator,
                ),
            }
        if gemm_type == "fp8_block":
            weight = torch.randn(
                (n, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=weight_generator,
            )
            return {
                "weight": weight.to(dtype=torch.float8_e4m3fn),
                "weight_scale": _block_scales(weight, group_size),
            }
        if gemm_type == "nvfp4":
            weight = torch.randn(
                (n, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=weight_generator,
            )
            weight_global_scale = (448 * 6) / weight.abs().max().float()
            weight_fp4, weight_block_scale = torch.ops.trtllm.fp4_quantize(
                weight,
                weight_global_scale,
                16,
                False,
            )
            if tensorrt_llm.__version__.startswith(("1.1.0", "1.2.0", "1.3.0")):
                weight_block_scale = torch.ops.trtllm.block_scale_interleave_reverse(
                    weight_block_scale.cpu().view(_pad_up(n, 128), -1)
                )
            else:
                weight_block_scale = torch.ops.trtllm.nvfp4_block_scale_interleave_reverse(
                    weight_block_scale.cpu().view(k, -1)
                )
            activation_global_scale = (448 * 6) / activation.abs().max().float()
            return {
                "weight": weight_fp4.cpu(),
                "weight_scale": weight_block_scale.view(torch.float8_e4m3fn),
                "weight_scale_2": 1.0 / weight_global_scale.cpu(),
                "input_scale": 1.0 / activation_global_scale.cpu(),
            }
        return {
            "weight": torch.randn(
                (n, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=weight_generator,
            )
        }

    bytes_per_element = {"bfloat16": 2, "fp8": 1, "fp8_block": 1, "nvfp4": 0.5}
    weight_bytes = int(n * k * bytes_per_element[gemm_type])
    outside_loop_count = max(1, min(5, math.ceil(_get_l2_cache_bytes(torch_device.index or 0) / weight_bytes)))
    operations = []
    weights_per_operation: list[dict[str, Any]]
    if outside_loop_count == 1:
        cache_key = (gemm_type, n, k, str(torch_device))
        if cache_key not in _WEIGHT_CACHE:
            _WEIGHT_CACHE.clear()
            _WEIGHT_CACHE[cache_key] = _build_weights()
        weights = _WEIGHT_CACHE[cache_key]
        if gemm_type == "nvfp4":
            activation_global_scale = (448 * 6) / activation.abs().max().float()
            weights = {**weights, "input_scale": 1.0 / activation_global_scale.cpu()}
        weights_per_operation = [weights]
    else:
        weights_per_operation = [_build_weights() for _ in range(outside_loop_count)]
    for weights in weights_per_operation:
        operation = Linear(
            k,
            n,
            bias=False,
            dtype=dtype,
            quant_config=quant_config,
            force_dynamic_quantization=False,
        )
        operation.load_weights([weights])
        if gemm_type == "fp8_block" and callable(getattr(operation, "post_load_weights", None)):
            operation.post_load_weights()
        operation.to(torch_device)
        operation.forward(activation)
        operations.append(operation)

    def kernel_func() -> None:
        for operation in operations:
            operation.forward(activation)

    return PreparedGemmCase(
        kernel_func=kernel_func,
        outside_loop_count=outside_loop_count,
        kernel_source="deepgemm" if gemm_type == "fp8_block" and sm_version >= 100 else "torch_flow",
        framework_version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(torch_device),
        device=torch_device,
    )


def get_gemm_test_cases() -> tuple[()]:
    """The online registry is exact-only and never expands an offline grid."""

    return ()


def run_gemm_case(
    gemm_type: str,
    m: int,
    n: int,
    k: int,
    *,
    protocol: MeasurementProtocol | None = None,
    device: str = "cuda:0",
) -> RawMeasurement:
    """Measure one exact GEMM case without opening or writing a perf file."""

    protocol = protocol or MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=6,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "trtllm-linear-v1"
        or protocol.statistic != "median"
    ):
        raise ValueError("GEMM measurement protocol is incompatible with the runner")
    prepared = _prepare_gemm_case(gemm_type, m, n, k, device)
    with benchmark_with_power(
        device=prepared.device,
        kernel_func=prepared.kernel_func,
        num_warmups=protocol.warmups,
        num_runs=protocol.samples,
        repeat_n=1,
        return_samples=True,
    ) as results:
        latency_ms = float(results["latency_ms"]) / prepared.outside_loop_count
        samples_ms = tuple(float(sample) / prepared.outside_loop_count for sample in results["samples_ms"])
        power_stats = results.get("power_stats")
        row = {
            "gemm_dtype": gemm_type,
            "m": m,
            "n": n,
            "k": k,
            "latency": latency_ms,
        }
        return RawMeasurement(
            latency_ms=latency_ms,
            energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
            samples_ms=samples_ms,
            statistic=protocol.statistic,
            perf_row=row,
            provenance={
                "framework": "TRTLLM",
                "framework_version": prepared.framework_version,
                "kernel_source": prepared.kernel_source,
                "device": prepared.device_name,
                "used_cuda_graph": bool(results["used_cuda_graph"]),
                "throttled": bool(results["throttled"]),
                "tensor_generator": "normal-v1",
                "seed": 0,
            },
            protocol_digest=protocol.digest,
            power_stats=power_stats,
        )


__all__ = ["PreparedGemmCase", "get_gemm_test_cases", "run_gemm_case"]
