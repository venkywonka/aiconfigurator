# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact SGLang GEMM runner for the frozen DeepSeek-V4 profile."""

from __future__ import annotations

import logging
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

from aiconfigurator.collector.benchmark import benchmark_with_power
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

logger = logging.getLogger(__name__)

_SUPPORTED_SGLANG_VERSIONS = frozenset({"0.5.10", "0.5.10rc0"})
_OUTSIDE_LOOP_COUNT = 6


@dataclass(frozen=True, slots=True)
class PreparedGemmCase:
    kernel_func: Callable[[], Any]
    cleanup_func: Callable[[], None]
    outside_loop_count: int
    kernel_source: str
    framework_version: str
    device_name: str
    device: Any


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _prepare_gemm_case(
    gemm_type: str,
    m: int,
    n: int,
    k: int,
    device: str,
) -> PreparedGemmCase:
    """Prepare one SGLang kernel after the worker has bound its GPU lease."""

    framework_version = version("sglang")
    if framework_version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            "SGLang GEMM requires 0.5.10 or the exact 0.5.10rc0 measurement runtime, "
            f"got {framework_version!r}"
        )
    if gemm_type not in {"bfloat16", "fp8_block"}:
        raise ValueError(f"unsupported frozen-profile SGLang GEMM type {gemm_type!r}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (m, n, k)):
        raise ValueError("GEMM m, n, and k must be positive integers")

    import torch
    import torch.nn.functional as functional

    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    device_name = torch.cuda.get_device_name(torch_device)
    if " ".join(device_name.split()).casefold() != "nvidia gb200":
        raise RuntimeError(f"frozen SGLang GEMM requires NVIDIA GB200, got {device_name!r}")

    operations: list[Callable[[], Any]] = []
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(0)

    if gemm_type == "bfloat16":
        for _ in range(_OUTSIDE_LOOP_COUNT):
            activation = torch.randn(
                (m, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=generator,
            )
            weight = torch.randn(
                (n, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=generator,
            )

            def operation(activation=activation, weight=weight):
                return functional.linear(activation, weight, None)

            operations.append(operation)
        kernel_source = "torch_flow"
    else:
        major, _minor = torch.cuda.get_device_capability(torch_device)
        if major < 10:
            raise RuntimeError("frozen fp8_block SGLang GEMM requires an SM100-or-newer GPU")

        # Match SGLang's coherent offline collector: compile only the exact M
        # requested by the lazy miss instead of precompiling an unrelated grid.
        os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
        from sglang.srt.layers.deep_gemm_wrapper import (
            DEEPGEMM_SCALE_UE8M0,
            gemm_nt_f8f8bf16,
        )
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        fp8_info = torch.finfo(torch.float8_e4m3fn)
        for _ in range(_OUTSIDE_LOOP_COUNT):
            activation = torch.randn(
                (m, k),
                dtype=torch.bfloat16,
                device=torch_device,
                generator=generator,
            )
            weight_fp32 = torch.rand(
                (n, k),
                dtype=torch.float32,
                device=torch_device,
                generator=generator,
            )
            weight_fp32 = (weight_fp32 - 0.5) * 2 * fp8_info.max
            weight = weight_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
            del weight_fp32
            weight_scale = torch.randn(
                (_ceil_div(n, 128), _ceil_div(k, 128)),
                dtype=torch.float32,
                device=torch_device,
                generator=generator,
            )
            output = torch.empty((m, n), dtype=torch.bfloat16, device=torch_device)

            def operation(
                activation=activation,
                weight=weight,
                weight_scale=weight_scale,
                output=output,
            ):
                activation_fp8, activation_scale = sglang_per_token_group_quant_fp8(
                    activation,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=DEEPGEMM_SCALE_UE8M0,
                )
                gemm_nt_f8f8bf16(
                    (activation_fp8, activation_scale),
                    (weight, weight_scale),
                    output,
                )
                return output

            operations.append(operation)
        kernel_source = "deepgemm"

    def kernel_func():
        result = None
        for operation in operations:
            result = operation()
        return result

    def cleanup_func() -> None:
        operations.clear()
        torch.cuda.empty_cache()

    return PreparedGemmCase(
        kernel_func=kernel_func,
        cleanup_func=cleanup_func,
        outside_loop_count=len(operations),
        kernel_source=kernel_source,
        framework_version=framework_version,
        device_name=device_name,
        device=torch_device,
    )


def get_gemm_test_cases() -> tuple[()]:
    """The packaged lazy runner is exact-only and never expands a grid."""

    return ()


def run_gemm_case(
    gemm_type: str,
    m: int,
    n: int,
    k: int,
    *,
    protocol: MeasurementProtocol,
    device: str = "cuda:0",
) -> RawMeasurement:
    """Measure one exact SGLang GEMM without mutating curated data."""

    if not isinstance(protocol, MeasurementProtocol):
        raise TypeError("SGLang GEMM protocol must be a MeasurementProtocol")
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "sglang-gemm-v1"
        or protocol.statistic != "median"
    ):
        raise ValueError("SGLang GEMM measurement protocol is incompatible with the runner")

    prepared = _prepare_gemm_case(gemm_type, m, n, k, device)
    try:
        with benchmark_with_power(
            device=prepared.device,
            kernel_func=prepared.kernel_func,
            num_warmups=protocol.warmups,
            num_runs=protocol.samples,
            repeat_n=1,
            return_samples=True,
        ) as results:
            samples_ms = tuple(
                float(sample) / prepared.outside_loop_count for sample in results["samples_ms"]
            )
            latency_ms = statistics.median(samples_ms)
            power_stats = results.get("power_stats")
            measurement = RawMeasurement(
                latency_ms=latency_ms,
                energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
                samples_ms=samples_ms,
                statistic=protocol.statistic,
                perf_row={
                    "gemm_dtype": gemm_type,
                    "m": m,
                    "n": n,
                    "k": k,
                    "latency": latency_ms,
                },
                provenance={
                    "framework": "SGLang",
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
    except BaseException:
        try:
            prepared.cleanup_func()
        except Exception:
            logger.exception("SGLang GEMM cleanup failed; preserving the primary measurement failure")
        raise
    prepared.cleanup_func()
    return measurement


__all__ = ["PreparedGemmCase", "get_gemm_test_cases", "run_gemm_case"]
