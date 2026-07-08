# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged lazy registrations for TensorRT-LLM collectors."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.types import LazyOpEntry
from aiconfigurator.sdk.perf_namespace import perf_namespace

GEMM_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.GEMM)),
    run_module="aiconfigurator.collector.trtllm.gemm",
    run_func="run_gemm_case",
    adapter_module="aiconfigurator.collector.trtllm.gemm_adapter",
    case_func="gemm_request_to_case",
    result_func="gemm_result_to_record",
    resource_func="gemm_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="trtllm-linear-v1",
)

TRTLLM_LAZY_REGISTRY = (
    OpEntry(
        op="gemm",
        module="aiconfigurator.collector.trtllm.gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm_case",
        perf_filename=PerfFile.GEMM,
        lazy=GEMM_LAZY_SPEC,
    ),
)

__all__ = ["GEMM_LAZY_SPEC", "TRTLLM_LAZY_REGISTRY"]
