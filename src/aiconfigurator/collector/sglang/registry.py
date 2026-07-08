# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged lazy registrations for the frozen SGLang profile."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC
from aiconfigurator.collector.types import LazyOpEntry
from aiconfigurator.sdk.perf_namespace import perf_namespace

MHC_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.MHC_MODULE)),
    run_module="aiconfigurator.collector.sglang.mhc",
    run_func="run_mhc_case",
    adapter_module="aiconfigurator.collector.sglang.mhc_adapter",
    case_func="mhc_request_to_case",
    result_func="mhc_result_to_record",
    resource_func="mhc_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="sglang-mhc-v1",
)

SGLANG_LAZY_REGISTRY = (
    OpEntry(
        op="gemm",
        module="aiconfigurator.collector.trtllm.gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm_case",
        perf_filename=PerfFile.GEMM,
        lazy=GEMM_LAZY_SPEC,
    ),
    OpEntry(
        op="mhc_module",
        module="aiconfigurator.collector.sglang.mhc",
        get_func="get_mhc_test_cases",
        run_func="run_mhc_case",
        perf_filename=PerfFile.MHC_MODULE,
        lazy=MHC_LAZY_SPEC,
    ),
)

__all__ = ["GEMM_LAZY_SPEC", "MHC_LAZY_SPEC", "SGLANG_LAZY_REGISTRY"]
