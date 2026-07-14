# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged lazy registrations for the frozen SGLang profile."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.types import FabricRequirement, LazyOpEntry, ResourceContract
from aiconfigurator.sdk.perf_namespace import perf_namespace

GEMM_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.GEMM)),
    run_module="aiconfigurator.collector.sglang.gemm",
    run_func="run_gemm_case",
    adapter_module="aiconfigurator.collector.trtllm.gemm_adapter",
    case_func="gemm_request_to_case",
    result_func="gemm_result_to_record",
    resource_func="gemm_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="sglang-gemm-v1",
    preflight_resource=ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE),
)

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
    preflight_resource=ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE),
)

MOE_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.MOE)),
    run_module="aiconfigurator.collector.sglang.moe",
    run_func="run_moe_case",
    adapter_module="aiconfigurator.collector.sglang.moe_adapter",
    case_func="moe_request_to_case",
    result_func="moe_result_to_record",
    resource_func="moe_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="sglang-moe-v1",
    preflight_resource=ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE),
)

CUSTOM_ALLREDUCE_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.CUSTOM_ALLREDUCE)),
    run_module="aiconfigurator.collector.sglang.custom_allreduce",
    run_func="run_custom_allreduce_case",
    adapter_module="aiconfigurator.collector.sglang.custom_allreduce_adapter",
    case_func="custom_allreduce_request_to_case",
    result_func="custom_allreduce_result_to_record",
    resource_func="custom_allreduce_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="sglang-custom-allreduce-v1",
    preflight_resource=ResourceContract(
        gpu_count=4,
        fabric=FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    ),
)


def _dsv4_attention_lazy_spec(perf_file: PerfFile) -> LazyOpEntry:
    return LazyOpEntry(
        namespace=perf_namespace(str(perf_file)),
        run_module="aiconfigurator.collector.sglang.dsv4_attn",
        run_func="run_dsv4_attn_case",
        adapter_module="aiconfigurator.collector.sglang.dsv4_attn_adapter",
        case_func="dsv4_attn_request_to_case",
        result_func="dsv4_attn_result_to_record",
        resource_func="dsv4_attn_resource_for_request",
        protocol_revision="cuda-event-samples-v1",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
        preflight_resource=ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE),
    )


DSV4_CSA_CONTEXT_LAZY_SPEC = _dsv4_attention_lazy_spec(PerfFile.DSV4_CSA_CONTEXT_MODULE)
DSV4_HCA_CONTEXT_LAZY_SPEC = _dsv4_attention_lazy_spec(PerfFile.DSV4_HCA_CONTEXT_MODULE)
DSV4_CSA_GENERATION_LAZY_SPEC = _dsv4_attention_lazy_spec(PerfFile.DSV4_CSA_GENERATION_MODULE)
DSV4_HCA_GENERATION_LAZY_SPEC = _dsv4_attention_lazy_spec(PerfFile.DSV4_HCA_GENERATION_MODULE)

SGLANG_LAZY_REGISTRY = (
    OpEntry(
        op="gemm",
        module="aiconfigurator.collector.sglang.gemm",
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
    OpEntry(
        op="moe",
        module="aiconfigurator.collector.sglang.moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_case",
        perf_filename=PerfFile.MOE,
        lazy=MOE_LAZY_SPEC,
    ),
    OpEntry(
        op="custom_allreduce",
        module="aiconfigurator.collector.sglang.custom_allreduce",
        get_func="get_custom_allreduce_test_cases",
        run_func="run_custom_allreduce_case",
        perf_filename=PerfFile.CUSTOM_ALLREDUCE,
        lazy=CUSTOM_ALLREDUCE_LAZY_SPEC,
    ),
    OpEntry(
        op="dsv4_csa_context_module",
        module="aiconfigurator.collector.sglang.dsv4_attn",
        get_func="get_dsv4_attn_test_cases",
        run_func="run_dsv4_attn_case",
        perf_filename=PerfFile.DSV4_CSA_CONTEXT_MODULE,
        lazy=DSV4_CSA_CONTEXT_LAZY_SPEC,
    ),
    OpEntry(
        op="dsv4_hca_context_module",
        module="aiconfigurator.collector.sglang.dsv4_attn",
        get_func="get_dsv4_attn_test_cases",
        run_func="run_dsv4_attn_case",
        perf_filename=PerfFile.DSV4_HCA_CONTEXT_MODULE,
        lazy=DSV4_HCA_CONTEXT_LAZY_SPEC,
    ),
    OpEntry(
        op="dsv4_csa_generation_module",
        module="aiconfigurator.collector.sglang.dsv4_attn",
        get_func="get_dsv4_attn_test_cases",
        run_func="run_dsv4_attn_case",
        perf_filename=PerfFile.DSV4_CSA_GENERATION_MODULE,
        lazy=DSV4_CSA_GENERATION_LAZY_SPEC,
    ),
    OpEntry(
        op="dsv4_hca_generation_module",
        module="aiconfigurator.collector.sglang.dsv4_attn",
        get_func="get_dsv4_attn_test_cases",
        run_func="run_dsv4_attn_case",
        perf_filename=PerfFile.DSV4_HCA_GENERATION_MODULE,
        lazy=DSV4_HCA_GENERATION_LAZY_SPEC,
    ),
)

__all__ = [
    "CUSTOM_ALLREDUCE_LAZY_SPEC",
    "DSV4_CSA_CONTEXT_LAZY_SPEC",
    "DSV4_CSA_GENERATION_LAZY_SPEC",
    "DSV4_HCA_CONTEXT_LAZY_SPEC",
    "DSV4_HCA_GENERATION_LAZY_SPEC",
    "GEMM_LAZY_SPEC",
    "MHC_LAZY_SPEC",
    "MOE_LAZY_SPEC",
    "SGLANG_LAZY_REGISTRY",
]
