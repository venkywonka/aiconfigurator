# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged lazy registrations for network collectors."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.types import LazyOpEntry
from aiconfigurator.sdk.perf_namespace import perf_namespace

NCCL_LAZY_SPEC = LazyOpEntry(
    namespace=perf_namespace(str(PerfFile.NCCL)),
    run_module="aiconfigurator.collector.network.nccl",
    run_func="run_nccl_case",
    adapter_module="aiconfigurator.collector.network.nccl_adapter",
    case_func="nccl_request_to_case",
    result_func="nccl_result_to_record",
    resource_func="nccl_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="torch-nccl-persistent-v1",
)

NETWORK_LAZY_REGISTRY = (
    OpEntry(
        op="nccl",
        module="aiconfigurator.collector.network.nccl",
        get_func="get_nccl_test_cases",
        run_func="run_nccl_case",
        perf_filename=PerfFile.NCCL,
        lazy=NCCL_LAZY_SPEC,
    ),
)

__all__ = ["NCCL_LAZY_SPEC", "NETWORK_LAZY_REGISTRY"]
