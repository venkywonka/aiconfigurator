# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaged lazy registrations for the frozen SGLang profile."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC

SGLANG_LAZY_REGISTRY = (
    OpEntry(
        op="gemm",
        module="aiconfigurator.collector.trtllm.gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm_case",
        perf_filename=PerfFile.GEMM,
        lazy=GEMM_LAZY_SPEC,
    ),
)

__all__ = ["GEMM_LAZY_SPEC", "SGLANG_LAZY_REGISTRY"]
