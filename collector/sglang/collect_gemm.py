# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline SGLang GEMM sweep backed by the installable exact runner."""

from __future__ import annotations

import os
import random

from aiconfigurator.collector.sglang.gemm import SUPPORTED_GEMM_TYPES, run_gemm_case
from collector.case_generator import get_gemm_case_specs
from collector.helper import get_sm_version, log_perf

__compat__ = "sglang>=0.5.10rc0"


def get_gemm_test_cases() -> list[list[object]]:
    """Preserve the existing exhaustive offline case expansion."""

    sm_version = get_sm_version()
    if sm_version < 89:
        gemm_types = ["bfloat16"]
    elif sm_version < 90:
        gemm_types = ["bfloat16", "fp8"]
    elif sm_version < 100:
        gemm_types = ["fp8_block", "bfloat16", "fp8"]
    elif sm_version < 110:
        gemm_types = ["fp8_block", "bfloat16", "fp8", "nvfp4"]
    else:
        gemm_types = ["bfloat16", "fp8", "nvfp4"]

    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_types = [gemm_type for gemm_type in gemm_types if gemm_type in requested]

    if not set(gemm_types) <= SUPPORTED_GEMM_TYPES:
        raise RuntimeError("offline SGLang GEMM grid contains a mode without a canonical runner")

    test_cases: list[list[object]] = []
    for case in get_gemm_case_specs():
        for gemm_type in gemm_types:
            if gemm_type in {"nvfp4", "fp8_block"} and (case.n < 128 or case.k < 128):
                continue
            test_cases.append([gemm_type, case.x, case.n, case.k])

    random.Random(42).shuffle(test_cases)
    return test_cases


def run_gemm(gemm_type, m, n, k, *, perf_filename, device="cuda:0") -> None:
    """Run one canonical exact case and retain the offline logging side effect."""

    raw = run_gemm_case(gemm_type, m, n, k, device=device)
    log_perf(
        item_list=[dict(raw.perf_row)],
        framework=str(raw.provenance["framework"]),
        version=str(raw.provenance["framework_version"]),
        device_name=str(raw.provenance["device"]),
        op_name="gemm",
        kernel_source=str(raw.provenance["kernel_source"]),
        perf_filename=perf_filename,
        power_stats=dict(raw.power_stats) if raw.power_stats is not None else None,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_gemm_test_cases():
        run_gemm(*test_case, perf_filename=PerfFile.GEMM)


__all__ = ["get_gemm_test_cases", "run_gemm"]
