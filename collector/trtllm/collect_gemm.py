# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline TensorRT-LLM GEMM sweep backed by the installable exact runner."""

from __future__ import annotations

import os
from collections import defaultdict

from aiconfigurator.collector.trtllm.gemm import run_gemm_case
from collector.case_generator import get_gemm_case_specs
from collector.helper import get_sm_version, log_perf


def _skip_trtllm_large_fp8_projection_gemm(gemm_type: str, m: int, n: int, k: int) -> bool:
    if gemm_type != "fp8" or not (1 <= m <= 8 and n >= 51200 and k >= 51200):
        return False
    import tensorrt_llm

    sm_version = get_sm_version()
    if tensorrt_llm.__version__.startswith(("1.3.0rc5", "1.3.0rc10")) and sm_version >= 120:
        return True
    return tensorrt_llm.__version__.startswith("1.3.0rc15") and sm_version == 89


def get_gemm_test_cases() -> list[list[object]]:
    """Preserve the existing exhaustive offline case expansion."""

    gemm_types = ["bfloat16"]
    sm_version = get_sm_version()
    if sm_version > 86:
        gemm_types += ["fp8", "fp8_block"]
    if sm_version >= 100:
        gemm_types.append("nvfp4")
    requested_gemm_types = os.environ.get("AIC_COLLECT_GEMM_TYPES")
    if requested_gemm_types:
        requested = {item.strip() for item in requested_gemm_types.split(",") if item.strip()}
        gemm_types = [gemm_type for gemm_type in gemm_types if gemm_type in requested]

    nk_to_x: dict[tuple[int, int], list[int]] = defaultdict(list)
    for case in get_gemm_case_specs():
        nk_to_x[(case.n, case.k)].append(case.x)

    test_cases: list[list[object]] = []
    for n, k in sorted(nk_to_x, key=lambda shape: (-shape[0], -shape[1])):
        for gemm_type in gemm_types:
            if gemm_type in {"nvfp4", "fp8_block"} and (n < 128 or k < 128):
                continue
            if gemm_type == "fp8_block" and n * k >= 2**31:
                continue
            for x in sorted(nk_to_x[(n, k)], reverse=True):
                if x * n >= 2**31 or x * k >= 2**31:
                    continue
                if _skip_trtllm_large_fp8_projection_gemm(gemm_type, x, n, k):
                    continue
                test_cases.append([gemm_type, x, n, k])
    return test_cases


def run_gemm(gemm_type, m, n, k, *, perf_filename, device="cuda:0") -> None:
    """Run one exact case and retain the legacy explicit logging side effect."""

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
