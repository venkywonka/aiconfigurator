# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline SGLang MoE sweep backed by the installable exact runner."""

from __future__ import annotations

import itertools
import statistics

from aiconfigurator.collector.sglang.moe import run_moe_case
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol
from collector.case_generator import get_moe_quantization_module_config
from collector.helper import log_perf

_SM120_NEMOTRON_NVFP4_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
}


def get_moe_test_cases():
    dependencies = globals()
    if "get_common_moe_test_cases" not in dependencies:
        from collector.case_generator import (
            get_common_moe_test_cases,
            get_moe_quantization_module_config,
            moe_model_allows_quantization,
        )
        from collector.helper import get_sm_version
    else:
        get_common_moe_test_cases = dependencies["get_common_moe_test_cases"]
        get_moe_quantization_module_config = dependencies["get_moe_quantization_module_config"]
        moe_model_allows_quantization = dependencies["moe_model_allows_quantization"]
        get_sm_version = dependencies["get_sm_version"]

    # fp8_block MOE requires SM90+ due to shared memory requirements
    # L40S (SM89) has 100KB shared memory, fp8_block kernel needs ~144KB
    sm_version = get_sm_version()
    if sm_version < 90:
        moe_list = ["bfloat16", "int4_wo"]
    elif sm_version < 100:
        moe_list = ["bfloat16", "fp8_block", "int4_wo"]
    elif sm_version in (100, 103):
        moe_list = [
            "bfloat16",
            "fp8_block",
            "nvfp4",
            "int4_wo",
            "w4a16_mxfp4",
            "w4a8_mxfp4_mxfp8",
        ]
    else:
        # SGLang 0.5.10 routes many nvfp4 MoE cases through FlashInfer paths
        # that were not validated on SM120. Add back only live-smoked Nemotron
        # NVFP4 model cases below instead of enabling the mode globally.
        moe_list = ["bfloat16", "fp8_block", "int4_wo"]

    common_cases = get_common_moe_test_cases()
    # When both GLM-5-NVFP4 and GLM-5.2-NVFP4 are present, collect only
    # GLM-5.2-NVFP4 (identical MoE; GLM-5.2 is the longest-context one).
    _present = {tc.model_name for tc in common_cases}
    _drop_models = {"nvidia/GLM-5-NVFP4"} if {"nvidia/GLM-5-NVFP4", "nvidia/GLM-5.2-NVFP4"} <= _present else set()

    test_cases = []

    for common_moe_testcase in common_cases:
        model_name = common_moe_testcase.model_name
        if model_name in _drop_models:
            continue

        model_moe_list = moe_list
        if model_name == "zai-org/GLM-5":
            model_moe_list = ["bfloat16"]
        elif model_name == "zai-org/GLM-5-FP8":
            model_moe_list = ["fp8_block"]
        elif model_name in ("nvidia/GLM-5-NVFP4", "nvidia/GLM-5.2-NVFP4"):
            # nvfp4 MoE; sweep the full EP/TP grid like DeepSeek-V4 (the EP/TP
            # validity is already bounded by get_common_moe_test_cases via
            # tp*ep==gpu / ep<=num_experts / num_experts%ep==0 / inter%tp==0).
            # (Previously hard-pinned to ep==1 & tp<32 as an nvfp4-EP>1 workaround.)
            model_moe_list = ["nvfp4"]
        elif sm_version >= 120 and model_name in _SM120_NEMOTRON_NVFP4_MODELS:
            model_moe_list = [*model_moe_list, "nvfp4"]

        num_tokens_list = [num_tokens for num_tokens in common_moe_testcase.num_tokens_list if num_tokens <= 20480]

        for moe_type, num_tokens in itertools.product(model_moe_list, num_tokens_list):
            if not moe_model_allows_quantization("sglang", model_name, moe_type):
                continue
            if (
                sm_version >= 120
                and moe_type == "nvfp4"
                and model_name in _SM120_NEMOTRON_NVFP4_MODELS
                and common_moe_testcase.ep == 1
                and (common_moe_testcase.inter_size // common_moe_testcase.tp) % 32 != 0
            ):
                # The SGLang 0.5.10 EP=1 NVFP4 path uses FlashInfer's TRTLLM
                # BF16xFP4 routed kernel, which requires the local intermediate
                # size to be divisible by 32. Keep non-divisible Nemotron
                # slices out of generated collection plans.
                continue
            # fp8_block requires hidden_size divisible by block group_size (128)
            if moe_type == "fp8_block" and (
                common_moe_testcase.hidden_size % 128 != 0 or common_moe_testcase.inter_size % 128 != 0
            ):
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 14336
                and common_moe_testcase.topk == 2
                and common_moe_testcase.num_experts == 8
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 16
                    or (common_moe_testcase.ep == 2 and num_tokens >= 8)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 4)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 2)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Mixtral on SM120 at this TP slice. These token counts require
                # 144 KiB shared memory, above the 99 KiB runtime limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 2688
                and common_moe_testcase.topk == 22
                and common_moe_testcase.num_experts == 512
                and (
                    (common_moe_testcase.tp == 2 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 64 and num_tokens >= 16)
                    or (common_moe_testcase.tp == 4 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 8 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 64)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 128)
                    or (common_moe_testcase.tp == 16 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                )
            ):
                # SGLang 0.5.10 falls back to the default Triton fp8 block MoE
                # config for Nemotron-3 Super on SM120 for these TP/EP slices.
                # That config requires 144 KiB shared memory, above the 99 KiB
                # runtime limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 2048
                and common_moe_testcase.inter_size == 768
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 128
                and common_moe_testcase.tp == 4
                and (
                    num_tokens >= 160
                    or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                    or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                    or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                    or (common_moe_testcase.ep == 64 and num_tokens >= 8)
                )
            ):
                # SGLang 0.5.10 also uses the default Triton fp8 block MoE config
                # for Qwen3-30B-A3B on SM120. For these larger token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 1536
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 128
                and (
                    (
                        common_moe_testcase.tp == 8
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                            or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 16
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 32
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                        )
                    )
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3-235B-A22B on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 6144
                and common_moe_testcase.inter_size == 2560
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 160
                and (
                    (
                        common_moe_testcase.tp == 8
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                            or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 16
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 32
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                        )
                    )
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3-Coder-480B-A35B on SM120. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 1024
                and common_moe_testcase.topk == 10
                and common_moe_testcase.num_experts == 512
                and (
                    (common_moe_testcase.tp == 16 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3.5-397B-A17B on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 6144
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # GLM-5 on SM120 at this TP slice. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V3 on SM120 at this TP slice. For these token counts
                # that config requires 144 KiB shared memory, above the 99 KiB
                # limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 6
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V4-Flash on SM120 at this TP slice. For these token
                # counts that config requires 144 KiB shared memory, above the
                # 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 3072
                and common_moe_testcase.topk == 6
                and common_moe_testcase.num_experts == 384
                and (
                    (common_moe_testcase.tp == 16 and num_tokens >= 512)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 64)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep >= 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 512)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 64)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep >= 16 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V4-Pro on SM120 for these TP/EP slices. For these
                # token counts that config requires 144 KiB shared memory,
                # above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 384
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 512
                    or (common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 64)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Kimi-K2 on SM120 at this TP slice. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 3072
                and common_moe_testcase.inter_size == 1536
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and (
                    common_moe_testcase.tp == 16
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 8 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 16 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 32 and num_tokens >= 16)
                    or (common_moe_testcase.tp == 8 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 320)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # MiniMax-M2.5 on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue

            if moe_type == "nvfp4":
                shard_k = common_moe_testcase.inter_size // common_moe_testcase.tp
                # fp4_quantize requires weight dims divisible by 16 after TP sharding.
                # CuteDSL grouped GEMM additionally requires 16-byte contiguous alignment:
                # for fp4 (4-bit), that's 32 elements (16 * 8 // 4 = 32).
                # See: flashinfer/cute_dsl/blockscaled_gemm.py
                #   Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment()
                if shard_k % 32 != 0:
                    continue

            if moe_type == "int4_wo":
                int4_group_size = int(
                    get_moe_quantization_module_config("sglang", moe_type, model_name=model_name).get("group_size", 128)
                )
                if (
                    common_moe_testcase.hidden_size % int4_group_size != 0
                    or (common_moe_testcase.inter_size // common_moe_testcase.tp) % int4_group_size != 0
                ):
                    continue
            if moe_type == "int4_wo" and common_moe_testcase.topk > (
                common_moe_testcase.num_experts // common_moe_testcase.ep
            ):
                # The SGLang int4 MoE path benchmarks the rank-0 local expert
                # slice. Cases where global top-k exceeds local experts fail
                # routing before kernel timing and are not valid single-rank
                # collector inputs.
                continue

            swiglu_limit = None
            # DeepSeek-V4 uses swiglu_limit=10
            if "DeepSeek-V4" in common_moe_testcase.model_name:
                swiglu_limit = 10

            base_case = [
                moe_type,
                num_tokens,
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.model_name,
                common_moe_testcase.token_expert_distribution,
                common_moe_testcase.power_law_alpha,
                swiglu_limit,
            ]
            test_cases.append(base_case)

    return test_cases


def _distribution_name(distributed: str, power_law_alpha: float | None) -> str:
    if distributed != "power_law":
        return distributed
    if power_law_alpha is None:
        raise ValueError("power_law collection requires power_law_alpha")
    return f"power_law_{power_law_alpha}"


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0,
    swiglu_limit=10,
    *,
    perf_filename,
    device="cuda:0",
):
    """Measure one offline row, sharing the online runner where supported."""

    int4_group_size = None
    if str(moe_type) == "int4_wo":
        int4_group_size = int(
            get_moe_quantization_module_config("sglang", str(moe_type), model_name=model_name).get("group_size", 128)
        )
    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=5,
        samples=10,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-moe-v1",
    )
    distribution_name = _distribution_name(distributed, power_law_alpha)
    if str(moe_type) in {"bfloat16", "fp8_block"}:
        raw = run_moe_case(
            num_tokens=int(num_tokens),
            hidden_size=int(hidden_size),
            inter_size=int(inter_size),
            topk=int(topk),
            num_experts=int(num_experts),
            moe_tp_size=int(moe_tp_size),
            moe_ep_size=int(moe_ep_size),
            quant_mode=str(moe_type),
            workload_distribution=distribution_name,
            swiglu_limit=swiglu_limit,
            protocol=protocol,
            device=device,
            model_path=model_name,
        )
    else:
        from collector.sglang import moe_runtime

        results = moe_runtime.run_moe_torch(
            str(moe_type),
            int(num_tokens),
            int(hidden_size),
            int(inter_size),
            int(topk),
            int(num_experts),
            int(moe_tp_size),
            int(moe_ep_size),
            model_name,
            distributed,
            power_law_alpha,
            swiglu_limit,
            device=device,
            num_warmups=protocol.warmups,
            num_iterations=protocol.samples,
            int4_group_size=int4_group_size,
        )
        if results.get("used_cuda_graph") is not True:
            raise RuntimeError("offline SGLang MoE collection requires CUDA Graph capture")
        samples_ms = tuple(float(sample) for sample in results["samples_ms"])
        if len(samples_ms) != protocol.samples:
            raise ValueError("offline SGLang MoE backend sample count does not match the protocol")
        latency_ms = float(statistics.median(samples_ms))
        power_stats = results.get("power_stats")
        raw = RawMeasurement(
            latency_ms=latency_ms,
            energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
            samples_ms=samples_ms,
            statistic=protocol.statistic,
            perf_row={
                "framework": "SGLang",
                "version": results["framework_version"],
                "device": results["device_name"],
                "op_name": "moe",
                "kernel_source": results["kernel_source"],
                **results["perf_row"],
                "latency": latency_ms,
            },
            provenance={
                "framework": "SGLang",
                "framework_version": results["framework_version"],
                "kernel_source": results["kernel_source"],
                "device": results["device_name"],
                "used_cuda_graph": True,
                "throttled": bool(results.get("throttled", False)),
                "model_artifact": model_name,
                "workload_generator": "power_law_v3" if distributed == "power_law" else "balanced-v1",
                "seed": 0,
                "rank_simulation": f"single-gpu-ep{moe_ep_size}-rank0",
            },
            protocol_digest=protocol.digest,
            power_stats=power_stats,
        )
    log_perf(
        item_list=[dict(raw.perf_row)],
        framework=str(raw.provenance["framework"]),
        version=str(raw.provenance["framework_version"]),
        device_name=str(raw.provenance["device"]),
        op_name="moe",
        kernel_source=str(raw.provenance["kernel_source"]),
        perf_filename=perf_filename,
        power_stats=dict(raw.power_stats) if raw.power_stats is not None else None,
    )
    return raw


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_moe_test_cases():
        print(test_case)
        run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
