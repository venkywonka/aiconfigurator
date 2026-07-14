# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only SGLang MoE backends for legacy offline quantization sweeps.

Benchmarks SGLang fused MoE kernels across BF16, FP8 block, NVFP4, and INT4
paths when supported. The installable online runner intentionally excludes the
NVFP4, INT4, and MXFP4 compatibility surface in this module. Shared MoE
model/sweep cases come from YAML; this module owns SGLang kernel compatibility,
server-args mocking, routing-logit synthesis, rank-local workload construction,
and quantized weight setup. It never owns persistence.
"""

import inspect
from contextlib import contextmanager, nullcontext
from importlib.metadata import version as get_version
from typing import TypedDict
from unittest.mock import MagicMock

import sglang.srt.server_args as _server_args_module
import torch

from aiconfigurator.collector.benchmark import benchmark_with_power
from aiconfigurator.collector.sglang.moe import (
    _balanced_logits as balanced_logits,
)
from aiconfigurator.collector.sglang.moe import (
    _build_rank0_local_workload as build_rank0_local_workload,
)
from aiconfigurator.collector.sglang.moe import (
    _power_law_logits_v3 as power_law_logits_v3,
)


@contextmanager
def _temporary_legacy_server_args():
    """Provide legacy import/runtime defaults without contaminating later cases."""

    original = _server_args_module._global_server_args
    if original is None:
        mock_server_args = MagicMock()
        mock_server_args.enable_deterministic_inference = False
        mock_server_args.enable_fused_moe_sum_all_reduce = False
        mock_server_args.kt_weight_path = None
        mock_server_args.flashinfer_mxfp4_moe_precision = "default"
        _server_args_module._global_server_args = mock_server_args
    try:
        yield
    finally:
        _server_args_module._global_server_args = original


# SGLang 0.5.5+ reads global server args while importing its legacy Triton MoE
# compatibility modules. Scope those defaults to import itself and restore the
# process global before this source-only helper becomes visible to its caller.
with _temporary_legacy_server_args():
    import sglang.srt.layers.moe.fused_moe_triton.layer as _moe_layer_mod
    import sglang.srt.layers.moe.token_dispatcher.standard as _std_dispatch_mod
    import sglang.srt.layers.moe.utils as _moe_utils

    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            get_default_config,
            get_moe_configs,
        )
    except ImportError:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
            get_config_dtype_str,
            get_default_config,
            get_moe_configs,
        )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import BypassedTopKOutput, StandardTopKOutput, TopKConfig, select_experts
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.utils import is_hip

    try:
        import sglang.srt.layers.quantization.mxfp4 as _mxfp4_mod
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

        _HAS_SGLANG_MXFP4 = True
    except ImportError:
        _HAS_SGLANG_MXFP4 = False

    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fmoe_mod
    except ImportError:
        import sglang.srt.layers.moe.fused_moe_triton.fused_moe as _fmoe_mod


@contextmanager
def _temporary_eager_moe_sum_reduce():
    """Scope the legacy eager reduction workaround to one offline benchmark."""

    original = getattr(_fmoe_mod, "moe_sum_reduce_torch_compile", None)
    if original is None:
        yield
        return

    # sglang >=0.5.10 can JIT this reduction for small token counts during CUDA
    # graph capture.  Some legacy quantized offline cases require eager execution.
    def eager_moe_sum_reduce(x, out, routed_scaling_factor):
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    _fmoe_mod.moe_sum_reduce_torch_compile = eager_moe_sum_reduce
    try:
        yield
    finally:
        _fmoe_mod.moe_sum_reduce_torch_compile = original

with _temporary_legacy_server_args():
    try:
        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )

        HAS_FLASHINFER_CUTE = True
    except ImportError:
        HAS_FLASHINFER_CUTE = False

    try:
        from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as _scaled_fp4_quant

        _HAS_SCALED_FP4_QUANT = True
    except ImportError:
        _HAS_SCALED_FP4_QUANT = False

    # Marlin int4 MoE kernel (W4A16) — much faster than the Triton GPTQ/AWQ path.
    _HAS_MARLIN_MOE = False
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
        from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
        from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

        _HAS_MARLIN_MOE = True
    except ImportError:
        pass

    _is_hip = is_hip()
    _MOE_RUNNER_CONFIG_PARAMS = set(inspect.signature(MoeRunnerConfig).parameters)
_NON_GATED_MOE_MODEL_PATTERNS = ("Nemotron-3", "nemotron-ultra", "Nemotron-H")


def _make_moe_runner_config(swiglu_limit: float | None = None) -> MoeRunnerConfig:
    kwargs = {}
    if "swiglu_limit" in _MOE_RUNNER_CONFIG_PARAMS:
        kwargs["swiglu_limit"] = swiglu_limit
    elif "gemm1_clamp_limit" in _MOE_RUNNER_CONFIG_PARAMS:
        kwargs["gemm1_clamp_limit"] = swiglu_limit
    return MoeRunnerConfig(**kwargs)


def _uses_relu2_moe_activation(model_name: str) -> bool:
    return any(pattern in model_name for pattern in _NON_GATED_MOE_MODEL_PATTERNS)


def _mxfp4_activation_precision(moe_type: str) -> str:
    """Map the persisted quant label to SGLang's explicit activation mode."""

    return "bf16" if moe_type == "w4a16_mxfp4" else "default"


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_nvfp4: bool = False,
    use_trtllm_bf16_fp4: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a16: bool = False,
    use_mxfp4_w4a8: bool = False,
    block_shape: list[int] | None = None,
    num_warmups: int = 5,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    model_name: str = "",
) -> float:
    device = torch.device("cuda")
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8
    workload_count = len(workloads) if workloads is not None else num_iters
    if workloads is not None:
        num_tokens = max(workload["hidden_states"].shape[0] for workload in workloads)

    # 1. Gating Output Generation (not needed for Marlin int4 path which builds its own)
    if not (use_int4_w4a16 and _HAS_MARLIN_MOE):
        if workloads is not None:
            gating_output = None
        elif distributed == "uniform":
            gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32, device=device)
        elif distributed == "balanced":
            gating_output = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
        elif distributed == "power_law":
            gating_output = [
                power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device)
                for _ in range(num_iters)
            ]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

    # 2. Setup based on Path
    if use_int4_w4a16 and _HAS_MARLIN_MOE:
        # Marlin int4 MoE path: repack GPTQ weights into Marlin tile layout
        # and call fused_marlin_moe which uses optimized CUDA kernels.
        num_bits = 4
        pack_factor = 8  # 32-bit int packs 8 x int4
        group_size = block_shape[1] if block_shape else 128

        # GPTQ-packed weights: (E, K // pack_factor, N) as int32
        w1_packed = torch.randint(
            -(2**31),
            2**31 - 1,
            (num_experts, hidden_size // pack_factor, shard_intermediate_size),
            dtype=torch.int32,
            device=device,
        )
        w2_packed = torch.randint(
            -(2**31),
            2**31 - 1,
            (num_experts, (shard_intermediate_size // 2) // pack_factor, hidden_size),
            dtype=torch.int32,
            device=device,
        )
        empty_perm = torch.empty((num_experts, 0), dtype=torch.int32, device=device)

        # Repack to Marlin layout: (E, K // 16, N * (num_bits // 2))
        w1_marlin = gptq_marlin_moe_repack(
            w1_packed,
            empty_perm,
            hidden_size,
            shard_intermediate_size,
            num_bits,
        )
        w2_marlin = gptq_marlin_moe_repack(
            w2_packed,
            empty_perm,
            shard_intermediate_size // 2,
            hidden_size,
            num_bits,
        )
        del w1_packed, w2_packed

        # Per-group scales: (E, K // group_size, N) — then permute for Marlin
        w1_scale = torch.randn(
            (num_experts, hidden_size // group_size, shard_intermediate_size),
            dtype=dtype,
            device=device,
        )
        w2_scale = torch.randn(
            (num_experts, (shard_intermediate_size // 2) // group_size, hidden_size),
            dtype=dtype,
            device=device,
        )
        w1_scale = marlin_moe_permute_scales(w1_scale, hidden_size, shard_intermediate_size, group_size)
        w2_scale = marlin_moe_permute_scales(w2_scale, shard_intermediate_size // 2, hidden_size, group_size)

        x = None if workloads is not None else torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        if workloads is None:
            if distributed == "power_law":
                gating_list = [
                    power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device)
                    for _ in range(num_iters)
                ]
            elif distributed == "balanced":
                gating_list = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
            else:
                gating_list = [
                    torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(num_iters)
                ]

        def run_op(i):
            if workloads is not None:
                current_hidden_states = workloads[i % workload_count]["hidden_states"]
                current_topk = workloads[i % workload_count]["topk_output"]
                # fused_marlin_moe asserts gating_output.shape[0] == hidden_states.shape[0],
                # but only uses topk_weights/topk_ids for routing. Provide a dummy.
                dummy_gating = torch.zeros(
                    current_hidden_states.shape[0],
                    num_experts,
                    device=current_hidden_states.device,
                    dtype=torch.float32,
                )
                # build_rank0_local_workload sets remote expert IDs to -1
                # and their weights to 0.  The Marlin CUDA kernel
                # (moe_wna16_marlin_gemm) indexes weight tensors by expert
                # ID without masking, so -1 causes illegal memory access.
                # Clamp to 0; the zero weight ensures no contribution.
                safe_topk_ids = current_topk.topk_ids.clamp(min=0)
                fused_marlin_moe(
                    current_hidden_states,
                    w1_marlin,
                    w2_marlin,
                    w1_scale,
                    w2_scale,
                    dummy_gating,
                    current_topk.topk_weights,
                    safe_topk_ids,
                    num_bits=num_bits,
                    is_k_full=True,
                )
            else:
                gating = gating_list[i % num_iters]
                new_topk = select_experts(x, gating, TopKConfig(top_k=topk))
                fused_marlin_moe(
                    x,
                    w1_marlin,
                    w2_marlin,
                    w1_scale,
                    w2_scale,
                    gating,
                    new_topk.topk_weights,
                    new_topk.topk_ids,
                    num_bits=num_bits,
                    is_k_full=True,
                )

    elif use_nvfp4 and use_trtllm_bf16_fp4 and workloads is None:
        from flashinfer.fused_moe import ActivationType, trtllm_fp4_block_scale_routed_moe
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import quantize_hidden_states_fp4

        try:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _pack_topk_for_flashinfer_routed
        except ImportError:
            # Newer sglang (glm52 dev image) dropped this helper from
            # flashinfer_trtllm; the same trtllm routed-MoE pack
            # ((expert_id<<16)|bf16_weight, bit-identical) now lives as
            # fused_pack_topk. Without this the trtllm BF16xFP4 MoE kernel can't
            # be collected on the dev image (ImportError) and only the slower
            # cutedsl path survives, making AIC over-predict MoE.
            from sglang.jit_kernel.trtllm_lora_temp.topk_pack import (
                fused_pack_topk as _pack_topk_for_flashinfer_routed,
            )

        if hidden_size % 32 != 0 or (shard_intermediate_size // 2) % 32 != 0:
            raise ValueError(
                "FlashInfer TRTLLM BF16xFP4 MoE requires hidden and intermediate dimensions "
                f"to be divisible by 32, got hidden_size={hidden_size} and "
                f"intermediate_size={shard_intermediate_size // 2}"
            )

        intermediate_size = shard_intermediate_size // 2
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        router_logits_list = (
            gating_output
            if isinstance(gating_output, list)
            else [gating_output[i] for i in range(gating_output.shape[0])]
        )

        # Match GLM-5's ModelOpt FP4 + FlashInfer TRTLLM routed path. That path
        # quantizes activations to FP4 before invoking the routed MoE kernel;
        # passing BF16 activations directly selects a much slower BF16xFP4 path.
        w13_weight = torch.randint(
            0,
            256,
            (num_experts, shard_intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2_weight = torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        sf_block_size = 16
        w13_scale = torch.ones(
            (num_experts, shard_intermediate_size, hidden_size // sf_block_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        w2_scale = torch.ones(
            (num_experts, hidden_size, intermediate_size // sf_block_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )

        output = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
        tune_max_num_tokens = max(1, 1 << (num_tokens - 1).bit_length())
        scale_ones = torch.ones(num_experts, dtype=torch.float32, device=device)
        input_scale_quant = torch.ones((), dtype=torch.float32, device=device)
        activation_type = ActivationType.Relu2 if _uses_relu2_moe_activation(model_name) else ActivationType.Swiglu
        topk_config = TopKConfig(
            top_k=topk,
            renormalize=True,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
        )
        packed_topk_list = []
        for logits in router_logits_list:
            topk_output = select_experts(x, logits, topk_config)
            packed_topk_list.append(
                _pack_topk_for_flashinfer_routed(
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                )
            )
        torch.cuda.synchronize()

        def run_op(i):
            x_fp4, x_scale = quantize_hidden_states_fp4(x, input_scale_quant)
            packed_topk = packed_topk_list[i % len(packed_topk_list)]
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed_topk,
                routing_bias=None,
                hidden_states=x_fp4,
                hidden_states_scale=x_scale,
                gemm1_weights=w13_weight,
                gemm1_weights_scale=w13_scale,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=w2_weight,
                gemm2_weights_scale=w2_scale,
                gemm2_bias=None,
                output1_scale_scalar=scale_ones,
                output1_scale_gate_scalar=scale_ones,
                output2_scale_scalar=scale_ones,
                num_experts=num_experts,
                top_k=packed_topk.shape[1],
                n_group=0,
                topk_group=0,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                routing_method_type=1,
                do_finalize=True,
                activation_type=activation_type,
                output=output,
                tune_max_num_tokens=tune_max_num_tokens,
            )

    elif use_nvfp4:
        if not HAS_FLASHINFER_CUTE:
            raise ImportError("FlashInfer CuteDSL not available")
        if not _HAS_SCALED_FP4_QUANT:
            raise ImportError(
                "scaled_fp4_quant not available (sglang.jit_kernel.nvfp4); "
                "NVFP4 MoE benchmarking requires this for correct weight layout"
            )

        # Global scales and Alpha
        input_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        a2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)

        # Weight quantization
        w1_bf16 = torch.randn(num_experts, shard_intermediate_size, hidden_size, device=device, dtype=dtype)
        w2_bf16 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, device=device, dtype=dtype)

        # Quantize weights per-expert using scaled_fp4_quant which produces
        # swizzled blockscales and maintains (num_experts, N, K//2) layout.
        w1_list_q, w1_list_bs = [], []
        for e in range(num_experts):
            q, bs = _scaled_fp4_quant(w1_bf16[e], w1_gs[e])
            w1_list_q.append(q)
            w1_list_bs.append(bs)
        w1 = torch.stack(w1_list_q)
        w1_bs = torch.stack(w1_list_bs)

        w2_list_q, w2_list_bs = [], []
        for e in range(num_experts):
            q, bs = _scaled_fp4_quant(w2_bf16[e], w2_gs[e])
            w2_list_q.append(q)
            w2_list_bs.append(bs)
        w2 = torch.stack(w2_list_q)
        w2_bs = torch.stack(w2_list_bs)

        def get_masked_m(logits):
            _, topk_idx = torch.topk(torch.softmax(logits, dim=1), topk, dim=-1)
            counts = [(topk_idx.view(-1) == i).sum() for i in range(num_experts)]
            return torch.tensor(counts, dtype=torch.int32, device=device)

        masked_m_list = (
            [workload["masked_m"] for workload in workloads]
            if workloads is not None
            else [get_masked_m(logits) for logits in gating_output]
        )

        # Calculate the maximum tokens any single expert will handle across all iterations
        max_m = 0
        for counts in masked_m_list:
            max_m = max(max_m, counts.max().item())
        # Align to 128 for kernel efficiency and safety
        max_m = (max_m + 127) // 128 * 128

        x_dispatched = torch.randn(num_experts, max_m, hidden_size, device=device, dtype=dtype)

        def run_op(i):
            flashinfer_cutedsl_moe_masked(
                hidden_states=(x_dispatched, None),
                input_global_scale=input_gs,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gs,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m_list[i % workload_count],
            )
    elif use_mxfp4_moe:
        # Reuse SGLang 0.5.10's production Mxfp4Config/FusedMoE path. It owns
        # the exact FlashInfer API, weight layout, TP padding, and EP-local
        # expert mapping for both W4A16 and W4A8.
        if not _HAS_SGLANG_MXFP4:
            raise ImportError("SGLang MXFP4 MoE support is not available")
        if workloads is not None:
            raise ValueError("MXFP4 benchmarking uses full-router logits, not rank-local workloads")

        previous_backend = _moe_utils.MOE_RUNNER_BACKEND
        server_args = _server_args_module._global_server_args
        previous_precision = server_args.flashinfer_mxfp4_moe_precision
        _moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend.FLASHINFER_MXFP4
        server_args.flashinfer_mxfp4_moe_precision = _mxfp4_activation_precision(
            "w4a16_mxfp4" if use_mxfp4_w4a16 else "w4a8_mxfp4_mxfp8"
        )
        try:
            intermediate_size = shard_intermediate_size // 2 * moe_tp_size
            mxfp4_config_kwargs = {}
            if "is_checkpoint_mxfp4_serialized" in inspect.signature(Mxfp4Config).parameters:
                mxfp4_config_kwargs["is_checkpoint_mxfp4_serialized"] = True
            quant_config = Mxfp4Config(**mxfp4_config_kwargs)
            moe_layer = FusedMoE(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_id=0,
                top_k=topk,
                params_dtype=dtype,
                reduce_results=False,
                quant_config=quant_config,
                prefix="aic_sglang_mxfp4_moe",
            ).to(device)
        finally:
            _moe_utils.MOE_RUNNER_BACKEND = previous_backend
            server_args.flashinfer_mxfp4_moe_precision = previous_precision

        with torch.no_grad():
            moe_layer.w13_weight.zero_()
            moe_layer.w2_weight.zero_()
            moe_layer.w13_weight_scale.copy_(
                torch.ones_like(moe_layer.w13_weight_scale, dtype=torch.float8_e4m3fn).view(torch.uint8)
            )
            moe_layer.w2_weight_scale.copy_(
                torch.ones_like(moe_layer.w2_weight_scale, dtype=torch.float8_e4m3fn).view(torch.uint8)
            )
            moe_layer.w13_weight_bias.zero_()
            moe_layer.w2_weight_bias.zero_()
        moe_layer.quant_method.process_weights_after_loading(moe_layer)

        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if distributed == "uniform":
            router_logits_list = [
                torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(num_iters)
            ]
        elif distributed == "balanced":
            router_logits_list = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
        elif distributed == "power_law":
            router_logits_list = [
                power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha).to(device)
                for _ in range(num_iters)
            ]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        def run_op(i):
            moe_layer(
                x,
                BypassedTopKOutput(
                    hidden_states=x,
                    router_logits=router_logits_list[i % num_iters],
                    topk_config=TopKConfig(top_k=topk),
                ),
            )
    else:
        init_dtype = torch.bfloat16 if use_fp8_w8a8 else dtype
        x = None if workloads is not None else torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if use_int8_w8a16 or use_int8_w8a8:
            w1 = torch.randint(
                -127, 127, (num_experts, shard_intermediate_size, hidden_size), dtype=torch.int8, device=device
            )
            w2 = torch.randint(
                -127, 127, (num_experts, hidden_size, shard_intermediate_size // 2), dtype=torch.int8, device=device
            )
        elif use_int4_w4a16:
            # W4A16: 2 int4 values packed per int8 byte — K dimension halved.
            # w1 shape: (E, N=shard_inter, K_packed=hidden//2)
            # w2 shape: (E, N=hidden, K_packed=shard_inter//4)
            w1 = torch.randint(
                0, 127, (num_experts, shard_intermediate_size, hidden_size // 2), dtype=torch.int8, device=device
            )
            w2 = torch.randint(
                0, 127, (num_experts, hidden_size, shard_intermediate_size // 4), dtype=torch.int8, device=device
            )
        else:
            w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device)
            w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device)

        w1_scale = w2_scale = a1_scale = a2_scale = None
        if use_int8_w8a16:
            w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device)
            w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
        elif use_int4_w4a16:
            # Per-group scales along K. The GPTQ kernel receives K = A.shape[1]
            # (unpacked hidden size), so scale groups are hidden_size // group_size,
            # NOT (hidden_size // 2) // group_size (the packed size).
            # w2's K is shard_intermediate_size // 2 (post silu_and_mul, unpacked).
            group_size = block_shape[1] if block_shape else 128
            w1_scale = torch.randn(
                (num_experts, shard_intermediate_size, hidden_size // group_size),
                dtype=torch.float32,
                device=device,
            )
            w2_scale = torch.randn(
                (num_experts, hidden_size, (shard_intermediate_size // 2) // group_size),
                dtype=torch.float32,
                device=device,
            )
        elif use_fp8_w8a8 or use_int8_w8a8:
            if use_int8_w8a8 and block_shape is None:
                w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
            elif block_shape is None:
                w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                a1_scale = torch.randn(1, dtype=torch.float32, device=device)
                a2_scale = torch.randn(1, dtype=torch.float32, device=device)
            else:
                bn, bk = block_shape
                w1_scale = torch.rand(
                    (num_experts, (shard_intermediate_size + bn - 1) // bn, (hidden_size + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )
                w2_scale = torch.rand(
                    (num_experts, (hidden_size + bn - 1) // bn, (shard_intermediate_size // 2 + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )

        if use_fp8_w8a8:
            f8_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w1, w2 = w1.to(f8_type), w2.to(f8_type)

        topk_output = (
            None
            if workloads is not None
            else select_experts(x, torch.randn(num_tokens, num_experts, device=device), TopKConfig(top_k=topk))
        )

        def run_op(i):
            from sglang.srt.layers.moe.fused_moe_triton import override_config

            if workloads is None:
                input_gating = gating_output[i % num_iters]
                new_topk = select_experts(x, input_gating, TopKConfig(top_k=topk))
                topk_output.topk_weights.copy_(new_topk.topk_weights)
                topk_output.topk_ids.copy_(new_topk.topk_ids)
                topk_output.router_logits.copy_(new_topk.router_logits)
                current_hidden_states = x
                current_topk_output = topk_output
            else:
                current_hidden_states = workloads[i % workload_count]["hidden_states"]
                current_topk_output = workloads[i % workload_count]["topk_output"]
                # build_rank0_local_workload sets remote expert IDs to -1
                # and their weights to 0.  The Triton fused_moe kernel
                # indexes weight tensors by expert ID without masking,
                # so -1 causes illegal memory access.
                # Clamp to 0; the zero weight ensures no contribution.
                current_topk_output = StandardTopKOutput(
                    topk_weights=current_topk_output.topk_weights,
                    topk_ids=current_topk_output.topk_ids.clamp(min=0),
                    router_logits=current_topk_output.router_logits,
                )

            with override_config(config):
                moe_runner_config = _make_moe_runner_config(swiglu_limit=swiglu_limit)
                fused_moe(
                    current_hidden_states,
                    w1,
                    w2,
                    current_topk_output,
                    moe_runner_config=moe_runner_config,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=use_int4_w4a16,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                )

    # 3. Unified Execution Loop
    outside_loop_count = 5  # Repeat ops within kernel_func to increase accuracy for fast kernels

    def kernel_func():
        for i in range(outside_loop_count):
            run_op(i)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=num_warmups,
        num_runs=num_iters,
        repeat_n=1,
        # sglang >=0.5.10 adds @torch.compile paths inside fused_experts_impl
        # (moe_sum_reduce_torch_compile) that can hang during CUDA graph capture.
        allow_graph_fail=False,
        return_samples=True,
    ) as results:
        pass

    return {
        **results,
        "latency_ms": results["latency_ms"] / outside_loop_count,
        "samples_ms": tuple(sample / outside_loop_count for sample in results["samples_ms"]),
    }


@contextmanager
def _patch_mxfp4_single_process_parallel(*, moe_tp_size: int, moe_ep_size: int):
    """Temporarily patch SGLang distributed helpers for a rank-0 MoE benchmark."""

    missing = object()
    originals = []

    def replace(module, name, value):
        originals.append((module, name, getattr(module, name, missing)))
        setattr(module, name, value)

    try:
        for module in (_moe_layer_mod, _std_dispatch_mod, _mxfp4_mod):
            if hasattr(module, "get_tp_group"):
                replace(module, "get_tp_group", lambda: None)
            if hasattr(module, "is_allocation_symmetric"):
                replace(module, "is_allocation_symmetric", lambda: False)
        replace(_moe_layer_mod, "get_moe_expert_parallel_world_size", lambda: moe_ep_size)
        replace(_moe_layer_mod, "get_moe_expert_parallel_rank", lambda: 0)
        replace(_moe_layer_mod, "get_moe_tensor_parallel_world_size", lambda: moe_tp_size)
        replace(_moe_layer_mod, "get_moe_tensor_parallel_rank", lambda: 0)
        replace(_moe_layer_mod, "create_kt_config_from_server_args", lambda _server_args, _layer_id: None)
        replace(_std_dispatch_mod, "get_moe_expert_parallel_world_size", lambda: moe_ep_size)
        replace(_std_dispatch_mod, "get_moe_expert_parallel_rank", lambda: 0)
        yield
    finally:
        for module, name, original in reversed(originals):
            if original is missing:
                delattr(module, name)
            else:
                setattr(module, name, original)


def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_nvfp4: bool = False,
    use_trtllm_bf16_fp4: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a16: bool = False,
    use_mxfp4_w4a8: bool = False,
    block_shape: list[int] | None = None,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    model_name: str = "",
    num_warmups: int = 5,
    num_iters: int = 10,
) -> dict:
    torch.cuda.manual_seed_all(0)
    benchmark_num_tokens = (
        max(workload["hidden_states"].shape[0] for workload in workloads) if workloads is not None else num_tokens
    )
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8

    if use_nvfp4 or use_mxfp4_moe or (use_int4_w4a16 and _HAS_MARLIN_MOE):
        # NVFP4 uses FlashInfer CuteDSL; MXFP4 uses SGLang's high-level
        # FlashInfer runner; INT4_W4A16 uses Marlin. None need Triton tuning.
        # MXFP4 reads the patched helpers during forward, so keep them scoped
        # through setup, warmup, and timing rather than only layer construction.
        parallel_context = (
            _patch_mxfp4_single_process_parallel(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size)
            if use_mxfp4_moe and _HAS_SGLANG_MXFP4
            else nullcontext()
        )
        with parallel_context:
            results = benchmark_config(
                None,
                benchmark_num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype,
                use_fp8_w8a8,
                use_int8_w8a8,
                use_int8_w8a16,
                use_nvfp4,
                use_trtllm_bf16_fp4,
                use_int4_w4a16,
                use_mxfp4_w4a16,
                use_mxfp4_w4a8,
                block_shape,
                num_warmups=num_warmups,
                num_iters=num_iters,
                distributed=distributed,
                power_law_alpha=power_law_alpha,
                workloads=workloads,
                swiglu_limit=swiglu_limit,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                model_name=model_name,
            )
        return results

    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_fp8_w8a8=use_fp8_w8a8,
    )
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    op_config = get_moe_configs(num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k)
    if op_config is None:
        config = get_default_config(
            benchmark_num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - benchmark_num_tokens))]
    results = benchmark_config(
        config,
        benchmark_num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_nvfp4,
        False,
        use_int4_w4a16,
        use_mxfp4_w4a16,
        use_mxfp4_w4a8,
        block_shape,
        num_warmups=num_warmups,
        num_iters=num_iters,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
        workloads=workloads,
        swiglu_limit=swiglu_limit,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        model_name=model_name,
    )
    return results


class Rank0Workload(TypedDict):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    masked_m: torch.Tensor


def build_rank0_workloads(
    num_workloads: int,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    moe_ep_size: int,
    distributed: str,
    power_law_alpha: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Rank0Workload]:
    workloads: list[Rank0Workload] = []
    experts_per_rank = num_experts // moe_ep_size

    for _ in range(num_workloads):
        if distributed == "power_law":
            if power_law_alpha is None:
                raise ValueError("power_law_alpha is required for power_law distribution")
            _, rank0_info = power_law_logits_v3(
                num_tokens,
                num_experts,
                topk,
                moe_ep_size,
                power_law_alpha,
                return_rank0_info=True,
            )
        elif distributed == "balanced":
            router_logits = balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)
            rank0_selected_slots = torch.topk(router_logits, topk, dim=-1).indices.to(torch.int64)
            rank0_token_mask = (rank0_selected_slots < experts_per_rank).any(dim=1)
            rank0_info = {
                "rank0_selected_slots": rank0_selected_slots[rank0_token_mask],
                "rank0_logits": router_logits[rank0_token_mask],
                "rank0_num_tokens": int(rank0_token_mask.sum().item()),
                "slots_per_rank": experts_per_rank,
            }
        else:
            raise ValueError(f"Unsupported distribution for rank0 workloads: {distributed}")

        rank0_local = build_rank0_local_workload(rank0_info)
        rank0_num_tokens = int(rank0_local["num_tokens"])
        workloads.append(
            {
                "hidden_states": torch.randn(rank0_num_tokens, hidden_size, dtype=dtype, device=device),
                "topk_output": StandardTopKOutput(
                    topk_weights=rank0_local["topk_weights"].to(device=device, dtype=torch.float32),
                    topk_ids=rank0_local["topk_ids"].to(device=device, dtype=torch.int32),
                    router_logits=torch.empty((rank0_num_tokens, 0), dtype=torch.float32, device=device),
                ),
                "masked_m": rank0_local["masked_m"].to(device=device, dtype=torch.int32),
            }
        )

    return workloads


def _run_moe_torch_impl(
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
    swiglu_limit=None,
    *,
    device="cuda:0",
    num_warmups=5,
    num_iterations=10,
    int4_group_size: int | None = None,
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert moe_type in [
        "fp8_block",
        "bfloat16",
        "nvfp4",
        "w4a8_mxfp4_mxfp8",
        "int4_wo",
        "w4a16_mxfp4",
    ], "only support moe type = fp8_block, bfloat16, nvfp4, int4_wo, w4a16_mxfp4, or w4a8_mxfp4_mxfp8"
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"
    assert num_experts % moe_ep_size == 0, "num_experts must be divisible by moe_ep_size"

    num_local_experts = num_experts // moe_ep_size
    use_int4_w4a16 = moe_type == "int4_wo"
    # GPT-OSS can use BF16 or MXFP8 activations with MXFP4 weights. Both labels
    # run through SGLang's version-matched high-level FlashInfer backend.
    use_mxfp4_w4a16 = moe_type == "w4a16_mxfp4"
    use_mxfp4_w4a8 = moe_type == "w4a8_mxfp4_mxfp8"
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8
    use_nvfp4_kernel = moe_type == "nvfp4"
    use_trtllm_bf16_fp4 = moe_type == "nvfp4" and moe_ep_size == 1
    if use_int4_w4a16:
        int4_group_size = 128 if int4_group_size is None else int4_group_size
        if isinstance(int4_group_size, bool) or not isinstance(int4_group_size, int) or int4_group_size <= 0:
            raise ValueError("SGLang INT4 group size must be a positive integer")
        block_shape = [0, int4_group_size]
    elif moe_type == "fp8_block" and (inter_size // moe_tp_size) % 128 == 0 and hidden_size % 128 == 0:
        block_shape = [128, 128]
    else:
        block_shape = None

    rank0_workloads: list[Rank0Workload] | None = None
    if moe_ep_size > 1 and distributed in ("power_law", "balanced") and not use_mxfp4_moe:
        rank0_workloads = build_rank0_workloads(
            num_workloads=5,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
            distributed=distributed,
            power_law_alpha=power_law_alpha if distributed == "power_law" else None,
            dtype=torch.bfloat16,
            device=torch.device(device),
        )

    with _temporary_eager_moe_sum_reduce():
        if rank0_workloads is not None:
            results = benchmark(
                num_tokens,
                num_local_experts,
                2 * inter_size // moe_tp_size,
                hidden_size,
                topk,
                torch.bfloat16,
                moe_type == "fp8_block",
                False,
                False,
                use_nvfp4=use_nvfp4_kernel,
                use_trtllm_bf16_fp4=use_trtllm_bf16_fp4,
                use_int4_w4a16=use_int4_w4a16,
                use_mxfp4_w4a16=False,
                use_mxfp4_w4a8=False,
                block_shape=block_shape,
                distributed=distributed,
                power_law_alpha=power_law_alpha,
                workloads=rank0_workloads,
                swiglu_limit=swiglu_limit,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                model_name=model_name,
                num_warmups=num_warmups,
                num_iters=num_iterations,
            )
        else:
            results = benchmark(
                num_tokens,
                num_experts if use_mxfp4_moe else num_local_experts,
                2 * inter_size // moe_tp_size,
                hidden_size,
                topk,
                torch.bfloat16,
                moe_type == "fp8_block",
                False,
                False,
                use_nvfp4=use_nvfp4_kernel,
                use_trtllm_bf16_fp4=use_trtllm_bf16_fp4,
                use_int4_w4a16=use_int4_w4a16,
                use_mxfp4_w4a16=use_mxfp4_w4a16,
                use_mxfp4_w4a8=use_mxfp4_w4a8,
                block_shape=block_shape,
                distributed=distributed,
                power_law_alpha=power_law_alpha,
                swiglu_limit=swiglu_limit,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                model_name=model_name,
                num_warmups=num_warmups,
                num_iters=num_iterations,
            )

    latency = results["latency_ms"]
    kernel_source = (
        "sglang_flashinfer_trtllm_bf16_fp4_moe"
        if use_trtllm_bf16_fp4
        else "sglang_flashinfer_cutedsl_moe"
        if use_nvfp4_kernel
        else "sglang_marlin_moe"
        if moe_type == "int4_wo" and _HAS_MARLIN_MOE
        else "sglang_flashinfer_mxfp4_moe"
        if moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
        else "sglang_fused_moe_triton"
    )
    row = {
        "moe_dtype": moe_type,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "topk": topk,
        "num_experts": num_experts,
        "moe_tp_size": moe_tp_size,
        "moe_ep_size": moe_ep_size,
        "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
        "latency": latency,
    }
    framework_version = get_version("sglang")
    device_name = torch.cuda.get_device_name(device)
    return {
        **results,
        "perf_row": row,
        "framework_version": framework_version,
        "device_name": device_name,
        "kernel_source": kernel_source,
    }


def run_moe_torch(*args, **kwargs):
    """Run one legacy quantized case without leaking SGLang server globals."""

    with _temporary_legacy_server_args():
        return _run_moe_torch_impl(*args, **kwargs)
