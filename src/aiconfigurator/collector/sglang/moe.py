# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact DSv4 V1.2 SGLang MoE runner."""

from __future__ import annotations

import inspect
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as get_version
from typing import Any
from unittest.mock import MagicMock

from aiconfigurator.collector.benchmark import benchmark_with_power
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"


@dataclass(frozen=True, slots=True)
class PreparedMoeCase:
    """One initialized rank-local EP4 case retained by a GPU worker."""

    kernel_func: Callable[[], Any]
    framework_version: str
    device_name: str
    device: Any
    model_artifact: str
    kernel_source: str
    workload_generator: str
    seed: int
    rank_simulation: str
    num_tokens: int
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    moe_tp_size: int
    moe_ep_size: int
    quant_mode: str
    workload_distribution: str
    latency_divisor: int = 1


def _round_robin_adjust_per_rank(counts_2d, remaining, is_valid, pick_local_index, step):
    while remaining > 0:
        progressed = False
        for rank_idx in range(counts_2d.size(0)):
            local_counts = counts_2d[rank_idx]
            valid_local = is_valid(local_counts).nonzero().flatten()
            if valid_local.numel() == 0:
                continue
            chosen = valid_local[pick_local_index(local_counts[valid_local])].item()
            counts_2d[rank_idx, chosen] += step
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return counts_2d


def _assign_experts_from_counts(num_tokens_per_expert, num_tokens: int, topk: int):
    import numpy as np
    import torch

    counts = num_tokens_per_expert.cpu().numpy().astype(np.int64)
    sorted_experts = np.argsort(-counts)
    expert_ids_flat = np.repeat(sorted_experts, counts[sorted_experts])
    selected = expert_ids_flat.reshape(topk, num_tokens).T.copy()
    return torch.from_numpy(selected).to(device=num_tokens_per_expert.device)


def _power_law_selected_experts(
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep_size: int,
    alpha: float,
    *,
    device,
    torch_module,
):
    xmin, xmax = (1.0, num_tokens * 0.8) if num_tokens * topk > num_experts else (0.01, 2.0)
    uniform = torch_module.rand(num_experts, device=device)
    sampled = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * uniform + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
    target_sum = num_tokens * topk
    counts = torch_module.round(sampled / sampled.sum() * target_sum).to(torch_module.int64)
    overflow = (counts - num_tokens).clamp(min=0).sum().item()
    counts = counts.clamp(max=num_tokens)
    experts_per_rank = num_experts // ep_size
    if overflow > 0:
        counts = _round_robin_adjust_per_rank(
            counts.view(ep_size, experts_per_rank),
            int(overflow),
            lambda local: local < num_tokens,
            torch_module.argmin,
            1,
        ).view(-1)
    delta = target_sum - counts.sum().item()
    if delta != 0:
        counts = _round_robin_adjust_per_rank(
            counts.view(ep_size, experts_per_rank),
            int(abs(delta)),
            (lambda local: local < num_tokens) if delta > 0 else (lambda local: local > 0),
            torch_module.argmin if delta > 0 else torch_module.argmax,
            1 if delta > 0 else -1,
        ).view(-1)
    rank_loads = counts.view(ep_size, experts_per_rank).sum(dim=1)
    max_rank = int(torch_module.argmax(rank_loads).item())
    if max_rank:
        by_rank = counts.view(ep_size, experts_per_rank)
        first = by_rank[0].clone()
        by_rank[0] = by_rank[max_rank]
        by_rank[max_rank] = first
        counts = by_rank.view(-1)
    return _assign_experts_from_counts(counts, num_tokens, topk)


def _rank0_workloads(
    *,
    num_workloads: int,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    moe_ep_size: int,
    power_law_alpha: float,
    device,
    torch_module,
    standard_topk_output,
):
    import torch.nn.functional as functional

    workloads = []
    experts_per_rank = num_experts // moe_ep_size
    for _ in range(num_workloads):
        selected = _power_law_selected_experts(
            num_tokens,
            num_experts,
            topk,
            moe_ep_size,
            power_law_alpha,
            device=device,
            torch_module=torch_module,
        )
        expert_map = functional.one_hot(selected.long(), num_classes=num_experts).sum(1)
        logits = functional.softmax(expert_map.bfloat16(), dim=1)
        token_mask = (selected < experts_per_rank).any(dim=1)
        selected = selected[token_mask]
        logits = logits[token_mask].to(torch_module.float32)
        weights = torch_module.gather(logits, 1, selected.long())
        local_mask = selected < experts_per_rank
        ids = selected.to(torch_module.int32)
        ids[~local_mask] = -1
        weights[~local_mask] = 0.0
        rank_tokens = int(token_mask.sum().item())
        generator = torch_module.Generator(device=device)
        generator.manual_seed(0)
        workloads.append(
            (
                torch_module.randn(
                    rank_tokens,
                    hidden_size,
                    dtype=torch_module.bfloat16,
                    device=device,
                    generator=generator,
                ),
                standard_topk_output(
                    topk_weights=weights.contiguous(),
                    topk_ids=ids.contiguous(),
                    router_logits=torch_module.empty((rank_tokens, 0), dtype=torch_module.float32, device=device),
                ),
            )
        )
    return workloads


def _make_dsv4_moe_runner_config(config_factory):
    parameters = inspect.signature(config_factory).parameters
    if "swiglu_limit" in parameters:
        return config_factory(swiglu_limit=10)
    if "gemm1_clamp_limit" in parameters:
        return config_factory(gemm1_clamp_limit=10)
    raise RuntimeError("SGLang MoeRunnerConfig does not expose the DSv4 SwiGLU clamp")


def _prepare_moe_case(
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    quant_mode: str,
    workload_distribution: str,
    device: str,
    model_path: str,
) -> PreparedMoeCase:
    """Import Torch/SGLang only after the worker has bound its GPU UUID."""

    import sglang.srt.server_args as server_args_module
    import torch

    if server_args_module._global_server_args is None:
        server_args = MagicMock()
        server_args.enable_deterministic_inference = False
        server_args.enable_fused_moe_sum_all_reduce = False
        server_args.kt_weight_path = None
        server_args.flashinfer_mxfp4_moe_precision = "default"
        server_args_module._global_server_args = server_args

    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            get_default_config,
            get_moe_configs,
        )
    except ImportError:
        import sglang.srt.layers.moe.fused_moe_triton.fused_moe as fused_moe_module
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
            get_config_dtype_str,
            get_default_config,
            get_moe_configs,
        )
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    def eager_sum_reduce(x, out, routed_scaling_factor):
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    if hasattr(fused_moe_module, "moe_sum_reduce_torch_compile"):
        fused_moe_module.moe_sum_reduce_torch_compile = eager_sum_reduce

    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    local_experts = num_experts // moe_ep_size
    shard_intermediate_size = 2 * inter_size // moe_tp_size
    workloads = _rank0_workloads(
        num_workloads=5,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        topk=topk,
        num_experts=num_experts,
        moe_ep_size=moe_ep_size,
        power_law_alpha=1.01,
        device=torch_device,
        torch_module=torch,
        standard_topk_output=StandardTopKOutput,
    )
    max_rank_tokens = max(hidden_states.shape[0] for hidden_states, _topk in workloads)
    block_shape = [128, 128]
    dtype_name = get_config_dtype_str(torch.bfloat16, use_fp8_w8a8=True)
    configs = get_moe_configs(local_experts, shard_intermediate_size // 2, dtype_name, *block_shape)
    config = (
        get_default_config(
            max_rank_tokens,
            local_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_name,
            False,
            block_shape,
        )
        if configs is None
        else configs[min(configs, key=lambda value: abs(value - max_rank_tokens))]
    )

    w1 = torch.randn(
        local_experts,
        shard_intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=torch_device,
    ).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        local_experts,
        hidden_size,
        shard_intermediate_size // 2,
        dtype=torch.bfloat16,
        device=torch_device,
    ).to(torch.float8_e4m3fn)
    block_n, block_k = block_shape
    w1_scale = torch.rand(
        local_experts,
        (shard_intermediate_size + block_n - 1) // block_n,
        (hidden_size + block_k - 1) // block_k,
        dtype=torch.float32,
        device=torch_device,
    )
    w2_scale = torch.rand(
        local_experts,
        (hidden_size + block_n - 1) // block_n,
        (shard_intermediate_size // 2 + block_k - 1) // block_k,
        dtype=torch.float32,
        device=torch_device,
    )
    runner_config = _make_dsv4_moe_runner_config(MoeRunnerConfig)

    def one_workload(hidden_states, topk_output):
        kernel_topk_output = StandardTopKOutput(
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids.clamp(min=0),
            router_logits=topk_output.router_logits,
        )
        with override_config(config):
            return fused_moe(
                hidden_states,
                w1,
                w2,
                kernel_topk_output,
                moe_runner_config=runner_config,
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=block_shape,
            )

    def kernel_func():
        with torch.no_grad():
            return [one_workload(hidden_states, topk_output) for hidden_states, topk_output in workloads]

    torch.cuda.synchronize(torch_device)
    return PreparedMoeCase(
        kernel_func=kernel_func,
        framework_version=get_version("sglang"),
        device_name=torch.cuda.get_device_name(torch_device),
        device=torch_device,
        model_artifact=model_path,
        kernel_source="sglang_fused_moe_triton",
        workload_generator="power_law_v3",
        seed=0,
        rank_simulation="single-gpu-ep4-rank0",
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        inter_size=inter_size,
        topk=topk,
        num_experts=num_experts,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        quant_mode=quant_mode,
        workload_distribution=workload_distribution,
        latency_divisor=len(workloads),
    )


def get_moe_test_cases() -> tuple[()]:
    """Lazy workers accept exact requests; they never enumerate a sweep."""

    return ()


def run_moe_case(
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    quant_mode: str,
    workload_distribution: str,
    *,
    protocol: MeasurementProtocol | None = None,
    device: str = "cuda:0",
    model_path: str = _MODEL_ARTIFACT,
) -> RawMeasurement:
    """Measure one exact FP8-block local-rank case without a perf-file write."""

    dimensions = (num_tokens, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("MoE dimensions must be positive integers")
    if (
        hidden_size != 4096
        or inter_size != 2048
        or topk != 6
        or num_experts != 256
        or moe_tp_size != 1
        or moe_ep_size != 4
        or quant_mode != "fp8_block"
        or workload_distribution != "power_law_1.01"
        or model_path != _MODEL_ARTIFACT
    ):
        raise ValueError("MoE case is outside the frozen DSv4 FP8-block TP1/EP4 capability envelope")
    protocol = protocol or MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=6,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-moe-v1",
    )
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "sglang-moe-v1"
        or protocol.statistic != "median"
        or protocol.samples < 3
    ):
        raise ValueError("MoE measurement protocol is incompatible with the exact runner")

    prepared = _prepare_moe_case(
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        quant_mode,
        workload_distribution,
        device,
        model_path,
    )
    expected = (
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        quant_mode,
        workload_distribution,
    )
    actual = (
        prepared.num_tokens,
        prepared.hidden_size,
        prepared.inter_size,
        prepared.topk,
        prepared.num_experts,
        prepared.moe_tp_size,
        prepared.moe_ep_size,
        prepared.quant_mode,
        prepared.workload_distribution,
    )
    if actual != expected or prepared.model_artifact != _MODEL_ARTIFACT:
        raise ValueError("prepared MoE case does not match the frozen request")

    with benchmark_with_power(
        device=prepared.device,
        kernel_func=prepared.kernel_func,
        num_warmups=protocol.warmups,
        num_runs=protocol.samples,
        repeat_n=1,
        allow_graph_fail=False,
        use_cuda_graph=True,
        return_samples=True,
    ) as results:
        if results.get("used_cuda_graph") is not True:
            raise RuntimeError("MoE exact runner requires CUDA Graph capture")
        divisor = int(getattr(prepared, "latency_divisor", 1))
        samples_ms = tuple(float(sample) / divisor for sample in results["samples_ms"])
        if len(samples_ms) != protocol.samples:
            raise ValueError("MoE benchmark sample count does not match the protocol")
        latency_ms = float(statistics.median(samples_ms))
        power_stats = results.get("power_stats")
        perf_row = {
            "framework": "SGLang",
            "version": prepared.framework_version,
            "device": prepared.device_name,
            "op_name": "moe",
            "kernel_source": prepared.kernel_source,
            "moe_dtype": prepared.quant_mode,
            "num_tokens": prepared.num_tokens,
            "hidden_size": prepared.hidden_size,
            "inter_size": prepared.inter_size,
            "topk": prepared.topk,
            "num_experts": prepared.num_experts,
            "moe_tp_size": prepared.moe_tp_size,
            "moe_ep_size": prepared.moe_ep_size,
            "distribution": prepared.workload_distribution,
            "latency": latency_ms,
        }
        return RawMeasurement(
            latency_ms=latency_ms,
            energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
            samples_ms=samples_ms,
            statistic=protocol.statistic,
            perf_row=perf_row,
            provenance={
                "framework": "SGLang",
                "framework_version": prepared.framework_version,
                "kernel_source": prepared.kernel_source,
                "device": prepared.device_name,
                "used_cuda_graph": True,
                "throttled": bool(results["throttled"]),
                "model_artifact": prepared.model_artifact,
                "workload_generator": prepared.workload_generator,
                "seed": prepared.seed,
                "rank_simulation": prepared.rank_simulation,
            },
            protocol_digest=protocol.digest,
            power_stats=power_stats,
        )


__all__ = ["PreparedMoeCase", "get_moe_test_cases", "run_moe_case"]
