# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact SGLang DeepSeek-V4 mHC module runner."""

from __future__ import annotations

import copy
import gc
import json
import logging
import os
import random
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

from aiconfigurator.collector.benchmark import benchmark_with_power
from aiconfigurator.collector.sglang.dsv4_runtime_contract import validate_deepseek_v4_runtime_contract
from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_ARCHITECTURE = "DeepseekV4ForCausalLM"
_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "model_configs"
_TEMPORARY_MODEL_DIRS: set[Path] = set()
logger = logging.getLogger(__name__)


def _noop_cleanup() -> None:
    return None


@dataclass(frozen=True, slots=True)
class PreparedMhcCase:
    """One initialized exact case retained by the UUID-bound worker."""

    kernel_func: Callable[[], Any]
    framework_version: str
    device_name: str
    device: Any
    architecture: str
    model_artifact: str
    num_sites: int
    hidden_size: int
    hc_mult: int
    sinkhorn_iters: int
    quant_mode: str
    cleanup: Callable[[], None] = _noop_cleanup


def _read_model_config(model_id: str) -> dict[str, Any]:
    config_file = _MODEL_CONFIG_DIR / f"{model_id.replace('/', '--')}_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: expected {config_file}")
    with config_file.open() as config_stream:
        value = json.load(config_stream)
    if not isinstance(value, dict):
        raise TypeError(f"AIC packaged config for model_id={model_id!r} must be a JSON object")
    return value


def _patched_model_dir(model_id: str) -> str:
    """Create the minimal dummy-load model directory required by SGLang."""

    original_config = _read_model_config(model_id)
    config = copy.deepcopy(original_config)
    config["num_hidden_layers"] = 2
    config["architectures"] = [_ARCHITECTURE]
    config["model_type"] = "deepseek_v3"

    temp_dir = Path(tempfile.mkdtemp(prefix=f"aic_mhc_{model_id.replace('/', '_')}_"))
    _TEMPORARY_MODEL_DIRS.add(temp_dir)
    with (temp_dir / "config.json").open("w") as config_stream:
        json.dump(config, config_stream)

    if "SGLANG_DSV4_FP4_EXPERTS" not in os.environ:
        expert_dtype = str(original_config.get("expert_dtype", "")).casefold()
        os.environ["SGLANG_DSV4_FP4_EXPERTS"] = "1" if expert_dtype == "fp4" else "0"
    return str(temp_dir)


def _cleanup_temporary_model_dirs() -> None:
    """Remove every worker-local dummy-load config created for an exact case."""

    cleanup_errors: list[OSError] = []
    for model_dir in tuple(_TEMPORARY_MODEL_DIRS):
        try:
            shutil.rmtree(model_dir)
        except FileNotFoundError:
            _TEMPORARY_MODEL_DIRS.discard(model_dir)
        except OSError as error:
            cleanup_errors.append(error)
        else:
            _TEMPORARY_MODEL_DIRS.discard(model_dir)
    if cleanup_errors:
        for secondary in cleanup_errors[1:]:
            logger.error(
                "secondary mHC temporary-directory cleanup failure; preserving the first error: %s",
                secondary,
            )
        raise cleanup_errors[0]


def _cleanup_mhc_runtime(
    model_runner,
    *,
    torch_module,
    cleanup_distributed: Callable[[], None],
    collect_garbage: Callable[[], Any] = gc.collect,
) -> None:
    """Release one mHC runner fully so the bound worker can serve another case."""

    cleanup_errors: list[Exception] = []
    if model_runner is not None:
        for pool_name in ("req_to_token_pool", "token_to_kv_pool_allocator"):
            try:
                pool = getattr(model_runner, pool_name)
                pool.clear()
            except Exception as error:
                cleanup_errors.append(error)
        for attribute in ("model", "req_to_token_pool", "token_to_kv_pool_allocator"):
            try:
                setattr(model_runner, attribute, None)
            except Exception as error:
                cleanup_errors.append(error)

    for cleanup in (
        cleanup_distributed,
        collect_garbage,
        torch_module.cuda.empty_cache,
        _cleanup_temporary_model_dirs,
    ):
        try:
            cleanup()
        except Exception as error:
            cleanup_errors.append(error)
    if torch_module.distributed.is_initialized():
        cleanup_errors.append(RuntimeError("SGLang distributed state remained initialized after mHC cleanup"))
    if cleanup_errors:
        for secondary in cleanup_errors[1:]:
            logger.error(
                "secondary mHC cleanup failure; preserving the first cleanup error: %s",
                secondary,
            )
        raise cleanup_errors[0]


def _preserve_primary_failure(cleanup: Callable[[], None], *, context: str) -> None:
    try:
        cleanup()
    except Exception:
        logger.exception("%s; preserving the primary mHC failure", context)


def _load_one_layer_runner(
    model_path: str,
    device: str,
    mem_fraction_static: float,
    *,
    torch_module,
):
    """Port the existing collector's dummy-load one-layer SGLang setup."""

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    validate_deepseek_v4_runtime_contract()
    suppress_other_loggers()
    torch_device = torch_module.device(device)
    torch_module.cuda.set_device(torch_device)
    local_model_path = _patched_model_dir(model_path)
    gpu_id = torch_device.index if torch_device.index is not None else torch_module.cuda.current_device()
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype="fp8_e4m3",
        max_total_tokens=4096,
        max_running_requests=16,
        max_prefill_tokens=4096,
    )
    server_args.disable_piecewise_cuda_graph = True
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "compressed"
    server_args.page_size = 256
    _set_envs_and_config(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    return ModelRunner(
        model_config=model_config,
        mem_fraction_static=mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        nccl_port=29500 + random.randint(0, 10000),
        server_args=server_args,
    )


def _hidden_size(layer) -> int:
    return int(layer.config.hidden_size)


def _make_residual(layer, num_tokens: int, device: str, *, torch_module):
    generator = torch_module.Generator(device=device)
    generator.manual_seed(0)
    return torch_module.randn(
        num_tokens,
        layer.hc_mult,
        _hidden_size(layer),
        dtype=torch_module.bfloat16,
        device=device,
        generator=generator,
    )


def _mhc_call_args(layer) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]]:
    """Return the attention-site and FFN-site arguments used by a real layer."""

    return (
        (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
        (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
    )


def _hc_pre_post_inputs(hc_pre_output):
    if len(hc_pre_output) == 3:
        return hc_pre_output
    if len(hc_pre_output) == 4:
        x, post, comb, _norm_fused = hc_pre_output
        return x, post, comb
    raise ValueError(f"unexpected hc_pre output arity: {len(hc_pre_output)}")


def _make_kernel(layer, op: str, residual, *, torch_module):
    """Port the legacy collector's two-site pre/post kernel construction."""

    if op == "pre":
        call_args = _mhc_call_args(layer)

        def kernel():
            return [layer.hc_pre(residual, *args) for args in call_args]

        return kernel

    if op == "post":
        with torch_module.no_grad():
            post_inputs = [_hc_pre_post_inputs(layer.hc_pre(residual, *args)) for args in _mhc_call_args(layer)]
        torch_module.cuda.synchronize()

        def kernel():
            return [layer.hc_post(x, residual, post, comb) for x, post, comb, *_ in post_inputs]

        return kernel

    raise ValueError(f"unsupported mHC op: {op}")


def _prepare_mhc_case(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    quant_mode: str,
    device: str,
    model_path: str,
) -> PreparedMhcCase:
    """Import Torch/SGLang only after the worker has bound its GPU UUID."""

    os.environ.setdefault("SGLANG_APPLY_CONFIG_BACKUP", "none")
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "0")

    import torch
    from sglang.srt.distributed import parallel_state

    model_runner = None
    cleaned = False
    kernel_state: dict[str, Callable[[], Any]] = {}

    def cleanup_distributed() -> None:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()

    def cleanup() -> None:
        nonlocal cleaned, model_runner
        if cleaned:
            return
        cleaned = True
        kernel_state.clear()
        try:
            _cleanup_mhc_runtime(
                model_runner,
                torch_module=torch,
                cleanup_distributed=cleanup_distributed,
            )
        finally:
            model_runner = None

    try:
        model_runner = _load_one_layer_runner(
            model_path,
            device,
            mem_fraction_static=0.5,
            torch_module=torch,
        )
        layer = model_runner.model.model.layers[0]
        actual_hidden_size = _hidden_size(layer)
        actual_hc_mult = int(layer.hc_mult)
        actual_sinkhorn_iters = int(getattr(layer.config, "hc_sinkhorn_iters", 20))
        architecture_values = getattr(layer.config, "architectures", None)
        architecture = architecture_values[0] if architecture_values else _ARCHITECTURE
        if (
            actual_hidden_size != hidden_size
            or actual_hc_mult != hc_mult
            or actual_sinkhorn_iters != sinkhorn_iters
            or architecture != _ARCHITECTURE
        ):
            raise ValueError("loaded SGLang mHC layer does not match the requested frozen case")

        residual = _make_residual(layer, num_tokens, device, torch_module=torch)
        kernel_state["raw_kernel"] = _make_kernel(layer, op, residual, torch_module=torch)

        def timed_kernel():
            with torch.no_grad():
                return kernel_state["raw_kernel"]()

        call_args = _mhc_call_args(layer)
        if len(call_args) != 2:
            raise ValueError("loaded SGLang mHC layer does not expose both full-module sites")
        return PreparedMhcCase(
            kernel_func=timed_kernel,
            framework_version=get_version("sglang"),
            device_name=torch.cuda.get_device_name(torch.device(device)),
            device=torch.device(device),
            architecture=architecture,
            model_artifact=model_path,
            num_sites=len(call_args),
            hidden_size=actual_hidden_size,
            hc_mult=actual_hc_mult,
            sinkhorn_iters=actual_sinkhorn_iters,
            quant_mode=quant_mode,
            cleanup=cleanup,
        )
    except BaseException:
        _preserve_primary_failure(cleanup, context="mHC preparation cleanup failed")
        raise


def get_mhc_test_cases() -> tuple[()]:
    """The lazy registry is exact-only and never expands an offline grid."""

    return ()


def run_mhc_case(
    op: str,
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    sinkhorn_iters: int,
    quant_mode: str,
    *,
    protocol: MeasurementProtocol | None = None,
    device: str = "cuda:0",
    model_path: str = _MODEL_ARTIFACT,
) -> RawMeasurement:
    """Measure one exact BF16 two-site mHC case without a perf-file write."""

    dimensions = (num_tokens, hidden_size, hc_mult, sinkhorn_iters)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("mHC num_tokens, hidden_size, hc_mult, and sinkhorn_iters must be positive integers")
    if (
        op not in {"pre", "post"}
        or hidden_size != 4096
        or hc_mult != 4
        or sinkhorn_iters != 20
        or quant_mode != "bfloat16"
        or model_path != _MODEL_ARTIFACT
    ):
        raise ValueError("mHC case is outside the frozen DSv4 BF16 full-module capability envelope")

    protocol = protocol or MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=6,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-mhc-v1",
    )
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "sglang-mhc-v1"
        or protocol.statistic != "median"
        or protocol.samples < 3
    ):
        raise ValueError("mHC measurement protocol is incompatible with the exact runner")

    prepared = _prepare_mhc_case(
        op,
        num_tokens,
        hidden_size,
        hc_mult,
        sinkhorn_iters,
        quant_mode,
        device,
        model_path,
    )
    try:
        if (
            prepared.architecture != _ARCHITECTURE
            or prepared.model_artifact != _MODEL_ARTIFACT
            or prepared.num_sites != 2
            or prepared.hidden_size != hidden_size
            or prepared.hc_mult != hc_mult
            or prepared.sinkhorn_iters != sinkhorn_iters
            or prepared.quant_mode != quant_mode
        ):
            raise ValueError("prepared mHC case does not match the frozen two-site full-module request")

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
                raise RuntimeError("mHC exact runner requires CUDA Graph capture")
            latency_ms = float(results["latency_ms"])
            samples_ms = tuple(float(sample) for sample in results["samples_ms"])
            if len(samples_ms) != protocol.samples:
                raise ValueError("mHC benchmark sample count does not match the protocol")
            power_stats = results.get("power_stats")
            perf_row = {
                "architecture": prepared.architecture,
                "op_name": op,
                "num_tokens": num_tokens,
                "num_sites": prepared.num_sites,
                "hc_mult": prepared.hc_mult,
                "hidden_size": prepared.hidden_size,
                "sinkhorn_iters": prepared.sinkhorn_iters,
                "quant_mode": prepared.quant_mode,
                "latency": latency_ms,
            }
            measurement = RawMeasurement(
                latency_ms=latency_ms,
                energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
                samples_ms=samples_ms,
                statistic=protocol.statistic,
                perf_row=perf_row,
                provenance={
                    "framework": "SGLang",
                    "framework_version": prepared.framework_version,
                    "kernel_source": "sglang_mhc",
                    "device": prepared.device_name,
                    "used_cuda_graph": True,
                    "throttled": bool(results["throttled"]),
                    "model_artifact": prepared.model_artifact,
                    "full_module": True,
                    "num_sites": prepared.num_sites,
                    "tensor_generator": "normal-v1",
                    "seed": 0,
                },
                protocol_digest=protocol.digest,
                power_stats=power_stats,
            )
    except BaseException:
        _preserve_primary_failure(prepared.cleanup, context="mHC execution cleanup failed")
        raise

    prepared.cleanup()
    return measurement


__all__ = ["PreparedMhcCase", "get_mhc_test_cases", "run_mhc_case"]
