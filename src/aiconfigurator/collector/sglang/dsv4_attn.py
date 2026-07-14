# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-light exact SGLang DeepSeek-V4 attention module runner."""

from __future__ import annotations

import contextlib
import copy
import errno
import gc
import inspect
import json
import logging
import math
import os
import shutil
import socket
import statistics
import tempfile
from collections.abc import Callable, Iterator
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
_ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}
_CANONICAL_NUM_HEADS = 16
_PADDED_NUM_HEADS = 64
_TP_SIZE = 4
_SUPPORTED_SGLANG_VERSIONS = frozenset({"0.5.10", "0.5.10rc0"})
_PROPER_INIT_STD = 0.05
_PROPER_INIT_SEED = 1234
_INPUT_SEED = 0
_TEMPORARY_MODEL_DIRS: set[Path] = set()

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedDsv4AttentionCase:
    """One initialized full-module attention case retained by its worker."""

    kernel_func: Callable[[], Any]
    framework_version: str
    device_name: str
    device: Any
    architecture: str
    model_artifact: str
    mode: str
    attn_kind: str
    compress_ratio: int
    tp_size: int
    canonical_num_heads: int
    padded_num_heads: int
    mla_dtype: str
    kv_cache_dtype: str
    gemm_type: str
    model_weight_generator: str = "proper-normal-v1"
    model_weight_std: float = _PROPER_INIT_STD
    model_weight_seed: int = _PROPER_INIT_SEED
    cleanup_func: Callable[[], None] | None = None


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"DSv4 attention {field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"DSv4 attention {field} must be a non-negative integer")
    return value


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"DSv4 attention {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"DSv4 attention {field} must be positive and finite")
    return number


def _read_model_config(model_id: str) -> dict[str, Any]:
    config_file = _MODEL_CONFIG_DIR / f"{model_id.replace('/', '--')}_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"AIC packaged config not found for model_id={model_id!r}: expected {config_file}")
    with config_file.open() as config_stream:
        value = json.load(config_stream)
    if not isinstance(value, dict):
        raise TypeError(f"AIC packaged config for model_id={model_id!r} must be a JSON object")
    return value


def _patched_model_dir(model_id: str, attn_kind: str, compress_ratio: int) -> str:
    """Create the minimal one-layer dummy-load config required by SGLang."""

    original_config = _read_model_config(model_id)
    config = copy.deepcopy(original_config)
    config.pop("auto_map", None)
    config["num_hidden_layers"] = 1
    config["num_key_value_heads"] = 1
    config["architectures"] = [_ARCHITECTURE]
    config["model_type"] = "deepseek_v3"
    config["compress_ratios"] = [compress_ratio]
    config["n_routed_experts"] = min(int(config.get("n_routed_experts", 8)), 8)
    config["num_experts_per_tok"] = min(int(config.get("num_experts_per_tok", 2)), 2)

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"aic_dsv4_attn_{attn_kind}_{os.getpid()}_",
            dir=tempfile.gettempdir(),
        )
    )
    _TEMPORARY_MODEL_DIRS.add(temp_dir)
    with (temp_dir / "config.json").open("w") as config_stream:
        json.dump(config, config_stream)

    expert_dtype = str(original_config.get("expert_dtype", "")).casefold()
    os.environ["SGLANG_DSV4_FP4_EXPERTS"] = "1" if expert_dtype == "fp4" else "0"
    return str(temp_dir)


@contextlib.contextmanager
def _forced_proper_init() -> Iterator[None]:
    """Force representative deterministic dummy weights for C4 top-k."""

    overrides = {
        "AIC_DSV4_PROPER_INIT": "1",
        "AIC_DSV4_PROPER_INIT_STD": str(_PROPER_INIT_STD),
    }
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def _tp_load_model_patch(tp_size: int) -> Iterator[None]:
    """Allocate TP-sharded projections while keeping one real process group."""

    import sglang.srt.distributed.parallel_state as parallel_state
    import sglang.srt.layers.dp_attention as dp_attention
    from sglang.srt.model_executor.model_runner import ModelRunner

    original_load = ModelRunner.load_model

    def patched_load(model_runner):
        tp_group = parallel_state._TP
        if tp_group is None:
            raise RuntimeError("SGLang tensor-parallel group was not initialized before model load")
        original_world_size = tp_group.world_size
        original_rank = tp_group.rank_in_group
        tp_group.world_size = tp_size
        tp_group.rank_in_group = 0
        dp_attention._ATTN_TP_SIZE = tp_size
        dp_attention._ATTN_TP_RANK = 0
        try:
            return original_load(model_runner)
        finally:
            tp_group.world_size = original_world_size
            tp_group.rank_in_group = original_rank

    ModelRunner.load_model = patched_load
    try:
        yield
    finally:
        ModelRunner.load_model = original_load


@contextlib.contextmanager
def _attention_only_dsv4_moe(deepseek_v2_module: Any) -> Iterator[None]:
    """Adapt eager DSv4 model construction without enabling an MoE measurement."""

    original_moe_type = deepseek_v2_module.DeepseekV2MoE
    supports_dsv4_flag = "is_deepseek_v4" in inspect.signature(original_moe_type.__init__).parameters

    class AttentionOnlyDeepseekV2MoE(original_moe_type):
        def __init__(
            self,
            *args,
            is_deepseek_v4: bool = False,
            **kwargs,
        ) -> None:
            if is_deepseek_v4 is not True:
                raise ValueError("attention-only DSv4 MoE compatibility requires is_deepseek_v4=True")
            if supports_dsv4_flag:
                kwargs["is_deepseek_v4"] = True
            super().__init__(*args, **kwargs)

        def forward(self, *args, **kwargs):
            raise RuntimeError("attention-only DSv4 setup must not execute the MoE path")

    deepseek_v2_module.DeepseekV2MoE = AttentionOnlyDeepseekV2MoE
    try:
        yield
    finally:
        deepseek_v2_module.DeepseekV2MoE = original_moe_type


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        return int(port_socket.getsockname()[1])


def _is_address_in_use(error: OSError | RuntimeError) -> bool:
    if isinstance(error, OSError) and error.errno == errno.EADDRINUSE:
        return True
    message = str(error).casefold()
    return "address already in use" in message or "eaddrinuse" in message


def _construct_model_runner(
    model_runner_factory: Callable[..., Any],
    model_runner_kwargs: dict[str, Any],
    *,
    cleanup_failed_attempt: Callable[[], None],
    port_factory: Callable[[], int] = _free_tcp_port,
    max_attempts: int = 3,
):
    """Construct a runner, retrying the unavoidable free-port bind race."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        try:
            return model_runner_factory(nccl_port=port_factory(), **model_runner_kwargs)
        except (OSError, RuntimeError) as error:
            if not _is_address_in_use(error) or attempt + 1 == max_attempts:
                raise
            cleanup_failed_attempt()
    raise AssertionError("unreachable")


def _proper_initialize_model_runner(model_runner, *, torch_module) -> None:
    if os.environ.get("AIC_DSV4_PROPER_INIT") != "1":
        raise RuntimeError("representative DSv4 proper initialization was not enabled")
    std = float(os.environ["AIC_DSV4_PROPER_INIT_STD"])
    if std != _PROPER_INIT_STD:
        raise ValueError("DSv4 proper initialization standard deviation must be 0.05")

    torch_module.manual_seed(_PROPER_INIT_SEED)
    with torch_module.no_grad():
        for name, parameter in model_runner.model.state_dict().items():
            if not torch_module.is_floating_point(parameter):
                continue
            normalized_name = name.casefold()
            if "scale" in normalized_name or "norm" in normalized_name:
                parameter.fill_(1.0)
            elif torch_module.finfo(parameter.dtype).bits < 16:
                sample = torch_module.empty_like(parameter, dtype=torch_module.float16).normal_(0.0, std)
                parameter.copy_(sample.to(parameter.dtype))
            else:
                parameter.normal_(0.0, std)


def _full_tokens_for_swa_capacity(
    *,
    logical_tokens: int,
    page_size: int,
    swa_full_tokens_ratio: float,
) -> int:
    """Return the page-aligned full pool needed for one exact DSv4 request."""

    if isinstance(logical_tokens, bool) or not isinstance(logical_tokens, int) or logical_tokens <= 0:
        raise ValueError("logical token count must be a positive integer")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size <= 0:
        raise ValueError("page size must be a positive integer")
    if (
        isinstance(swa_full_tokens_ratio, bool)
        or not isinstance(swa_full_tokens_ratio, (int, float))
        or not math.isfinite(float(swa_full_tokens_ratio))
        or not 0.0 < float(swa_full_tokens_ratio) <= 1.0
    ):
        raise ValueError("SWA/full token ratio must be finite and in (0, 1]")

    required_swa_pages = math.ceil(logical_tokens / page_size)
    required_full_pages = math.ceil(required_swa_pages / float(swa_full_tokens_ratio))
    return required_full_pages * page_size


def _page_rounded_swa_capacity_tokens(
    *,
    batch_size: int,
    tokens_per_request: int,
    page_size: int,
) -> int:
    """Return SWA capacity after page-rounding every request independently."""

    for field_name, value in (
        ("batch size", batch_size),
        ("tokens per request", tokens_per_request),
        ("page size", page_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    return batch_size * math.ceil(tokens_per_request / page_size) * page_size


def _load_model_runner(
    model_path: str,
    *,
    attn_kind: str,
    compress_ratio: int,
    kv_cache_dtype: str,
    gemm_type: str,
    tp_size: int,
    batch_size: int,
    max_total_tokens: int,
    required_swa_tokens: int,
    device: str,
    torch_module,
    cleanup_failed_attempt: Callable[[], None],
):
    """Port the legacy one-layer load and TP4 single-GPU simulation."""

    os.environ["SGLANG_APPLY_CONFIG_BACKUP"] = "none"
    os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.models import deepseek_v2
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import suppress_other_loggers

    validate_deepseek_v4_runtime_contract()
    suppress_other_loggers()

    torch_device = torch_module.device(device)
    torch_module.cuda.set_device(torch_device)
    local_model_path = _patched_model_dir(model_path, attn_kind, compress_ratio)
    gpu_id = torch_device.index if torch_device.index is not None else torch_module.cuda.current_device()
    server_args = ServerArgs(
        model_path=local_model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        kv_cache_dtype={"bfloat16": "bfloat16", "fp8": "fp8_e4m3"}[kv_cache_dtype],
        max_running_requests=max(16, batch_size + 1),
        max_total_tokens=max_total_tokens,
    )
    server_args.quantization = "fp8" if gemm_type == "fp8_block" else None
    server_args.disable_piecewise_cuda_graph = True
    server_args.enable_piecewise_cuda_graph = False
    server_args.attention_backend = "compressed"
    server_args.page_size = 256
    _set_envs_and_config(server_args)
    server_args.max_total_tokens = max(
        max_total_tokens,
        _full_tokens_for_swa_capacity(
            logical_tokens=required_swa_tokens,
            page_size=server_args.page_size,
            swa_full_tokens_ratio=server_args.swa_full_tokens_ratio,
        ),
    )
    model_config = ModelConfig.from_server_args(server_args)
    with (
        _tp_load_model_patch(tp_size),
        _attention_only_dsv4_moe(deepseek_v2),
    ):
        model_runner = _construct_model_runner(
            ModelRunner,
            {
                "model_config": model_config,
                "mem_fraction_static": server_args.mem_fraction_static,
                "gpu_id": gpu_id,
                "tp_rank": 0,
                "tp_size": 1,
                "pp_rank": 0,
                "pp_size": 1,
                "moe_ep_rank": 0,
                "moe_ep_size": 1,
                "server_args": server_args,
            },
            cleanup_failed_attempt=cleanup_failed_attempt,
        )
    _proper_initialize_model_runner(model_runner, torch_module=torch_module)
    return model_runner


def _runtime_chunk_size(model_runner) -> int:
    chunk_size = getattr(model_runner.server_args, "chunked_prefill_size", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise RuntimeError("SGLang did not initialize a positive chunked_prefill_size")
    return chunk_size


def _chunked_alloc_extend(original_alloc_extend, chunk_size: int, *, torch_module):
    def wrapped(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu, last_loc, extend_num_tokens):
        batch_size = int(prefix_lens.shape[0])
        if extend_num_tokens <= chunk_size or batch_size == 0:
            return original_alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
            )

        extend_per_request = (seq_lens_cpu - prefix_lens_cpu).tolist()
        chunk_per_request = max(1, chunk_size // batch_size)
        current_prefix = prefix_lens.clone()
        current_prefix_cpu = prefix_lens_cpu.clone()
        current_last_location = last_loc.clone()
        advanced = [0] * batch_size
        per_request_indices: list[list[Any]] = [[] for _ in range(batch_size)]
        while True:
            chunk_extends = [
                min(chunk_per_request, extend_per_request[index] - advanced[index]) for index in range(batch_size)
            ]
            chunk_total = sum(chunk_extends)
            if chunk_total == 0:
                break
            chunk_tensor = torch_module.tensor(chunk_extends, dtype=current_prefix_cpu.dtype)
            new_sequence_cpu = current_prefix_cpu + chunk_tensor
            new_sequence = new_sequence_cpu.to(seq_lens.device)
            indices = original_alloc_extend(
                current_prefix,
                current_prefix_cpu,
                new_sequence,
                new_sequence_cpu,
                current_last_location,
                chunk_total,
            )
            if indices is None:
                return None
            offset = 0
            for index, count in enumerate(chunk_extends):
                if count:
                    request_chunk = indices[offset : offset + count]
                    per_request_indices[index].append(request_chunk)
                    current_last_location[index] = request_chunk[-1]
                    advanced[index] += count
                    offset += count
            current_prefix = new_sequence
            current_prefix_cpu = new_sequence_cpu
        flattened = [torch_module.cat(chunks) for chunks in per_request_indices if chunks]
        if not flattened:
            return torch_module.empty((0,), dtype=torch_module.int64, device=prefix_lens.device)
        return torch_module.cat(flattened)

    return wrapped


@contextlib.contextmanager
def _temporarily_chunked_alloc_extend(model_runner, extend_num_tokens: int, *, torch_module) -> Iterator[None]:
    allocator = model_runner.token_to_kv_pool_allocator
    chunk_size = _runtime_chunk_size(model_runner)
    original_alloc_extend = None
    if extend_num_tokens > chunk_size:
        original_alloc_extend = allocator.alloc_extend
        allocator.alloc_extend = _chunked_alloc_extend(
            original_alloc_extend,
            chunk_size,
            torch_module=torch_module,
        )
    try:
        yield
    finally:
        if original_alloc_extend is not None:
            allocator.alloc_extend = original_alloc_extend


def _alloc_prefix_indices(model_runner, batch_size: int, prefix_len: int, *, torch_module) -> list[Any]:
    device = getattr(model_runner, "device", "cuda")
    if prefix_len == 0:
        return [torch_module.empty((0,), dtype=torch_module.int64, device=device) for _ in range(batch_size)]

    allocator = model_runner.token_to_kv_pool_allocator
    alloc_extend = allocator.alloc_extend
    chunk_size = _runtime_chunk_size(model_runner)
    if batch_size * prefix_len > chunk_size:
        alloc_extend = _chunked_alloc_extend(alloc_extend, chunk_size, torch_module=torch_module)
    prefix_lens_cpu = torch_module.zeros(batch_size, dtype=torch_module.int64)
    sequence_lens_cpu = torch_module.full((batch_size,), prefix_len, dtype=torch_module.int64)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    sequence_lens = sequence_lens_cpu.to(device, non_blocking=True)
    last_location = torch_module.full((batch_size,), -1, dtype=torch_module.int64, device=device)
    flattened = alloc_extend(
        prefix_lens,
        prefix_lens_cpu,
        sequence_lens,
        sequence_lens_cpu,
        last_location,
        batch_size * prefix_len,
    )
    if flattened is None:
        raise RuntimeError("SGLang failed to allocate the requested DSv4 attention prefix cache")
    return [flattened[index * prefix_len : (index + 1) * prefix_len].contiguous() for index in range(batch_size)]


def _make_requests(
    batch_size: int,
    sequence_length: int,
    *,
    decode: bool,
    prefix_len: int,
    prefix_indices: list[Any],
    torch_module,
):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    if decode and prefix_len:
        raise ValueError("prefix_len is only valid for DSv4 context attention")
    token_generator = torch_module.Generator(device="cpu")
    token_generator.manual_seed(_INPUT_SEED)
    full_length = prefix_len + sequence_length
    requests = []
    for index in range(batch_size):
        token_ids = torch_module.randint(0, 10000, (full_length,), generator=token_generator).tolist()
        request = Req(
            rid=str(index),
            origin_input_text="",
            origin_input_ids=token_ids,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        request.prefix_indices = prefix_indices[index]
        request.fill_ids = request.origin_input_ids
        request.extend_input_len = sequence_length if prefix_len else len(request.fill_ids)
        request.logprob_start_len = 0
        if decode:
            request.cached_tokens = 0
            request.already_computed = 0
        requests.append(request)
    return requests


def _build_forward_batch(
    model_runner,
    batch_size: int,
    sequence_length: int,
    *,
    is_prefill: bool,
    prefix_len: int,
    torch_module,
):
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    prefix_indices = _alloc_prefix_indices(
        model_runner,
        batch_size,
        prefix_len,
        torch_module=torch_module,
    )
    requests = _make_requests(
        batch_size,
        sequence_length,
        decode=not is_prefill,
        prefix_len=prefix_len,
        prefix_indices=prefix_indices,
        torch_module=torch_module,
    )
    cache_params = CacheInitParams(
        disable=True,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        page_size=model_runner.token_to_kv_pool_allocator.page_size,
    )
    batch = ScheduleBatch.init_new(
        reqs=requests,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=ChunkCache(cache_params),
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    with _temporarily_chunked_alloc_extend(
        model_runner,
        batch_size * sequence_length,
        torch_module=torch_module,
    ):
        batch.prepare_for_extend()
        if not is_prefill:
            output_generator = torch_module.Generator(device=model_runner.device)
            output_generator.manual_seed(_INPUT_SEED)
            batch.output_ids = torch_module.randint(
                0,
                10000,
                (batch_size,),
                dtype=torch_module.int64,
                device=model_runner.device,
                generator=output_generator,
            )
            batch.prepare_for_decode()
    worker_batch = batch.get_model_worker_batch() if hasattr(batch, "get_model_worker_batch") else batch
    forward_batch = ForwardBatch.init_new(worker_batch, model_runner)
    model_runner.attn_backend.init_forward_metadata(forward_batch)
    return forward_batch


def _make_inputs(
    model_runner,
    batch_size: int,
    sequence_length: int,
    *,
    is_prefill: bool,
    prefix_len: int,
    device: str,
    torch_module,
):
    hidden_size = int(model_runner.model.config.hidden_size)
    if is_prefill:
        token_count = batch_size * sequence_length
        positions = (
            torch_module.arange(prefix_len, prefix_len + sequence_length, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
            .flatten()
        )
    else:
        token_count = batch_size
        positions = torch_module.full(
            (batch_size,),
            sequence_length,
            dtype=torch_module.int64,
            device=device,
        )
    generator = torch_module.Generator(device=device)
    generator.manual_seed(_INPUT_SEED)
    hidden_states = torch_module.randn(
        token_count,
        hidden_size,
        dtype=torch_module.bfloat16,
        device=device,
        generator=generator,
    )
    return hidden_states, positions


def _cleanup_distributed_runtime(
    *,
    torch_module,
    cleanup_distributed: Callable[[], None],
    collect_garbage: Callable[[], Any] = gc.collect,
) -> None:
    """Release SGLang/Torch distributed state, including failed construction."""

    cleanup_errors: list[Exception] = []
    for cleanup in (
        cleanup_distributed,
        torch_module.cuda.empty_cache,
        collect_garbage,
        _cleanup_temporary_model_dirs,
    ):
        try:
            cleanup()
        except Exception as error:
            cleanup_errors.append(error)
    if torch_module.distributed.is_initialized():
        cleanup_errors.append(
            RuntimeError("SGLang distributed state remained initialized after DSv4 attention cleanup")
        )
    if cleanup_errors:
        for secondary in cleanup_errors[1:]:
            logger.error(
                "secondary DSv4 attention cleanup failure; preserving the first cleanup error: %s",
                secondary,
            )
        raise cleanup_errors[0]


def _cleanup_temporary_model_dirs() -> None:
    """Remove worker-local patched model configs after each exact case."""

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
                "secondary DSv4 attention temporary-directory cleanup failure; preserving the first error: %s",
                secondary,
            )
        raise cleanup_errors[0]


def _preserve_primary_failure(cleanup: Callable[[], None], *, context: str) -> None:
    """Attempt teardown without replacing the exception that triggered it."""

    try:
        cleanup()
    except Exception:
        logger.exception("%s; preserving the primary DSv4 attention failure", context)


def _cleanup_model_runner(
    model_runner,
    *,
    torch_module,
    cleanup_distributed: Callable[[], None],
    collect_garbage: Callable[[], Any] = gc.collect,
) -> None:
    """Release one exact runner completely so its worker can serve another case."""

    pool_errors: list[Exception] = []
    for pool_name in ("req_to_token_pool", "token_to_kv_pool_allocator"):
        try:
            pool = getattr(model_runner, pool_name)
            pool.clear()
        except Exception as error:
            pool_errors.append(error)
    cleanup_errors = list(pool_errors)
    try:
        _cleanup_distributed_runtime(
            torch_module=torch_module,
            cleanup_distributed=cleanup_distributed,
            collect_garbage=collect_garbage,
        )
    except Exception as error:
        cleanup_errors.append(error)
    if pool_errors:
        for secondary in cleanup_errors[1:]:
            logger.error(
                "secondary DSv4 attention runner cleanup failure; preserving the first model-pool error: %s",
                secondary,
            )
        raise RuntimeError("DSv4 attention model-pool cleanup failed") from cleanup_errors[0]
    if cleanup_errors:
        raise cleanup_errors[0]


def _prepare_dsv4_attn_case(
    *,
    mode: str,
    attn_kind: str,
    tp_size: int,
    canonical_num_heads: int,
    num_heads: int,
    compress_ratio: int,
    batch_size: int,
    mla_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    isl: int | None,
    prefix: int | None,
    s_total: int | None,
    device: str,
    model_path: str,
) -> PreparedDsv4AttentionCase:
    """Import Torch/SGLang after GPU binding and prepare exactly one case."""

    import torch
    from sglang.srt.distributed import parallel_state

    def cleanup_distributed() -> None:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()

    def cleanup_failed_attempt() -> None:
        _cleanup_distributed_runtime(
            torch_module=torch,
            cleanup_distributed=cleanup_distributed,
        )

    is_prefill = mode == "context"
    sequence_length = int(isl) if is_prefill else int(s_total) - 1
    prefix_length = int(prefix or 0) if is_prefill else 0
    tokens_per_request = sequence_length + prefix_length + (0 if is_prefill else 1)
    total_tokens = batch_size * tokens_per_request
    max_total_tokens = max(4096, math.ceil(total_tokens * 1.05))
    required_swa_tokens = _page_rounded_swa_capacity_tokens(
        batch_size=batch_size,
        tokens_per_request=tokens_per_request,
        page_size=256,
    )
    try:
        with _forced_proper_init():
            model_runner = _load_model_runner(
                model_path,
                attn_kind=attn_kind,
                compress_ratio=compress_ratio,
                kv_cache_dtype=kv_cache_dtype,
                gemm_type=gemm_type,
                tp_size=tp_size,
                batch_size=batch_size,
                max_total_tokens=max_total_tokens,
                required_swa_tokens=required_swa_tokens,
                device=device,
                torch_module=torch,
                cleanup_failed_attempt=cleanup_failed_attempt,
            )
    except BaseException:
        _preserve_primary_failure(
            lambda: _cleanup_distributed_runtime(
                torch_module=torch,
                cleanup_distributed=cleanup_distributed,
            ),
            context="distributed cleanup after model construction failed",
        )
        raise

    try:
        attention_module = model_runner.model.model.layers[0].self_attn
        actual_ratio = int(getattr(attention_module, "compress_ratio", -1))
        padded_num_heads = int(getattr(attention_module, "n_heads", -1))
        architecture_values = getattr(model_runner.model.config, "architectures", None)
        architecture = architecture_values[0] if architecture_values else None
        if (
            actual_ratio != compress_ratio
            or padded_num_heads != num_heads
            or padded_num_heads // tp_size != canonical_num_heads
            or architecture != _ARCHITECTURE
        ):
            raise ValueError("loaded SGLang attention module does not match the frozen padded64/TP4 case")

        forward_batch = _build_forward_batch(
            model_runner,
            batch_size,
            sequence_length,
            is_prefill=is_prefill,
            prefix_len=prefix_length,
            torch_module=torch,
        )
        hidden_states, positions = _make_inputs(
            model_runner,
            batch_size,
            sequence_length,
            is_prefill=is_prefill,
            prefix_len=prefix_length,
            device=device,
            torch_module=torch,
        )
        framework_version = get_version("sglang")

        def kernel_func():
            with torch.no_grad():
                if framework_version.startswith("0.5.13"):
                    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

                    forward_scope = forward_context(ForwardContext(attn_backend=model_runner.attn_backend))
                else:
                    forward_scope = contextlib.nullcontext()
                with forward_scope:
                    return attention_module(
                        x=hidden_states,
                        positions=positions,
                        forward_batch=forward_batch,
                    )

        def cleanup_func() -> None:
            _cleanup_model_runner(
                model_runner,
                torch_module=torch,
                cleanup_distributed=cleanup_distributed,
            )

        prepared = PreparedDsv4AttentionCase(
            kernel_func=kernel_func,
            framework_version=framework_version,
            device_name=torch.cuda.get_device_name(torch.device(device)),
            device=torch.device(device),
            architecture=architecture,
            model_artifact=model_path,
            mode=mode,
            attn_kind=attn_kind,
            compress_ratio=actual_ratio,
            tp_size=tp_size,
            canonical_num_heads=canonical_num_heads,
            padded_num_heads=padded_num_heads,
            mla_dtype=mla_dtype,
            kv_cache_dtype=kv_cache_dtype,
            gemm_type=gemm_type,
            cleanup_func=cleanup_func,
        )
    except BaseException:
        _preserve_primary_failure(
            lambda: _cleanup_model_runner(
                model_runner,
                torch_module=torch,
                cleanup_distributed=cleanup_distributed,
            ),
            context="model-runner cleanup after attention preparation failed",
        )
        raise
    return prepared


def get_dsv4_attn_test_cases() -> tuple[()]:
    """The packaged lazy runner is exact-only and never expands a sweep."""

    return ()


def _validate_case(
    *,
    mode: str,
    attn_kind: str,
    tp_size: int,
    canonical_num_heads: int,
    num_heads: int,
    compress_ratio: int,
    batch_size: int,
    mla_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    isl: int | None,
    prefix: int | None,
    s_total: int | None,
    model_path: str,
) -> None:
    expected_ratio = _ATTN_KIND_TO_COMPRESS_RATIO.get(attn_kind)
    if (
        mode not in {"context", "generation"}
        or expected_ratio != compress_ratio
        or tp_size != _TP_SIZE
        or canonical_num_heads != _CANONICAL_NUM_HEADS
        or num_heads != _PADDED_NUM_HEADS
        or mla_dtype != "bfloat16"
        or kv_cache_dtype != "fp8"
        or gemm_type != "fp8_block"
        or model_path != _MODEL_ARTIFACT
    ):
        raise ValueError("DSv4 attention case is outside the frozen V1.2 padded64/TP4 capability envelope")
    _positive_int(batch_size, field="batch_size")
    if mode == "context":
        _positive_int(isl, field="isl")
        _non_negative_int(prefix, field="prefix")
        if s_total is not None:
            raise ValueError("DSv4 context attention does not accept s_total")
    else:
        if isl is not None or prefix is not None:
            raise ValueError("DSv4 generation attention does not accept context isl/prefix")
        if _positive_int(s_total, field="s_total") < 2:
            raise ValueError("DSv4 generation s_total must include at least one persisted decode step")


def _validate_protocol(protocol: MeasurementProtocol) -> None:
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "sglang-dsv4-attn-v1"
        or protocol.statistic != "median"
        or protocol.samples < 3
    ):
        raise ValueError("DSv4 attention measurement protocol is incompatible with the exact runner")


def run_dsv4_attn_case(
    mode: str,
    attn_kind: str,
    tp_size: int,
    canonical_num_heads: int,
    num_heads: int,
    compress_ratio: int,
    batch_size: int,
    mla_dtype: str,
    kv_cache_dtype: str,
    gemm_type: str,
    isl: int | None = None,
    prefix: int | None = None,
    s_total: int | None = None,
    *,
    protocol: MeasurementProtocol | None = None,
    device: str = "cuda:0",
    model_path: str = _MODEL_ARTIFACT,
) -> RawMeasurement:
    """Measure one exact full-module attention case without offline output."""

    _validate_case(
        mode=mode,
        attn_kind=attn_kind,
        tp_size=tp_size,
        canonical_num_heads=canonical_num_heads,
        num_heads=num_heads,
        compress_ratio=compress_ratio,
        batch_size=batch_size,
        mla_dtype=mla_dtype,
        kv_cache_dtype=kv_cache_dtype,
        gemm_type=gemm_type,
        isl=isl,
        prefix=prefix,
        s_total=s_total,
        model_path=model_path,
    )
    protocol = protocol or MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=6,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )
    _validate_protocol(protocol)
    prepared = _prepare_dsv4_attn_case(
        mode=mode,
        attn_kind=attn_kind,
        tp_size=tp_size,
        canonical_num_heads=canonical_num_heads,
        num_heads=num_heads,
        compress_ratio=compress_ratio,
        batch_size=batch_size,
        mla_dtype=mla_dtype,
        kv_cache_dtype=kv_cache_dtype,
        gemm_type=gemm_type,
        isl=isl,
        prefix=prefix,
        s_total=s_total,
        device=device,
        model_path=model_path,
    )
    try:
        if (
            prepared.framework_version not in _SUPPORTED_SGLANG_VERSIONS
            or " ".join(prepared.device_name.split()).casefold() != "nvidia gb200"
            or prepared.architecture != _ARCHITECTURE
            or prepared.model_artifact != _MODEL_ARTIFACT
            or prepared.mode != mode
            or prepared.attn_kind != attn_kind
            or prepared.compress_ratio != compress_ratio
            or prepared.tp_size != _TP_SIZE
            or prepared.canonical_num_heads != _CANONICAL_NUM_HEADS
            or prepared.padded_num_heads != _PADDED_NUM_HEADS
            or prepared.mla_dtype != "bfloat16"
            or prepared.kv_cache_dtype != "fp8"
            or prepared.gemm_type != "fp8_block"
            or prepared.model_weight_generator != "proper-normal-v1"
            or prepared.model_weight_std != _PROPER_INIT_STD
            or prepared.model_weight_seed != _PROPER_INIT_SEED
        ):
            raise ValueError("prepared DSv4 attention case does not match the frozen exact request")

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
                raise RuntimeError("DSv4 attention exact runner requires CUDA Graph capture")
            latency_ms = _positive_finite_number(results.get("latency_ms"), field="latency")
            raw_samples = results.get("samples_ms")
            if not isinstance(raw_samples, (list, tuple)):
                raise TypeError("DSv4 attention benchmark must return CUDA-event samples")
            samples_ms = tuple(_positive_finite_number(sample, field="sample") for sample in raw_samples)
            if len(samples_ms) != protocol.samples or statistics.median(samples_ms) != latency_ms:
                raise ValueError("DSv4 attention samples do not match the requested median protocol")
            power_stats = results.get("power_stats")
            throttled = bool(results.get("throttled", False))
    except BaseException:
        if prepared.cleanup_func is not None:
            _preserve_primary_failure(
                prepared.cleanup_func,
                context="model-runner cleanup after attention measurement failed",
            )
        raise
    else:
        if prepared.cleanup_func is not None:
            prepared.cleanup_func()

    persisted_isl = int(isl) if mode == "context" else 1
    persisted_step = int(prefix) if mode == "context" else int(s_total) - 1
    perf_row = {
        "model": prepared.model_artifact,
        "architecture": prepared.architecture,
        "mla_dtype": prepared.mla_dtype,
        "kv_cache_dtype": prepared.kv_cache_dtype,
        "gemm_type": prepared.gemm_type,
        "num_heads": prepared.padded_num_heads,
        "batch_size": batch_size,
        "isl": persisted_isl,
        "tp_size": prepared.tp_size,
        "step": persisted_step,
        "compress_ratio": prepared.compress_ratio,
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
            "kernel_source": "compressed_flashmla",
            "device": prepared.device_name,
            "used_cuda_graph": True,
            "throttled": throttled,
            "model_artifact": prepared.model_artifact,
            "full_module": True,
            "mode": prepared.mode,
            "attn_kind": prepared.attn_kind,
            "tp_simulation": "single-gpu-tp4",
            "canonical_num_heads": prepared.canonical_num_heads,
            "padded_num_heads": prepared.padded_num_heads,
            "tensor_generator": "normal-v1",
            "seed": _INPUT_SEED,
            "model_weight_generator": prepared.model_weight_generator,
            "model_weight_std": prepared.model_weight_std,
            "model_weight_seed": prepared.model_weight_seed,
        },
        protocol_digest=protocol.digest,
        power_stats=power_stats,
    )


__all__ = ["PreparedDsv4AttentionCase", "get_dsv4_attn_test_cases", "run_dsv4_attn_case"]
