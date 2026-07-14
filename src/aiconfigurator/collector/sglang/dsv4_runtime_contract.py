# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed runtime contract checks for DSv4 SGLang collectors.

AIC's production-shaped DSv4 collection path uses SGLang as the selected
runtime artifact provides it.  These checks deliberately validate the native
runtime closure instead of monkeypatching SGLang or assembling source overlays.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from inspect import Parameter
from pathlib import Path

_REQUIRED_MODULES = (
    "sglang.srt.models.deepseek_v4",
    "sglang.srt.layers.attention.deepseek_v4_backend_radix",
    "sglang.srt.mem_cache.deepseekv4_memory_pool",
    "sglang.srt.model_executor.model_runner_kv_cache_mixin",
    "sglang.jit_kernel.deepseek_v4",
)
_REQUIRED_JIT_FILES = (
    "csrc/deepseek_v4/common.cuh",
    "include/sgl_kernel/deepseek_v4/compress.cuh",
)


class DeepSeekV4RuntimeContractError(RuntimeError):
    """The installed SGLang artifact cannot serve AIC DSv4 exact collection."""


@dataclass(frozen=True, slots=True)
class DeepSeekV4RuntimeContract:
    """Minimal provenance for the selected coherent SGLang artifact."""

    version: str
    commit_id: str | None
    jit_root: str


def validate_deepseek_v4_runtime_contract() -> DeepSeekV4RuntimeContract:
    """Require native DSv4 compressed support in the selected SGLang artifact.

    The failure message is intentionally explicit: choose a coherent SGLang
    artifact with DSv4 compressed attention support, or mark the route
    unsupported.  AIC does not silently graft missing SGLang internals.
    """

    missing_modules = tuple(name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None)
    if missing_modules:
        raise DeepSeekV4RuntimeContractError(
            "SGLang artifact is missing native DeepSeek-V4 compressed modules: "
            f"{', '.join(missing_modules)}. Use a coherent DSv4-capable SGLang "
            "artifact/container or mark this route unsupported; AIC does not "
            "patch SGLang internals at runtime."
        )

    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if not _forward_mode_accepts_include_draft_extend_v2(ForwardMode.is_prefill):
        raise DeepSeekV4RuntimeContractError(
            "SGLang ForwardMode.is_prefill does not accept include_draft_extend_v2. "
            "Use a coherent DSv4-capable SGLang artifact/container or mark this "
            "route unsupported; AIC does not monkeypatch SGLang ForwardMode."
        )

    import sglang.version as sglang_version
    from sglang.srt.model_executor import model_runner_kv_cache_mixin

    source = _best_effort_source(model_runner_kv_cache_mixin)
    if "DeepSeekV4TokenToKVPool" not in source or "is_deepseek_compressed" not in source:
        raise DeepSeekV4RuntimeContractError(
            "SGLang ModelRunner KV-cache path does not natively initialize "
            "DeepSeekV4TokenToKVPool. Use a coherent DSv4-capable SGLang "
            "artifact/container or mark this route unsupported; AIC does not "
            "replace ModelRunner pool initialization."
        )

    jit_root = _deepseek_v4_jit_root()
    missing_jit = _missing_required_jit_files(jit_root)
    if missing_jit:
        raise DeepSeekV4RuntimeContractError(
            "SGLang artifact is missing DeepSeek-V4 JIT source files: "
            f"{', '.join(missing_jit)}. Use a coherent DSv4-capable SGLang "
            "artifact/container or mark this route unsupported; AIC does not "
            "graft SGLang JIT headers at runtime."
        )

    return DeepSeekV4RuntimeContract(
        version=_sglang_version(),
        commit_id=getattr(sglang_version, "__commit_id__", None),
        jit_root=str(jit_root),
    )


def _sglang_version() -> str:
    try:
        return package_version("sglang")
    except PackageNotFoundError:
        version_module = importlib.import_module("sglang.version")
        return str(getattr(version_module, "__version__", "unknown"))


def _forward_mode_accepts_include_draft_extend_v2(method: object) -> bool:
    try:
        parameters = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == "include_draft_extend_v2"
        for parameter in parameters
    )


def _deepseek_v4_jit_root() -> Path:
    spec = importlib.util.find_spec("sglang.jit_kernel.deepseek_v4")
    origin = getattr(spec, "origin", None)
    if not origin:
        raise DeepSeekV4RuntimeContractError("cannot locate SGLang deepseek_v4 JIT module origin")
    return Path(origin).resolve().parent


def _missing_required_jit_files(jit_root: Path) -> tuple[str, ...]:
    return tuple(relative for relative in _REQUIRED_JIT_FILES if not (jit_root / relative).is_file())


def _best_effort_source(module: object) -> str:
    try:
        return inspect.getsource(module)
    except (OSError, TypeError):
        return ""


__all__ = [
    "DeepSeekV4RuntimeContract",
    "DeepSeekV4RuntimeContractError",
    "validate_deepseek_v4_runtime_contract",
]
