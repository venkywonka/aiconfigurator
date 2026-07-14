# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional backend-selection contracts for every active DSv4 SGLang factory."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector.sglang import dsv4_attn as packaged_attention
from aiconfigurator.collector.sglang import mhc as packaged_mhc

pytestmark = pytest.mark.unit

_EXPECTED_BACKEND = "compressed"
_EXPECTED_PAGE_SIZE = 256
_EXPECTED_DISABLE_PIECEWISE_CUDA_GRAPH = True


class _BackendCapturedError(RuntimeError):
    """Stop a factory immediately after SGLang receives its configured backend."""


class _FakeServerArgs:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.mem_fraction_static = kwargs.get("mem_fraction_static", 0.5)
        self.chunked_prefill_size = kwargs.get("chunked_prefill_size", 1024)
        self.max_prefill_tokens = kwargs.get("max_prefill_tokens", 4096)


class _FakeModelConfig:
    @classmethod
    def from_server_args(cls, server_args: _FakeServerArgs) -> object:
        raise AssertionError(f"backend capture should stop before ModelConfig construction: {server_args!r}")


class _FakeModelRunner:
    pass


class _FakeForwardMode:
    def is_prefill(self) -> bool:
        return True


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: object) -> ModuleType:
    parent: ModuleType | None = None
    qualified = ""
    for part in name.split("."):
        qualified = f"{qualified}.{part}" if qualified else part
        module = sys.modules.get(qualified)
        if not isinstance(module, ModuleType):
            module = ModuleType(qualified)
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, qualified, module)
        if parent is not None:
            monkeypatch.setattr(parent, part, module, raising=False)
        parent = module
    assert parent is not None
    for attribute, value in attributes.items():
        monkeypatch.setattr(parent, attribute, value, raising=False)
    return parent


def _install_backend_capture_runtime(monkeypatch: pytest.MonkeyPatch) -> list[_FakeServerArgs]:
    captured: list[_FakeServerArgs] = []

    def _capture(server_args: _FakeServerArgs) -> None:
        captured.append(server_args)
        raise _BackendCapturedError

    _install_module(monkeypatch, "sglang.srt.configs.model_config", ModelConfig=_FakeModelConfig)
    _install_module(monkeypatch, "sglang.srt.entrypoints.engine", _set_envs_and_config=_capture)
    _install_module(monkeypatch, "sglang.srt.model_executor.model_runner", ModelRunner=_FakeModelRunner)
    _install_module(monkeypatch, "sglang.srt.server_args", ServerArgs=_FakeServerArgs)
    _install_module(monkeypatch, "sglang.srt.utils", suppress_other_loggers=lambda: None)
    deepseek_v2 = _install_module(monkeypatch, "sglang.srt.models.deepseek_v2")
    models = _install_module(monkeypatch, "sglang.srt.models")
    monkeypatch.setattr(models, "deepseek_v2", deepseek_v2, raising=False)
    return captured


def _fake_torch() -> ModuleType:
    device = SimpleNamespace(index=0)
    module = ModuleType("torch")
    module.device = lambda _value: device
    module.cuda = SimpleNamespace(set_device=lambda _device: None, current_device=lambda: 0)
    return module


def _assert_dsv4_server_args(captured: list[_FakeServerArgs]) -> None:
    assert [args.attention_backend for args in captured] == [_EXPECTED_BACKEND]
    assert [args.page_size for args in captured] == [_EXPECTED_PAGE_SIZE]
    assert [getattr(args, "disable_piecewise_cuda_graph", False) for args in captured] == [
        _EXPECTED_DISABLE_PIECEWISE_CUDA_GRAPH
    ]


def _assert_no_runtime_patch_dependency(module: ModuleType) -> None:
    source = inspect.getsource(module)
    assert "dsv4_runtime_patch" not in source
    assert "apply_deepseek_v4_forward_mode_compat_patch" not in source
    assert "deepseek_v4_compressed_kv_cache_patch" not in source


def _load_source_collector(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    relative_path: str,
) -> ModuleType:
    helper = ModuleType("helper")
    helper.benchmark_with_power = lambda *_args, **_kwargs: None
    helper.log_perf = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "helper", helper)

    spec = importlib.util.spec_from_file_location(module_name, Path(relative_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_packaged_attention_runner_selects_registered_compressed_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_backend_capture_runtime(monkeypatch)
    monkeypatch.setattr(packaged_attention, "validate_deepseek_v4_runtime_contract", lambda: None)
    monkeypatch.setattr(packaged_attention, "_patched_model_dir", lambda *_args: "/tmp/model")

    with pytest.raises(_BackendCapturedError):
        packaged_attention._load_model_runner(
            "model",
            attn_kind="csa",
            compress_ratio=8,
            kv_cache_dtype="fp8",
            gemm_type="fp8_block",
            tp_size=4,
            batch_size=1,
            max_total_tokens=128,
            required_swa_tokens=128,
            device="cuda:0",
            torch_module=_fake_torch(),
            cleanup_failed_attempt=lambda: None,
        )

    _assert_dsv4_server_args(captured)


def test_packaged_mhc_runner_selects_registered_compressed_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_backend_capture_runtime(monkeypatch)
    monkeypatch.setattr(packaged_mhc, "validate_deepseek_v4_runtime_contract", lambda: None)
    monkeypatch.setattr(packaged_mhc, "_patched_model_dir", lambda *_args: "/tmp/model")

    with pytest.raises(_BackendCapturedError):
        packaged_mhc._load_one_layer_runner(
            "model",
            "cuda:0",
            0.5,
            torch_module=_fake_torch(),
        )

    _assert_dsv4_server_args(captured)


def test_source_attention_collector_delegates_backend_selection_to_packaged_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_attention = _load_source_collector(
        monkeypatch,
        module_name="_r25_source_attention",
        relative_path="collector/sglang/collect_dsv4_attn.py",
    )
    assert source_attention.run_dsv4_attn_case is packaged_attention.run_dsv4_attn_case
    assert not hasattr(source_attention, "_load_model_runner")


def test_source_mhc_collector_delegates_backend_selection_to_packaged_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mhc = _load_source_collector(
        monkeypatch,
        module_name="_r25_source_mhc",
        relative_path="collector/sglang/collect_mhc_module.py",
    )
    assert source_mhc.run_mhc_case is packaged_mhc.run_mhc_case
    assert not hasattr(source_mhc, "_load_one_layer_runner")


def test_active_dsv4_collectors_do_not_depend_on_runtime_sglang_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_attention = _load_source_collector(
        monkeypatch,
        module_name="_r25_source_attention_no_patch_audit",
        relative_path="collector/sglang/collect_dsv4_attn.py",
    )
    source_mhc = _load_source_collector(
        monkeypatch,
        module_name="_r25_source_mhc_no_patch_audit",
        relative_path="collector/sglang/collect_mhc_module.py",
    )

    for module in (packaged_attention, packaged_mhc, source_attention, source_mhc):
        _assert_no_runtime_patch_dependency(module)
