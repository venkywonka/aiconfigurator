# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for coherent, unpatched SGLang DSv4 runtime validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aiconfigurator.collector.sglang import dsv4_runtime_contract as contract

pytestmark = pytest.mark.unit


def test_forward_mode_contract_accepts_native_dsv4_signature() -> None:
    def native_is_prefill(self: object, include_draft_extend_v2: bool = False) -> bool:
        return bool(self) or include_draft_extend_v2

    assert contract._forward_mode_accepts_include_draft_extend_v2(native_is_prefill)


def test_forward_mode_contract_accepts_variadic_runtime_signature() -> None:
    def variadic_is_prefill(self: object, **kwargs: object) -> bool:
        return bool(self) or bool(kwargs)

    assert contract._forward_mode_accepts_include_draft_extend_v2(variadic_is_prefill)


def test_forward_mode_contract_rejects_older_stable_signature() -> None:
    def old_is_prefill(self: object) -> bool:
        return bool(self)

    assert not contract._forward_mode_accepts_include_draft_extend_v2(old_is_prefill)


def test_missing_required_jit_files_reports_transitive_csrc_closure(tmp_path) -> None:
    assert contract._missing_required_jit_files(tmp_path) == (
        "csrc/deepseek_v4/common.cuh",
        "include/sgl_kernel/deepseek_v4/compress.cuh",
    )

    (tmp_path / "csrc/deepseek_v4").mkdir(parents=True)
    (tmp_path / "csrc/deepseek_v4/common.cuh").write_text("// common\n")

    assert contract._missing_required_jit_files(tmp_path) == ("include/sgl_kernel/deepseek_v4/compress.cuh",)


def test_validate_runtime_contract_fails_closed_when_native_sglang_modules_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(contract.DeepSeekV4RuntimeContractError) as exc_info:
        contract.validate_deepseek_v4_runtime_contract()

    message = str(exc_info.value)
    assert "missing native DeepSeek-V4 compressed modules" in message
    assert "AIC does not patch SGLang internals at runtime" in message


def test_deepseek_v4_jit_root_uses_module_origin(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = tmp_path / "sglang/jit_kernel/deepseek_v4.py"
    module.parent.mkdir(parents=True)
    module.write_text("# jit\n")
    monkeypatch.setattr(contract.importlib.util, "find_spec", lambda _name: SimpleNamespace(origin=str(module)))

    assert contract._deepseek_v4_jit_root() == module.parent.resolve()
