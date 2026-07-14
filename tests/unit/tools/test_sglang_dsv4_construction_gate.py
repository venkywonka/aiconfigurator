# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the four-route R25-P construction-only harness."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.run_sglang_dsv4_construction_gate import (
    R25_PREPARATION_ROUTES,
    ConstructionGateError,
    validate_constructed_backend,
)

pytestmark = pytest.mark.unit


class _AttentionBackend:
    pass


class _C4IndexerBackend:
    pass


class _CompressorBackend:
    pass


class _DeepseekV4BackendRadix(_AttentionBackend, _C4IndexerBackend, _CompressorBackend):
    pass


def test_r25p_matrix_is_the_four_terminal_attention_routes() -> None:
    assert [
        (route.route_id, route.mode, route.attn_kind, route.attention_backend) for route in R25_PREPARATION_ROUTES
    ] == [
        ("context-csa", "context", "csa", "compressed"),
        ("context-hca", "context", "hca", "compressed"),
        ("generation-csa", "generation", "csa", "compressed"),
        ("generation-hca", "generation", "hca", "compressed"),
    ]


def test_constructed_runner_must_expose_compressed_dsv4_mro() -> None:
    runner = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend="compressed"),
        attn_backend=_DeepseekV4BackendRadix(),
    )

    evidence = validate_constructed_backend(
        runner,
        backend_type=_DeepseekV4BackendRadix,
        indexer_type=_C4IndexerBackend,
        compressor_type=_CompressorBackend,
    )

    assert evidence["attention_backend"] == "compressed"
    assert evidence["backend_class"] == "_DeepseekV4BackendRadix"


def test_constructed_runner_rejects_flashmla_or_dsv4_alias() -> None:
    wrong_runner = SimpleNamespace(
        server_args=SimpleNamespace(attention_backend="dsv4"),
        attn_backend=_AttentionBackend(),
    )
    with pytest.raises(ConstructionGateError):
        validate_constructed_backend(
            wrong_runner,
            backend_type=_DeepseekV4BackendRadix,
            indexer_type=_C4IndexerBackend,
            compressor_type=_CompressorBackend,
        )
