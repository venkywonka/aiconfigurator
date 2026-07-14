# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One executable runner owns every offline/JIT-capable framework route."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest

from aiconfigurator.collector.network.registry import NETWORK_LAZY_REGISTRY
from aiconfigurator.collector.registry_types import OpEntry
from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
from aiconfigurator.collector.trtllm.registry import TRTLLM_LAZY_REGISTRY
from collector.sglang.registry import REGISTRY as SGLANG_OFFLINE_REGISTRY
from collector.trtllm.registry import REGISTRY as TRTLLM_OFFLINE_REGISTRY

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _Route:
    backend: str
    op: str
    packaged_registry: tuple[OpEntry, ...]
    offline_module: str
    offline_canonical_symbol: str
    offline_registry: tuple[OpEntry, ...] | list[OpEntry] | None


_ROUTES = (
    _Route(
        "trtllm",
        "gemm",
        TRTLLM_LAZY_REGISTRY,
        "collector.trtllm.collect_gemm",
        "run_gemm_case",
        TRTLLM_OFFLINE_REGISTRY,
    ),
    _Route(
        "network",
        "nccl",
        NETWORK_LAZY_REGISTRY,
        "collector.network.collect_nccl",
        "run_nccl_case",
        None,
    ),
    _Route(
        "sglang",
        "gemm",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_gemm",
        "run_gemm_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "mhc_module",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_mhc_module",
        "run_mhc_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "moe",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_moe",
        "run_moe_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "custom_allreduce",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.custom_allreduce",
        "run_custom_allreduce_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "dsv4_csa_context_module",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_dsv4_attn",
        "run_dsv4_attn_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "dsv4_hca_context_module",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_dsv4_attn",
        "run_dsv4_attn_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "dsv4_csa_generation_module",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_dsv4_attn",
        "run_dsv4_attn_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
    _Route(
        "sglang",
        "dsv4_hca_generation_module",
        SGLANG_LAZY_REGISTRY,
        "collector.sglang.collect_dsv4_attn",
        "run_dsv4_attn_case",
        SGLANG_OFFLINE_REGISTRY,
    ),
)


def _entry(registry: tuple[OpEntry, ...] | list[OpEntry], op: str) -> OpEntry:
    matches = [entry for entry in registry if entry.op == op]
    assert len(matches) == 1, f"expected one {op!r} registry entry, found {len(matches)}"
    return matches[0]


@pytest.mark.parametrize("route", _ROUTES, ids=lambda route: f"{route.backend}-{route.op}")
def test_offline_and_jit_routes_resolve_to_one_canonical_runner(route: _Route) -> None:
    packaged = _entry(route.packaged_registry, route.op)
    assert packaged.lazy is not None
    assert (packaged.module, packaged.run_func) == (
        packaged.lazy.run_module,
        packaged.lazy.run_func,
    )

    canonical_module = importlib.import_module(packaged.lazy.run_module)
    canonical_runner = getattr(canonical_module, packaged.lazy.run_func)
    offline_module = importlib.import_module(route.offline_module)
    assert getattr(offline_module, route.offline_canonical_symbol) is canonical_runner

    if route.offline_registry is not None:
        offline = _entry(route.offline_registry, route.op)
        assert offline.lazy == packaged.lazy


def test_lazy_route_identity_is_unambiguous_per_backend_and_namespace() -> None:
    identities = [(route.backend, _entry(route.packaged_registry, route.op).lazy.namespace) for route in _ROUTES]
    assert len(identities) == len(set(identities))
