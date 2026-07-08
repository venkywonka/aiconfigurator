# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-only contracts for packaged GEMM and NCCL collector surfaces."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from aiconfigurator.collector.types import RawMeasurement

pytestmark = pytest.mark.unit

_SRC_ROOT = Path(__file__).resolve().parents[4] / "src"
_LIGHTWEIGHT_MODULES = (
    "aiconfigurator.collector.trtllm.registry",
    "aiconfigurator.collector.trtllm.gemm_adapter",
    "aiconfigurator.collector.trtllm.gemm",
    "aiconfigurator.collector.sglang.registry",
    "aiconfigurator.collector.network.registry",
    "aiconfigurator.collector.network.nccl_adapter",
    "aiconfigurator.collector.network.nccl",
)
_PACKAGED_REGISTRIES = (
    ("aiconfigurator.collector.trtllm.registry", "TRTLLM_LAZY_REGISTRY"),
    ("aiconfigurator.collector.sglang.registry", "SGLANG_LAZY_REGISTRY"),
    ("aiconfigurator.collector.network.registry", "NETWORK_LAZY_REGISTRY"),
)


def test_packaged_modules_import_from_src_only_without_heavy_or_legacy_dependencies(
    tmp_path: Path,
) -> None:
    """Model an installed wheel where the repository-level collector is absent."""

    probe = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        blocked_roots = frozenset(("collector", "tensorrt_llm", "torch"))

        class BlockedDependency(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname.partition(".")[0] in blocked_roots:
                    raise AssertionError(f"lightweight import requested {{fullname}}")
                return None

        sys.path.insert(0, {str(_SRC_ROOT)!r})
        sys.meta_path.insert(0, BlockedDependency())
        for module_name in {_LIGHTWEIGHT_MODULES!r}:
            importlib.import_module(module_name)

        from aiconfigurator.collector.adapters import LazyAdapterIndex
        from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
        from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC

        routes = LazyAdapterIndex.from_registries({{"sglang": SGLANG_LAZY_REGISTRY}}).routes_for(
            (GEMM_LAZY_SPEC.namespace, "sglang", "0.5.10")
        )
        assert len(routes) == 1
        assert routes[0].collector_module == "aiconfigurator.collector.trtllm.gemm"

        loaded = sorted(
            name for name in sys.modules if name.partition(".")[0] in blocked_roots
        )
        assert not loaded, loaded
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


@pytest.mark.parametrize(("registry_module", "registry_name"), _PACKAGED_REGISTRIES)
def test_every_packaged_registry_callable_resolves(
    registry_module: str,
    registry_name: str,
) -> None:
    registry = importlib.import_module(registry_module)

    for entry in getattr(registry, registry_name):
        offline_module = importlib.import_module(entry.module)
        assert callable(getattr(offline_module, entry.get_func))
        assert callable(getattr(offline_module, entry.run_func))

        lazy = entry.lazy
        assert lazy is not None
        assert entry.module == lazy.run_module
        runner = importlib.import_module(lazy.run_module)
        adapter = importlib.import_module(lazy.adapter_module)
        assert callable(getattr(runner, lazy.run_func))
        assert callable(getattr(adapter, lazy.case_func))
        assert callable(getattr(adapter, lazy.result_func))
        assert callable(getattr(adapter, lazy.resource_func))


def test_source_trtllm_registry_reuses_packaged_gemm_spec_by_identity() -> None:
    packaged_registry = importlib.import_module("aiconfigurator.collector.trtllm.registry")
    source_registry = importlib.import_module("collector.trtllm.registry")

    source_gemm = next(entry for entry in source_registry.REGISTRY if entry.op == "gemm")

    assert source_gemm.lazy is packaged_registry.GEMM_LAZY_SPEC


def test_source_sglang_registry_reuses_packaged_gemm_spec_by_identity() -> None:
    packaged_registry = importlib.import_module("aiconfigurator.collector.sglang.registry")
    source_registry = importlib.import_module("collector.sglang.registry")

    source_gemm = next(entry for entry in source_registry.REGISTRY if entry.op == "gemm")

    assert source_gemm.lazy is packaged_registry.GEMM_LAZY_SPEC


def test_packaged_sglang_registry_resolves_the_frozen_gemm_route() -> None:
    from aiconfigurator.collector.adapters import LazyAdapterIndex
    from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
    from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC

    routes = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY}).routes_for(
        (GEMM_LAZY_SPEC.namespace, "sglang", "0.5.10")
    )

    assert len(routes) == 1
    assert routes[0].collector_module == "aiconfigurator.collector.trtllm.gemm"
    assert routes[0].lazy is GEMM_LAZY_SPEC


def test_raw_measurement_mapping_survives_pickle_round_trip() -> None:
    measurement = RawMeasurement(
        latency_ms=1.25,
        energy_wms=62.5,
        samples_ms=(1.0, 1.25, 1.5),
        statistic="median",
        perf_row={"op": "all_reduce", "latency": 1.25},
        provenance={"runtime": "worker", "rank_count": 2},
        protocol_digest="protocol-digest",
        power_stats={"power": 50.0, "power_limit": 700.0},
    )

    restored = pickle.loads(pickle.dumps(measurement))

    assert isinstance(restored, RawMeasurement)
    assert isinstance(restored, Mapping)
    assert restored == measurement
    assert dict(restored) == dict(measurement)
    assert tuple(restored) == RawMeasurement._FIELDS
