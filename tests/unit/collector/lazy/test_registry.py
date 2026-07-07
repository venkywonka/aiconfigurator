# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for packaged lazy collector registry metadata."""

from __future__ import annotations

import importlib
import json
import os
import site
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from aiconfigurator.collector.registry_types import OpEntry, PerfFile, VersionRoute
from aiconfigurator.collector.types import FabricRequirement, LazyOpEntry, ResourceContract
from aiconfigurator.collector.version_resolver import (
    _check_compat,
    _normalize_version,
    build_collections,
    resolve_module,
)

pytestmark = pytest.mark.unit


def _lazy_entry() -> LazyOpEntry:
    return LazyOpEntry(
        namespace="trtllm/gemm/v1",
        run_module="aiconfigurator.collector.fake_runner",
        run_func="run_case",
        adapter_module="aiconfigurator.collector.fake_adapter",
        case_func="request_to_case",
        result_func="result_to_record",
        resource_func="resource_for_request",
        protocol_revision="cuda-event-v1",
        timer="cuda_event",
        tuning_revision="fake-v1",
    )


def test_offline_collection_dict_is_unchanged_when_lazy_adapter_exists() -> None:
    entry = OpEntry(
        op="gemm",
        module="collector.fake",
        get_func="all_cases",
        run_func="run_case",
        perf_filename=PerfFile.GEMM,
        lazy=_lazy_entry(),
    )

    assert build_collections([entry], "trtllm", "1.2.0") == [
        {
            "name": "trtllm",
            "type": "gemm",
            "module": "collector.fake",
            "get_func": "all_cases",
            "run_func": "run_case",
            "perf_filename": PerfFile.GEMM,
        }
    ]


@pytest.mark.parametrize("gpu_count", [0, -1])
def test_resource_contract_rejects_non_positive_counts(gpu_count: int) -> None:
    with pytest.raises(ValueError, match="gpu_count"):
        ResourceContract(gpu_count=gpu_count, fabric=FabricRequirement.NONE)


@pytest.mark.parametrize("gpu_count", [True, 1.5, "2"])
def test_resource_contract_rejects_untyped_counts(gpu_count: object) -> None:
    with pytest.raises(TypeError, match="gpu_count"):
        ResourceContract(gpu_count=gpu_count, fabric=FabricRequirement.NONE)


@pytest.mark.parametrize("fabric", [FabricRequirement.P2P, FabricRequirement.NVLINK])
def test_single_gpu_resource_contract_rejects_fabric(fabric: FabricRequirement) -> None:
    with pytest.raises(ValueError, match="single-GPU"):
        ResourceContract(gpu_count=1, fabric=fabric)


@pytest.mark.parametrize("fabric", ["none", "nvlink", "bad", None, 1])
def test_resource_contract_rejects_untyped_fabric(fabric: object) -> None:
    with pytest.raises(TypeError, match="fabric"):
        ResourceContract(gpu_count=2, fabric=fabric)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exclusive_devices", "true"),
        ("exclusive_devices", 1),
        ("reserve_fabric_domain", "false"),
        ("reserve_fabric_domain", 0),
    ],
)
def test_resource_contract_rejects_untyped_boolean_flags(field: str, value: object) -> None:
    values = {"gpu_count": 2, "fabric": FabricRequirement.NONE, field: value}
    with pytest.raises(TypeError, match=field):
        ResourceContract(**values)


def test_fabric_requirement_preserves_string_enum_behavior_on_python_310() -> None:
    assert str(FabricRequirement.NVLINK) == "nvlink"
    assert FabricRequirement.NVLINK == "nvlink"
    assert json.loads(json.dumps({"fabric": FabricRequirement.NVLINK})) == {"fabric": "nvlink"}


@pytest.mark.parametrize(
    ("bad_value", "error_type"),
    [("", ValueError), (" padded ", ValueError), (1, TypeError)],
)
def test_lazy_entry_rejects_empty_or_untyped_metadata(bad_value: object, error_type: type[Exception]) -> None:
    lazy = _lazy_entry()
    values = {field.name: getattr(lazy, field.name) for field in fields(lazy)}
    for field_name in values:
        invalid = values | {field_name: bad_value}
        with pytest.raises(error_type, match=field_name):
            LazyOpEntry(**invalid)


def test_legacy_registry_and_resolver_imports_are_identity_reexports() -> None:
    from collector.registry_types import OpEntry as LegacyOpEntry
    from collector.registry_types import PerfFile as LegacyPerfFile
    from collector.registry_types import VersionRoute as LegacyVersionRoute
    from collector.version_resolver import OpEntry as LegacyResolverOpEntry
    from collector.version_resolver import _check_compat as legacy_check_compat
    from collector.version_resolver import _normalize_version as legacy_normalize_version
    from collector.version_resolver import build_collections as legacy_build_collections
    from collector.version_resolver import resolve_module as legacy_resolve_module

    assert LegacyOpEntry is OpEntry
    assert LegacyPerfFile is PerfFile
    assert LegacyVersionRoute is VersionRoute
    assert legacy_check_compat is _check_compat
    assert legacy_normalize_version is _normalize_version
    assert LegacyResolverOpEntry is OpEntry
    assert legacy_build_collections is build_collections
    assert legacy_resolve_module is resolve_module


def test_bare_registry_types_import_is_an_identity_reexport(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("registry_types", None)
    monkeypatch.syspath_prepend(str(Path("collector").resolve()))
    try:
        bare_registry = importlib.import_module("registry_types")
        assert bare_registry.OpEntry is OpEntry
        assert bare_registry.PerfFile is PerfFile
        assert bare_registry.VersionRoute is VersionRoute
    finally:
        sys.modules.pop("registry_types", None)


def test_source_tree_collector_imports_without_an_installed_distribution() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    site_packages = Path(site.getsitepackages()[0])
    command = f"""
import importlib.metadata
import sys

sys.path.append({str(site_packages)!r})
real_version = importlib.metadata.version

def source_tree_version(distribution_name):
    if distribution_name == "aiconfigurator":
        raise importlib.metadata.PackageNotFoundError(distribution_name)
    return real_version(distribution_name)

importlib.metadata.version = source_tree_version
sys.path.insert(0, "collector")
from registry_types import OpEntry, PerfFile
from version_resolver import build_collections
from aiconfigurator import __version__

assert OpEntry.__module__ == "aiconfigurator.collector.registry_types"
assert PerfFile.__module__ == "aiconfigurator.collector.registry_types"
assert build_collections.__module__ == "aiconfigurator.collector.version_resolver"
assert __version__ == "0.0.0+source"
assert {str(repo_root / "src")!r} not in sys.path
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_tree_wrappers_prefer_colocated_source_to_an_unloaded_installed_package(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    site_packages = Path(site.getsitepackages()[0])
    package = tmp_path / "aiconfigurator"
    collector_package = package / "collector"
    collector_package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (collector_package / "__init__.py").write_text("", encoding="utf-8")
    (collector_package / "registry_types.py").write_text(
        "class OpEntry: origin = 'installed'; __dataclass_fields__ = {'lazy': None}\n"
        "class PerfFile: origin = 'installed'\n"
        "class VersionRoute: origin = 'installed'\n",
        encoding="utf-8",
    )
    (collector_package / "version_resolver.py").write_text(
        "from .registry_types import OpEntry\n"
        "def _check_compat(): return 'installed'\n"
        "def _normalize_version(): return 'installed'\n"
        "def _parse_compat_specifier(): return 'installed'\n"
        "def _strip_local_metadata(): return 'installed'\n"
        "def build_collections(): return 'installed'\n"
        "def resolve_module(): return 'installed'\n",
        encoding="utf-8",
    )
    command = f"""
import sys

sys.path.append({str(site_packages)!r})
sys.path.insert(0, {str(tmp_path)!r})
sys.path.insert(0, "collector")
import registry_types
import version_resolver

assert registry_types.OpEntry.__module__ == "aiconfigurator.collector.registry_types"
assert version_resolver.build_collections.__module__ == "aiconfigurator.collector.version_resolver"
assert not hasattr(registry_types.OpEntry, "origin")
assert version_resolver.build_collections([], "trtllm", "1.0.0") == []
assert {str(repo_root / "src")!r} not in sys.path
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_tree_wrappers_ignore_an_unloaded_older_package_without_collector(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    site_packages = Path(site.getsitepackages()[0])
    package = tmp_path / "aiconfigurator"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'old-installed'\n", encoding="utf-8")
    command = f"""
import sys

sys.path.append({str(site_packages)!r})
sys.path.insert(0, {str(tmp_path)!r})
sys.path.insert(0, "collector")
from registry_types import OpEntry
from version_resolver import OpEntry as ResolverOpEntry

assert OpEntry.__module__ == "aiconfigurator.collector.registry_types"
assert ResolverOpEntry is OpEntry
assert {str(repo_root / "src")!r} not in sys.path
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_tree_wrappers_ignore_unloaded_incompatible_installed_collector_modules(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    site_packages = Path(site.getsitepackages()[0])
    package = tmp_path / "aiconfigurator"
    collector_package = package / "collector"
    collector_package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'old-installed'\n", encoding="utf-8")
    (collector_package / "__init__.py").write_text("", encoding="utf-8")
    (collector_package / "registry_types.py").write_text(
        "class OpEntry: pass\nclass PerfFile: pass\nclass VersionRoute: pass\n",
        encoding="utf-8",
    )
    (collector_package / "version_resolver.py").write_text(
        "from .registry_types import OpEntry\n"
        "def _check_compat(): return 'old'\n"
        "def _normalize_version(): return 'old'\n"
        "def _parse_compat_specifier(): return 'old'\n"
        "def _strip_local_metadata(): return 'old'\n"
        "def build_collections(): return 'old'\n"
        "def resolve_module(): return 'old'\n",
        encoding="utf-8",
    )
    command = f"""
import sys

sys.path.append({str(site_packages)!r})
sys.path.insert(0, {str(tmp_path)!r})
sys.path.insert(0, "collector")
import registry_types
import version_resolver

assert registry_types.OpEntry.__module__ == "aiconfigurator.collector.registry_types"
assert version_resolver.build_collections.__module__ == "aiconfigurator.collector.version_resolver"
assert not hasattr(registry_types.OpEntry, "origin")
assert version_resolver.build_collections([], "trtllm", "1.0.0") == []
assert {str(repo_root / "src")!r} not in sys.path
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_tree_wrappers_reject_an_already_loaded_foreign_package(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    package = tmp_path / "aiconfigurator"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'foreign'\n", encoding="utf-8")
    command = f"""
import sys

sys.path.insert(0, {str(tmp_path)!r})
import aiconfigurator
sys.path.insert(0, "collector")
import registry_types
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-S", "-c", command],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "already loaded from another location" in completed.stderr


def test_op_entry_retains_positional_compatibility_and_runtime_type_hints() -> None:
    entry = OpEntry("gemm", "all_cases", "run_case", PerfFile.GEMM, "collector.fake")

    assert entry.module == "collector.fake"
    assert entry.versions == ()
    assert entry.lazy is None
    assert LazyOpEntry in get_args(get_type_hints(OpEntry)["lazy"])


def test_version_routing_retains_lazy_metadata_without_emitting_it_offline() -> None:
    lazy = _lazy_entry()
    entry = OpEntry(
        op="gemm",
        get_func="all_cases",
        run_func="run_case",
        perf_filename=PerfFile.GEMM,
        versions=(VersionRoute("1.0.0", "collector.fake_v1"),),
        lazy=lazy,
    )

    assert resolve_module(entry, "1.2.0") == "collector.fake_v1"
    assert entry.lazy is lazy
    assert "lazy" not in build_collections([entry], "trtllm", "1.2.0")[0]
