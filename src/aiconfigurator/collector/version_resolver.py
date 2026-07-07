# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve version-specific collector modules from registry entries."""

from __future__ import annotations

import re

from packaging.specifiers import InvalidSpecifier, Specifier
from packaging.version import InvalidVersion, Version

from aiconfigurator.collector.registry_types import OpEntry

_FRAMEWORK_PREFIX_RE = re.compile(r"^[a-zA-Z_]+")


def _strip_local_metadata(v: str) -> str:
    """Drop local build metadata suffix to preserve legacy collector behavior."""
    return v.split("+", 1)[0].strip()


def _normalize_version(v: str) -> Version:
    """Parse a version while preserving the collector's historical fallbacks."""
    normalized = _strip_local_metadata(v)
    if not normalized:
        return Version("0")
    if normalized.startswith("0.0.0.dev"):
        return Version("9999.0.0")
    try:
        return Version(normalized)
    except InvalidVersion:
        return Version("0")


def _parse_compat_specifier(compat_str: str) -> list[Specifier]:
    """Parse ``<framework><constraints>`` into validated specifier clauses."""
    spec = _FRAMEWORK_PREFIX_RE.sub("", compat_str, count=1).strip()
    if not spec:
        raise ValueError(f"Invalid __compat__ {compat_str!r}: missing version constraints")

    clauses = [clause.strip() for clause in spec.split(",") if clause.strip()]
    if not clauses:
        raise ValueError(f"Invalid __compat__ {compat_str!r}: no valid constraint clauses found")

    parsed: list[Specifier] = []
    for clause in clauses:
        try:
            parsed.append(Specifier(clause))
        except InvalidSpecifier as error:
            raise ValueError(f"Invalid __compat__ {compat_str!r}: {error}") from error
    return parsed


def _check_compat(compat_str: str, runtime_version: str) -> bool:
    """Check whether a runtime version satisfies a collector compatibility specifier."""
    specifiers = _parse_compat_specifier(compat_str)
    runtime = _normalize_version(runtime_version)

    for specifier in specifiers:
        operator = specifier.operator
        version = _normalize_version(specifier.version)
        if operator == ">=":
            compatible = runtime >= version
        elif operator == "<=":
            compatible = runtime <= version
        elif operator == ">":
            compatible = runtime > version
        elif operator == "<":
            compatible = runtime < version
        elif operator == "==":
            compatible = runtime == version
        elif operator == "!=":
            compatible = runtime != version
        else:
            raise ValueError(
                f"Invalid __compat__ {compat_str!r}: unsupported operator {operator!r}. Use one of >=, <=, >, <, ==, !="
            )
        if not compatible:
            return False
    return True


def resolve_module(entry: OpEntry, runtime_version: str) -> str | None:
    """Return the collector module selected for one registry entry."""
    if not entry.versions:
        return entry.module

    runtime = _normalize_version(runtime_version)
    for route in entry.versions:
        if runtime >= _normalize_version(route.min_version):
            return route.module
    return None


def build_collections(
    registry: list[OpEntry],
    backend_name: str,
    runtime_version: str,
    ops: list[str] | None = None,
    *,
    logger=None,
) -> list[dict]:
    """Build legacy offline collection dictionaries from registry entries."""
    collections = []
    for entry in registry:
        if ops and entry.op not in ops:
            continue

        module = resolve_module(entry, runtime_version)
        if module is None:
            if logger:
                logger.warning(f"Skipping {backend_name}.{entry.op} — no collector for v{runtime_version}")
            continue

        collections.append(
            {
                "name": backend_name,
                "type": entry.op,
                "module": module,
                "get_func": entry.get_func,
                "run_func": entry.run_func,
                "perf_filename": entry.perf_filename,
            }
        )

    return collections
