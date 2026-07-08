# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight reverse lookup for exact on-demand collector adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from aiconfigurator.collector.preflight import RouteIdentity
from aiconfigurator.collector.registry_types import OpEntry
from aiconfigurator.collector.types import LazyOpEntry, ResourceContract
from aiconfigurator.collector.version_resolver import resolve_module
from aiconfigurator.sdk.resolution.types import (
    MeasurementRecord,
    MeasurementRequest,
    RecordStatus,
)


@dataclass(frozen=True, slots=True)
class PreparedMeasurement:
    """One validated exact case plus its hardware contract."""

    request: MeasurementRequest
    case: Mapping[str, Any]
    contract: ResourceContract


@dataclass(frozen=True, slots=True)
class ResolvedLazyAdapter:
    """One version-resolved registry entry and its lightweight adapter hooks."""

    entry: OpEntry
    lazy: LazyOpEntry
    collector_module: str
    backend: str
    backend_version: str
    adapter_module: ModuleType
    _case_func: Callable[[MeasurementRequest], Mapping[str, Any]]
    _resource_func: Callable[[MeasurementRequest, Mapping[str, Any]], ResourceContract]
    _result_func: Callable[[MeasurementRequest, Mapping[str, Any], Mapping[str, Any]], MeasurementRecord]

    def prepare(self, request: MeasurementRequest) -> PreparedMeasurement:
        """Validate identity/capability before returning a schedulable case."""
        protocol = request.protocol
        for field_name, actual, expected in (
            ("revision", protocol.revision, self.lazy.protocol_revision),
            ("timer", protocol.timer, self.lazy.timer),
            ("tuning_revision", protocol.tuning_revision, self.lazy.tuning_revision),
        ):
            if actual != expected:
                raise ValueError(f"request protocol {field_name}={actual!r} does not match adapter {expected!r}")

        environment = request.environment
        if environment.backend != self.backend:
            raise ValueError(
                f"request environment backend={environment.backend!r} does not match route {self.backend!r}"
            )
        if environment.backend_version != self.backend_version:
            raise ValueError(
                "request environment backend_version="
                f"{environment.backend_version!r} does not match route {self.backend_version!r}"
            )
        if request.key.namespace != self.lazy.namespace:
            raise ValueError(
                f"request PerfKey namespace={request.key.namespace!r} does not match adapter {self.lazy.namespace!r}"
            )

        case = self._case_func(request)
        if not isinstance(case, Mapping):
            raise TypeError("lazy adapter case function must return a Mapping")
        contract = self._resource_func(request, case)
        if not isinstance(contract, ResourceContract):
            raise TypeError("lazy adapter resource function must return ResourceContract")
        return PreparedMeasurement(request=request, case=case, contract=contract)

    def record(
        self,
        prepared: PreparedMeasurement,
        raw_result: Mapping[str, Any],
    ) -> MeasurementRecord:
        """Convert worker output in the parent and enforce physical identity."""
        if not isinstance(prepared, PreparedMeasurement):
            raise TypeError("prepared must be a PreparedMeasurement")
        if not isinstance(raw_result, Mapping):
            raise TypeError("raw_result must be a Mapping")
        record = self._result_func(prepared.request, prepared.case, raw_result)
        if not isinstance(record, MeasurementRecord):
            raise TypeError("lazy adapter result function must return MeasurementRecord")
        if record.key != prepared.request.key:
            raise ValueError("adapter record PerfKey does not match the requested key")
        if record.protocol != prepared.request.protocol:
            raise ValueError("adapter record protocol does not match the request")
        if record.status is not RecordStatus.VALID:
            raise ValueError("adapter result function must return a valid MeasurementRecord")
        return record


class LazyAdapterIndex:
    """Reverse index over existing backend collector registries."""

    def __init__(self, registries: Mapping[str, tuple[OpEntry, ...]]) -> None:
        self._registries = dict(registries)
        self._cache: dict[RouteIdentity, tuple[ResolvedLazyAdapter, ...]] = {}

    @classmethod
    def from_registries(
        cls,
        registries: Mapping[str, Sequence[OpEntry]],
    ) -> LazyAdapterIndex:
        if not isinstance(registries, Mapping):
            raise TypeError("registries must be a Mapping")
        copied: dict[str, tuple[OpEntry, ...]] = {}
        for backend, entries in registries.items():
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError("registry backend names must be non-empty strings")
            resolved_entries = tuple(entries)
            if any(not isinstance(entry, OpEntry) for entry in resolved_entries):
                raise TypeError(f"registry {backend!r} must contain only OpEntry values")
            copied[backend] = resolved_entries
        return cls(copied)

    def routes_for(self, identity: RouteIdentity) -> tuple[ResolvedLazyAdapter, ...]:
        if not isinstance(identity, tuple) or len(identity) != 3:
            raise TypeError("route identity must be a (namespace, backend, backend_version) tuple")
        namespace, backend, backend_version = identity
        if any(not isinstance(value, str) or not value.strip() for value in identity):
            raise ValueError("route identity components must be non-empty strings")
        cached = self._cache.get(identity)
        if cached is not None:
            return cached

        routes: list[ResolvedLazyAdapter] = []
        for entry in self._registries.get(backend, ()):
            lazy = entry.lazy
            if lazy is None or lazy.namespace != namespace:
                continue
            collector_module = resolve_module(entry, backend_version)
            if collector_module is None:
                continue
            _validate_namespaced_module(lazy.adapter_module, "adapter_module")
            _validate_namespaced_module(lazy.run_module, "run_module")
            module = importlib.import_module(lazy.adapter_module)
            routes.append(
                ResolvedLazyAdapter(
                    entry=entry,
                    lazy=lazy,
                    collector_module=collector_module,
                    backend=backend,
                    backend_version=backend_version,
                    adapter_module=module,
                    _case_func=_required_callable(module, lazy.case_func),
                    _resource_func=_required_callable(module, lazy.resource_func),
                    _result_func=_required_callable(module, lazy.result_func),
                )
            )
        result = tuple(routes)
        self._cache[identity] = result
        return result


def _validate_namespaced_module(module: str, field_name: str) -> None:
    if not module.startswith("aiconfigurator."):
        raise ValueError(
            f"lazy adapter {field_name} must be an installable aiconfigurator-namespaced module, got {module!r}"
        )


def _required_callable(module: ModuleType, name: str) -> Callable:
    value = getattr(module, name, None)
    if not callable(value):
        raise TypeError(f"adapter module {module.__name__!r} has no callable {name!r}")
    return value
