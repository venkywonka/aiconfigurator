# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic operation classification and registry-route preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

_Route = TypeVar("_Route")
RouteIdentity = tuple[str, str, str]


class OperationKind(str, Enum):
    """How one operation participates in an exact-resolution walk."""

    MEASURED = "measured"
    DETERMINISTIC = "deterministic"
    COMPOSITION_ONLY = "composition_only"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class OperationCapability:
    """Generic resolution classification for one reachable operation."""

    operation: str
    kind: OperationKind
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str):
            raise TypeError("operation must be a string")
        if not self.operation.strip():
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.kind, OperationKind):
            raise TypeError("kind must be an OperationKind")
        if self.namespace is not None:
            if not isinstance(self.namespace, str):
                raise TypeError("namespace must be a string or None")
            if not self.namespace.strip():
                raise ValueError("namespace must be a non-empty string when provided")


class CapabilityPreflightError(RuntimeError):
    """Raised when reachable operations cannot be resolved unambiguously."""


def preflight_capabilities(
    capabilities: Sequence[OperationCapability],
    *,
    backend: str,
    backend_version: str,
    routes_for: Callable[[RouteIdentity], Sequence[_Route]],
) -> Mapping[RouteIdentity, _Route]:
    """Validate classifications and resolve each measured route exactly once."""
    if not isinstance(backend, str) or not backend.strip():
        raise CapabilityPreflightError("backend must be a non-empty string")
    if not isinstance(backend_version, str) or not backend_version.strip():
        raise CapabilityPreflightError("backend_version must be a non-empty string")

    classifications: dict[str, tuple[OperationKind, str | None]] = {}
    for capability in capabilities:
        if capability.kind is OperationKind.UNSUPPORTED:
            raise CapabilityPreflightError(f"{capability.operation}: operation is unsupported")

        if capability.kind is OperationKind.MEASURED:
            if not capability.namespace:
                raise CapabilityPreflightError(f"{capability.operation}: measured operation requires a namespace")
        elif capability.namespace is not None:
            raise CapabilityPreflightError(
                f"{capability.operation}: {capability.kind.value} operation cannot declare a namespace"
            )

        classification = (capability.kind, capability.namespace)
        previous = classifications.get(capability.operation)
        if previous is not None and previous != classification:
            raise CapabilityPreflightError(
                f"{capability.operation}: conflicting capability classifications {previous} and {classification}"
            )
        classifications[capability.operation] = classification

    resolved: dict[RouteIdentity, _Route] = {}
    for capability in capabilities:
        if capability.kind is not OperationKind.MEASURED:
            continue
        assert capability.namespace is not None
        identity = (capability.namespace, backend, backend_version)
        if identity in resolved:
            continue
        routes = tuple(routes_for(identity))
        if len(routes) != 1:
            raise CapabilityPreflightError(
                f"{capability.operation}: expected one route for {identity}, got {len(routes)}"
            )
        resolved[identity] = routes[0]

    return resolved
