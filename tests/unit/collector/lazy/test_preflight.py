# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic capability-classification and route-preflight contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aiconfigurator.collector.preflight import (
    CapabilityPreflightError,
    OperationCapability,
    OperationKind,
    preflight_capabilities,
)

pytestmark = pytest.mark.unit


def test_preflight_routes_each_measured_namespace_once_by_backend_identity() -> None:
    capabilities = (
        OperationCapability("gemm-a", OperationKind.MEASURED, "gemm_perf.txt/v1"),
        OperationCapability("gemm-b", OperationKind.MEASURED, "gemm_perf.txt/v1"),
        OperationCapability("embedding", OperationKind.DETERMINISTIC),
        OperationCapability("overlap", OperationKind.COMPOSITION_ONLY),
    )
    expected_identity = ("gemm_perf.txt/v1", "sglang", "0.5.10")
    route_calls: list[tuple[str, str, str]] = []

    def routes_for(identity: tuple[str, str, str]) -> tuple[str, ...]:
        route_calls.append(identity)
        return ("gemm-route",)

    resolved = preflight_capabilities(
        capabilities,
        backend="sglang",
        backend_version="0.5.10",
        routes_for=routes_for,
    )

    assert resolved == {expected_identity: "gemm-route"}
    assert route_calls == [expected_identity]
    assert [kind.value for kind in OperationKind] == [
        "measured",
        "deterministic",
        "composition_only",
        "unsupported",
    ]
    with pytest.raises(FrozenInstanceError):
        capabilities[0].namespace = "another-namespace"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("capability", "detail"),
    (
        (OperationCapability("missing-namespace", OperationKind.MEASURED), "missing-namespace"),
        (
            OperationCapability("deterministic-with-key", OperationKind.DETERMINISTIC, "unexpected/v1"),
            "deterministic-with-key",
        ),
        (
            OperationCapability("composition-with-key", OperationKind.COMPOSITION_ONLY, "unexpected/v1"),
            "composition-with-key",
        ),
    ),
)
def test_preflight_rejects_invalid_kind_namespace_combinations(capability, detail: str) -> None:
    with pytest.raises(CapabilityPreflightError, match=detail):
        preflight_capabilities(
            (capability,),
            backend="sglang",
            backend_version="0.5.10",
            routes_for=lambda identity: (identity,),
        )


@pytest.mark.parametrize("routes", ((), ("route-a", "route-b")))
def test_preflight_rejects_missing_or_ambiguous_measured_routes(routes: tuple[str, ...]) -> None:
    identity = ("gemm_perf.txt/v1", "sglang", "0.5.10")

    with pytest.raises(CapabilityPreflightError, match="gemm"):
        preflight_capabilities(
            (OperationCapability("gemm", OperationKind.MEASURED, identity[0]),),
            backend=identity[1],
            backend_version=identity[2],
            routes_for=lambda candidate: routes if candidate == identity else (),
        )


def test_preflight_fails_unsupported_without_attempting_route_lookup() -> None:
    def unexpected_route_lookup(identity: tuple[str, str, str]) -> tuple[object, ...]:
        raise AssertionError(f"unsupported operation must not route: {identity}")

    with pytest.raises(CapabilityPreflightError, match="unsupported-op"):
        preflight_capabilities(
            (OperationCapability("unsupported-op", OperationKind.UNSUPPORTED),),
            backend="sglang",
            backend_version="0.5.10",
            routes_for=unexpected_route_lookup,
        )


@pytest.mark.parametrize(
    "capabilities",
    (
        (
            OperationCapability("compound", OperationKind.MEASURED, "module_perf.txt/v1"),
            OperationCapability("compound", OperationKind.COMPOSITION_ONLY),
        ),
        (
            OperationCapability("compound", OperationKind.MEASURED, "module_a.txt/v1"),
            OperationCapability("compound", OperationKind.MEASURED, "module_b.txt/v1"),
        ),
    ),
)
def test_preflight_rejects_conflicting_classifications_for_one_operation_before_route_lookup(capabilities) -> None:
    route_calls = []

    with pytest.raises(CapabilityPreflightError, match=r"compound.*conflicting"):
        preflight_capabilities(
            capabilities,
            backend="sglang",
            backend_version="0.5.10",
            routes_for=lambda identity: route_calls.append(identity) or ("route",),
        )

    assert route_calls == []


def test_preflight_allows_repeated_identical_classification_for_shared_graph_consumer() -> None:
    capability = OperationCapability("shared-gemm", OperationKind.MEASURED, "gemm_perf.txt/v1")
    route_calls = []

    resolved = preflight_capabilities(
        (capability, capability),
        backend="sglang",
        backend_version="0.5.10",
        routes_for=lambda identity: route_calls.append(identity) or ("route",),
    )

    assert resolved == {("gemm_perf.txt/v1", "sglang", "0.5.10"): "route"}
    assert route_calls == [("gemm_perf.txt/v1", "sglang", "0.5.10")]


@pytest.mark.parametrize(
    ("kwargs", "error", "detail"),
    (
        ({"operation": "", "kind": OperationKind.MEASURED, "namespace": "gemm/v1"}, ValueError, "operation"),
        ({"operation": 7, "kind": OperationKind.MEASURED, "namespace": "gemm/v1"}, TypeError, "operation"),
        ({"operation": "gemm", "kind": "measured", "namespace": "gemm/v1"}, TypeError, "kind"),
        ({"operation": "gemm", "kind": OperationKind.MEASURED, "namespace": ""}, ValueError, "namespace"),
        ({"operation": "gemm", "kind": OperationKind.MEASURED, "namespace": 7}, TypeError, "namespace"),
    ),
)
def test_operation_capability_rejects_blank_or_untyped_identity_fields(kwargs, error, detail: str) -> None:
    with pytest.raises(error, match=detail):
        OperationCapability(**kwargs)


@pytest.mark.parametrize(
    ("backend", "backend_version", "detail"),
    (
        ("", "1.0", "backend"),
        ("sglang", "", "backend_version"),
        (7, "1.0", "backend"),
        ("sglang", 7, "backend_version"),
    ),
)
def test_preflight_rejects_blank_or_untyped_backend_identity_before_route_lookup(
    backend,
    backend_version,
    detail: str,
) -> None:
    route_calls = []

    with pytest.raises(CapabilityPreflightError, match=detail):
        preflight_capabilities(
            (OperationCapability("gemm", OperationKind.MEASURED, "gemm/v1"),),
            backend=backend,
            backend_version=backend_version,
            routes_for=lambda identity: route_calls.append(identity) or ("route",),
        )

    assert route_calls == []
