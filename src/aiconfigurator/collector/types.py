# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata contracts for on-demand collector adapters."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Generic, TypeVar

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _FrozenMapping(Mapping[_Key, _Value], Generic[_Key, _Value]):
    """Small pickle-safe immutable copy of a mapping."""

    __slots__ = ("_data", "_hash")

    def __init__(self, value: Mapping[_Key, _Value]) -> None:
        self._data = dict(value)
        self._hash: int | None = None

    def __getitem__(self, key: _Key) -> _Value:
        return self._data[key]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __reduce__(self) -> tuple[type[_FrozenMapping], tuple[dict[_Key, _Value]]]:
        return _FrozenMapping, (self._data,)


class FabricRequirement(str, Enum):
    """GPU interconnect capability required by one measurement."""

    NONE = "none"
    P2P = "p2p"
    NVLINK = "nvlink"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ResourceContract:
    """Resources that must be leased together for one measurement."""

    gpu_count: int
    fabric: FabricRequirement
    exclusive_devices: bool = True
    reserve_fabric_domain: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.gpu_count, bool) or not isinstance(self.gpu_count, int):
            raise TypeError("gpu_count must be an integer")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be a positive integer")
        if not isinstance(self.fabric, FabricRequirement):
            raise TypeError("fabric must be a FabricRequirement")
        if not isinstance(self.exclusive_devices, bool):
            raise TypeError("exclusive_devices must be a bool")
        if not isinstance(self.reserve_fabric_domain, bool):
            raise TypeError("reserve_fabric_domain must be a bool")
        if self.gpu_count == 1 and self.fabric is not FabricRequirement.NONE:
            raise ValueError("single-GPU work cannot require a GPU fabric")


@dataclass(frozen=True, slots=True)
class CollectionJob:
    """One deduplicated measurement request ready for resource placement."""

    request_digest: str
    adapter_namespace: str
    contract: ResourceContract
    payload: bytes


@dataclass(frozen=True, slots=True)
class Assignment:
    """A collection job bound to physical GPUs and reservation tokens."""

    job: CollectionJob
    gpu_ids: tuple[int, ...]
    reserved_domains: frozenset[str]


@dataclass(frozen=True, slots=True)
class LazyOpEntry:
    """Optional adapter metadata for exact on-demand measurements."""

    namespace: str
    run_module: str
    run_func: str
    adapter_module: str
    case_func: str
    result_func: str
    resource_func: str
    protocol_revision: str
    timer: str
    tuning_revision: str

    def __post_init__(self) -> None:
        for metadata_field in fields(self):
            value = getattr(self, metadata_field.name)
            if not isinstance(value, str):
                raise TypeError(f"{metadata_field.name} must be a string")
            if not value or value != value.strip():
                raise ValueError(f"{metadata_field.name} must be non-empty and trimmed")


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """One GPU visible to the collector coordinator."""

    index: int
    uuid: str
    name: str
    pci_bus_id: str


@dataclass(frozen=True, slots=True)
class HardwareDiscoveryEvidence:
    """Raw hardware probe output retained for measurement provenance."""

    raw_gpu_query: str
    raw_topology: str
    raw_p2p_read: str
    raw_p2p_write: str
    p2p_errors: tuple[str, ...] = ()


_NVLINK_TOKEN = re.compile(r"NV[1-9][0-9]*\Z")
_NON_NVLINK_TOKENS = frozenset({"PIX", "PXB", "PHB", "NODE", "SYS"})
_FABRIC_DOMAIN = re.compile(r"nvlink:[0-9]+\Z")


def _is_nvlink_token(token: str) -> bool:
    return _NVLINK_TOKEN.fullmatch(token) is not None


def _is_normalized_link_token(token: str) -> bool:
    return _is_nvlink_token(token) or token in _NON_NVLINK_TOKENS


def _derive_nvlink_domains(
    device_ids: tuple[int, ...],
    links: Mapping[tuple[int, int], str],
) -> dict[int, str]:
    remaining = set(device_ids)
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(
                peer for peer in remaining if peer != current and _is_nvlink_token(links.get((current, peer), ""))
            )
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))

    return {
        gpu_id: f"nvlink:{component_index}"
        for component_index, component in enumerate(components)
        if len(component) > 1
        for gpu_id in component
    }


def canonical_topology_fingerprint(
    schema_revision: str,
    devices: tuple[GpuDevice, ...],
    links: Mapping[tuple[int, int], str],
    p2p_read: Mapping[tuple[int, int], bool],
    p2p_write: Mapping[tuple[int, int], bool],
) -> str:
    """Hash normalized capability while excluding host-local identifiers."""
    ordered = tuple(sorted(devices, key=lambda device: device.index))
    local_index = {device.index: position for position, device in enumerate(ordered)}
    expected_pairs = {(left.index, right.index) for left in ordered for right in ordered if left.index != right.index}
    if set(links) != expected_pairs:
        raise ValueError("topology links must be complete for every directed GPU pair")
    for field_name, capability in (("p2p_read", p2p_read), ("p2p_write", p2p_write)):
        if set(capability) != expected_pairs:
            raise ValueError(f"{field_name} must be complete for every directed GPU pair")
        if any(type(value) is not bool for value in capability.values()):
            raise TypeError(f"{field_name} values must be bool")
    if any(not isinstance(token, str) for token in links.values()):
        raise TypeError("topology link values must be strings")
    if any(not _is_normalized_link_token(token) for token in links.values()):
        raise ValueError("topology link values must be normalized NVIDIA path tokens")

    edges: list[list[object]] = []
    for left_position, left_device in enumerate(ordered):
        for right_device in ordered[left_position + 1 :]:
            left = left_device.index
            right = right_device.index
            forward = links[(left, right)]
            reverse = links[(right, left)]
            if forward != reverse:
                raise ValueError(f"asymmetric topology link GPU{left}/GPU{right}")
            edges.append(
                [
                    local_index[left],
                    local_index[right],
                    forward,
                    p2p_read[(left, right)],
                    p2p_read[(right, left)],
                    p2p_write[(left, right)],
                    p2p_write[(right, left)],
                ]
            )
    payload = {
        "schema_revision": schema_revision,
        "gpu_classes": [" ".join(device.name.split()).casefold() for device in ordered],
        "edges": edges,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    """Normalized GPU topology, peer capability, and full-host fabric contention labels.

    Fabric-domain membership is a reservation identity, never proof that a
    selected GPU subset is connected. Placement must inspect ``links`` and all
    four directed peer-access bits independently.
    """

    schema_revision: str
    devices: tuple[GpuDevice, ...]
    links: Mapping[tuple[int, int], str]
    p2p_read: Mapping[tuple[int, int], bool]
    p2p_write: Mapping[tuple[int, int], bool]
    fabric_domains: Mapping[int, str]
    topology_fingerprint: str
    evidence: HardwareDiscoveryEvidence = field(
        compare=False,
        hash=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple):
            raise TypeError("hardware inventory devices must be a tuple")
        if any(not isinstance(device, GpuDevice) for device in self.devices):
            raise TypeError("hardware inventory devices must contain only GpuDevice values")
        device_ids = tuple(device.index for device in self.devices)
        if not device_ids or len(set(device_ids)) != len(device_ids):
            raise ValueError("hardware inventory GPU ids must be non-empty and unique")
        if not isinstance(self.evidence, HardwareDiscoveryEvidence):
            raise TypeError("hardware inventory evidence must be HardwareDiscoveryEvidence")
        expected_fingerprint = canonical_topology_fingerprint(
            self.schema_revision,
            self.devices,
            self.links,
            self.p2p_read,
            self.p2p_write,
        )
        if self.topology_fingerprint != expected_fingerprint:
            raise ValueError("topology_fingerprint does not match normalized inventory")
        device_id_set = set(device_ids)
        if any(gpu_id not in device_id_set for gpu_id in self.fabric_domains):
            raise ValueError("fabric_domains contains a GPU not present in the inventory")
        if any(
            not isinstance(domain, str) or _FABRIC_DOMAIN.fullmatch(domain) is None
            for domain in self.fabric_domains.values()
        ):
            raise ValueError("fabric_domains labels must match 'nvlink:<number>'")
        domain_sizes = {
            domain: sum(candidate == domain for candidate in self.fabric_domains.values())
            for domain in set(self.fabric_domains.values())
        }
        if any(size < 2 for size in domain_sizes.values()):
            raise ValueError("fabric_domains must not contain singleton domains")
        for (left, right), token in self.links.items():
            if _is_nvlink_token(token) and (
                self.fabric_domains.get(left) is None or self.fabric_domains.get(left) != self.fabric_domains.get(right)
            ):
                raise ValueError("every NVLink edge must belong to one stable fabric_domains label")
        object.__setattr__(self, "links", _FrozenMapping(self.links))
        object.__setattr__(self, "p2p_read", _FrozenMapping(self.p2p_read))
        object.__setattr__(self, "p2p_write", _FrozenMapping(self.p2p_write))
        object.__setattr__(self, "fabric_domains", _FrozenMapping(self.fabric_domains))

    def has_bidirectional_peer_access(self, left: int, right: int) -> bool:
        """Return true only when read and write are supported both ways."""
        return all(
            (
                self.p2p_read.get((left, right), False),
                self.p2p_read.get((right, left), False),
                self.p2p_write.get((left, right), False),
                self.p2p_write.get((right, left), False),
            )
        )

    def restrict(self, gpu_ids: tuple[int, ...]) -> HardwareInventory:
        """Return a subset while retaining its stable full-host contention labels."""
        if not gpu_ids:
            raise ValueError("assigned GPU ids must not be empty")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("assigned GPU ids must be unique")
        by_id = {device.index: device for device in self.devices}
        try:
            devices = tuple(by_id[gpu_id] for gpu_id in gpu_ids)
        except KeyError as error:
            raise ValueError(f"assigned GPU id is not present: {error.args[0]}") from error

        allowed = set(gpu_ids)
        links = {pair: value for pair, value in self.links.items() if pair[0] in allowed and pair[1] in allowed}
        p2p_read = {pair: value for pair, value in self.p2p_read.items() if pair[0] in allowed and pair[1] in allowed}
        p2p_write = {pair: value for pair, value in self.p2p_write.items() if pair[0] in allowed and pair[1] in allowed}
        domain_counts = {
            domain: sum(self.fabric_domains.get(gpu_id) == domain for gpu_id in gpu_ids)
            for domain in set(self.fabric_domains.values())
        }
        fabric_domains = {
            gpu_id: domain
            for gpu_id in gpu_ids
            if (domain := self.fabric_domains.get(gpu_id)) is not None and domain_counts[domain] > 1
        }
        return HardwareInventory(
            schema_revision=self.schema_revision,
            devices=devices,
            links=links,
            p2p_read=p2p_read,
            p2p_write=p2p_write,
            fabric_domains=fabric_domains,
            topology_fingerprint=canonical_topology_fingerprint(
                self.schema_revision,
                devices,
                links,
                p2p_read,
                p2p_write,
            ),
            evidence=self.evidence,
        )
