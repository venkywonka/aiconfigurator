# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata contracts for on-demand collector adapters."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


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
