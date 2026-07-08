# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installable contracts for offline and on-demand performance collection."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile, VersionRoute
from aiconfigurator.collector.types import (
    Assignment,
    CollectionJob,
    FabricRequirement,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    LazyOpEntry,
    RawMeasurement,
    ResourceContract,
    canonical_topology_fingerprint,
)

__all__ = [
    "Assignment",
    "CollectionJob",
    "FabricRequirement",
    "GpuDevice",
    "HardwareDiscoveryEvidence",
    "HardwareInventory",
    "LazyOpEntry",
    "OpEntry",
    "PerfFile",
    "RawMeasurement",
    "ResourceContract",
    "VersionRoute",
    "canonical_topology_fingerprint",
]
