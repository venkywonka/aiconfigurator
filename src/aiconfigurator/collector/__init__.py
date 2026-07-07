# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installable contracts for offline and on-demand performance collection."""

from aiconfigurator.collector.registry_types import OpEntry, PerfFile, VersionRoute
from aiconfigurator.collector.types import FabricRequirement, LazyOpEntry, ResourceContract

__all__ = [
    "FabricRequirement",
    "LazyOpEntry",
    "OpEntry",
    "PerfFile",
    "ResourceContract",
    "VersionRoute",
]
