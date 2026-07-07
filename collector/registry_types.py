# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for source-tree collector scripts."""

if __package__:
    from ._canonical_import import load_canonical
else:
    from _canonical_import import load_canonical

_canonical = load_canonical("aiconfigurator.collector.registry_types")

OpEntry = _canonical.OpEntry
PerfFile = _canonical.PerfFile
VersionRoute = _canonical.VersionRoute

__all__ = ["OpEntry", "PerfFile", "VersionRoute"]
