# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for source-tree collector scripts."""

if __package__:
    from ._canonical_import import load_canonical
else:
    from _canonical_import import load_canonical

_canonical = load_canonical("aiconfigurator.collector.version_resolver")

_check_compat = _canonical._check_compat
_normalize_version = _canonical._normalize_version
_parse_compat_specifier = _canonical._parse_compat_specifier
_strip_local_metadata = _canonical._strip_local_metadata
OpEntry = _canonical.OpEntry
build_collections = _canonical.build_collections
resolve_module = _canonical.resolve_module

__all__ = [
    "OpEntry",
    "_check_compat",
    "_normalize_version",
    "_parse_compat_specifier",
    "_strip_local_metadata",
    "build_collections",
    "resolve_module",
]
