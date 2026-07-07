# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load packaged collector contracts from installed or source-only environments."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType


def _is_colocated_source(package: ModuleType, source_root: Path) -> bool:
    package_file = getattr(package, "__file__", None)
    if package_file is None:
        return False
    try:
        Path(package_file).resolve().relative_to(source_root.resolve())
    except ValueError:
        return False
    return True


def load_canonical(module_name: str) -> ModuleType:
    """Load the canonical module paired with these source-tree shims."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    loaded_package = sys.modules.get("aiconfigurator")
    if loaded_package is not None:
        if not _is_colocated_source(loaded_package, source_root):
            raise ImportError("aiconfigurator is already loaded from another location")
        return import_module(module_name)

    original_path = sys.path.copy()
    sys.path.insert(0, str(source_root))
    try:
        return import_module(module_name)
    finally:
        sys.path[:] = original_path
