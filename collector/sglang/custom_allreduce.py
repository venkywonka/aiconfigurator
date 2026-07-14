# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-tree compatibility surface for exact CustomAllReduce collection."""

from __future__ import annotations

from typing import Any


def get_custom_allreduce_test_cases() -> tuple[()]:
    """Expose the package runner's deliberately empty offline grid."""

    from aiconfigurator.collector.sglang.custom_allreduce import get_custom_allreduce_test_cases as get_cases

    return get_cases()


def run_custom_allreduce_case(*args: Any, **kwargs: Any) -> Any:
    """Delegate an exact case to the package-owned import-light runner."""

    from aiconfigurator.collector.sglang.custom_allreduce import run_custom_allreduce_case as run_case

    return run_case(*args, **kwargs)


__all__ = ["get_custom_allreduce_test_cases", "run_custom_allreduce_case"]
