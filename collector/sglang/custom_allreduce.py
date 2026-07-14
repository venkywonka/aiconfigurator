# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-tree aliases for the package-owned CustomAllReduce runner."""

from __future__ import annotations

from aiconfigurator.collector.sglang.custom_allreduce import (
    get_custom_allreduce_test_cases,
    run_custom_allreduce_case,
)

__all__ = ["get_custom_allreduce_test_cases", "run_custom_allreduce_case"]
