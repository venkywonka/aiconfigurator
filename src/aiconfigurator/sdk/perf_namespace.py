# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical persisted namespace construction without collector imports."""


def perf_namespace(perf_filename: str, schema_revision: int = 1) -> str:
    if not isinstance(perf_filename, str) or not perf_filename.strip():
        raise ValueError("perf_filename must be a non-empty string")
    if isinstance(schema_revision, bool) or not isinstance(schema_revision, int) or schema_revision <= 0:
        raise ValueError("schema_revision must be a positive integer")
    return f"{perf_filename}/v{schema_revision}"


__all__ = ["perf_namespace"]
