# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aiconfigurator")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+source"
