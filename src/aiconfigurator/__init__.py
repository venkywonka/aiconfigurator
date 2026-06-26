# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

try:
    __version__ = importlib.metadata.version("aiconfigurator")
except importlib.metadata.PackageNotFoundError:
    # Source checkout without an installed dist (e.g. PYTHONPATH=src on a fresh box,
    # where an editable install is blocked by the build backend's missing PEP 660
    # hook). Fall back to a local sentinel so `import aiconfigurator` succeeds for
    # tooling that only needs the package importable, not its precise version.
    __version__ = "0.0.0+local"
