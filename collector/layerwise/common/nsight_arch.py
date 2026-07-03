# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-specific Nsight Systems layout policy."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

_LAYOUTS = {
    "x86_64": ("target-linux-x64", "host-linux-x64"),
    "aarch64": ("target-linux-sbsa-armv8", "host-linux-armv8"),
}


class NsightArchitectureError(ValueError):
    """The requested or observed node architecture is not safe to use."""


class NsightInstallationError(RuntimeError):
    """The selected Nsight installation is incomplete for its architecture."""


def nsight_layout(architecture: str) -> tuple[str, str]:
    """Return ``(target, importer)`` directory names for an exact architecture."""

    try:
        return _LAYOUTS[architecture]
    except KeyError as error:
        raise NsightArchitectureError(f"unsupported Nsight architecture {architecture!r}") from error


def validate_nsight_architecture(expected: str, observed: str) -> tuple[str, str]:
    """Require the registry architecture to match the allocated node exactly."""

    layout = nsight_layout(expected)
    nsight_layout(observed)
    if expected != observed:
        raise NsightArchitectureError(f"expected architecture {expected!r} but observed {observed!r}")
    return layout


def validate_nsight_install(root: Path, architecture: str) -> tuple[Path, Path, Path]:
    """Validate the target, importer, and executable below an Nsight root."""

    target_name, importer_name = nsight_layout(architecture)
    target = root / target_name
    importer = root / importer_name
    binary = root / "bin" / "nsys"
    if not target.is_dir():
        raise NsightInstallationError(f"Nsight target directory missing: {target}")
    if not importer.is_dir():
        raise NsightInstallationError(f"Nsight importer directory missing: {importer}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise NsightInstallationError(f"Nsight nsys executable missing: {binary}")
    return target, importer, binary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--validate-install", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate architecture and print target/importer directory names, one per line."""

    args = _parser().parse_args(argv)
    try:
        layout = validate_nsight_architecture(args.expected, args.observed)
        if args.validate_install:
            if args.root is None:
                raise NsightInstallationError("Nsight root is required for install validation")
            validate_nsight_install(args.root, args.expected)
    except (NsightArchitectureError, NsightInstallationError) as error:
        print(error, file=sys.stderr)
        return 2
    print(*layout, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
