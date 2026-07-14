# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover normalized GPU topology and explicit peer-access capability."""

from __future__ import annotations

import csv
import io
import re
import subprocess
from collections.abc import Callable, Mapping

from aiconfigurator.collector.types import (
    _NON_NVLINK_TOKENS,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    _derive_nvlink_domains,
    _is_nvlink_token,
    canonical_topology_fingerprint,
)

_SCHEMA_REVISION = "nvidia-smi-topology-v2"
_GPU_QUERY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,pci.bus_id",
    "--format=csv,noheader",
)
_TOPOLOGY_COMMAND = ("nvidia-smi", "topo", "-m")
_P2P_READ_COMMAND = ("nvidia-smi", "topo", "-p2p", "r")
_P2P_WRITE_COMMAND = ("nvidia-smi", "topo", "-p2p", "w")
_PROBE_TIMEOUT_SECONDS = 10.0
_P2P_NEGATIVE_STATUSES = frozenset({"CNS", "GNS", "TNS", "NS", "U", "DR"})
_ANSI_SGR_SEQUENCE = re.compile(r"\x1b\[[0-9;]*m")


class HardwareDiscoveryError(RuntimeError):
    """Hardware discovery output is unavailable or structurally unsafe."""


def normalize_link_token(raw_token: str) -> str:
    """Normalize one documented physical topology token."""
    token = raw_token.strip().upper()
    if _is_nvlink_token(token) or token in _NON_NVLINK_TOKENS:
        return token
    raise HardwareDiscoveryError(f"unsupported topology link token {raw_token!r}")


def parse_gpu_query(text: str) -> tuple[GpuDevice, ...]:
    """Parse strict nvidia-smi CSV GPU identity output."""
    devices: list[GpuDevice] = []
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(field.strip() for field in row):
            continue
        fields = [field.strip() for field in row]
        if len(fields) != 4 or any(not field for field in fields):
            raise HardwareDiscoveryError(f"invalid GPU query row: {row!r}")
        try:
            index = int(fields[0])
        except ValueError as error:
            raise HardwareDiscoveryError(f"invalid GPU index {fields[0]!r}") from error
        devices.append(GpuDevice(index=index, uuid=fields[1], name=fields[2], pci_bus_id=fields[3]))

    devices.sort(key=lambda device: device.index)
    expected_indices = list(range(len(devices)))
    if not devices or [device.index for device in devices] != expected_indices:
        raise HardwareDiscoveryError("GPU query must contain unique contiguous indices starting at zero")
    if len({device.uuid for device in devices}) != len(devices):
        raise HardwareDiscoveryError("GPU query contains duplicate UUIDs")
    if len({device.pci_bus_id for device in devices}) != len(devices):
        raise HardwareDiscoveryError("GPU query contains duplicate PCI bus ids")
    return tuple(devices)


def _matrix_rows(
    text: str,
    devices: tuple[GpuDevice, ...],
    *,
    allow_trailing_fields: bool,
) -> dict[int, list[str]]:
    parsing_text = _ANSI_SGR_SEQUENCE.sub("", text)
    lines = [line for line in parsing_text.splitlines() if line.strip()]
    if not lines:
        raise HardwareDiscoveryError("empty nvidia-smi matrix")
    columns = [int(value) for value in re.findall(r"\bGPU([0-9]+)\b", lines[0])]
    expected = [device.index for device in devices]
    if columns != expected:
        raise HardwareDiscoveryError(f"matrix columns {columns} do not match GPUs {expected}")

    rows: dict[int, list[str]] = {}
    for line in lines[1:]:
        fields = line.split()
        match = re.fullmatch(r"GPU([0-9]+)", fields[0]) if fields else None
        if match is None:
            continue
        row_gpu = int(match.group(1))
        if row_gpu not in expected or row_gpu in rows:
            raise HardwareDiscoveryError(f"invalid or duplicate matrix row GPU{row_gpu}")
        row_fields = fields[1:]
        if len(row_fields) < len(columns) or (not allow_trailing_fields and len(row_fields) != len(columns)):
            raise HardwareDiscoveryError(f"invalid matrix row width for GPU{row_gpu}")
        tokens = row_fields[: len(columns)]
        rows[row_gpu] = tokens
    if set(rows) != set(expected):
        raise HardwareDiscoveryError("nvidia-smi matrix is incomplete")
    return rows


def parse_topology(text: str, devices: tuple[GpuDevice, ...]) -> Mapping[tuple[int, int], str]:
    """Parse a complete symmetric physical path matrix from ``topo -m``."""
    rows = _matrix_rows(text, devices, allow_trailing_fields=True)
    columns = [device.index for device in devices]
    links: dict[tuple[int, int], str] = {}
    for row_gpu, tokens in rows.items():
        for column_gpu, raw_token in zip(columns, tokens, strict=True):
            if row_gpu == column_gpu:
                if raw_token.upper() != "X":
                    raise HardwareDiscoveryError(f"invalid topology diagonal GPU{row_gpu}: {raw_token!r}")
                continue
            links[(row_gpu, column_gpu)] = normalize_link_token(raw_token)

    for (left, right), token in links.items():
        if links.get((right, left)) != token:
            raise HardwareDiscoveryError(f"asymmetric topology link GPU{left}/GPU{right}")
    return links


def parse_p2p_matrix(text: str, devices: tuple[GpuDevice, ...]) -> Mapping[tuple[int, int], bool]:
    """Parse one directed P2P capability matrix; only literal ``OK`` is usable."""
    rows = _matrix_rows(text, devices, allow_trailing_fields=False)
    columns = [device.index for device in devices]
    capability: dict[tuple[int, int], bool] = {}
    for row_gpu, tokens in rows.items():
        for column_gpu, raw_token in zip(columns, tokens, strict=True):
            token = raw_token.strip().upper()
            if row_gpu == column_gpu:
                if token != "X":
                    raise HardwareDiscoveryError(f"invalid P2P diagonal GPU{row_gpu}: {raw_token!r}")
                continue
            if token == "X":
                raise HardwareDiscoveryError(f"invalid off-diagonal P2P status GPU{row_gpu}/GPU{column_gpu}")
            if token != "OK" and token not in _P2P_NEGATIVE_STATUSES:
                raise HardwareDiscoveryError(f"unsupported P2P status GPU{row_gpu}/GPU{column_gpu}: {raw_token!r}")
            capability[(row_gpu, column_gpu)] = token == "OK"
    return capability


def _command_error(label: str, error: BaseException) -> HardwareDiscoveryError:
    if isinstance(error, subprocess.TimeoutExpired):
        return HardwareDiscoveryError(f"{label} timed out after {error.timeout} seconds")
    stderr = getattr(error, "stderr", None)
    detail = f": {stderr}" if stderr else ""
    return HardwareDiscoveryError(f"{label} failed{detail}")


def _run_required(
    run: Callable[..., subprocess.CompletedProcess[str]],
    command: tuple[str, ...],
    label: str,
) -> str:
    try:
        completed = run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise _command_error(label, error) from error
    return completed.stdout


def _run_p2p_probe(
    run: Callable[..., subprocess.CompletedProcess[str]],
    command: tuple[str, ...],
    label: str,
    devices: tuple[GpuDevice, ...],
) -> tuple[str, Mapping[tuple[int, int], bool] | None, str | None]:
    try:
        completed = run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raw_output = getattr(error, "output", "") or ""
        if isinstance(raw_output, bytes):
            raw_output = raw_output.decode("utf-8", errors="replace")
        return str(raw_output), None, str(_command_error(label, error))
    try:
        return completed.stdout, parse_p2p_matrix(completed.stdout, devices), None
    except HardwareDiscoveryError as error:
        return completed.stdout, None, f"{label} parse failed: {error}"


def discover_hardware(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> HardwareInventory:
    """Probe locally visible GPUs; the caller guarantees they equal its assigned pool."""
    raw_query = _run_required(run, _GPU_QUERY_COMMAND, "GPU query")
    devices = parse_gpu_query(raw_query)
    raw_topology = _run_required(run, _TOPOLOGY_COMMAND, "GPU topology query")
    links = parse_topology(raw_topology, devices)

    raw_read, p2p_read, read_error = _run_p2p_probe(run, _P2P_READ_COMMAND, "P2P read query", devices)
    raw_write, p2p_write, write_error = _run_p2p_probe(run, _P2P_WRITE_COMMAND, "P2P write query", devices)
    p2p_errors = tuple(error for error in (read_error, write_error) if error is not None)
    if p2p_errors:
        all_pairs = {(left.index, right.index): False for left in devices for right in devices if left != right}
        p2p_read = all_pairs
        p2p_write = dict(all_pairs)

    assert p2p_read is not None
    assert p2p_write is not None
    fabric_domains = _derive_nvlink_domains(tuple(device.index for device in devices), links)
    evidence = HardwareDiscoveryEvidence(
        raw_gpu_query=raw_query,
        raw_topology=raw_topology,
        raw_p2p_read=raw_read,
        raw_p2p_write=raw_write,
        p2p_errors=p2p_errors,
    )
    return HardwareInventory(
        schema_revision=_SCHEMA_REVISION,
        devices=devices,
        links=links,
        p2p_read=p2p_read,
        p2p_write=p2p_write,
        fabric_domains=fabric_domains,
        topology_fingerprint=canonical_topology_fingerprint(
            _SCHEMA_REVISION,
            devices,
            links,
            p2p_read,
            p2p_write,
        ),
        evidence=evidence,
    )
