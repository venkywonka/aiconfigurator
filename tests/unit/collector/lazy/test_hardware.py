# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware discovery contracts for lazy collector scheduling."""

from __future__ import annotations

import hashlib
import pickle
import subprocess
from dataclasses import replace
from textwrap import dedent

import pytest

from aiconfigurator.collector import GpuDevice, HardwareDiscoveryEvidence, HardwareInventory
from aiconfigurator.collector.hardware import (
    HardwareDiscoveryError,
    canonical_topology_fingerprint,
    discover_hardware,
    parse_gpu_query,
    parse_p2p_matrix,
    parse_topology,
)

pytestmark = pytest.mark.unit

QUERY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,pci.bus_id",
    "--format=csv,noheader",
)
TOPOLOGY_COMMAND = ("nvidia-smi", "topo", "-m")
P2P_READ_COMMAND = ("nvidia-smi", "topo", "-p2p", "r")
P2P_WRITE_COMMAND = ("nvidia-smi", "topo", "-p2p", "w")

GPU_QUERY = dedent(
    """\
    0, GPU-a, NVIDIA H100 80GB HBM3, 00000000:1B:00.0
    1, GPU-b, NVIDIA H100 80GB HBM3, 00000000:43:00.0
    2, GPU-c, NVIDIA H100 80GB HBM3, 00000000:52:00.0
    3, GPU-d, NVIDIA H100 80GB HBM3, 00000000:7A:00.0
    """
)

TOPOLOGY = dedent(
    """\
            GPU0 GPU1 GPU2 GPU3 CPU Affinity NUMA Affinity
    GPU0    X    NV4  SYS  SYS  0-31         0
    GPU1    NV4  X    SYS  SYS  0-31         0
    GPU2    SYS  SYS  X    NV4  32-63        1
    GPU3    SYS  SYS  NV4  X    32-63        1
    """
)

# Exact container bytes captured by the bounded no-measurement OCI-HSG probe
# 4277198 on nvl72078-T01. Keep the SGR sequences: they are the regression.
GB200_ANSI_TOPOLOGY = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tNIC1\tNIC2\tNIC3\tNIC4\tNIC5\tCPU Affinity\t"
    "NUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV18\tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t0-35\t0\t\tN/A\n"
    "GPU1\tNV18\t X \tNV18\tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t0-35\t0\t\tN/A\n"
    "GPU2\tNV18\tNV18\t X \tNV18\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t\t1\t\tN/A\n"
    "GPU3\tNV18\tNV18\tNV18\t X \tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t\t1\t\tN/A\n"
    "NIC0\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\tNODE\tNODE\tNODE\t\t\t\t\n"
    "NIC1\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\tNODE\tNODE\t\t\t\t\n"
    "NIC2\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\tNODE\t\t\t\t\n"
    "NIC3\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\tNODE\t\t\t\t\n"
    "NIC4\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \tNODE\t\t\t\t\n"
    "NIC5\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\tNODE\t X \t\t\t\t\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)\n"
    "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node\n"
    "  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)\n"
    "  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)\n"
    "  PIX  = Connection traversing at most a single PCIe bridge\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
    "\n"
    "NIC Legend:\n"
    "\n"
    "  NIC0: mlx5_0\n"
    "  NIC1: mlx5_1\n"
    "  NIC2: mlx5_2\n"
    "  NIC3: mlx5_3\n"
    "  NIC4: mlx5_4\n"
    "  NIC5: mlx5_5\n"
    "\n"
)

P2P_OK_WITHIN_DOMAINS = dedent(
    """\
            GPU0 GPU1 GPU2 GPU3
    GPU0    X    OK   NS   NS
    GPU1    OK   X    NS   NS
    GPU2    NS   NS   X    OK
    GPU3    NS   NS   OK   X
    """
)

P2P_NONE = dedent(
    """\
            GPU0 GPU1 GPU2 GPU3
    GPU0    X    NS   NS   NS
    GPU1    NS   X    NS   NS
    GPU2    NS   NS   X    NS
    GPU3    NS   NS   NS   X
    """
)


class FakeRun:
    def __init__(self, results: dict[tuple[str, ...], str | BaseException]) -> None:
        self.results = results
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        command = tuple(args)
        self.calls.append((command, kwargs))
        result = self.results[command]
        if isinstance(result, BaseException):
            raise result
        return subprocess.CompletedProcess(args, 0, stdout=result, stderr="")


def _runner(
    *,
    query: str | BaseException = GPU_QUERY,
    topology: str | BaseException = TOPOLOGY,
    p2p_read: str | BaseException = P2P_OK_WITHIN_DOMAINS,
    p2p_write: str | BaseException = P2P_OK_WITHIN_DOMAINS,
) -> FakeRun:
    return FakeRun(
        {
            QUERY_COMMAND: query,
            TOPOLOGY_COMMAND: topology,
            P2P_READ_COMMAND: p2p_read,
            P2P_WRITE_COMMAND: p2p_write,
        }
    )


def test_gpu_query_normalizes_order_and_rejects_invalid_identity_rows() -> None:
    devices = parse_gpu_query("\n".join(reversed(GPU_QUERY.strip().splitlines())))
    assert tuple(device.index for device in devices) == (0, 1, 2, 3)
    assert devices[0].uuid == "GPU-a"
    assert devices[0].name == "NVIDIA H100 80GB HBM3"
    assert devices[0].pci_bus_id == "00000000:1B:00.0"

    invalid_queries = (
        "",
        "0, GPU-a, NVIDIA H100 80GB HBM3",
        "x, GPU-a, NVIDIA H100 80GB HBM3, 0000:01:00.0",
        "0, GPU-a, NVIDIA H100 80GB HBM3, 0000:01:00.0\n0, GPU-b, H100, 0000:02:00.0",
        "0, GPU-a, NVIDIA H100 80GB HBM3, 0000:01:00.0\n2, GPU-c, H100, 0000:03:00.0",
        "0, GPU-a, H100, 0000:01:00.0\n1, GPU-a, H100, 0000:02:00.0",
        "0, GPU-a, H100, 0000:01:00.0\n1, GPU-b, H100, 0000:01:00.0",
        "0, , NVIDIA H100 80GB HBM3, 0000:01:00.0",
        "0, GPU-a, , 0000:01:00.0",
        "0, GPU-a, NVIDIA H100 80GB HBM3, ",
    )
    for query in invalid_queries:
        with pytest.raises(HardwareDiscoveryError):
            parse_gpu_query(query)


def test_topology_normalizes_row_order_and_rejects_unsafe_structure() -> None:
    devices = parse_gpu_query(GPU_QUERY)
    lines = TOPOLOGY.strip().splitlines()
    permuted = "\n".join((lines[0], lines[3], lines[1], lines[4], lines[2]))
    links = parse_topology(permuted, devices)
    assert links[(0, 1)] == links[(1, 0)] == "NV4"
    assert links[(0, 2)] == links[(2, 0)] == "SYS"
    assert len(links) == 12

    invalid_topologies = (
        "",
        TOPOLOGY.replace("NV4", "FOO", 1),
        TOPOLOGY.replace("NV4", "NVX", 1),
        TOPOLOGY.replace("GPU0    X", "GPU0    OK", 1),
        TOPOLOGY.replace("GPU1    NV4", "GPU1    SYS", 1),
        "\n".join(TOPOLOGY.strip().splitlines()[:-1]),
        TOPOLOGY.replace("GPU3    SYS  SYS  NV4  X", "GPU2    SYS  SYS  NV4  X", 1),
    )
    for topology in invalid_topologies:
        with pytest.raises(HardwareDiscoveryError):
            parse_topology(topology, devices)

    two_devices = parse_gpu_query("0, GPU-a, NVIDIA H100, 0000:01:00.0\n1, GPU-b, NVIDIA H100, 0000:02:00.0")
    topology_with_nic_and_legend = dedent(
        """\
                GPU0 GPU1 NIC0 CPU Affinity NUMA Affinity GPU NUMA ID
        GPU0    X    NV4  NODE 0-31         0             N/A
        GPU1    NV4  X    NODE 0-31         0             N/A
        NIC0    NODE NODE X

        Legend:
          X    = Self
          NODE = Connection traversing PCIe and NUMA links
          NV#  = Bonded set of NVLinks
        """
    )
    assert parse_topology(topology_with_nic_and_legend, two_devices) == {(0, 1): "NV4", (1, 0): "NV4"}

    for token in ("PIX", "PXB", "PHB", "NODE", "SYS"):
        topology = f"GPU0 GPU1\nGPU0 X {token}\nGPU1 {token} X"
        assert parse_topology(topology, two_devices) == {(0, 1): token, (1, 0): token}


def test_topology_parser_accepts_exact_ansi_sgr_gb200_capture() -> None:
    assert hashlib.sha256(GB200_ANSI_TOPOLOGY.encode()).hexdigest() == (
        "cca8b1e86930da4776987fdd3fe8160e6e3093342c4c99a9a030cfd9c5b29999"
    )
    devices = parse_gpu_query(GPU_QUERY)
    links = parse_topology(GB200_ANSI_TOPOLOGY, devices)
    assert len(links) == 12
    assert set(links.values()) == {"NV18"}


def test_topology_parser_accepts_compound_sgr_but_not_other_csi_sequences() -> None:
    devices = parse_gpu_query(GPU_QUERY)
    compound_sgr = TOPOLOGY.replace("GPU0 GPU1 GPU2 GPU3", "\x1b[1;4mGPU0 GPU1 GPU2 GPU3\x1b[0m", 1)
    assert len(parse_topology(compound_sgr, devices)) == 12

    non_sgr_csi = TOPOLOGY.replace("GPU0 GPU1 GPU2 GPU3", "\x1b[2KGPU0 GPU1 GPU2 GPU3", 1)
    with pytest.raises(HardwareDiscoveryError, match="matrix columns"):
        parse_topology(non_sgr_csi, devices)


@pytest.mark.parametrize(
    "header",
    (
        "\x1b[4mGPU1 GPU2 GPU3\x1b[0m",
        "\x1b[4mGPU1 GPU0 GPU2 GPU3\x1b[0m",
        "\x1b[4mGPU0 GPU0 GPU2 GPU3\x1b[0m",
        "\x1b[4mGPX0 GPU1 GPU2 GPU3\x1b[0m",
    ),
)
def test_sgr_wrapped_unsafe_topology_headers_still_fail_closed(header: str) -> None:
    devices = parse_gpu_query(GPU_QUERY)
    topology = TOPOLOGY.replace("GPU0 GPU1 GPU2 GPU3", header, 1)
    with pytest.raises(HardwareDiscoveryError, match="matrix columns"):
        parse_topology(topology, devices)


@pytest.mark.parametrize(
    "topology",
    (
        GB200_ANSI_TOPOLOGY.replace("GPU0\t X ", "GPU0\t OK ", 1),
        GB200_ANSI_TOPOLOGY.replace("GPU0\t X \tNV18", "GPU0\t X \tFOO", 1),
        GB200_ANSI_TOPOLOGY.replace("GPU1\tNV18", "GPU1\tSYS", 1),
    ),
)
def test_sgr_wrapped_invalid_topology_payloads_still_fail_closed(topology: str) -> None:
    with pytest.raises(HardwareDiscoveryError):
        parse_topology(topology, parse_gpu_query(GPU_QUERY))


def test_p2p_parser_keeps_directionality_and_treats_only_ok_as_capable() -> None:
    devices = parse_gpu_query("0, GPU-a, NVIDIA H100, 0000:01:00.0\n1, GPU-b, NVIDIA H100, 0000:02:00.0")
    negative_statuses = ("CNS", "GNS", "TNS", "NS", "U", "DR")
    for status in negative_statuses:
        matrix = dedent(
            f"""\
                    GPU0 GPU1
            GPU0    X    OK
            GPU1    {status}  X
            """
        )
        capability = parse_p2p_matrix(matrix, devices)
        assert capability == {(0, 1): True, (1, 0): False}

    malformed = (
        "GPU0 GPU1\nGPU0 X OK",
        "GPU0 GPU2\nGPU0 X OK\nGPU1 OK X",
        "GPU0 GPU1\nGPU0 X OK\nGPU0 OK X",
        "GPU0 GPU1\nGPU0 X OK EXTRA\nGPU1 OK X EXTRA",
        "GPU0 GPU1\nGPU0 X X\nGPU1 OK X",
        "GPU0 GPU1\nGPU0 OK OK\nGPU1 OK X",
        "GPU0 GPU1\nGPU0 X WTF\nGPU1 OK X",
    )
    for matrix in malformed:
        with pytest.raises(HardwareDiscoveryError):
            parse_p2p_matrix(matrix, devices)

    with_legend = "GPU0 GPU1\nGPU0 X OK\nGPU1 OK X\n\nLegend:\n  OK = Supported\n  NS = Not supported"
    assert parse_p2p_matrix(with_legend, devices) == {(0, 1): True, (1, 0): True}

    sgr_matrix = P2P_OK_WITHIN_DOMAINS.replace("GPU0 GPU1 GPU2 GPU3", "\x1b[1;4mGPU0 GPU1 GPU2 GPU3\x1b[0m", 1)
    four_devices = parse_gpu_query(GPU_QUERY)
    assert parse_p2p_matrix(sgr_matrix, four_devices) == parse_p2p_matrix(P2P_OK_WITHIN_DOMAINS, four_devices)
    with pytest.raises(HardwareDiscoveryError, match="invalid matrix row width"):
        parse_p2p_matrix(
            sgr_matrix.replace("GPU0    X    OK   NS   NS", "GPU0    X    OK   NS   NS EXTRA"), four_devices
        )


def test_discovery_requires_explicit_four_direction_peer_capability() -> None:
    no_peer = discover_hardware(run=_runner(p2p_read=P2P_NONE, p2p_write=P2P_NONE))
    assert no_peer.links[(0, 1)] == "NV4"
    assert not no_peer.has_bidirectional_peer_access(0, 1)

    read_only = P2P_OK_WITHIN_DOMAINS.replace("GPU1    OK", "GPU1    NS", 1)
    partial = discover_hardware(run=_runner(p2p_read=read_only))
    assert partial.p2p_read[(0, 1)]
    assert not partial.p2p_read[(1, 0)]
    assert not partial.has_bidirectional_peer_access(0, 1)

    read_forward_only = P2P_OK_WITHIN_DOMAINS.replace("GPU0    X    OK", "GPU0    X    NS", 1)
    partial_read_forward = discover_hardware(run=_runner(p2p_read=read_forward_only))
    assert not partial_read_forward.p2p_read[(0, 1)]
    assert partial_read_forward.p2p_read[(1, 0)]
    assert not partial_read_forward.has_bidirectional_peer_access(0, 1)

    write_only = P2P_OK_WITHIN_DOMAINS.replace("GPU1    OK", "GPU1    NS", 1)
    partial_write = discover_hardware(run=_runner(p2p_write=write_only))
    assert partial_write.p2p_write[(0, 1)]
    assert not partial_write.p2p_write[(1, 0)]
    assert not partial_write.has_bidirectional_peer_access(0, 1)

    write_forward_only = P2P_OK_WITHIN_DOMAINS.replace("GPU0    X    OK", "GPU0    X    NS", 1)
    partial_write_forward = discover_hardware(run=_runner(p2p_write=write_forward_only))
    assert not partial_write_forward.p2p_write[(0, 1)]
    assert partial_write_forward.p2p_write[(1, 0)]
    assert not partial_write_forward.has_bidirectional_peer_access(0, 1)

    capable = discover_hardware(run=_runner())
    assert capable.has_bidirectional_peer_access(0, 1)
    assert capable.has_bidirectional_peer_access(2, 3)
    assert not capable.has_bidirectional_peer_access(0, 2)
    assert capable.fabric_domains == {0: "nvlink:0", 1: "nvlink:0", 2: "nvlink:1", 3: "nvlink:1"}

    nvswitch_topology = TOPOLOGY.replace("NV4", "NV18").replace("SYS", "NV18")
    nvswitch = discover_hardware(run=_runner(topology=nvswitch_topology))
    assert nvswitch.fabric_domains == {0: "nvlink:0", 1: "nvlink:0", 2: "nvlink:0", 3: "nvlink:0"}
    assert nvswitch.topology_fingerprint != capable.topology_fingerprint

    pcie_topology = TOPOLOGY.replace("NV4", "SYS")
    explicit_peer = discover_hardware(run=_runner(topology=pcie_topology))
    assert explicit_peer.has_bidirectional_peer_access(0, 1)
    assert explicit_peer.fabric_domains == {}


def test_peer_probe_failure_fails_closed_but_required_discovery_fails_loudly() -> None:
    read_failure = subprocess.CalledProcessError(1, P2P_READ_COMMAND, output="", stderr="unsupported")
    inventory = discover_hardware(run=_runner(p2p_read=read_failure))
    assert not any(inventory.p2p_read.values())
    assert not any(inventory.p2p_write.values())
    assert inventory.evidence.p2p_errors

    malformed = discover_hardware(run=_runner(p2p_write="not a matrix"))
    assert not any(malformed.p2p_read.values())
    assert not any(malformed.p2p_write.values())
    assert malformed.evidence.p2p_errors

    unknown_status = discover_hardware(run=_runner(p2p_read=P2P_OK_WITHIN_DOMAINS.replace("NS", "WTF", 1)))
    assert not any(unknown_status.p2p_read.values())
    assert not any(unknown_status.p2p_write.values())
    assert unknown_status.evidence.p2p_errors

    timeout = subprocess.TimeoutExpired(
        P2P_READ_COMMAND,
        10.0,
        output=b"partial peer output",
        stderr=b"probe stalled",
    )
    timed_out = discover_hardware(run=_runner(p2p_read=timeout))
    assert not any(timed_out.p2p_read.values())
    assert not any(timed_out.p2p_write.values())
    assert timed_out.evidence.raw_p2p_read == "partial peer output"
    assert "timed out" in timed_out.evidence.p2p_errors[0]

    for required in ("query", "topology"):
        failure = subprocess.CalledProcessError(1, ("nvidia-smi",), output="", stderr="missing")
        runner = _runner(**{required: failure})
        with pytest.raises(HardwareDiscoveryError):
            discover_hardware(run=runner)
        timeout = subprocess.TimeoutExpired(("nvidia-smi",), 10.0, output="partial", stderr="stalled")
        runner = _runner(**{required: timeout})
        with pytest.raises(HardwareDiscoveryError, match="timed out"):
            discover_hardware(run=runner)

    one_gpu_query = "0, GPU-a, NVIDIA H100, 0000:01:00.0"
    one_gpu_topology = "GPU0\nGPU0 X"
    one_gpu = discover_hardware(
        run=_runner(
            query=one_gpu_query,
            topology=one_gpu_topology,
            p2p_read=read_failure,
            p2p_write=read_failure,
        )
    )
    assert tuple(device.index for device in one_gpu.devices) == (0,)


def test_fingerprint_tracks_capability_and_pair_structure_not_host_identity() -> None:
    devices = parse_gpu_query(GPU_QUERY)
    links = parse_topology(TOPOLOGY, devices)
    p2p_read = parse_p2p_matrix(P2P_OK_WITHIN_DOMAINS, devices)
    p2p_write = parse_p2p_matrix(P2P_OK_WITHIN_DOMAINS, devices)
    fingerprint = canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, links, p2p_read, p2p_write)

    changed_identity = tuple(
        replace(device, index=device.index + 10, uuid=f"foreign-{device.uuid}", pci_bus_id=f"bus-{device.index}")
        for device in devices
    )
    remap = {device.index: changed.index for device, changed in zip(devices, changed_identity, strict=True)}
    remapped_links = {(remap[left], remap[right]): token for (left, right), token in links.items()}
    remapped_read = {(remap[left], remap[right]): value for (left, right), value in p2p_read.items()}
    remapped_write = {(remap[left], remap[right]): value for (left, right), value in p2p_write.items()}
    assert (
        canonical_topology_fingerprint(
            "nvidia-smi-topology-v2", changed_identity, remapped_links, remapped_read, remapped_write
        )
        == fingerprint
    )

    assert canonical_topology_fingerprint("changed-schema", devices, links, p2p_read, p2p_write) != fingerprint
    changed_class = (replace(devices[0], name="NVIDIA B200"), *devices[1:])
    assert (
        canonical_topology_fingerprint("nvidia-smi-topology-v2", changed_class, links, p2p_read, p2p_write)
        != fingerprint
    )
    changed_link = dict(links)
    changed_link[(0, 1)] = changed_link[(1, 0)] = "SYS"
    assert (
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, changed_link, p2p_read, p2p_write)
        != fingerprint
    )
    for capability_name, direction in (
        ("read", (0, 1)),
        ("read", (1, 0)),
        ("write", (0, 1)),
        ("write", (1, 0)),
    ):
        changed_read = dict(p2p_read)
        changed_write = dict(p2p_write)
        target = changed_read if capability_name == "read" else changed_write
        target[direction] = False
        assert (
            canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, links, changed_read, changed_write)
            != fingerprint
        )

    cross_links = dict(links)
    for left, right in ((0, 1), (1, 0), (2, 3), (3, 2)):
        cross_links[(left, right)] = "SYS"
    for left, right in ((0, 2), (2, 0), (1, 3), (3, 1)):
        cross_links[(left, right)] = "NV4"
    assert (
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, cross_links, p2p_read, p2p_write)
        != fingerprint
    )

    heterogeneous = tuple(replace(device, name=f"GPU class {position}") for position, device in enumerate(devices))
    swapped_classes = (
        replace(heterogeneous[0], name=heterogeneous[1].name),
        replace(heterogeneous[1], name=heterogeneous[0].name),
        *heterogeneous[2:],
    )
    heterogeneous_fingerprint = canonical_topology_fingerprint(
        "nvidia-smi-topology-v2", heterogeneous, links, p2p_read, p2p_write
    )
    assert (
        canonical_topology_fingerprint("nvidia-smi-topology-v2", swapped_classes, links, p2p_read, p2p_write)
        != heterogeneous_fingerprint
    )

    with pytest.raises(ValueError, match="complete"):
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, links, {}, p2p_write)
    with pytest.raises(ValueError, match="complete"):
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, links, p2p_read, {})
    with pytest.raises(ValueError, match="complete"):
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, {}, p2p_read, p2p_write)
    invalid_bool = dict(p2p_read)
    invalid_bool[(0, 1)] = 1  # type: ignore[assignment]
    with pytest.raises(TypeError, match="bool"):
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, links, invalid_bool, p2p_write)
    unnormalized_links = dict(links)
    unnormalized_links[(0, 1)] = unnormalized_links[(1, 0)] = " nv4 "
    with pytest.raises(ValueError, match="normalized"):
        canonical_topology_fingerprint("nvidia-smi-topology-v2", devices, unnormalized_links, p2p_read, p2p_write)


def test_restrict_preserves_global_domains_evidence_and_immutable_subset_identity() -> None:
    inventory = discover_hardware(run=_runner())
    restored = pickle.loads(pickle.dumps(inventory))
    assert restored == inventory
    assert restored.evidence == inventory.evidence

    changed_evidence = replace(inventory.evidence, raw_gpu_query="different raw host output")
    same_identity = replace(inventory, evidence=changed_evidence)
    assert same_identity == inventory
    assert hash(same_identity) == hash(inventory)
    assert same_identity.topology_fingerprint == inventory.topology_fingerprint

    renamed_domains = replace(
        inventory,
        fabric_domains={
            gpu_id: "nvlink:7" if domain == "nvlink:0" else "nvlink:8"
            for gpu_id, domain in inventory.fabric_domains.items()
        },
    )
    assert renamed_domains.topology_fingerprint == inventory.topology_fingerprint

    with pytest.raises(ValueError, match="fabric_domains"):
        replace(inventory, fabric_domains={})
    with pytest.raises(ValueError, match="fabric_domains"):
        replace(inventory, fabric_domains={0: "garbage", 99: "nvlink:0"})
    with pytest.raises(ValueError, match="stable fabric_domains"):
        replace(inventory, fabric_domains={0: "nvlink:0", 2: "nvlink:0", 1: "nvlink:1", 3: "nvlink:1"})

    with pytest.raises(ValueError, match="complete"):
        replace(inventory, p2p_read={})
    with pytest.raises(ValueError, match="topology_fingerprint"):
        replace(inventory, topology_fingerprint="wrong")
    with pytest.raises(TypeError, match="devices must be a tuple"):
        replace(inventory, devices=list(inventory.devices))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        HardwareDiscoveryEvidence()
    with pytest.raises(TypeError):
        HardwareInventory(
            schema_revision=inventory.schema_revision,
            devices=inventory.devices,
            links=inventory.links,
            p2p_read=inventory.p2p_read,
            p2p_write=inventory.p2p_write,
            fabric_domains=inventory.fabric_domains,
            topology_fingerprint=inventory.topology_fingerprint,
        )

    subset = inventory.restrict((2, 3))
    assert tuple(device.index for device in subset.devices) == (2, 3)
    assert set(subset.links) == {(2, 3), (3, 2)}
    assert set(subset.p2p_read) == {(2, 3), (3, 2)}
    assert set(subset.p2p_write) == {(2, 3), (3, 2)}
    assert subset.fabric_domains == {2: "nvlink:1", 3: "nvlink:1"}
    assert subset.evidence is inventory.evidence
    assert subset.topology_fingerprint != inventory.topology_fingerprint
    assert inventory.restrict((2,)).fabric_domains == {}

    chain_devices = tuple(GpuDevice(index, f"GPU-{index}", "NVIDIA H100", f"bus-{index}") for index in range(5))
    chain_links = {
        (left, right): "NV4" if abs(left - right) == 1 else "SYS"
        for left in range(5)
        for right in range(5)
        if left != right
    }
    chain_capability = dict.fromkeys(chain_links, False)
    chain_fingerprint = canonical_topology_fingerprint(
        "nvidia-smi-topology-v2",
        chain_devices,
        chain_links,
        chain_capability,
        chain_capability,
    )
    chain = HardwareInventory(
        schema_revision="nvidia-smi-topology-v2",
        devices=chain_devices,
        links=chain_links,
        p2p_read=chain_capability,
        p2p_write=chain_capability,
        fabric_domains=dict.fromkeys(range(5), "nvlink:0"),
        topology_fingerprint=chain_fingerprint,
        evidence=inventory.evidence,
    )
    assert set(chain.restrict((0, 1, 3, 4)).fabric_domains.values()) == {"nvlink:0"}
    assert chain.restrict((0, 2)).fabric_domains == {0: "nvlink:0", 2: "nvlink:0"}

    with pytest.raises(ValueError, match="empty"):
        inventory.restrict(())
    with pytest.raises(ValueError, match="unique"):
        inventory.restrict((2, 2))
    with pytest.raises(ValueError, match="not present"):
        inventory.restrict((8,))
    for immutable_mapping in (subset.links, subset.p2p_read, subset.p2p_write, subset.fabric_domains):
        with pytest.raises(TypeError):
            immutable_mapping[next(iter(immutable_mapping))] = "invalid"  # type: ignore[index]


def test_discovery_invokes_exact_argument_arrays_without_a_shell() -> None:
    runner = _runner()
    inventory = discover_hardware(run=runner)
    assert inventory.evidence.raw_gpu_query == GPU_QUERY
    assert inventory.evidence.raw_topology == TOPOLOGY
    assert inventory.evidence.raw_p2p_read == P2P_OK_WITHIN_DOMAINS
    assert inventory.evidence.raw_p2p_write == P2P_OK_WITHIN_DOMAINS
    assert [command for command, _ in runner.calls] == [
        QUERY_COMMAND,
        TOPOLOGY_COMMAND,
        P2P_READ_COMMAND,
        P2P_WRITE_COMMAND,
    ]
    for _, kwargs in runner.calls:
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 10.0}
