# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hardware-aware wave scheduling contracts."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from aiconfigurator.collector import (
    FabricRequirement,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    ResourceContract,
    canonical_topology_fingerprint,
)
from aiconfigurator.collector.scheduler import (
    Assignment,
    CollectionJob,
    HardwareAwareScheduler,
    UnschedulableRequest,
)

pytestmark = pytest.mark.unit

_EVIDENCE = HardwareDiscoveryEvidence(
    raw_gpu_query="synthetic query",
    raw_topology="synthetic topology",
    raw_p2p_read="synthetic read matrix",
    raw_p2p_write="synthetic write matrix",
)


def _inventory(
    *,
    gpu_ids: tuple[int, ...],
    nvlink_pairs: tuple[tuple[int, int], ...],
    p2p_read_directions: tuple[tuple[int, int], ...],
    p2p_write_directions: tuple[tuple[int, int], ...],
    fabric_domains: Mapping[int, str],
) -> HardwareInventory:
    devices = tuple(
        GpuDevice(
            index=gpu_id,
            uuid=f"GPU-{gpu_id}",
            name="NVIDIA H100 80GB HBM3",
            pci_bus_id=f"00000000:{gpu_id:02X}:00.0",
        )
        for gpu_id in gpu_ids
    )
    normalized_nvlink_pairs = {tuple(sorted(pair)) for pair in nvlink_pairs}
    links = {
        (left, right): "NV4" if tuple(sorted((left, right))) in normalized_nvlink_pairs else "SYS"
        for left in gpu_ids
        for right in gpu_ids
        if left != right
    }
    read_directions = set(p2p_read_directions)
    write_directions = set(p2p_write_directions)
    p2p_read = {pair: pair in read_directions for pair in links}
    p2p_write = {pair: pair in write_directions for pair in links}
    fingerprint = canonical_topology_fingerprint(
        "scheduler-test-v1",
        devices,
        links,
        p2p_read,
        p2p_write,
    )
    return HardwareInventory(
        schema_revision="scheduler-test-v1",
        devices=devices,
        links=links,
        p2p_read=p2p_read,
        p2p_write=p2p_write,
        fabric_domains=fabric_domains,
        topology_fingerprint=fingerprint,
        evidence=_EVIDENCE,
    )


def _two_domain_inventory() -> HardwareInventory:
    directions = ((0, 1), (1, 0), (2, 3), (3, 2))
    return _inventory(
        gpu_ids=(0, 1, 2, 3),
        nvlink_pairs=((0, 1), (2, 3)),
        p2p_read_directions=directions,
        p2p_write_directions=directions,
        fabric_domains={
            0: "nvlink:0",
            1: "nvlink:0",
            2: "nvlink:1",
            3: "nvlink:1",
        },
    )


def _one_domain_clique() -> HardwareInventory:
    nvlink_pairs = tuple((left, right) for left in range(4) for right in range(left + 1, 4))
    directions = tuple((left, right) for left in range(4) for right in range(4) if left != right)
    return _inventory(
        gpu_ids=(0, 1, 2, 3),
        nvlink_pairs=nvlink_pairs,
        p2p_read_directions=directions,
        p2p_write_directions=directions,
        fabric_domains=dict.fromkeys(range(4), "nvlink:0"),
    )


def _job(
    request_digest: str,
    gpu_count: int,
    fabric: FabricRequirement = FabricRequirement.NONE,
    *,
    reserve_fabric_domain: bool = False,
) -> CollectionJob:
    return CollectionJob(
        request_digest=request_digest,
        adapter_namespace="scheduler-test",
        contract=ResourceContract(
            gpu_count=gpu_count,
            fabric=fabric,
            reserve_fabric_domain=reserve_fabric_domain,
        ),
        payload=request_digest.encode("utf-8"),
    )


def _shape(
    waves: tuple[tuple[Assignment, ...], ...],
) -> tuple[tuple[tuple[str, tuple[int, ...], frozenset[str]], ...], ...]:
    return tuple(
        tuple(
            (
                assignment.job.request_digest,
                assignment.gpu_ids,
                assignment.reserved_domains,
            )
            for assignment in wave
        )
        for wave in waves
    )


def test_none_jobs_pack_deterministically_and_always_reserve_exact_gpu_tokens() -> None:
    inventory = _inventory(
        gpu_ids=(0, 1, 2, 3),
        nvlink_pairs=(),
        p2p_read_directions=(),
        p2p_write_directions=(),
        fabric_domains={},
    )
    scheduler = HardwareAwareScheduler(inventory)

    singles = tuple(_job(digest, 1) for digest in ("d", "b", "a", "c"))
    expected_singles = (
        (
            ("a", (0,), frozenset({"gpu:0"})),
            ("b", (1,), frozenset({"gpu:1"})),
            ("c", (2,), frozenset({"gpu:2"})),
            ("d", (3,), frozenset({"gpu:3"})),
        ),
    )
    assert _shape(scheduler.plan(singles)) == expected_singles
    assert _shape(scheduler.plan(tuple(reversed(singles)))) == expected_singles

    mixed = (_job("single-b", 1), _job("pair-z", 2), _job("single-a", 1))
    expected_mixed = (
        (
            ("pair-z", (0, 1), frozenset({"gpu:0", "gpu:1"})),
            ("single-a", (2,), frozenset({"gpu:2"})),
            ("single-b", (3,), frozenset({"gpu:3"})),
        ),
    )
    assert _shape(scheduler.plan(mixed)) == expected_mixed
    assert _shape(scheduler.plan(tuple(reversed(mixed)))) == expected_mixed


def test_single_gpu_jobs_never_reserve_the_shared_stable_fabric_domain() -> None:
    scheduler = HardwareAwareScheduler(_one_domain_clique())

    ordinary_jobs = (_job("b-plain", 1), _job("a-plain", 1))
    assert _shape(scheduler.plan(ordinary_jobs)) == (
        (
            ("a-plain", (0,), frozenset({"gpu:0"})),
            ("b-plain", (1,), frozenset({"gpu:1"})),
        ),
    )

    explicitly_reserving = (
        _job("b-plain", 1),
        _job("a-reserving", 1, reserve_fabric_domain=True),
    )
    assert _shape(scheduler.plan(explicitly_reserving)) == (
        (
            ("a-reserving", (0,), frozenset({"gpu:0"})),
            ("b-plain", (1,), frozenset({"gpu:1"})),
        ),
    )


def test_p2p_and_nvlink_gates_preserve_physical_gpu_and_stable_domain_ids() -> None:
    p2p_directions = ((0, 2), (2, 0))
    p2p_inventory = _inventory(
        gpu_ids=(0, 1, 2),
        nvlink_pairs=(),
        p2p_read_directions=p2p_directions,
        p2p_write_directions=p2p_directions,
        fabric_domains={},
    )
    p2p_assignment = HardwareAwareScheduler(p2p_inventory).plan((_job("p2p", 2, FabricRequirement.P2P),))[0][0]
    assert p2p_assignment.gpu_ids == (0, 2)
    assert p2p_assignment.reserved_domains == frozenset({"gpu:0", "gpu:2"})
    assert p2p_inventory.links[(0, 2)] == "SYS"

    restricted = _two_domain_inventory().restrict((2, 3))
    nvlink_assignment = HardwareAwareScheduler(restricted).plan(
        (
            _job(
                "nvlink",
                2,
                FabricRequirement.NVLINK,
                reserve_fabric_domain=True,
            ),
        )
    )[0][0]
    assert nvlink_assignment.gpu_ids == (2, 3)
    assert nvlink_assignment.reserved_domains == frozenset({"gpu:2", "gpu:3", "fabric:nvlink:1"})


def test_three_gpu_p2p_requires_every_selected_pair() -> None:
    all_directions = tuple((left, right) for left in range(3) for right in range(3) if left != right)
    complete = _inventory(
        gpu_ids=(0, 1, 2),
        nvlink_pairs=(),
        p2p_read_directions=all_directions,
        p2p_write_directions=all_directions,
        fabric_domains={},
    )
    assignment = HardwareAwareScheduler(complete).plan((_job("complete-three-gpu-p2p", 3, FabricRequirement.P2P),))[0][
        0
    ]
    assert assignment.gpu_ids == (0, 1, 2)
    assert assignment.reserved_domains == frozenset({"gpu:0", "gpu:1", "gpu:2"})

    missing_pair_directions = tuple(pair for pair in all_directions if frozenset(pair) != frozenset({1, 2}))
    incomplete = _inventory(
        gpu_ids=(0, 1, 2),
        nvlink_pairs=(),
        p2p_read_directions=missing_pair_directions,
        p2p_write_directions=missing_pair_directions,
        fabric_domains={},
    )
    job = _job("missing-three-gpu-p2p-pair", 3, FabricRequirement.P2P)
    with pytest.raises(UnschedulableRequest, match=job.request_digest) as caught:
        HardwareAwareScheduler(incomplete).plan((job,))
    assert caught.value.request_digest == job.request_digest


def test_fabric_reservation_requires_one_shared_stable_domain() -> None:
    peer_directions = ((0, 1), (1, 0))
    no_domain = _inventory(
        gpu_ids=(0, 1),
        nvlink_pairs=(),
        p2p_read_directions=peer_directions,
        p2p_write_directions=peer_directions,
        fabric_domains={},
    )
    split_domain_directions = ((0, 2), (2, 0))
    split_domains = _inventory(
        gpu_ids=(0, 1, 2, 3),
        nvlink_pairs=(),
        p2p_read_directions=split_domain_directions,
        p2p_write_directions=split_domain_directions,
        fabric_domains={
            0: "nvlink:0",
            1: "nvlink:0",
            2: "nvlink:1",
            3: "nvlink:1",
        },
    )

    for inventory, digest in (
        (no_domain, "missing-stable-domain"),
        (split_domains, "split-stable-domains"),
    ):
        job = _job(
            digest,
            2,
            FabricRequirement.P2P,
            reserve_fabric_domain=True,
        )
        with pytest.raises(UnschedulableRequest, match=digest) as caught:
            HardwareAwareScheduler(inventory).plan((job,))
        assert caught.value.request_digest == digest


def test_pairwise_gate_failures_raise_the_unschedulable_request_digest() -> None:
    missing_write_direction = _inventory(
        gpu_ids=(0, 1),
        nvlink_pairs=((0, 1),),
        p2p_read_directions=((0, 1), (1, 0)),
        p2p_write_directions=((0, 1),),
        fabric_domains={0: "nvlink:0", 1: "nvlink:0"},
    )
    pcie_only_peer = _inventory(
        gpu_ids=(0, 1),
        nvlink_pairs=(),
        p2p_read_directions=((0, 1), (1, 0)),
        p2p_write_directions=((0, 1), (1, 0)),
        fabric_domains={},
    )
    cases = (
        (
            missing_write_direction,
            _job("missing-write", 2, FabricRequirement.P2P),
        ),
        (
            pcie_only_peer,
            _job("missing-nvlink", 2, FabricRequirement.NVLINK),
        ),
        (
            _two_domain_inventory(),
            _job(
                "missing-three-gpu-clique",
                3,
                FabricRequirement.NVLINK,
                reserve_fabric_domain=True,
            ),
        ),
    )
    for inventory, job in cases:
        with pytest.raises(UnschedulableRequest, match=job.request_digest) as caught:
            HardwareAwareScheduler(inventory).plan((job,))
        assert caught.value.request_digest == job.request_digest


def test_collectives_pack_disjoint_domains_and_defer_the_third_collective() -> None:
    scheduler = HardwareAwareScheduler(_two_domain_inventory())
    jobs = tuple(
        _job(
            digest,
            2,
            FabricRequirement.NVLINK,
            reserve_fabric_domain=True,
        )
        for digest in ("c", "a", "b")
    )
    expected = (
        (
            (
                "a",
                (0, 1),
                frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"}),
            ),
            (
                "b",
                (2, 3),
                frozenset({"gpu:2", "gpu:3", "fabric:nvlink:1"}),
            ),
        ),
        (
            (
                "c",
                (0, 1),
                frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"}),
            ),
        ),
    )
    assert _shape(scheduler.plan(jobs)) == expected
    assert _shape(scheduler.plan(tuple(reversed(jobs)))) == expected


def test_frozen_profile_four_gpu_collective_isolates_single_gpu_jobs() -> None:
    scheduler = HardwareAwareScheduler(_one_domain_clique())
    jobs = (
        _job("b-single", 1),
        _job(
            "world4-collective",
            4,
            FabricRequirement.NVLINK,
            reserve_fabric_domain=True,
        ),
        _job("a-single", 1),
    )
    expected = (
        (
            (
                "world4-collective",
                (0, 1, 2, 3),
                frozenset(
                    {
                        "gpu:0",
                        "gpu:1",
                        "gpu:2",
                        "gpu:3",
                        "fabric:nvlink:0",
                    }
                ),
            ),
        ),
        (
            ("a-single", (0,), frozenset({"gpu:0"})),
            ("b-single", (1,), frozenset({"gpu:1"})),
        ),
    )

    assert _shape(scheduler.plan(jobs)) == expected
    assert _shape(scheduler.plan(tuple(reversed(jobs)))) == expected


def test_fabric_reservation_excludes_every_occupant_of_the_stable_domain() -> None:
    scheduler = HardwareAwareScheduler(_one_domain_clique())

    reserving_first = (
        _job(
            "a-reserving",
            2,
            FabricRequirement.NVLINK,
            reserve_fabric_domain=True,
        ),
        _job("b-plain", 2),
    )
    assert _shape(scheduler.plan(tuple(reversed(reserving_first)))) == (
        (
            (
                "a-reserving",
                (0, 1),
                frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"}),
            ),
        ),
        (("b-plain", (0, 1), frozenset({"gpu:0", "gpu:1"})),),
    )

    plain_first = (
        _job("a-plain", 2),
        _job(
            "b-reserving",
            2,
            FabricRequirement.NVLINK,
            reserve_fabric_domain=True,
        ),
    )
    assert _shape(scheduler.plan(tuple(reversed(plain_first)))) == (
        (("a-plain", (0, 1), frozenset({"gpu:0", "gpu:1"})),),
        (
            (
                "b-reserving",
                (0, 1),
                frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"}),
            ),
        ),
    )

    outside_domain = (
        _job(
            "a-collective",
            2,
            FabricRequirement.NVLINK,
            reserve_fabric_domain=True,
        ),
        _job("b-compute", 1),
    )
    assert _shape(HardwareAwareScheduler(_two_domain_inventory()).plan(outside_domain)) == (
        (
            (
                "a-collective",
                (0, 1),
                frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"}),
            ),
            ("b-compute", (2,), frozenset({"gpu:2"})),
        ),
    )
