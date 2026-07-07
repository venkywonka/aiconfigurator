# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hardware-aware placement for lazy collection jobs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import combinations

from aiconfigurator.collector.types import (
    Assignment,
    CollectionJob,
    FabricRequirement,
    HardwareInventory,
)

__all__ = [
    "Assignment",
    "CollectionJob",
    "HardwareAwareScheduler",
    "UnschedulableRequest",
]

_NVLINK_TOKEN = re.compile(r"NV[1-9][0-9]*\Z")


class UnschedulableRequest(RuntimeError):  # noqa: N818 - public contract uses this name
    """Raised when an inventory cannot satisfy a collection request."""

    def __init__(self, request_digest: str, detail: str) -> None:
        self.request_digest = request_digest
        super().__init__(f"{request_digest}: {detail}")


class HardwareAwareScheduler:
    """Pack collection jobs into deterministic, resource-safe waves."""

    def __init__(self, inventory: HardwareInventory) -> None:
        self.inventory = inventory
        self._gpu_ids = tuple(sorted(device.index for device in inventory.devices))

    def plan(self, jobs: Sequence[CollectionJob]) -> tuple[tuple[Assignment, ...], ...]:
        """Greedily pack jobs after applying a deterministic priority order."""
        pending = sorted(
            jobs,
            key=lambda job: (-job.contract.gpu_count, job.request_digest),
        )
        waves: list[tuple[Assignment, ...]] = []

        while pending:
            used_gpus: set[int] = set()
            occupied_fabric_domains: set[str] = set()
            exclusive_fabric_domains: set[str] = set()
            assignments: list[Assignment] = []
            deferred: list[CollectionJob] = []

            for job in pending:
                assignment = self._place(
                    job,
                    used_gpus,
                    occupied_fabric_domains,
                    exclusive_fabric_domains,
                )
                if assignment is None:
                    deferred.append(job)
                    continue

                assignments.append(assignment)
                used_gpus.update(assignment.gpu_ids)
                assignment_domains = self._fabric_domains(assignment.gpu_ids)
                occupied_fabric_domains.update(assignment_domains)
                if job.contract.reserve_fabric_domain and job.contract.gpu_count > 1:
                    exclusive_fabric_domains.update(assignment_domains)

            if not assignments:
                job = pending[0]
                raise UnschedulableRequest(
                    job.request_digest,
                    "no compatible GPU group",
                )

            waves.append(tuple(assignments))
            pending = deferred

        return tuple(waves)

    def _place(
        self,
        job: CollectionJob,
        used_gpus: set[int],
        occupied_fabric_domains: set[str],
        exclusive_fabric_domains: set[str],
    ) -> Assignment | None:
        for gpu_ids in combinations(self._gpu_ids, job.contract.gpu_count):
            if used_gpus.intersection(gpu_ids):
                continue
            if not self._supports_fabric(gpu_ids, job.contract.fabric):
                continue

            candidate_domains = self._fabric_domains(gpu_ids)
            if candidate_domains.intersection(exclusive_fabric_domains):
                continue

            reserved_fabric_domain: str | None = None
            if job.contract.reserve_fabric_domain and job.contract.gpu_count > 1:
                reserved_fabric_domain = self._shared_fabric_domain(gpu_ids)
                if reserved_fabric_domain is None:
                    continue
                if reserved_fabric_domain in occupied_fabric_domains:
                    continue

            reserved_domains = {f"gpu:{gpu_id}" for gpu_id in gpu_ids}
            if reserved_fabric_domain is not None:
                reserved_domains.add(f"fabric:{reserved_fabric_domain}")
            return Assignment(
                job=job,
                gpu_ids=gpu_ids,
                reserved_domains=frozenset(reserved_domains),
            )

        return None

    def _supports_fabric(
        self,
        gpu_ids: tuple[int, ...],
        requirement: FabricRequirement,
    ) -> bool:
        if requirement is FabricRequirement.NONE:
            return True

        for left, right in combinations(gpu_ids, 2):
            if not self.inventory.has_bidirectional_peer_access(left, right):
                return False
            if requirement is FabricRequirement.NVLINK and not all(
                _NVLINK_TOKEN.fullmatch(self.inventory.links.get(pair, "")) for pair in ((left, right), (right, left))
            ):
                return False
        return True

    def _fabric_domains(self, gpu_ids: tuple[int, ...]) -> frozenset[str]:
        return frozenset(
            domain for gpu_id in gpu_ids if (domain := self.inventory.fabric_domains.get(gpu_id)) is not None
        )

    def _shared_fabric_domain(self, gpu_ids: tuple[int, ...]) -> str | None:
        first_domain = self.inventory.fabric_domains.get(gpu_ids[0])
        if first_domain is None:
            return None
        if any(self.inventory.fabric_domains.get(gpu_id) != first_domain for gpu_id in gpu_ids[1:]):
            return None
        return first_domain
