# AIC Hardware-Aware Lazy Collector Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve exact AIC perf misses through the existing collector functions while safely saturating independent GPUs and reserving whole GPU/fabric groups for collectives.

**Architecture:** Installable runtime and pilot adapter code lives under `aiconfigurator.collector`; the repository's top-level `collector/` scripts remain offline/CLI compatibility wrappers and are not published as a generic site-packages namespace. Existing registry entries reference the same namespaced adapter specifications and one-case implementations. A deterministic planner packs single-GPU and collective jobs into non-conflicting waves from live GPU/topology inventory. Long-lived subprocess workers own CUDA runtimes; collective workers additionally own persistent rank processes and NCCL communicators. Correlated messages and fail-closed worker eviction prevent stale replies after timeout. The executor returns records only—the SDK resolution session remains the sole overlay writer.

**Tech Stack:** Python 3.10+, dataclasses, `multiprocessing` with `spawn`, `concurrent.futures`, PyTorch CUDA/distributed, `nvidia-smi`, pytest.

---

## Source prerequisites

Read these first:

- `docs/plans/2026-07-06-dynamic-lazy-perf-collection-design.md`
- `docs/superpowers/plans/2026-07-06-aic-lazy-perf-core.md`

Complete the lazy core plan before this one. Execute this plan in the AIC repository on a branch containing `upstream/main` commit `0828d6b7e4a7880079443b1c6f9c148d85bdbf54` plus the completed core series. All unit tests in Tasks 1–5 use fake collectors and fake hardware; Tasks 6–7 add opt-in GPU tests for the two pilot adapters.

### Task 0: Verify source, namespace, and wheel baselines

**Files:** no changes

- [ ] **Step 1: Verify the reviewed source anchors**

Run:

```bash
git merge-base --is-ancestor 0828d6b7e4a7880079443b1c6f9c148d85bdbf54 HEAD
rg -n "class OpEntry|def build_collections|def benchmark_with_power|def run_gemm|def nccl_benchmark" \
  collector/registry_types.py collector/version_resolver.py collector/helper.py \
  collector/trtllm/collect_gemm.py collector/network/collect_nccl.py
```

Expected: the ancestor check succeeds and every offline source anchor exists. If a newer base moved one, update the delegation steps before editing.

- [ ] **Step 2: Prove the initial wheel does not publish top-level collector**

Build the unmodified wheel and inspect it. Record the exact commit and wheel listing in the implementation PR. Assert `aiconfigurator` imports successfully and `import collector` does not resolve from the wheel-only environment. This is the namespace baseline; the feature must add `aiconfigurator.collector` without publishing a generic top-level package.

## File map

- Create `src/aiconfigurator/collector/__init__.py` — namespaced runtime exports.
- Create `src/aiconfigurator/collector/types.py` — lazy adapter and resource declarations shared with offline registries.
- Create `src/aiconfigurator/collector/registry_types.py` — canonical packaged `PerfFile`, `VersionRoute`, and lazy-aware `OpEntry`.
- Create `src/aiconfigurator/collector/version_resolver.py` — canonical packaged version routing.
- Modify `collector/registry_types.py` — source-checkout compatibility re-export.
- Modify `collector/version_resolver.py` — source-checkout compatibility re-export.
- Create `src/aiconfigurator/collector/hardware.py` — GPU and topology discovery.
- Create `src/aiconfigurator/collector/scheduler.py` — deterministic conflict-aware wave packing.
- Create `src/aiconfigurator/collector/adapters.py` — registry lookup and dynamic function loading.
- Create `src/aiconfigurator/collector/executor.py` — persistent worker ownership and `MeasurementExecutor` implementation.
- Modify `collector/helper.py` — optional per-sample latency reporting without changing existing callers.
- Modify `collector/trtllm/collect_gemm.py` — delegate the exact-case path to the namespaced implementation.
- Create `src/aiconfigurator/collector/trtllm/gemm.py` — packaged exact-case implementation.
- Create `src/aiconfigurator/collector/trtllm/gemm_adapter.py` — lightweight request/result/resource mapping; no CUDA imports.
- Create `src/aiconfigurator/collector/trtllm/registry.py` — packaged GEMM lazy registration/specification.
- Modify `collector/trtllm/registry.py` — register the GEMM lazy adapter.
- Modify `src/aiconfigurator/sdk/operations/gemm.py` — construct exact GEMM requests.
- Modify `collector/network/collect_nccl.py` — delegate the exact-case path to the namespaced implementation.
- Create `src/aiconfigurator/collector/network/nccl.py` — packaged one-case API plus persistent-runtime hook.
- Create `src/aiconfigurator/collector/network/nccl_adapter.py` — lightweight NCCL request/result/resource mapping.
- Create `src/aiconfigurator/collector/network/registry.py` — packaged NCCL lazy registration.
- Modify `src/aiconfigurator/sdk/operations/communication.py` — construct exact NCCL requests.
- Create `tests/unit/collector/lazy/test_registry.py`.
- Create `tests/unit/collector/lazy/test_hardware.py`.
- Create `tests/unit/collector/lazy/test_scheduler.py`.
- Create `tests/unit/collector/lazy/test_adapters.py`.
- Create `tests/unit/collector/lazy/test_executor.py`.
- Create `tests/unit/collector/lazy/test_benchmark_samples.py`.
- Create `tests/unit/collector/lazy/test_import_surface.py`.
- Create `tests/integration/collector/test_lazy_gemm_gpu.py`.
- Create `tests/integration/collector/test_lazy_nccl_gpu.py`.

### Task 1: Add optional lazy metadata without changing offline collection

**Files:**
- Create: `src/aiconfigurator/collector/__init__.py`
- Create: `src/aiconfigurator/collector/types.py`
- Create: `src/aiconfigurator/collector/registry_types.py`
- Create: `src/aiconfigurator/collector/version_resolver.py`
- Modify: `collector/registry_types.py` (compatibility re-export)
- Modify: `collector/version_resolver.py` (compatibility re-export)
- Modify: `tests/unit/collector/test_version_resolver.py`
- Create: `tests/unit/collector/lazy/test_registry.py`

- [ ] **Step 1: Write backward-compatibility and validation tests**

```python
import pytest

from aiconfigurator.collector.types import FabricRequirement, LazyOpEntry, ResourceContract
from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.version_resolver import build_collections

pytestmark = pytest.mark.unit


def test_offline_collection_dict_is_unchanged_when_lazy_adapter_exists() -> None:
    entry = OpEntry(
        op="gemm",
        module="collector.fake",
        get_func="all_cases",
        run_func="run_case",
        perf_filename=PerfFile.GEMM,
        lazy=LazyOpEntry(
            namespace="trtllm/gemm/v1",
            run_module="aiconfigurator.collector.fake_runner",
            run_func="run_case",
            adapter_module="aiconfigurator.collector.fake_adapter",
            case_func="request_to_case",
            result_func="result_to_record",
            resource_func="resource_for_request",
            protocol_revision="cuda-event-v1",
            timer="cuda_event",
            tuning_revision="fake-v1",
        ),
    )
    assert build_collections([entry], "trtllm", "1.2.0") == [
        {
            "name": "trtllm",
            "type": "gemm",
            "module": "collector.fake",
            "get_func": "all_cases",
            "run_func": "run_case",
            "perf_filename": PerfFile.GEMM,
        }
    ]


def test_resource_contract_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="gpu_count"):
        ResourceContract(gpu_count=0, fabric=FabricRequirement.NONE)


def test_legacy_registry_imports_are_identity_reexports() -> None:
    from collector.registry_types import OpEntry as LegacyOpEntry
    from collector.version_resolver import build_collections as legacy_build_collections

    assert LegacyOpEntry is OpEntry
    assert legacy_build_collections is build_collections
```

- [ ] **Step 2: Run the tests and verify the missing lazy module failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py -v`

Expected: collection fails because `aiconfigurator.collector.types` does not exist.

- [ ] **Step 3: Define the shared lazy types**

```python
# src/aiconfigurator/collector/types.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class FabricRequirement(StrEnum):
    NONE = "none"
    P2P = "p2p"
    NVLINK = "nvlink"


@dataclass(frozen=True, slots=True)
class ResourceContract:
    gpu_count: int
    fabric: FabricRequirement
    exclusive_devices: bool = True
    reserve_fabric_domain: bool = False

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be at least one")
        if self.gpu_count == 1 and self.fabric is not FabricRequirement.NONE:
            raise ValueError("single-GPU work cannot require a GPU fabric")


@dataclass(frozen=True, slots=True)
class LazyOpEntry:
    namespace: str
    run_module: str
    run_func: str
    adapter_module: str
    case_func: str
    result_func: str
    resource_func: str
    protocol_revision: str
    timer: str
    tuning_revision: str


@dataclass(frozen=True, slots=True)
class CaseInvocation:
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawMeasurement:
    latency_ms: float
    energy_wms: float
    samples_ms: tuple[float, ...]
    statistic: str
    perf_row: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    uuid: str
    name: str
    pci_bus_id: str


_NVLINK_TOKEN = re.compile(r"NV[1-9][0-9]*\Z")


def _is_nvlink_token(token: str) -> bool:
    return _NVLINK_TOKEN.fullmatch(token) is not None


def _derive_nvlink_domains(
    device_ids: tuple[int, ...],
    links: Mapping[tuple[int, int], str],
) -> dict[int, str]:
    remaining = set(device_ids)
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(
                peer
                for peer in remaining
                if peer != current and _is_nvlink_token(links[(current, peer)])
            )
        remaining.difference_update(component)
        components.append(tuple(sorted(component)))
    return {
        gpu: f"nvlink:{component_index}"
        for component_index, component in enumerate(components)
        if len(component) > 1
        for gpu in component
    }


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    schema_revision: str
    devices: tuple[GpuDevice, ...]
    links: Mapping[tuple[int, int], str]
    fabric_domains: Mapping[int, str]
    topology_fingerprint: str

    def restrict(self, gpu_ids: tuple[int, ...]) -> "HardwareInventory":
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("assigned GPU ids must be unique")
        by_id = {device.index: device for device in self.devices}
        try:
            devices = tuple(by_id[gpu_id] for gpu_id in gpu_ids)
        except KeyError as error:
            raise ValueError(f"assigned GPU id is not present: {error.args[0]}") from error
        allowed = set(gpu_ids)
        links = {pair: link for pair, link in self.links.items() if set(pair) <= allowed}
        fabric_domains = _derive_nvlink_domains(tuple(sorted(allowed)), links)
        return HardwareInventory(
            schema_revision=self.schema_revision,
            devices=devices,
            links=links,
            fabric_domains=fabric_domains,
            topology_fingerprint=canonical_topology_fingerprint(
                self.schema_revision,
                devices,
                links,
            ),
        )


@dataclass(frozen=True, slots=True)
class CollectionJob:
    request_digest: str
    adapter_namespace: str
    contract: ResourceContract
    payload: bytes


@dataclass(frozen=True, slots=True)
class Assignment:
    job: CollectionJob
    gpu_ids: tuple[int, ...]
    reserved_domains: frozenset[str]


@dataclass(frozen=True, slots=True)
class WorkerLeaseKey:
    run_module: str
    adapter_namespace: str
    protocol_digest: str
    gpu_uuids: tuple[str, ...]
    topology_fingerprint: str
```

Export these names from `aiconfigurator.collector.__init__`. `canonical_topology_fingerprint` sorts device classes and symmetric links, excludes physical UUID/index from compatibility, and includes the parser schema revision plus normalized link tokens.

- [ ] **Step 4: Move shared registry contracts under the packaged namespace**

Move the current `PerfFile`, `VersionRoute`, `OpEntry`, `resolve_module`, and `build_collections` implementations into `aiconfigurator.collector.registry_types` and `.version_resolver` without behavior changes. In canonical `OpEntry`, import `LazyOpEntry` under `TYPE_CHECKING`, then add this field after `versions` so every existing positional constructor remains valid:

```python
lazy: LazyOpEntry | None = None
```

Make the top-level `collector/registry_types.py` and `collector/version_resolver.py` narrow compatibility re-exports with explicit `__all__`; source-only offline registries and scripts must receive the identical class/function objects. Do not add lazy fields to `build_collections()` output. Add a version-routed test proving `resolve_module()` selects the same module and retains `entry.lazy` on the original immutable entry. Run the existing full version-resolver/collector registry tests through both import paths.

- [ ] **Step 5: Run and commit the registry changes**

Run: `pytest -m unit tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py -v`

Expected: all tests pass.

```bash
git add src/aiconfigurator/collector collector/registry_types.py collector/version_resolver.py tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py
git commit -m "feat: declare optional lazy collector adapters"
```

### Task 2: Discover GPUs, links, and contention domains

**Files:**
- Create: `src/aiconfigurator/collector/hardware.py`
- Test: `tests/unit/collector/lazy/test_hardware.py`

- [ ] **Step 1: Write parser tests from fixed command output**

Use this four-GPU fixture:

```text
0, GPU-a, NVIDIA H100 80GB HBM3, 00000000:1B:00.0
1, GPU-b, NVIDIA H100 80GB HBM3, 00000000:43:00.0
2, GPU-c, NVIDIA H100 80GB HBM3, 00000000:52:00.0
3, GPU-d, NVIDIA H100 80GB HBM3, 00000000:7A:00.0
```

```text
        GPU0 GPU1 GPU2 GPU3 CPU Affinity
GPU0     X   NV4  SYS  SYS  0-31
GPU1    NV4   X   SYS  SYS  0-31
GPU2    SYS  SYS   X   NV4  32-63
GPU3    SYS  SYS  NV4   X   32-63
```

Assert device indices are `(0, 1, 2, 3)`, links are symmetric, fabric domains are `{0: "nvlink:0", 1: "nvlink:0", 2: "nvlink:1", 3: "nvlink:1"}`, and the topology fingerprint is stable across row/order changes. Add `inventory.restrict((2, 3))` and assert it preserves physical ids/links while excluding GPUs 0/1 and recomputes both fabric domains and the subset fingerprint; `inventory.restrict((2,))` must drop the former `nvlink:1` domain rather than retaining a singleton fabric tag. Duplicate or absent requested ids raise `ValueError`.

Add three more fixtures: a PCIe-only system where singleton GPUs have no `nvlink:*` domain, an NVSwitch/Blackwell-style matrix using the driver's documented bonded-link `NV#` tokens whose normalized graph produces one shared domain and a fingerprint distinct from the H100 fixture, and matrices containing `FOO` plus an unknown NV-prefixed token such as `NVX`, both of which raise `HardwareDiscoveryError`. Also add malformed-row and missing-GPU tests that fail rather than silently returning a partial inventory. Parser behavior changes require a new schema revision and therefore a new fingerprint.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_hardware.py -v`

Expected: import failure for `aiconfigurator.collector.hardware`.

- [ ] **Step 3: Implement discovery with injectable command execution**

Define these public functions:

```python
import csv
import io
import re
import subprocess
from collections.abc import Callable

from .types import GpuDevice, HardwareInventory, _derive_nvlink_domains, _is_nvlink_token


class HardwareDiscoveryError(RuntimeError):
    pass


_NON_NVLINK_TOKENS = frozenset({"PIX", "PXB", "PHB", "NODE", "SYS"})


def normalize_link_token(raw_token: str) -> str:
    token = raw_token.strip().upper()
    if _is_nvlink_token(token) or token in _NON_NVLINK_TOKENS:
        return token
    raise HardwareDiscoveryError(f"unsupported topology link token {raw_token!r}")


def parse_gpu_query(text: str) -> tuple[GpuDevice, ...]:
    devices: list[GpuDevice] = []
    for row in csv.reader(io.StringIO(text)):
        fields = [field.strip() for field in row]
        if len(fields) != 4:
            raise HardwareDiscoveryError(f"invalid GPU query row: {row!r}")
        try:
            index = int(fields[0])
        except ValueError as error:
            raise HardwareDiscoveryError(f"invalid GPU index {fields[0]!r}") from error
        devices.append(GpuDevice(index, fields[1], fields[2], fields[3]))
    devices.sort(key=lambda device: device.index)
    if not devices or [device.index for device in devices] != list(range(len(devices))):
        raise HardwareDiscoveryError("GPU query must contain contiguous indices starting at zero")
    return tuple(devices)


def parse_topology(text: str, devices: tuple[GpuDevice, ...]) -> HardwareInventory:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise HardwareDiscoveryError("empty nvidia-smi topology")
    columns = [int(value) for value in re.findall(r"GPU(\d+)", lines[0])]
    expected = [device.index for device in devices]
    if columns != expected:
        raise HardwareDiscoveryError(f"topology columns {columns} do not match GPUs {expected}")

    links: dict[tuple[int, int], str] = {}
    for line in lines[1:]:
        fields = line.split()
        if not fields or not fields[0].startswith("GPU"):
            continue
        row_gpu = int(fields[0][3:])
        tokens = fields[1 : 1 + len(columns)]
        if len(tokens) != len(columns):
            raise HardwareDiscoveryError(f"short topology row for GPU{row_gpu}")
        for column_gpu, raw_token in zip(columns, tokens, strict=True):
            if row_gpu != column_gpu:
                links[(row_gpu, column_gpu)] = normalize_link_token(raw_token)
    if len(links) != len(devices) * (len(devices) - 1):
        raise HardwareDiscoveryError("topology matrix is incomplete")
    for (left, right), token in links.items():
        if links.get((right, left)) != token:
            raise HardwareDiscoveryError(f"asymmetric topology link GPU{left}/GPU{right}")

    fabric_domains = _derive_nvlink_domains(tuple(expected), links)
    schema_revision = "nvidia-smi-topology-v1"
    return HardwareInventory(
        schema_revision=schema_revision,
        devices=devices,
        links=links,
        fabric_domains=fabric_domains,
        topology_fingerprint=canonical_topology_fingerprint(schema_revision, devices, links),
    )


def discover_hardware(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> HardwareInventory:
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    topology = run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_topology(topology.stdout, parse_gpu_query(query.stdout))
```

Recognize only the documented normalized link classes: bonded NVLink/NVSwitch links must match `NV[1-9][0-9]*` exactly, and non-NV classes are the exact set `PIX`, `PXB`, `PHB`, `NODE`, and `SYS`. Reject every other token—including arbitrary `NV*` strings—before inserting it into `links`. Build NVLink connected components only from validated `NV#` edges and assign an `nvlink:*` domain only when a component has at least two GPUs. Preserve all normalized edges in the fingerprint even when they do not form an NVLink domain, and retain the raw `nvidia-smi topo -m` output in collection/session provenance for auditability.

`discover_hardware()` must invoke argument arrays, never a shell string:

```python
[
    "nvidia-smi",
    "--query-gpu=index,uuid,name,pci.bus_id",
    "--format=csv,noheader",
]
["nvidia-smi", "topo", "-m"]
```

Treat recognized `NV#`/NVSwitch tokens as NVLink, `PIX`/`PXB`/`PHB` as P2P-capable, and `SYS`/`NODE` as non-P2P for initial placement. Build multi-GPU NVLink connected components in ascending GPU order and name them `nvlink:0`, `nvlink:1`, and so on. Preserve normalized and raw link tokens for provenance.

- [ ] **Step 4: Run and commit hardware discovery**

Run: `pytest -m unit tests/unit/collector/lazy/test_hardware.py -v`

Expected: all parser and failure tests pass without a GPU.

```bash
git add src/aiconfigurator/collector/hardware.py src/aiconfigurator/collector/types.py tests/unit/collector/lazy/test_hardware.py
git commit -m "feat: inventory GPU and fabric resources"
```

### Task 3: Pack work into deterministic non-conflicting waves

**Files:**
- Create: `src/aiconfigurator/collector/scheduler.py`
- Test: `tests/unit/collector/lazy/test_scheduler.py`

- [ ] **Step 1: Write placement tests**

With the Task 2 inventory, assert:

- four one-GPU jobs form one wave and receive GPUs 0, 1, 2, 3;
- two two-GPU NVLink collective jobs form one wave on `(0, 1)` and `(2, 3)` because their fabric domains are disjoint;
- a third collective forms a second wave;
- a collective reserving `nvlink:0` can share a wave with compute on GPU 2 but not GPU 0 or 1;
- a three-GPU NVLink request raises `UnschedulableRequest` with the request digest;
- input order changes do not change assignments because jobs sort by `(-gpu_count, request_digest)`.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_scheduler.py -v`

Expected: import failure for `aiconfigurator.collector.scheduler`.

- [ ] **Step 3: Implement greedy wave packing**

```python
class UnschedulableRequest(RuntimeError):
    def __init__(self, request_digest: str, detail: str) -> None:
        self.request_digest = request_digest
        super().__init__(f"{request_digest}: {detail}")


class HardwareAwareScheduler:
    def __init__(self, inventory: HardwareInventory) -> None:
        self.inventory = inventory

    def plan(self, jobs: Sequence[CollectionJob]) -> tuple[tuple[Assignment, ...], ...]:
        pending = sorted(jobs, key=lambda job: (-job.contract.gpu_count, job.request_digest))
        waves: list[tuple[Assignment, ...]] = []
        while pending:
            used_gpus: set[int] = set()
            used_domains: set[str] = set()
            assignments: list[Assignment] = []
            deferred: list[CollectionJob] = []
            for job in pending:
                assignment = self._place(job, used_gpus, used_domains)
                if assignment is None:
                    deferred.append(job)
                    continue
                assignments.append(assignment)
                used_gpus.update(assignment.gpu_ids)
                used_domains.update(assignment.reserved_domains)
            if not assignments:
                job = pending[0]
                raise UnschedulableRequest(job.request_digest, "no compatible GPU group")
            waves.append(tuple(assignments))
            pending = deferred
        return tuple(waves)
```

Implement `_place()` by enumerating ascending GPU combinations of exactly `gpu_count`. Reject used GPUs; enforce every pair for `P2P` or `NVLINK`; and, when `reserve_fabric_domain` is true, require one shared domain and reserve `fabric:<domain>`. A single-GPU job reserves only `gpu:<index>`.

- [ ] **Step 4: Run and commit the scheduler**

Run: `pytest -m unit tests/unit/collector/lazy/test_scheduler.py -v`

Expected: all scheduling tests pass.

```bash
git add src/aiconfigurator/collector/scheduler.py tests/unit/collector/lazy/test_scheduler.py
git commit -m "feat: schedule lazy collection across hardware domains"
```

### Task 4: Resolve registry adapters and validate record round trips

**Files:**
- Create: `src/aiconfigurator/collector/adapters.py`
- Test: `tests/unit/collector/lazy/test_adapters.py`

- [ ] **Step 1: Write fake-module adapter tests**

Create a synthetic module in `sys.modules` with `request_to_case`, `run_case`, `result_to_record`, and `resource_for_request`. Assert `LazyAdapterRegistry.resolve(request)`:

- selects by exact `PerfKey.namespace`;
- applies existing `resolve_module(entry, runtime_version)` routing;
- rejects a request protocol revision, timer method, or tuning revision that differs from `LazyOpEntry`;
- returns a `ResolvedLazyAdapter` containing lightweight mapping callables, the heavy run-function name, and the original `perf_filename`;
- raises `MissingLazyAdapter` for an unregistered namespace.
- rejects typed environment mismatches for system/GPU class, backend/runtime version, topology schema, and topology fingerprint before any worker starts.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_adapters.py -v`

Expected: import failure for `aiconfigurator.collector.adapters`.

- [ ] **Step 3: Implement exact namespace lookup and dynamic loading**

Import `OpEntry` and `resolve_module` only from `aiconfigurator.collector.registry_types` and `aiconfigurator.collector.version_resolver`. The installable runtime must have no import-time dependency on the repository-root `collector` compatibility package.

```python
@dataclass(frozen=True, slots=True)
class ResolvedLazyAdapter:
    namespace: str
    module_name: str
    adapter_module_name: str
    perf_filename: str
    case_func: Callable[[MeasurementRequest], CaseInvocation]
    run_func_name: str
    result_func: Callable[[MeasurementRequest, RawMeasurement], MeasurementRecord]
    resource_func: Callable[[MeasurementRequest], ResourceContract]
    protocol_revision: str
    timer: str
    tuning_revision: str


class LazyAdapterRegistry:
    def __init__(self, entries: Sequence[OpEntry], runtime_version: str) -> None:
        self._by_namespace: dict[str, ResolvedLazyAdapter] = {}
        for entry in entries:
            if entry.lazy is None:
                continue
            offline_module_name = resolve_module(entry, runtime_version)
            if offline_module_name is None:
                continue
            lazy = entry.lazy
            adapter_module = importlib.import_module(lazy.adapter_module)
            if lazy.namespace in self._by_namespace:
                raise ValueError(f"duplicate lazy namespace {lazy.namespace}")
            self._by_namespace[lazy.namespace] = ResolvedLazyAdapter(
                namespace=lazy.namespace,
                module_name=lazy.run_module,
                adapter_module_name=lazy.adapter_module,
                perf_filename=str(entry.perf_filename),
                case_func=getattr(adapter_module, lazy.case_func),
                run_func_name=lazy.run_func,
                result_func=getattr(adapter_module, lazy.result_func),
                resource_func=getattr(adapter_module, lazy.resource_func),
                protocol_revision=lazy.protocol_revision,
                timer=lazy.timer,
                tuning_revision=lazy.tuning_revision,
            )

    def resolve(self, request: MeasurementRequest) -> ResolvedLazyAdapter:
        adapter = self._by_namespace.get(request.key.namespace)
        if adapter is None:
            raise MissingLazyAdapter(request.key.namespace)
        return adapter
```

The existing `resolve_module(entry, runtime_version)` call remains a capability/version gate, but worker imports come from the namespaced `lazy.run_module`/`lazy.run_func`; an installed wheel never imports the generic top-level `collector` package. Before returning, compare `request.protocol.revision`, `request.protocol.timer`, and `request.protocol.tuning_revision` to the entry declaration retained on `ResolvedLazyAdapter`. Warmup and sample counts remain runtime policy inputs, but the emitted record must echo the complete request protocol exactly. Include expected capability values on the dataclass so mismatch errors are deterministic.

- [ ] **Step 4: Run and commit adapter loading**

Run: `pytest -m unit tests/unit/collector/lazy/test_adapters.py -v`

Expected: all tests pass.

```bash
git add src/aiconfigurator/collector/adapters.py tests/unit/collector/lazy/test_adapters.py
git commit -m "feat: resolve exact requests to collector functions"
```

### Task 5: Execute waves in persistent subprocess workers

**Files:**
- Create: `src/aiconfigurator/collector/executor.py`
- Test: `tests/unit/collector/lazy/test_executor.py`

- [ ] **Step 1: Write executor tests with fake worker channels**

Inject a `WorkerFactory` and assert:

- two one-GPU assignments in a wave are submitted before either result is awaited;
- the same `(run module, adapter namespace, protocol/tuning revision, GPU UUID tuple, topology fingerprint)` lease reuses one worker across two `execute()` calls, while any differing field creates a fresh lease;
- the next wave starts only after every assignment in the current wave returns;
- `close()` sends shutdown and joins every worker;
- an adapter exception becomes one `MeasurementRecord(status=FAILED)` with collector traceback provenance;
- parent-side hardware mismatch and unschedulable jobs become failed records without starting a worker;
- cancellation stops submitting later waves, drains or terminates active work according to the worker capability, and returns failed records for unstarted keys;
- a monotonic deadline terminates an overrun worker and returns a timeout record without blocking sibling records;
- successful results preserve request order even when workers finish out of order.
- every reply echoes a unique invocation id plus request digest; a deliberately delayed reply from a timed-out invocation can never satisfy a later request on a reused lease;
- timeout, cancellation, malformed reply, EOF, or child death terminates and evicts the affected worker, while every other outstanding assignment in that wave is either received and preserved or explicitly terminated and evicted before `execute()` returns.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_executor.py -v`

Expected: import failure for `aiconfigurator.collector.executor`.

- [ ] **Step 3: Implement the worker message protocol**

Use `multiprocessing.get_context("spawn")`. Define only JSON/pickle-safe frozen messages:

```python
@dataclass(frozen=True, slots=True)
class RunMessage:
    invocation_id: str
    request: MeasurementRequest
    module_name: str
    adapter_module_name: str
    run_func: str
    case_func: str
    result_func: str


@dataclass(frozen=True, slots=True)
class StopMessage:
    pass


@dataclass(frozen=True, slots=True)
class WorkerReply:
    invocation_id: str
    request_digest: str
    record: MeasurementRecord
```

The parent resolves configured discovery indices to `GpuDevice.uuid` values. The child entry point must set `CUDA_VISIBLE_DEVICES` from the assigned UUID tuple before importing the heavy collector module, avoiding host/container ordinal-remapping ambiguity; inside the child, visible CUDA ordinals are local `0..N-1`. It imports the run function from `module_name` and the case/result functions from `adapter_module_name`, creates `CaseInvocation`, and invokes the existing run function as:

```python
raw = run_func(*invocation.args, **invocation.kwargs)
record = result_func(message.request, raw)
```

It catches `BaseException`, serializes `traceback.format_exc()` into provenance, and returns a failed record with the matching key and complete `MeasurementProtocol`. It never opens the overlay database. The parent accepts a reply only when both correlation fields match the submitted invocation; any mismatch is a protocol violation that kills and evicts the worker.

- [ ] **Step 4: Implement the resource-aware executor**

```python
class ResourceAwareMeasurementExecutor:
    def __init__(
        self,
        adapters: LazyAdapterRegistry,
        inventory: HardwareInventory,
        worker_factory: WorkerFactory = ProcessWorkerFactory(),
    ) -> None:
        self.adapters = adapters
        self.scheduler = HardwareAwareScheduler(inventory)
        self.inventory = inventory
        self.worker_factory = worker_factory
        self._workers: dict[WorkerLeaseKey, WorkerChannel] = {}

    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> Sequence[MeasurementRecord]:
        prepared = tuple(self._prepare(request) for request in requests)
        waves = self.scheduler.plan(tuple(item.job for item in prepared))
        records: dict[str, MeasurementRecord] = {}
        for wave in waves:
            if cancellation.cancelled():
                self._record_cancelled_wave(wave, records)
                continue
            pending = [self._submit(assignment, prepared) for assignment in wave]
            for invocation_id, digest, lease_key, channel in pending:
                timeout = max(0.0, deadline_monotonic - time.monotonic())
                try:
                    reply = channel.receive(timeout=timeout)
                    self._validate_reply(invocation_id, digest, reply)
                    records[digest] = reply.record
                except BaseException as error:
                    self._terminate_and_evict(lease_key)
                    records[digest] = self._failed_record(digest, error)
            self._settle_wave(pending, records, deadline_monotonic)
        return tuple(records[request.key.digest] for request in requests)
```

`_prepare()` must verify actual GPU/system class, backend/runtime versions, topology schema, and topology fingerprint against the typed `request.environment`; it must not parse ad hoc fields back out of `PerfKey.environment_json`. Request construction already proves the typed environment canonicalizes to the key environment. Then resolve the adapter, calculate its contract, and serialize only the request digest into `CollectionJob.payload`; retain the full request in a parent lookup. Physical GPU UUIDs belong in provenance, not `PerfKey` compatibility. `_submit()` assigns a collision-resistant invocation id and sends every message in a wave before returning any receive handle.

No receive exception may escape before `_settle_wave` accounts for every submitted invocation. On timeout, cancellation, malformed/mismatched reply, EOF, or child death, terminate, join with a deadline, close channels, and evict that lease before it can be reused; convert the incident to a failed record with the specific `UnresolvedCode`. Continue receiving unrelated healthy siblings so successful partial records survive. If a sibling cannot be safely drained by the deadline, terminate and evict it too. `_record_cancelled_wave()` applies the same rule to active assignments and creates failed records for unstarted keys. Provide context-manager methods and idempotent `close()`.

- [ ] **Step 5: Run CPU-only executor tests and commit**

Run: `pytest -m unit tests/unit/collector/lazy -v`

Expected: all lazy collector tests pass without importing CUDA frameworks in the parent test process.

```bash
git add src/aiconfigurator/collector/executor.py tests/unit/collector/lazy/test_executor.py
git commit -m "feat: execute lazy measurements in resource waves"
```

### Task 6: Add the single-GPU TRT-LLM GEMM pilot

**Files:**
- Create: `src/aiconfigurator/collector/benchmark.py`
- Create: `src/aiconfigurator/collector/trtllm/gemm.py`
- Create: `src/aiconfigurator/collector/trtllm/gemm_adapter.py`
- Modify: `collector/helper.py` (`benchmark_with_power` compatibility wrapper)
- Modify: `collector/trtllm/collect_gemm.py` (offline logging wrapper)
- Modify: `collector/trtllm/registry.py` (GEMM entry)
- Modify: `src/aiconfigurator/sdk/operations/gemm.py` (`GEMM.query` normalization and request hook)
- Modify: `tests/unit/sdk/resolution/test_operations.py`
- Create: `tests/unit/collector/lazy/test_benchmark_samples.py`
- Create: `tests/integration/collector/test_lazy_gemm_gpu.py`

- [ ] **Step 1: Add CPU-only request-mapping tests**

Instantiate `GEMM("mlp", 1.0, n=4096, k=4096, quant_mode=GEMMQuantMode.bfloat16)`, call `measurement_request()` with `x=8`, and assert query fields are exactly `{"gemm_type": "bfloat16", "m": 8, "n": 4096, "k": 4096}`. Extract one private normalization helper used by `query()`, `curated_exact_result()`, and `measurement_request()`; tests must prove `_scale_num_tokens`, context-parallel `seq_split` ceil division, and a runtime `quant_mode` override produce the same final `(m, n, k, quant_mode)` in all paths. Assert `_scale_factor` affects only conversion of the exact row/record, and identical dictionaries with a different insertion order yield the same key. V1's lazy adapter capability allowlist is BF16 only: a simple non-BF16 GEMM may construct its normalized request, but capability preflight must reject it as `UNSUPPORTED_SHAPE` before resource acquisition if no literal exact row exists. `fp8_static` returns no lazy request at all because that query composes GEMM, compute-scale, and optional scale-matrix evidence and cannot be represented by one base-GEMM measurement.

Use a real temporary `PerfDatabase` GEMM table containing literal `m=8` and neighboring rows. Prove `curated_exact_result(x=8)` hits without collection; `query(x=9)` succeeds by current interpolation/clamping; but `query_with_resolution(x=9)` records one exact miss and invokes the adapter on the cold callback. Also choose a shape inserted only by `_extrapolate_gemm_data` and prove it remains an exact miss, plus test an entirely absent GEMM quant-mode table. This real-operation test is the release gate for the exact-miss predicate.

- [ ] **Step 2: Make collector timing optionally return raw samples**

Move the reusable timing primitive into `aiconfigurator.collector.benchmark` and make top-level `collector.helper.benchmark_with_power` a compatibility delegate. Add `return_samples: bool = False`. When false, preserve the existing two-event path byte-for-byte. When true, allocate one start/end CUDA event pair per measured replay, synchronize once after all replays, and return `samples_ms` as each pair's elapsed time divided by `repeat_n`; set `latency_ms` to the median using `statistics.median`. Add fake-event tests for both the namespaced primitive and legacy wrapper.

- [ ] **Step 3: Factor a side-effect-free exact GEMM case while preserving offline logging**

Implement packaged `run_gemm_case()` in `aiconfigurator.collector.trtllm.gemm`. Call the namespaced timing primitive with `repeat_n=1, return_samples=True`, construct the existing perf row once, and return the following without opening or writing a perf file:

```python
return RawMeasurement(
    latency_ms=results["latency_ms"] / outside_loop_count,
    energy_wms=(results["power_stats"] or {}).get("power", 0.0)
    * results["latency_ms"]
    / outside_loop_count,
    samples_ms=tuple(sample / outside_loop_count for sample in results["samples_ms"]),
    statistic="median",
    perf_row=row,
    provenance={
        "framework": "TRTLLM",
        "framework_version": tensorrt_llm.__version__,
        "kernel_source": kernel_source,
        "device": torch.cuda.get_device_name(device),
        "used_cuda_graph": results["used_cuda_graph"],
    },
)
```

The top-level offline `collector/trtllm/collect_gemm.py` calls `run_gemm_case()`, passes the returned row to its existing `log_perf` path, and preserves current CLI/output behavior. Lazy workers call `run_gemm_case()` directly and never invoke `log_perf`; their only durable write is the parent-owned overlay.

Implement `gemm_request_to_case`, `gemm_resource_for_request`, and `gemm_result_to_record` in lightweight `aiconfigurator.collector.trtllm.gemm_adapter`; that module must not import Torch or TensorRT-LLM. Reject any `gemm_type` outside the V1 BF16 allowlist before resource acquisition. The case tuple is `(gemm_type, m, n, k)`. The resource contract is one GPU/no fabric. The result adapter requires positive finite latency, exact row/query equality, and exact complete protocol identity before returning a valid record.

- [ ] **Step 4: Register and expose the GEMM request**

Define one packaged `GEMM_LAZY_SPEC` in `aiconfigurator.collector.trtllm.registry` and import that exact object from the existing source-only TRT-LLM registry. Set the field on the existing GEMM `OpEntry` as `lazy=GEMM_LAZY_SPEC`; the packaged built-in lazy registry consumes the same specification:

```python
GEMM_LAZY_SPEC = LazyOpEntry(
    namespace="trtllm/gemm/v1",
    run_module="aiconfigurator.collector.trtllm.gemm",
    run_func="run_gemm_case",
    adapter_module="aiconfigurator.collector.trtllm.gemm_adapter",
    case_func="gemm_request_to_case",
    result_func="gemm_result_to_record",
    resource_func="gemm_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="trtllm-linear-v1",
)
```

Create a packaged `TRTLLM_LAZY_REGISTRY` entry whose `module`, `get_func`, and `run_func` all resolve inside `aiconfigurator.collector.trtllm`; the source-only registry keeps its existing offline module/get/run fields and attaches `GEMM_LAZY_SPEC`. Tests must prove both entries expose the identical spec while a wheel-only `LazyAdapterRegistry` imports no top-level `collector` module.

During GEMM load, capture a provenance-bearing set/map of source-row `(quant_mode, m, n, k)` identities before `_extrapolate_gemm_data` or any other grid-synthesis/correction pass mutates the table. Key this index with the same full database cache key as `_data_cache`, preserve the originating file/version for inherited compatible rows, and clear it from `GEMM.clear_cache()` so fixture/database swaps cannot reuse stale membership. Implement `GEMM.curated_exact_result()` by checking that source-row index first and then reading the fully normalized tuple with non-mutating `.get()` calls. It must not call interpolation/clamping helpers and must not treat a load-time synthesized point as exact. Convert a literal row with the same latency, energy, and `_scale_factor` semantics as `query()` and tag it `source="curated_exact"`. Return `None` for an absent row and for composite `fp8_static` cases deferred by the pilot.

In `GEMM.measurement_request()`, call the same normalization helper as `query()`, derive a typed `MeasurementEnvironment` from the database system spec plus backend/runtime version, and include a deterministic semantic descriptor with `tensor_generator="normal-v1"` and `seed=0`; do not capture a runtime tensor.

- [ ] **Step 5: Add and run the opt-in GPU test**

Mark the test `pytest.mark.gpu` and skip unless CUDA and TensorRT-LLM are importable. Run one small BF16 shape through the adapter twice. Assert the first result is valid and finite, the second uses the same persistent worker PID, neither call creates or mutates a perf text file, and the legacy offline wrapper still logs exactly one identical row when explicitly invoked.

Run CPU tests: `pytest -m unit tests/unit/collector/lazy tests/unit/sdk/resolution/test_operations.py -v`

Run on a GPU node: `pytest -m gpu tests/integration/collector/test_lazy_gemm_gpu.py -v`

- [ ] **Step 6: Commit the GEMM pilot**

```bash
git add src/aiconfigurator/collector collector/helper.py collector/trtllm/collect_gemm.py collector/trtllm/registry.py src/aiconfigurator/sdk/operations/gemm.py tests/unit tests/integration/collector/test_lazy_gemm_gpu.py
git commit -m "feat: lazily measure exact TRT-LLM GEMM points"
```

### Task 7: Add the multi-GPU NCCL pilot with persistent groups

**Files:**
- Modify: `collector/network/collect_nccl.py` (offline wrapper)
- Create: `src/aiconfigurator/collector/network/nccl.py`
- Create: `src/aiconfigurator/collector/network/nccl_adapter.py`
- Create: `src/aiconfigurator/collector/network/registry.py`
- Modify: `src/aiconfigurator/collector/executor.py`
- Modify: `src/aiconfigurator/sdk/operations/communication.py` (`NCCL.query` normalization and request hook)
- Modify: `tests/unit/sdk/resolution/test_operations.py`
- Create: `tests/integration/collector/test_lazy_nccl_gpu.py`

- [ ] **Step 1: Add CPU-only NCCL mapping and placement tests**

For an `NCCL` operation, assert `measurement_request(x=16)` records dtype, op, `num_gpus`, and the exact element count produced by the same normalization helper as `query()` and `curated_exact_result()`. Test context-parallel `seq_split` ceil division before multiplying by `_num_elements_per_token`. Against a temporary populated NCCL table, prove a literal message-size row is an exact hit while a size that ordinary `query()` would interpolate becomes an exact miss. Its resource function must return `ResourceContract(num_gpus, NVLINK, reserve_fabric_domain=True)` for an intra-node NVLink pilot. Assert two requests with different `num_gpus` or topology fingerprints cannot reuse one worker lease.

- [ ] **Step 2: Factor a one-case API from the existing sweep**

Create this side-effect-free packaged API and make top-level `nccl_benchmark()` loop over it and perform the existing explicit logging so CLI/offline behavior uses the same measurement function while lazy collection never writes a perf file:

```python
def run_nccl_case(
    dtype: str,
    nccl_op: str,
    element_count: int,
    num_gpus: int,
    *,
    runtime: PersistentNcclRuntime | None = None,
    measure_power: bool = False,
) -> RawMeasurement:
    if runtime is None:
        samples_ms, power_stats, provenance = _run_nccl_tests_case(
            dtype=dtype,
            nccl_op=nccl_op,
            message_size_bytes=element_count * (2 if dtype == "half" else 1),
            num_gpus=num_gpus,
            measure_power=measure_power,
        )
    else:
        samples_ms = runtime.measure(dtype, nccl_op, element_count)
        power_stats = None
        provenance = {"runtime": "persistent_torch_distributed"}
    latency_ms = statistics.median(samples_ms)
    row = {
        "nccl_dtype": dtype,
        "num_gpus": num_gpus,
        "message_size": element_count,
        "latency": latency_ms,
    }
    return RawMeasurement(
        latency_ms=latency_ms,
        energy_wms=(power_stats or {}).get("power", 0.0) * latency_ms,
        samples_ms=tuple(samples_ms),
        statistic="median",
        perf_row=row,
        provenance=provenance,
    )
```

With `runtime=None`, convert `element_count` to bytes and preserve the current `nccl-tests` command/parser for one size. With a runtime, call `runtime.measure(dtype, nccl_op, element_count)` and use its per-replay CUDA-event samples. Both branches construct the same element-count row and return `RawMeasurement`. Only the top-level offline wrapper calls `log_perf`; the lazy worker does not receive a writable perf filename.

- [ ] **Step 3: Implement one persistent collective runtime per GPU tuple**

Add `PersistentNcclRuntime` to `aiconfigurator.collector.executor`. It spawns one rank process per assigned GPU once, initializes a `torch.distributed` NCCL process group using a parent-owned rendezvous whose lifetime is tied to the rank group, and keeps correlated command/reply channels alive. A measure command contains `(invocation_id, dtype, op, element_count, warmups, samples)`; every rank allocates the correct operation-specific input/output tensors, synchronizes before the measured series, performs warmups, records one CUDA event pair per sample, and returns only rank 0 timings after all ranks have completed. Support `all_reduce`, `all_gather`, `reduce_scatter`, and `alltoall`; reject unsupported operations before launch.

Healthy explicit close may attempt a bounded cooperative barrier and `destroy_process_group`. After any rank exception, timeout, cancellation, EOF, or protocol mismatch, never enter another collective or barrier: terminate the entire rank group, close the rendezvous socket/store and queues, join each rank with a deadline, force-kill survivors, and evict the communicator lease. Add fault-injection tests with one rank hanging and one rank crashing; both must return without deadlock and a later request must create fresh PIDs/channels.

The collective worker constructs this runtime on its first NCCL message and passes it through the invocation kwargs. This is the only adapter-specific worker capability in the pilot; single-GPU workers never import `torch.distributed`.

- [ ] **Step 4: Register NCCL and expose its exact request**

Define `NCCL_LAZY_SPEC` once in the packaged network registry. Its packaged `NETWORK_LAZY_REGISTRY` uses only namespaced modules; the source-only offline registry imports the same spec into its existing top-level entry:

```python
NCCL_LAZY_SPEC = LazyOpEntry(
    namespace="nccl/collective/v1",
    run_module="aiconfigurator.collector.network.nccl",
    run_func="run_nccl_case",
    adapter_module="aiconfigurator.collector.network.nccl_adapter",
    case_func="nccl_request_to_case",
    result_func="nccl_result_to_record",
    resource_func="nccl_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    timer="cuda_event",
    tuning_revision="torch-nccl-persistent-v1",
)

NETWORK_LAZY_REGISTRY = [
    OpEntry(
        op="nccl",
        module="aiconfigurator.collector.network.nccl",
        get_func="get_nccl_test_cases",
        run_func="run_nccl_case",
        perf_filename=PerfFile.NCCL,
        lazy=NCCL_LAZY_SPEC,
    )
]
```

`NCCL_LAZY_SPEC` contains the namespaced run/adapter functions and protocol fields shown in the GEMM pattern. Expose `get_nccl_test_cases()` from both the namespaced implementation and the top-level wrapper by expanding the current CLI ranges into `(dtype, op, element_count, num_gpus)` tuples, and make `nccl_benchmark()` consume that generator. Put the case/result/resource functions in namespaced `nccl_adapter.py` without Torch imports. Combine only packaged backend/network registries when constructing `LazyAdapterRegistry`; top-level compatibility modules are never imported in a wheel-only worker.

Implement `NCCL.curated_exact_result()` with a non-mutating exact lookup of dtype, world size, operation, and normalized element count in the loaded NCCL table; do not call nearest/interpolation helpers. Convert a literal row with the same scale/energy semantics as `query()` and tag it `source="curated_exact"`.

In `NCCL.measurement_request()`, call the same normalization helper as `query()`, then use database NCCL version, typed environment topology schema/fingerprint, dtype, op, group size, and exact element count in `PerfKey`; use deterministic tensor seed 0 in the semantic descriptor. Do not add a request hook to analytical `P2P` in this pilot.

- [ ] **Step 5: Run a two-or-more-GPU integration test**

Skip unless at least two mutually NVLink-connected GPUs exist. Resolve two message sizes in one callback and assert both run on the same persistent rank PIDs. For each operation, verify output shape/content and compare the warm persistent measurement with a freshly constructed `PersistentNcclRuntime` using the same Torch/NCCL harness, protocol, and tensor semantics; gate on a documented robust sample interval derived from repeated isolated controls rather than an invented fixed 20% threshold. If matching `nccl-tests` binaries are installed, record their result as diagnostic provenance only because that harness has different launch/warmup semantics; absence is not a lazy-runtime test failure. Then request the first size again and assert the overlay/core path performs no collective command.

Run CPU tests: `pytest -m unit tests/unit/collector/lazy tests/unit/sdk/resolution/test_operations.py -v`

Run on a GPU node: `pytest -m gpu tests/integration/collector/test_lazy_nccl_gpu.py -v -s`

- [ ] **Step 6: Commit the NCCL pilot**

```bash
git add src/aiconfigurator/collector/network src/aiconfigurator/collector/executor.py collector/network/collect_nccl.py src/aiconfigurator/sdk/operations/communication.py tests/unit tests/integration/collector/test_lazy_nccl_gpu.py
git commit -m "feat: lazily measure exact NCCL collective points"
```

### Task 8: Validate concurrency, failure isolation, packaging, and provenance end to end

**Files:**
- Modify: `tests/unit/collector/lazy/test_executor.py`
- Modify: `tests/integration/collector/test_lazy_gemm_gpu.py`
- Modify: `tests/integration/collector/test_lazy_nccl_gpu.py`
- Create: `tests/integration/collector/test_lazy_mixed_gpu.py`
- Create: `tests/unit/collector/lazy/test_import_surface.py`

- [ ] **Step 1: Add deterministic CPU fault-injection coverage**

Use fake worker channels to simulate one successful GEMM, one rejected GEMM, one timed-out worker with a delayed stale reply, and one crashed collective in the same executor call. Assert all records return in request order, successful records remain valid, every failed/protocol-violating worker is terminated and recreated on the next request, the stale reply is never consumed, and unrelated workers remain alive. Add a cancellation race while two assignments are active and prove `execute()` accounts for both before returning.

- [ ] **Step 2: Add a mixed hardware utilization test**

On four or more GPUs, submit one two-GPU collective and two GEMMs. Record worker start/end monotonic timestamps. Assert GEMMs on GPUs outside the collective lease overlap it, no assignment shares a GPU, and two collectives in the same fabric domain never overlap. The test reports achieved peak occupied GPUs but does not enforce a timing speedup threshold.

- [ ] **Step 3: Validate record provenance and round-trip lookup**

For every pilot record, assert provenance contains physical GPU UUIDs, topology/fabric domain and schema fingerprint, framework and collector versions, the complete protocol, worker PID, invocation id, command/case arguments, sample count, statistic, and throttling state. Append the records through `ResolutionSession`, close/reopen the overlay, and assert the operations return the same latency with `source="overlay"`.

- [ ] **Step 4: Validate the namespaced runtime in a non-editable AIC wheel**

All installable files are already under `src/aiconfigurator/collector`, so the existing `python-packages = ["aiconfigurator", "spica"]` includes them without adding a top-level include. Keep heavy framework imports confined to namespaced `trtllm/gemm.py` and `network/nccl.py`; importing `aiconfigurator.collector`, its scheduler/executor, or the lightweight adapters must not import `torch` or `tensorrt_llm`. Add source-checkout regression tests proving the top-level offline scripts delegate to the same implementations and retain current CLI/logging behavior.

In `test_import_surface.py`, launch a fresh Python subprocess that imports the namespaced runtime and both lightweight adapters and asserts `"torch" not in sys.modules` and `"tensorrt_llm" not in sys.modules`. Then build and inspect a wheel:

```bash
mkdir -p /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist
uv run maturin build --release --out /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist
python -m zipfile -l /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist/aiconfigurator-*.whl
python -m venv /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv
/home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv/bin/pip install /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist/aiconfigurator-*.whl
/home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv/bin/python -c "import importlib.util, aiconfigurator.collector, aiconfigurator.collector.trtllm.gemm_adapter, aiconfigurator.collector.network.nccl_adapter; assert importlib.util.find_spec('collector') is None"
```

Expected: the wheel listing contains the namespaced runtime, adapters, and pilot heavy modules; lightweight imports succeed without a source checkout; and the wheel does not publish a generic top-level `collector`. GPU integration tests exercise heavy modules only inside worker subprocesses.

- [ ] **Step 5: Run completion checks**

```bash
git diff --check
pytest -m unit tests/unit/collector/lazy tests/unit/collector/test_version_resolver.py tests/unit/sdk/resolution -v
pytest -m gpu tests/integration/collector/test_lazy_gemm_gpu.py tests/integration/collector/test_lazy_nccl_gpu.py tests/integration/collector/test_lazy_mixed_gpu.py -v -s
```

Expected: CPU tests pass anywhere; GPU tests pass on a compatible multi-GPU TRT-LLM host. The runtime is complete when compute fills every independent GPU, collectives reserve exactly their required clique and fabric domain, repeated keys avoid workers entirely, and one failed job does not discard valid sibling records.

- [ ] **Step 6: Commit final validation and packaging coverage**

```bash
git add src/aiconfigurator/collector collector tests/unit/collector/lazy tests/integration/collector
git commit -m "test: package and validate hardware-aware lazy collection"
```
