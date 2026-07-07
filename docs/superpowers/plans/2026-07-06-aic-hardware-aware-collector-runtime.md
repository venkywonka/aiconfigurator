# AIC Hardware-Aware Lazy Collector Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve exact AIC perf misses through the existing collector functions while safely saturating independent GPUs and reserving whole GPU/fabric groups for collectives.

**Architecture:** Collector registry entries gain an optional lazy adapter that maps a `MeasurementRequest` to the collector's existing case/run function and converts its raw result back to a `MeasurementRecord`. A deterministic planner packs single-GPU and collective jobs into non-conflicting waves from live GPU/topology inventory. Long-lived subprocess workers own CUDA runtimes; collective workers additionally own persistent rank processes and NCCL communicators. The executor returns records only—the SDK resolution session remains the sole overlay writer.

**Tech Stack:** Python 3.10+, dataclasses, `multiprocessing` with `spawn`, `concurrent.futures`, PyTorch CUDA/distributed, `nvidia-smi`, pytest.

---

## Source prerequisites

Read these first:

- `docs/plans/2026-07-06-dynamic-lazy-perf-collection-design.md`
- `docs/superpowers/plans/2026-07-06-aic-lazy-perf-core.md`

Complete the lazy core plan before this one. Execute this plan in the AIC repository. All unit tests in Tasks 1–5 use fake collectors and fake hardware; Tasks 6–7 add opt-in GPU tests for the two pilot adapters.

## File map

- Modify `collector/registry_types.py` — optional lazy adapter declaration.
- Modify `pyproject.toml` — ship collector Python modules needed by resolving mode.
- Modify `collector/version_resolver.py` — preserve optional lazy metadata through version routing.
- Create `collector/lazy/__init__.py` — public collector-runtime exports.
- Create `collector/lazy/types.py` — resource, raw-result, invocation, assignment, and hardware types.
- Create `collector/lazy/hardware.py` — GPU and topology discovery.
- Create `collector/lazy/scheduler.py` — deterministic conflict-aware wave packing.
- Create `collector/lazy/adapters.py` — registry lookup and dynamic function loading.
- Create `collector/lazy/executor.py` — persistent worker ownership and `MeasurementExecutor` implementation.
- Modify `collector/helper.py` — optional per-sample latency reporting without changing existing callers.
- Modify `collector/trtllm/collect_gemm.py` — exact-case raw result from the existing heavy collector.
- Create `collector/trtllm/lazy_gemm.py` — lightweight request/result/resource mapping; no CUDA imports.
- Modify `collector/trtllm/registry.py` — register the GEMM lazy adapter.
- Modify `src/aiconfigurator/sdk/operations/gemm.py` — construct exact GEMM requests.
- Modify `collector/network/collect_nccl.py` — one-case API plus persistent-runtime hook.
- Create `collector/network/lazy_nccl.py` — lightweight NCCL request/result/resource mapping.
- Create `collector/network/registry.py` — register the NCCL lazy adapter.
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
- Modify: `collector/registry_types.py:69-101`
- Modify: `collector/version_resolver.py:109-177`
- Modify: `tests/unit/collector/test_version_resolver.py`
- Create: `tests/unit/collector/lazy/test_registry.py`

- [ ] **Step 1: Write backward-compatibility and validation tests**

```python
import pytest

from collector.lazy.types import FabricRequirement, LazyOpEntry, ResourceContract
from collector.registry_types import OpEntry, PerfFile
from collector.version_resolver import build_collections

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
            adapter_module="collector.fake_adapter",
            case_func="request_to_case",
            result_func="result_to_record",
            resource_func="resource_for_request",
            protocol_revision="cuda-event-v1",
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
```

- [ ] **Step 2: Run the tests and verify the missing lazy module failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py -v`

Expected: collection fails because `collector.lazy.types` does not exist.

- [ ] **Step 3: Define the shared lazy types**

```python
# collector/lazy/types.py
from __future__ import annotations

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
    adapter_module: str
    case_func: str
    result_func: str
    resource_func: str
    protocol_revision: str
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


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    devices: tuple[GpuDevice, ...]
    links: Mapping[tuple[int, int], str]
    fabric_domains: Mapping[int, str]

    def restrict(self, gpu_ids: tuple[int, ...]) -> "HardwareInventory":
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("assigned GPU ids must be unique")
        by_id = {device.index: device for device in self.devices}
        try:
            devices = tuple(by_id[gpu_id] for gpu_id in gpu_ids)
        except KeyError as error:
            raise ValueError(f"assigned GPU id is not present: {error.args[0]}") from error
        allowed = set(gpu_ids)
        return HardwareInventory(
            devices=devices,
            links={pair: link for pair, link in self.links.items() if set(pair) <= allowed},
            fabric_domains={gpu: domain for gpu, domain in self.fabric_domains.items() if gpu in allowed},
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
```

Export these names from `collector/lazy/__init__.py`.

- [ ] **Step 4: Extend `OpEntry` with a final optional field**

Under `TYPE_CHECKING`, import `LazyOpEntry`, then add this field after `versions` so every existing positional constructor remains valid:

```python
lazy: LazyOpEntry | None = None
```

Do not add lazy fields to `build_collections()` output. Add a version-routed test proving `resolve_module()` selects the same module and retains `entry.lazy` on the original immutable entry.

- [ ] **Step 5: Run and commit the registry changes**

Run: `pytest -m unit tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py -v`

Expected: all tests pass.

```bash
git add collector/registry_types.py collector/version_resolver.py collector/lazy tests/unit/collector/lazy/test_registry.py tests/unit/collector/test_version_resolver.py
git commit -m "feat: declare optional lazy collector adapters"
```

### Task 2: Discover GPUs, links, and contention domains

**Files:**
- Create: `collector/lazy/hardware.py`
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

Assert device indices are `(0, 1, 2, 3)`, links are symmetric, and fabric domains are `{0: "nvlink:0", 1: "nvlink:0", 2: "nvlink:1", 3: "nvlink:1"}`. Add `inventory.restrict((2, 3))` and assert it preserves physical ids/links while excluding GPUs 0/1; duplicate or absent requested ids raise `ValueError`. Add malformed-row and missing-GPU tests that raise `HardwareDiscoveryError` rather than silently returning a partial inventory.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_hardware.py -v`

Expected: import failure for `collector.lazy.hardware`.

- [ ] **Step 3: Implement discovery with injectable command execution**

Define these public functions:

```python
import csv
import io
import re
import subprocess
from collections.abc import Callable

from .types import GpuDevice, HardwareInventory


class HardwareDiscoveryError(RuntimeError):
    pass


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
        for column_gpu, token in zip(columns, tokens, strict=True):
            if row_gpu != column_gpu:
                links[(row_gpu, column_gpu)] = token
    if len(links) != len(devices) * (len(devices) - 1):
        raise HardwareDiscoveryError("topology matrix is incomplete")
    for (left, right), token in links.items():
        if links.get((right, left)) != token:
            raise HardwareDiscoveryError(f"asymmetric topology link GPU{left}/GPU{right}")

    remaining = set(expected)
    components: list[list[int]] = []
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
                if peer != current and links[(current, peer)].startswith("NV")
            )
        remaining.difference_update(component)
        components.append(sorted(component))
    fabric_domains = {
        gpu: f"nvlink:{component_index}"
        for component_index, component in enumerate(components)
        for gpu in component
    }
    return HardwareInventory(devices, links, fabric_domains)


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

`discover_hardware()` must invoke argument arrays, never a shell string:

```python
[
    "nvidia-smi",
    "--query-gpu=index,uuid,name,pci.bus_id",
    "--format=csv,noheader",
]
["nvidia-smi", "topo", "-m"]
```

Treat `NV1` through `NV18` as NVLink, `PIX`/`PXB`/`PHB` as P2P-capable, and `SYS`/`NODE` as non-P2P for initial placement. Build NVLink connected components in ascending GPU order and name them `nvlink:0`, `nvlink:1`, and so on. Preserve the raw link token for provenance.

- [ ] **Step 4: Run and commit hardware discovery**

Run: `pytest -m unit tests/unit/collector/lazy/test_hardware.py -v`

Expected: all parser and failure tests pass without a GPU.

```bash
git add collector/lazy/hardware.py collector/lazy/types.py tests/unit/collector/lazy/test_hardware.py
git commit -m "feat: inventory GPU and fabric resources"
```

### Task 3: Pack work into deterministic non-conflicting waves

**Files:**
- Create: `collector/lazy/scheduler.py`
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

Expected: import failure for `collector.lazy.scheduler`.

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
git add collector/lazy/scheduler.py tests/unit/collector/lazy/test_scheduler.py
git commit -m "feat: schedule lazy collection across hardware domains"
```

### Task 4: Resolve registry adapters and validate record round trips

**Files:**
- Create: `collector/lazy/adapters.py`
- Test: `tests/unit/collector/lazy/test_adapters.py`

- [ ] **Step 1: Write fake-module adapter tests**

Create a synthetic module in `sys.modules` with `request_to_case`, `run_case`, `result_to_record`, and `resource_for_request`. Assert `LazyAdapterRegistry.resolve(request)`:

- selects by exact `PerfKey.namespace`;
- applies existing `resolve_module(entry, runtime_version)` routing;
- rejects a request protocol/tuning revision that differs from `LazyOpEntry`;
- returns a `ResolvedLazyAdapter` containing lightweight mapping callables, the heavy run-function name, and the original `perf_filename`;
- raises `MissingLazyAdapter` for an unregistered namespace.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_adapters.py -v`

Expected: import failure for `collector.lazy.adapters`.

- [ ] **Step 3: Implement exact namespace lookup and dynamic loading**

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
    tuning_revision: str


class LazyAdapterRegistry:
    def __init__(self, entries: Sequence[OpEntry], runtime_version: str) -> None:
        self._by_namespace: dict[str, ResolvedLazyAdapter] = {}
        for entry in entries:
            if entry.lazy is None:
                continue
            module_name = resolve_module(entry, runtime_version)
            if module_name is None:
                continue
            lazy = entry.lazy
            adapter_module = importlib.import_module(lazy.adapter_module)
            if lazy.namespace in self._by_namespace:
                raise ValueError(f"duplicate lazy namespace {lazy.namespace}")
            self._by_namespace[lazy.namespace] = ResolvedLazyAdapter(
                namespace=lazy.namespace,
                module_name=module_name,
                adapter_module_name=lazy.adapter_module,
                perf_filename=str(entry.perf_filename),
                case_func=getattr(adapter_module, lazy.case_func),
                run_func_name=entry.run_func,
                result_func=getattr(adapter_module, lazy.result_func),
                resource_func=getattr(adapter_module, lazy.resource_func),
                protocol_revision=lazy.protocol_revision,
                tuning_revision=lazy.tuning_revision,
            )

    def resolve(self, request: MeasurementRequest) -> ResolvedLazyAdapter:
        adapter = self._by_namespace.get(request.key.namespace)
        if adapter is None:
            raise MissingLazyAdapter(request.key.namespace)
        return adapter
```

Before returning, compare `request.protocol.revision` and `request.protocol.tuning_revision` to the entry declaration retained on `ResolvedLazyAdapter`. Include both expected values as fields on that dataclass so mismatch errors are deterministic.

- [ ] **Step 4: Run and commit adapter loading**

Run: `pytest -m unit tests/unit/collector/lazy/test_adapters.py -v`

Expected: all tests pass.

```bash
git add collector/lazy/adapters.py tests/unit/collector/lazy/test_adapters.py
git commit -m "feat: resolve exact requests to collector functions"
```

### Task 5: Execute waves in persistent subprocess workers

**Files:**
- Create: `collector/lazy/executor.py`
- Test: `tests/unit/collector/lazy/test_executor.py`

- [ ] **Step 1: Write executor tests with fake worker channels**

Inject a `WorkerFactory` and assert:

- two one-GPU assignments in a wave are submitted before either result is awaited;
- the same `(adapter namespace, gpu_ids)` lease reuses one worker across two `execute()` calls;
- the next wave starts only after every assignment in the current wave returns;
- `close()` sends shutdown and joins every worker;
- an adapter exception becomes one `MeasurementRecord(status=FAILED)` with collector traceback provenance;
- parent-side hardware mismatch and unschedulable jobs become failed records without starting a worker;
- cancellation stops submitting later waves, drains or terminates active work according to the worker capability, and returns failed records for unstarted keys;
- a monotonic deadline terminates an overrun worker and returns a timeout record without blocking sibling records;
- successful results preserve request order even when workers finish out of order.

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/collector/lazy/test_executor.py -v`

Expected: import failure for `collector.lazy.executor`.

- [ ] **Step 3: Implement the worker message protocol**

Use `multiprocessing.get_context("spawn")`. Define only JSON/pickle-safe frozen messages:

```python
@dataclass(frozen=True, slots=True)
class RunMessage:
    request: MeasurementRequest
    module_name: str
    adapter_module_name: str
    run_func: str
    case_func: str
    result_func: str
    perf_filename: str


@dataclass(frozen=True, slots=True)
class StopMessage:
    pass


@dataclass(frozen=True, slots=True)
class WorkerReply:
    record: MeasurementRecord
```

The child entry point must set `CUDA_VISIBLE_DEVICES` from the assigned physical GPU tuple before importing the heavy collector module. It imports the run function from `module_name` and the case/result functions from `adapter_module_name`, creates `CaseInvocation`, and invokes the existing run function as:

```python
raw = run_func(*invocation.args, perf_filename=message.perf_filename, **invocation.kwargs)
record = result_func(message.request, raw)
```

It catches `BaseException`, serializes `traceback.format_exc()` into provenance, and returns a failed record with matching key and revisions. It never opens the overlay database.

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
        self._workers: dict[tuple[str, tuple[int, ...]], WorkerChannel] = {}

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
            for digest, channel in pending:
                timeout = max(0.0, deadline_monotonic - time.monotonic())
                records[digest] = channel.receive(timeout=timeout).record
        return tuple(records[request.key.digest] for request in requests)
```

`_prepare()` must verify the actual GPU/system class and required topology class against the request environment JSON, resolve the adapter, calculate its contract, and serialize only the request digest into `CollectionJob.payload`; retain the full request in a parent lookup. Physical GPU UUIDs belong in provenance, not `PerfKey` compatibility. `_submit()` must send every message in a wave before returning any receive handle. Catch `TimeoutError`, terminate and discard only that worker, and create a failed record with `failure_code=UnresolvedCode.TIMEOUT`; `_record_cancelled_wave()` creates failed records with `failure_code=UnresolvedCode.CANCELLED`. Provide context-manager methods and idempotent `close()`.

- [ ] **Step 5: Run CPU-only executor tests and commit**

Run: `pytest -m unit tests/unit/collector/lazy -v`

Expected: all lazy collector tests pass without importing CUDA frameworks in the parent test process.

```bash
git add collector/lazy/executor.py tests/unit/collector/lazy/test_executor.py
git commit -m "feat: execute lazy measurements in resource waves"
```

### Task 6: Add the single-GPU TRT-LLM GEMM pilot

**Files:**
- Modify: `collector/helper.py:178-371`
- Modify: `collector/trtllm/collect_gemm.py:161-269`
- Create: `collector/trtllm/lazy_gemm.py`
- Modify: `collector/trtllm/registry.py:13-20`
- Modify: `src/aiconfigurator/sdk/operations/gemm.py:639-702`
- Modify: `tests/unit/sdk/resolution/test_operations.py`
- Create: `tests/unit/collector/lazy/test_benchmark_samples.py`
- Create: `tests/integration/collector/test_lazy_gemm_gpu.py`

- [ ] **Step 1: Add CPU-only request-mapping tests**

Instantiate `GEMM("mlp", 1.0, n=4096, k=4096, quant_mode=GEMMQuantMode.bfloat16)`, call `measurement_request()` with `x=8`, and assert query fields are exactly `{"gemm_type": "bfloat16", "m": 8, "n": 4096, "k": 4096}`. Assert `_scale_num_tokens` is applied before key construction, `_scale_factor` affects only `performance_from_record()`, and identical dictionaries with a different insertion order yield the same key. Assert `fp8_static` returns `None` in the pilot because that query composes GEMM, compute-scale, and optional scale-matrix evidence.

- [ ] **Step 2: Make collector timing optionally return raw samples**

Add `return_samples: bool = False` to `benchmark_with_power`. When false, preserve the existing two-event path byte-for-byte. When true, allocate one start/end CUDA event pair per measured replay, synchronize once after all replays, and return `samples_ms` as each pair's elapsed time divided by `repeat_n`; set `latency_ms` to the median using `statistics.median`. Add a fake-event unit test for both branches.

- [ ] **Step 3: Return `RawMeasurement` from `run_gemm` while preserving logging**

Call `benchmark_with_power(device=device, kernel_func=kernel_func, repeat_n=1, return_samples=True)`, construct the existing perf row once, pass it to `log_perf`, and return:

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

Implement `gemm_request_to_case`, `gemm_resource_for_request`, and `gemm_result_to_record` in the lightweight `lazy_gemm.py`; that module must not import Torch or TensorRT-LLM. The case tuple is `(gemm_type, m, n, k)`. The resource contract is one GPU/no fabric. The result adapter requires positive finite latency, exact row/query equality, and matching protocol/tuning revisions before returning a valid record.

- [ ] **Step 4: Register and expose the GEMM request**

Set this field on the existing TRT-LLM GEMM `OpEntry`:

```python
lazy=LazyOpEntry(
    namespace="trtllm/gemm/v1",
    adapter_module="collector.trtllm.lazy_gemm",
    case_func="gemm_request_to_case",
    result_func="gemm_result_to_record",
    resource_func="gemm_resource_for_request",
    protocol_revision="cuda-event-samples-v1",
    tuning_revision="trtllm-linear-v1",
)
```

In `GEMM.measurement_request()`, derive environment identity from the database system spec plus backend version and include a deterministic semantic descriptor with `tensor_generator="normal-v1"` and `seed=0`; do not capture a runtime tensor.

- [ ] **Step 5: Add and run the opt-in GPU test**

Mark the test `pytest.mark.gpu` and skip unless CUDA and TensorRT-LLM are importable. Run one small BF16 shape through the adapter twice. Assert the first result is valid and finite, the second uses the same persistent worker PID, and neither call mutates the curated perf file fixture.

Run CPU tests: `pytest -m unit tests/unit/collector/lazy tests/unit/sdk/resolution/test_operations.py -v`

Run on a GPU node: `pytest -m gpu tests/integration/collector/test_lazy_gemm_gpu.py -v`

- [ ] **Step 6: Commit the GEMM pilot**

```bash
git add collector/helper.py collector/trtllm/collect_gemm.py collector/trtllm/lazy_gemm.py collector/trtllm/registry.py src/aiconfigurator/sdk/operations/gemm.py tests/unit tests/integration/collector/test_lazy_gemm_gpu.py
git commit -m "feat: lazily measure exact TRT-LLM GEMM points"
```

### Task 7: Add the multi-GPU NCCL pilot with persistent groups

**Files:**
- Modify: `collector/network/collect_nccl.py:24-116`
- Create: `collector/network/lazy_nccl.py`
- Create: `collector/network/registry.py`
- Modify: `collector/lazy/executor.py`
- Modify: `src/aiconfigurator/sdk/operations/communication.py:242-434`
- Modify: `tests/unit/sdk/resolution/test_operations.py`
- Create: `tests/integration/collector/test_lazy_nccl_gpu.py`

- [ ] **Step 1: Add CPU-only NCCL mapping and placement tests**

For an `NCCL` operation, assert `measurement_request(x=16)` records dtype, op, `num_gpus`, and the exact `element_count = 16 * _num_elements_per_token`. Its resource function must return `ResourceContract(num_gpus, NVLINK, reserve_fabric_domain=True)` for an intra-node NVLink pilot. Assert two requests with different `num_gpus` cannot reuse one worker lease.

- [ ] **Step 2: Factor a one-case API from the existing sweep**

Create this API and make `nccl_benchmark()` loop over it so CLI/offline behavior uses the same measurement function:

```python
def run_nccl_case(
    dtype: str,
    nccl_op: str,
    element_count: int,
    num_gpus: int,
    *,
    perf_filename: str,
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
    log_perf(
        item_list=[row],
        framework="TRTLLM",
        version=_nccl_version(),
        device_name=torch.cuda.get_device_name(),
        op_name=nccl_op,
        kernel_source="NCCL",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )
    return RawMeasurement(
        latency_ms=latency_ms,
        energy_wms=(power_stats or {}).get("power", 0.0) * latency_ms,
        samples_ms=tuple(samples_ms),
        statistic="median",
        perf_row=row,
        provenance=provenance,
    )
```

With `runtime=None`, convert `element_count` to bytes and preserve the current `nccl-tests` command/parser for one size. With a runtime, call `runtime.measure(dtype, nccl_op, element_count)` and use its per-replay CUDA-event samples. Both branches construct the same element-count perf row and call the same `log_perf` block before returning `RawMeasurement`.

- [ ] **Step 3: Implement one persistent collective runtime per GPU tuple**

Add `PersistentNcclRuntime` to `collector/lazy/executor.py`. It spawns one rank process per assigned GPU once, initializes a `torch.distributed` NCCL process group using a parent-selected localhost TCP port, and keeps command/reply queues alive. A measure command contains `(dtype, op, element_count, warmups, samples)`; every rank allocates a deterministic tensor, performs warmups, records one CUDA event pair per sample, and returns only rank 0 timings after a final barrier. Support `all_reduce`, `all_gather`, `reduce_scatter`, and `alltoall`; reject unsupported operations before launch. Shutdown performs a barrier, destroys the process group, joins all ranks, and terminates only ranks that miss the join timeout.

The collective worker constructs this runtime on its first NCCL message and passes it through the invocation kwargs. This is the only adapter-specific worker capability in the pilot; single-GPU workers never import `torch.distributed`.

- [ ] **Step 4: Register NCCL and expose its exact request**

Create the network registry entry:

```python
NETWORK_LAZY_REGISTRY = [
    OpEntry(
        op="nccl",
        module="collector.network.collect_nccl",
        get_func="get_nccl_test_cases",
        run_func="run_nccl_case",
        perf_filename=PerfFile.NCCL,
        lazy=LazyOpEntry(
            namespace="nccl/collective/v1",
            adapter_module="collector.network.lazy_nccl",
            case_func="nccl_request_to_case",
            result_func="nccl_result_to_record",
            resource_func="nccl_resource_for_request",
            protocol_revision="cuda-event-samples-v1",
            tuning_revision="torch-nccl-persistent-v1",
        ),
    )
]
```

Expose `get_nccl_test_cases()` from `collect_nccl.py` by expanding the current CLI ranges into `(dtype, op, element_count, num_gpus)` tuples, and make `nccl_benchmark()` consume that generator. Put the case/result/resource functions in `lazy_nccl.py` without Torch imports. Combine this registry with the selected backend registry when constructing `LazyAdapterRegistry`.

In `NCCL.measurement_request()`, use database NCCL version, topology fingerprint, dtype, op, group size, and exact element count in `PerfKey`; use deterministic tensor seed 0 in the semantic descriptor. Do not add a request hook to analytical `P2P` in this pilot.

- [ ] **Step 5: Run a two-or-more-GPU integration test**

Skip unless at least two mutually NVLink-connected GPUs exist. Resolve two message sizes in one callback, assert both run on the same persistent rank PIDs, and compare each median to a one-shot `nccl-tests` result within a documented 20% pilot tolerance. Then request the first size again and assert the overlay/core path performs no collective command.

Run CPU tests: `pytest -m unit tests/unit/collector/lazy tests/unit/sdk/resolution/test_operations.py -v`

Run on a GPU node: `pytest -m gpu tests/integration/collector/test_lazy_nccl_gpu.py -v -s`

- [ ] **Step 6: Commit the NCCL pilot**

```bash
git add collector/network collector/lazy/executor.py src/aiconfigurator/sdk/operations/communication.py tests/unit tests/integration/collector/test_lazy_nccl_gpu.py
git commit -m "feat: lazily measure exact NCCL collective points"
```

### Task 8: Validate concurrency, failure isolation, packaging, and provenance end to end

**Files:**
- Modify: `tests/unit/collector/lazy/test_executor.py`
- Modify: `tests/integration/collector/test_lazy_gemm_gpu.py`
- Modify: `tests/integration/collector/test_lazy_nccl_gpu.py`
- Create: `tests/integration/collector/test_lazy_mixed_gpu.py`
- Modify: `pyproject.toml:119-148`
- Create: `tests/unit/collector/lazy/test_import_surface.py`

- [ ] **Step 1: Add deterministic CPU fault-injection coverage**

Use fake worker channels to simulate one successful GEMM, one rejected GEMM, and one crashed collective in the same executor call. Assert all three records return in request order, the successful record remains valid, failed workers are discarded and recreated on the next request, and unrelated workers remain alive.

- [ ] **Step 2: Add a mixed hardware utilization test**

On four or more GPUs, submit one two-GPU collective and two GEMMs. Record worker start/end monotonic timestamps. Assert GEMMs on GPUs outside the collective lease overlap it, no assignment shares a GPU, and two collectives in the same fabric domain never overlap. The test reports achieved peak occupied GPUs but does not enforce a timing speedup threshold.

- [ ] **Step 3: Validate record provenance and round-trip lookup**

For every pilot record, assert provenance contains physical GPU UUIDs, topology/fabric domain, framework and collector versions, protocol/tuning revisions, worker PID, command/case arguments, sample count, statistic, and throttling state. Append the records through `ResolutionSession`, close/reopen the overlay, and assert the operations return the same latency with `source="overlay"`.

- [ ] **Step 4: Package the lazy runtime for non-editable AIC installs**

Add `collector/**/*.py` to `[tool.maturin].include`. Keep heavy framework imports confined to `collect_gemm.py`/`collect_nccl.py`; importing `collector.lazy`, `collector.trtllm.lazy_gemm`, and `collector.network.lazy_nccl` must not import `torch` or `tensorrt_llm`. Use package-safe imports such as `from collector.helper import ...` and `from collector.case_generator import ...`, with the existing top-level import as a narrow `ModuleNotFoundError` fallback for legacy `python collector/collect.py` execution. Add a regression test for both module import and legacy script-path import so wheel support does not break offline collection.

In `test_import_surface.py`, launch a fresh Python subprocess that imports the three lightweight modules and asserts `"torch" not in sys.modules` and `"tensorrt_llm" not in sys.modules`. Then build and inspect a wheel:

```bash
mkdir -p /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist
uv run maturin build --release --out /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist
python -m zipfile -l /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist/aiconfigurator-*.whl
python -m venv /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv
/home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv/bin/pip install /home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/dist/aiconfigurator-*.whl
/home/gvenkatarama/scratch_big/aic-lazy-wheel-20260706/venv/bin/python -c "import collector.lazy, collector.trtllm.lazy_gemm, collector.network.lazy_nccl"
```

Expected: the wheel listing contains the collector runtime/adapters, and all three imports succeed without a source checkout. The last command does not import the heavy collector modules; GPU integration tests exercise those in worker subprocesses.

- [ ] **Step 5: Run completion checks**

```bash
git diff --check
pytest -m unit tests/unit/collector/lazy tests/unit/collector/test_version_resolver.py tests/unit/sdk/resolution -v
pytest -m gpu tests/integration/collector/test_lazy_gemm_gpu.py tests/integration/collector/test_lazy_nccl_gpu.py tests/integration/collector/test_lazy_mixed_gpu.py -v -s
```

Expected: CPU tests pass anywhere; GPU tests pass on a compatible multi-GPU TRT-LLM host. The runtime is complete when compute fills every independent GPU, collectives reserve exactly their required clique and fabric domain, repeated keys avoid workers entirely, and one failed job does not discard valid sibling records.

- [ ] **Step 6: Commit final validation and packaging coverage**

```bash
git add pyproject.toml collector tests/unit/collector/lazy tests/integration/collector
git commit -m "test: package and validate hardware-aware lazy collection"
```
