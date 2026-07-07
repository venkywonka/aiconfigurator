# AIC Lazy Performance Resolution Core Implementation Plan

> **V1.2 scope note (2026-07-07):** The completed core remains the substrate,
> but covered operations now follow
> `../specs/2026-07-07-aic-dsv4-online-collection-v1-2-design.md` and
> `2026-07-07-aic-dsv4-online-collection-v1-2.md`. Their ordinary lookup,
> literal-exact probe, and resolving request must share one normalization path.
> Do not add `EvidenceQuery`, `dataset_id`, or `collector_ref`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AIC's pure-Python exact-key, overlay, miss-deduplication, and one-requery resolution core without changing ordinary prediction behavior.

**Architecture:** Existing `Operation.query()` remains the pure/default contract. An optional `ResolutionSession` adds overlay-first lookup, shape-local `MissSet` discovery, executor dispatch, record commit, and one requery. Composite operations recurse through the same resolution-aware method so one callback discovers every visible miss.

**Tech Stack:** Python 3.10+, dataclasses, `sqlite3`, `hashlib`, AIC `PerformanceResult`, pytest.

---

## Source prerequisite

Read `docs/plans/2026-07-06-dynamic-lazy-perf-collection-design.md` first. Execute this plan in the AIC repository on a branch containing `upstream/main` commit `0828d6b7e4a7880079443b1c6f9c148d85bdbf54` or a newer commit that passes Task 0. Do not touch Dynamo or real GPU collectors in this plan; use a fake executor so this subsystem is independently testable.

### Task 0: Verify the implementation baseline

**Files:** no changes

- [ ] **Step 1: Confirm the branch contains the reviewed source baseline**

Run:

```bash
git merge-base --is-ancestor 0828d6b7e4a7880079443b1c6f9c148d85bdbf54 HEAD
test -f src/spica/evaluator.py
rg -n "def _get_configured_database_view|should_use_rust_engine_step|class FallbackOp|_CP_AWARE" \
  src/aiconfigurator/sdk/perf_database.py \
  src/aiconfigurator/sdk/backends/base_backend.py \
  src/aiconfigurator/sdk/operations/overlap.py \
  src/aiconfigurator/sdk/operations/base.py
```

Expected: the ancestor check and file check succeed; the source anchors show immutable configured database views, the Rust engine-step gate, the non-sticky current `FallbackOp`, and context-parallel operation contracts. If a newer source moved any anchor, update this plan before implementation rather than copying stale pseudocode.

- [ ] **Step 2: Establish a clean behavioral baseline**

Run:

```bash
pytest -m unit tests/unit/sdk/operations tests/unit/sdk/backends/test_base_backend.py tests/unit/sdk/test_inference_session.py -v
```

Expected: the selected upstream tests pass before feature changes. Record the exact AIC commit and command result in the implementation PR.

## File map

- Create `src/aiconfigurator/sdk/resolution/__init__.py` — public resolution exports.
- Create `src/aiconfigurator/sdk/resolution/types.py` — canonical keys, requests, records, protocols, failures.
- Create `src/aiconfigurator/sdk/resolution/overlay.py` — append-only SQLite evidence store.
- Create `src/aiconfigurator/sdk/resolution/session.py` — `MissSet`, executor protocol, budgets, callback lifecycle, report.
- Modify `src/aiconfigurator/sdk/operations/base.py` — optional request construction and overlay conversion hooks.
- Modify `src/aiconfigurator/sdk/operations/overlap.py` — recursive discovery for `FallbackOp` and `OverlapOp`.
- Modify `src/aiconfigurator/sdk/backends/base_backend.py` — pass the optional session through op walks and perform one requery.
- Modify `src/aiconfigurator/sdk/inference_session.py` — expose the optional session at the SDK boundary.
- Create `tests/unit/sdk/resolution/test_types.py`.
- Create `tests/unit/sdk/resolution/test_overlay.py`.
- Create `tests/unit/sdk/resolution/test_session.py`.
- Create `tests/unit/sdk/resolution/test_operations.py`.
- Modify `tests/unit/sdk/backends/test_base_backend.py`.
- Modify `tests/unit/sdk/test_inference_session.py`.

### Task 1: Canonical evidence types

**Files:**
- Create: `src/aiconfigurator/sdk/resolution/types.py`
- Create: `src/aiconfigurator/sdk/resolution/__init__.py`
- Test: `tests/unit/sdk/resolution/test_types.py`

- [ ] **Step 1: Write canonical-key and record tests**

```python
import pytest

from aiconfigurator.sdk.resolution.types import MeasurementProtocol, MeasurementRecord, PerfKey

pytestmark = pytest.mark.unit


def test_perf_key_is_order_independent() -> None:
    left = PerfKey.build(
        namespace="trtllm/gemm/v1",
        query={"m": 8, "n": 4096, "k": 4096, "dtype": "fp8"},
        environment={"system": "h100_sxm", "backend_version": "1.2.0"},
    )
    right = PerfKey.build(
        namespace="trtllm/gemm/v1",
        query={"dtype": "fp8", "k": 4096, "n": 4096, "m": 8},
        environment={"backend_version": "1.2.0", "system": "h100_sxm"},
    )
    assert left == right
    assert left.digest == right.digest


def test_record_converts_to_overlay_performance_result() -> None:
    key = PerfKey.build("trtllm/gemm/v1", {"m": 8}, {"system": "h100_sxm"})
    protocol = MeasurementProtocol(
        revision="microbench-v1",
        warmups=3,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )
    record = MeasurementRecord.valid(
        key=key,
        latency_ms=0.125,
        energy_wms=0.5,
        samples_ms=(0.126, 0.124, 0.125),
        protocol=protocol,
        perf_row={"m": 8, "latency": 0.125},
        provenance={"collector_revision": "abc123"},
    )
    result = record.performance_result(scale_factor=2.0)
    assert float(result) == pytest.approx(0.25)
    assert result.energy == pytest.approx(1.0)
    assert result.source == "overlay"
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `pytest -m unit tests/unit/sdk/resolution/test_types.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'aiconfigurator.sdk.resolution'`.

- [ ] **Step 3: Implement canonical JSON identity and evidence dataclasses**

```python
# src/aiconfigurator/sdk/resolution/types.py
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from aiconfigurator.sdk.performance_result import PerformanceResult


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True, slots=True)
class PerfKey:
    namespace: str
    query_json: str
    environment_json: str
    semantic_json: str = "{}"

    @classmethod
    def build(
        cls,
        namespace: str,
        query: Mapping[str, Any],
        environment: Mapping[str, Any],
        semantic: Mapping[str, Any] | None = None,
    ) -> "PerfKey":
        return cls(namespace, canonical_json(query), canonical_json(environment), canonical_json(semantic or {}))

    @property
    def canonical(self) -> str:
        return canonical_json(
            {
                "namespace": self.namespace,
                "query": json.loads(self.query_json),
                "environment": json.loads(self.environment_json),
                "semantic": json.loads(self.semantic_json),
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    revision: str
    warmups: int
    samples: int
    statistic: str = "median"
    timer: str = "cuda_event"
    tuning_revision: str = "none"

    @property
    def canonical(self) -> str:
        return canonical_json(
            {
                "revision": self.revision,
                "warmups": self.warmups,
                "samples": self.samples,
                "statistic": self.statistic,
                "timer": self.timer,
                "tuning_revision": self.tuning_revision,
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MeasurementEnvironment:
    system: str
    backend: str
    backend_version: str
    gpu_class: str
    runtime_versions: Mapping[str, str]
    topology_schema: str | None = None
    topology_fingerprint: str | None = None

    @property
    def canonical(self) -> str:
        return canonical_json(
            {
                "system": self.system,
                "backend": self.backend,
                "backend_version": self.backend_version,
                "gpu_class": self.gpu_class,
                "runtime_versions": self.runtime_versions,
                "topology_schema": self.topology_schema,
                "topology_fingerprint": self.topology_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    op_id: str
    key: PerfKey
    query: Mapping[str, Any]
    environment: MeasurementEnvironment
    semantic_descriptor: Mapping[str, Any]
    protocol: MeasurementProtocol

    def __post_init__(self) -> None:
        if self.key.query_json != canonical_json(self.query):
            raise ValueError("request query does not match PerfKey query")
        if self.key.environment_json != self.environment.canonical:
            raise ValueError("request environment does not match PerfKey environment")
        if self.key.semantic_json != canonical_json(self.semantic_descriptor):
            raise ValueError("request semantic descriptor does not match PerfKey semantic identity")


class RecordStatus(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"
    FAILED = "failed"


class ResolutionPolicy(StrEnum):
    PURE = "pure"
    OBSERVE_ONLY = "observe_only"
    MEASURE_ON_MISS = "measure_on_miss"


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    key: PerfKey
    status: RecordStatus
    latency_ms: float | None
    energy_wms: float
    samples_ms: tuple[float, ...]
    protocol: MeasurementProtocol
    perf_row: Mapping[str, Any]
    provenance: Mapping[str, Any]
    failure_code: UnresolvedCode | None = None
    failure_reason: str | None = None
    sequence: int | None = field(default=None, compare=False)

    @classmethod
    def valid(cls, **kwargs: Any) -> "MeasurementRecord":
        return cls(status=RecordStatus.VALID, failure_code=None, failure_reason=None, **kwargs)

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_wms) or self.energy_wms < 0:
            raise ValueError("energy_wms must be finite and non-negative")
        if self.status is RecordStatus.VALID:
            if self.latency_ms is None or not math.isfinite(self.latency_ms) or self.latency_ms < 0:
                raise ValueError("valid records require finite non-negative latency_ms")
            if len(self.samples_ms) != self.protocol.samples:
                raise ValueError("sample count must match the measurement protocol")
            if any(not math.isfinite(sample) or sample < 0 for sample in self.samples_ms):
                raise ValueError("measurement samples must be finite and non-negative")
        elif self.latency_ms is not None:
            raise ValueError("non-valid records cannot provide latency_ms")

    def performance_result(self, scale_factor: float = 1.0) -> PerformanceResult:
        if self.status is not RecordStatus.VALID or self.latency_ms is None:
            raise ValueError("only valid measurement records produce performance results")
        return PerformanceResult(
            self.latency_ms * scale_factor,
            energy=self.energy_wms * scale_factor,
            source="overlay",
        )


class UnresolvedCode(StrEnum):
    MISSING_ADAPTER = "missing_adapter"
    UNSUPPORTED_SHAPE = "unsupported_shape"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    COLLECTOR_FAILED = "collector_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    IDENTITY_MISMATCH = "identity_mismatch"
    INVALID_MEASUREMENT = "invalid_measurement"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RETRY_EXHAUSTED = "retry_exhausted"
    REQUERY_STILL_MISSING = "requery_still_missing"
    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True, slots=True)
class UnresolvedReason:
    code: UnresolvedCode
    operation: str
    detail: str
```

Import `math` for record validation and export these names from `resolution/__init__.py`. Extend the tests with protocol canonicalization, environment/key round-trip and mismatch, a warmup/sample/statistic/timer mismatch, NaN/infinite/negative latency, a sample-count mismatch, and invalid status/latency combinations. `MeasurementEnvironment.runtime_versions`, `MeasurementRequest.query`, semantic descriptors, perf rows, and provenance must be copied into JSON-safe immutable mappings at construction; a caller mutating its original dict after construction must not change a request, record, environment, or digest.

- [ ] **Step 4: Run the type tests**

Run: `pytest -m unit tests/unit/sdk/resolution/test_types.py -v`

Expected: all canonical type, environment, protocol, and record-validation tests pass.

- [ ] **Step 5: Commit the types**

```bash
git add src/aiconfigurator/sdk/resolution tests/unit/sdk/resolution/test_types.py
git commit -m "feat: add lazy performance evidence types"
```

### Task 2: Append-only SQLite overlay

**Files:**
- Create: `src/aiconfigurator/sdk/resolution/overlay.py`
- Test: `tests/unit/sdk/resolution/test_overlay.py`

- [ ] **Step 1: Write overlay precedence and rejection tests**

```python
import pytest

from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.types import MeasurementProtocol, MeasurementRecord, PerfKey, RecordStatus

pytestmark = pytest.mark.unit


def _record(key: PerfKey, latency: float, status: RecordStatus = RecordStatus.VALID) -> MeasurementRecord:
    protocol = MeasurementProtocol("microbench-v1", warmups=3, samples=1)
    return MeasurementRecord(
        key=key,
        status=status,
        latency_ms=latency if status is RecordStatus.VALID else None,
        energy_wms=0.0,
        samples_ms=(latency,),
        protocol=protocol,
        perf_row={"latency": latency},
        provenance={"collector_revision": "r1"},
        failure_reason=None if status is RecordStatus.VALID else "boom",
    )


def test_latest_valid_commit_wins(tmp_path) -> None:
    key = PerfKey.build("gemm/v1", {"m": 8}, {"system": "h100"})
    store = OverlayStore(tmp_path / "overlay.sqlite")
    first = store.append(_record(key, 0.2))
    store.append(_record(key, 9.9, RecordStatus.REJECTED))
    last = store.append(_record(key, 0.1))
    protocol = MeasurementProtocol(
        "microbench-v1",
        warmups=3,
        samples=1,
        statistic="median",
        timer="cuda_event",
        tuning_revision="none",
    )
    hit = store.lookup(key, protocol)
    assert first < last
    assert hit is not None
    assert hit.sequence == last
    assert hit.latency_ms == pytest.approx(0.1)
```

- [ ] **Step 2: Verify the overlay test fails**

Run: `pytest -m unit tests/unit/sdk/resolution/test_overlay.py -v`

Expected: import failure for `aiconfigurator.sdk.resolution.overlay`.

- [ ] **Step 3: Implement the immutable-record schema and queries**

Use one SQLite transaction per append, WAL mode, `PRAGMA busy_timeout=30000`, and `INTEGER PRIMARY KEY AUTOINCREMENT` as the monotonic sequence. Each process opens its own connection; connections are never pickled or inherited across `fork`/`spawn`. Store canonical key JSON and JSON-encoded evidence; never update a record row.

```sql
CREATE TABLE IF NOT EXISTS measurement_records (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    key_digest TEXT NOT NULL,
    key_json TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    energy_wms REAL NOT NULL,
    samples_json TEXT NOT NULL,
    statistic TEXT NOT NULL,
    protocol_digest TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    perf_row_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    failure_code TEXT,
    failure_reason TEXT
);
CREATE INDEX IF NOT EXISTS measurement_key_sequence
ON measurement_records(key_digest, sequence DESC);
```

Open SQLite with `isolation_level=None`, WAL mode, and `PRAGMA busy_timeout=30000`. Reads and `PRAGMA data_version` must be statement-scoped autocommit operations so another connection's commit becomes visible immediately. `append()` uses an explicit short `BEGIN IMMEDIATE` / `COMMIT` transaction and rolls back on every exception; no connection or cursor crosses a process boundary.

Implement `OverlayStore.append(record) -> int`, `lookup(key, protocol) -> MeasurementRecord | None`, and `close()`. Maintain an in-memory map keyed by `(key.digest, protocol.digest)`. Before using it, compare `PRAGMA data_version` with the value seen on the previous lookup; clear the map when another process/connection committed. Lookup then checks the map, queries SQLite on a miss, and populates the map. SQLite lookup must filter `status='valid'` and `protocol_digest=protocol.digest`; compare both stored `key_json` to `key.canonical` and stored `protocol_json` to `protocol.canonical` after digest matches; and order by `sequence DESC`. Append updates the local map only when the appended record is valid and has a greater sequence. Update the test to pass a matching full `MeasurementProtocol`, monkeypatch the SQLite query method to prove the second unchanged lookup is in-memory, add individual warmup/sample/statistic/timer/tuning changes that are not reused, hold two stores open while one appends to prove autocommit plus `data_version` invalidates the first store's cache, and verify an append exception leaves no partial row or open transaction.

- [ ] **Step 4: Run overlay tests including process reopen**

Add a test that closes and reopens the store before lookup.

Run: `pytest -m unit tests/unit/sdk/resolution/test_overlay.py -v`

Expected: all overlay tests pass.

- [ ] **Step 5: Commit the overlay**

```bash
git add src/aiconfigurator/sdk/resolution/overlay.py tests/unit/sdk/resolution/test_overlay.py
git commit -m "feat: add append-only performance overlay"
```

### Task 3: MissSet, budgets, executor protocol, and one-requery lifecycle

**Files:**
- Create: `src/aiconfigurator/sdk/resolution/session.py`
- Test: `tests/unit/sdk/resolution/test_session.py`

- [ ] **Step 1: Write deduplication and callback lifecycle tests**

```python
def test_miss_set_measures_one_key_for_two_consumers(session, request, fake_executor) -> None:
    session.record_miss(request, "layer.0.qkv")
    session.record_miss(request, "layer.1.qkv")
    session.resolve_pending()
    assert fake_executor.request_batches == [[request]]
    assert session.report.unique_misses == 1
    assert session.report.consumer_misses == 2


def test_execute_callback_requeries_exactly_once(session) -> None:
    calls = 0

    def query():
        nonlocal calls
        calls += 1
        if calls == 1:
            session.record_miss(session.test_request, "gemm")
            return 0.0
        return 1.25

    assert session.execute_callback(query) == 1.25
    assert calls == 2
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest -m unit tests/unit/sdk/resolution/test_session.py -v`

Expected: import failure for `resolution.session`.

- [ ] **Step 3: Implement the session contracts**

```python
from __future__ import annotations

import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, NoReturn, Protocol, Sequence, TypeVar

from .overlay import OverlayStore
from .types import (
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    ResolutionPolicy,
    UnresolvedCode,
    UnresolvedReason,
)

T = TypeVar("T")


class CancellationToken(Protocol):
    def cancelled(self) -> bool:
        return False


class _NeverCancelled:
    def cancelled(self) -> bool:
        return False


class MeasurementExecutor(Protocol):
    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> Sequence[MeasurementRecord]:
        raise NotImplementedError


@dataclass(slots=True)
class MissEntry:
    request: MeasurementRequest
    consumers: list[str] = field(default_factory=list)


class MissSet:
    def __init__(self) -> None:
        self._entries: "OrderedDict[PerfKey, MissEntry]" = OrderedDict()

    def record(self, request: MeasurementRequest, consumer: str) -> None:
        entry = self._entries.get(request.key)
        if entry is None:
            self._entries[request.key] = MissEntry(request, [consumer])
            return
        if entry.request != request:
            raise ValueError(f"conflicting requests share PerfKey {request.key.digest}")
        entry.consumers.append(consumer)

    def requests(self) -> tuple[MeasurementRequest, ...]:
        return tuple(entry.request for entry in self._entries.values())

    def entries(self) -> tuple[MissEntry, ...]:
        return tuple(self._entries.values())

    def clear(self) -> None:
        self._entries.clear()

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class ResolutionBudget:
    max_new_keys: int
    max_wall_seconds: float
    max_transient_attempts_per_key: int = 2


@dataclass(slots=True)
class ResolutionReport:
    overlay_hits: int = 0
    unique_misses: int = 0
    consumer_misses: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    collection_seconds: float = 0.0
    unresolved: list[UnresolvedReason] = field(default_factory=list)


class ResolutionFailed(RuntimeError):
    def __init__(self, reasons: Sequence[UnresolvedReason]):
        self.reasons = tuple(reasons)
        super().__init__("; ".join(f"{r.code}: {r.operation}: {r.detail}" for r in reasons))


class ResolutionSession:
    def __init__(
        self,
        overlay: OverlayStore,
        executor: MeasurementExecutor,
        budget: ResolutionBudget,
        protocol: MeasurementProtocol,
        *,
        policy: ResolutionPolicy = ResolutionPolicy.MEASURE_ON_MISS,
        clock: Callable[[], float] = time.monotonic,
        cancellation: CancellationToken | None = None,
    ) -> None:
        if policy is ResolutionPolicy.PURE:
            raise ValueError("pure prediction does not construct a ResolutionSession")
        self.overlay = overlay
        self.executor = executor
        self.budget = budget
        self.protocol = protocol
        self.policy = policy
        self.report = ResolutionReport()
        self._clock = clock
        self._cancellation = cancellation or _NeverCancelled()
        self._charged_keys: set[PerfKey] = set()
        self._callback_lock = threading.RLock()
        self._misses = MissSet()
        self._unresolved: list[UnresolvedReason] = []
        self._negative: dict[PerfKey, UnresolvedReason] = {}
        self._transient_attempts: dict[PerfKey, int] = {}

    def lookup(self, key: PerfKey) -> MeasurementRecord | None:
        record = self.overlay.lookup(key, self.protocol)
        if record is not None:
            self.report.overlay_hits += 1
        return record

    def record_miss(self, request: MeasurementRequest, consumer: str) -> None:
        if request.protocol != self.protocol:
            self.record_unresolved(
                UnresolvedReason(
                    UnresolvedCode.IDENTITY_MISMATCH,
                    consumer,
                    "request protocol does not match resolution session",
                )
            )
            return
        previous_failure = self._negative.get(request.key)
        if previous_failure is not None:
            self.record_unresolved(previous_failure)
            return
        if self._transient_attempts.get(request.key, 0) >= self.budget.max_transient_attempts_per_key:
            self.record_unresolved(
                UnresolvedReason(
                    UnresolvedCode.RETRY_EXHAUSTED,
                    consumer,
                    f"transient retry budget exhausted for {request.key.digest}",
                )
            )
            return
        self._misses.record(request, consumer)
        self.report.consumer_misses += 1

    def record_missing_adapter(self, operation: str, error: Exception) -> None:
        self.record_unresolved(
            UnresolvedReason(UnresolvedCode.MISSING_ADAPTER, operation, str(error))
        )

    def record_unresolved(self, reason: UnresolvedReason) -> None:
        self._unresolved.append(reason)

    def checkpoint(self) -> tuple[int, int]:
        return len(self._misses), len(self._unresolved)

    def changed_since(self, checkpoint: tuple[int, int]) -> bool:
        return self.checkpoint() != checkpoint

    def _fail(self, reasons: Sequence[UnresolvedReason]) -> NoReturn:
        self.report.unresolved.extend(reasons)
        raise ResolutionFailed(reasons)

    def _remember_failure(self, key: PerfKey, reason: UnresolvedReason) -> None:
        deterministic = {
            UnresolvedCode.MISSING_ADAPTER,
            UnresolvedCode.UNSUPPORTED_SHAPE,
            UnresolvedCode.TOPOLOGY_MISMATCH,
            UnresolvedCode.IDENTITY_MISMATCH,
            UnresolvedCode.INVALID_MEASUREMENT,
        }
        if reason.code in deterministic:
            self._negative[key] = reason
        elif reason.code not in {UnresolvedCode.CANCELLED, UnresolvedCode.BUDGET_EXHAUSTED}:
            self._transient_attempts[key] = self._transient_attempts.get(key, 0) + 1

    def resolve_pending(self) -> None:
        if self._unresolved:
            self._fail(self._unresolved)
        discovered = self._misses.requests()
        if not discovered:
            return
        self.report.unique_misses += len(discovered)
        requests = tuple(request for request in discovered if self.lookup(request.key) is None)
        if not requests:
            return

        if self._cancellation.cancelled():
            self._fail(
                [
                    UnresolvedReason(UnresolvedCode.CANCELLED, request.op_id, "session cancelled")
                    for request in requests
                ]
            )

        if self.policy is ResolutionPolicy.OBSERVE_ONLY:
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.OBSERVE_ONLY,
                        request.op_id,
                        f"observed unresolved key {request.key.digest}",
                    )
                    for request in requests
                ]
            )

        newly_charged = tuple(request for request in requests if request.key not in self._charged_keys)
        remaining_keys = self.budget.max_new_keys - len(self._charged_keys)
        remaining_seconds = self.budget.max_wall_seconds - self.report.collection_seconds
        if len(newly_charged) > remaining_keys or remaining_seconds <= 0:
            self._fail(
                [
                    UnresolvedReason(
                        UnresolvedCode.BUDGET_EXHAUSTED,
                        request.op_id,
                        f"need {len(newly_charged)} new keys/{remaining_seconds:.3f}s remaining; "
                        f"budget has {remaining_keys} keys",
                    )
                    for request in requests
                ]
            )

        self._charged_keys.update(request.key for request in newly_charged)

        started = self._clock()
        try:
            records = tuple(
                self.executor.execute(
                    requests,
                    deadline_monotonic=started + remaining_seconds,
                    cancellation=self._cancellation,
                )
            )
        except Exception as error:
            self.report.collection_seconds += self._clock() - started
            reasons = [
                UnresolvedReason(UnresolvedCode.COLLECTOR_FAILED, request.op_id, str(error))
                for request in requests
            ]
            for request, reason in zip(requests, reasons, strict=True):
                self._remember_failure(request.key, reason)
            self._fail(reasons)
        self.report.collection_seconds += self._clock() - started

        requested = {request.key: request for request in requests}
        records_by_key: dict[PerfKey, list[MeasurementRecord]] = {}
        failures: list[UnresolvedReason] = []
        invalid_keys: set[PerfKey] = set()
        for record in records:
            request = requested.get(record.key)
            if request is None:
                failures.append(
                    UnresolvedReason(
                        UnresolvedCode.INVALID_MEASUREMENT,
                        record.key.namespace,
                        f"executor returned unrequested key {record.key.digest}",
                    )
                )
                continue
            if record.protocol != request.protocol:
                reason = UnresolvedReason(
                    UnresolvedCode.IDENTITY_MISMATCH,
                    request.op_id,
                    "record protocol does not exactly match request",
                )
                failures.append(reason)
                self._remember_failure(record.key, reason)
                invalid_keys.add(record.key)
                self.report.rejected_records += 1
                continue
            self.overlay.append(record)
            records_by_key.setdefault(record.key, []).append(record)
            self.report.accepted_records += int(record.status is RecordStatus.VALID)
            self.report.rejected_records += int(record.status is not RecordStatus.VALID)

        for request in requests:
            if request.key in invalid_keys:
                continue
            matches = records_by_key.get(request.key, [])
            if len(matches) != 1:
                reason = UnresolvedReason(
                    UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    f"expected one record for {request.key.digest}, got {len(matches)}",
                )
                failures.append(reason)
                self._remember_failure(request.key, reason)
                continue
            record = matches[0]
            if record.status is RecordStatus.FAILED:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.COLLECTOR_FAILED,
                    request.op_id,
                    record.failure_reason or "collector failed",
                )
                failures.append(reason)
                self._remember_failure(request.key, reason)
            elif record.status is not RecordStatus.VALID or record.latency_ms is None:
                reason = UnresolvedReason(
                    record.failure_code or UnresolvedCode.INVALID_MEASUREMENT,
                    request.op_id,
                    record.failure_reason or "measurement rejected",
                )
                failures.append(reason)
                self._remember_failure(request.key, reason)
        if self._cancellation.cancelled():
            failures.append(
                UnresolvedReason(UnresolvedCode.CANCELLED, "resolution_session", "session cancelled during collection")
            )
        if failures:
            self._fail(failures)

    def execute_callback(self, query: Callable[[], T]) -> T:
        with self._callback_lock:
            return self._execute_callback_locked(query)

    def _execute_callback_locked(self, query: Callable[[], T]) -> T:
        self._misses.clear()
        self._unresolved.clear()
        first = query()
        if not self._misses and not self._unresolved:
            return first
        self.resolve_pending()
        self._misses.clear()
        self._unresolved.clear()
        result = query()
        if self._misses or self._unresolved:
            reasons = list(self._unresolved)
            reasons.extend(
                UnresolvedReason(
                    UnresolvedCode.REQUERY_STILL_MISSING,
                    entry.request.op_id,
                    f"key {entry.request.key.digest} remained absent after collection",
                )
                for entry in self._misses.entries()
            )
            self._fail(reasons)
        return result
```

Define `_NeverCancelled.cancelled()` to return `False`. Add tests for two threads calling the same cold session (one executor batch, second callback warms after the callback lock), deterministic negative caching, transient timeout/worker-loss retry followed by `RETRY_EXHAUSTED`, cancellation that does not poison later work, pre-dispatch cancellation, cancellation while the fake executor is active, cumulative unique-key budget exhaustion, wall-budget exhaustion with an injected clock, executor exceptions, a missing returned key, duplicate returned records, full protocol mismatch, rejected/failed records being appended for provenance, and a second walk that still misses. Include the exact regression: `max_new_keys=1`, one key times out, the same key is allowed a second dispatch without another key charge, and the next attempt returns `RETRY_EXHAUSTED`; a different key is rejected by `BUDGET_EXHAUSTED`. Assert every failure exposes the exact `UnresolvedCode` above and that partial valid records remain queryable after another request in the same executor batch fails.

- [ ] **Step 4: Run session tests**

Run: `pytest -m unit tests/unit/sdk/resolution/test_session.py -v`

Expected: dedup, budget, unresolved, and one-requery tests pass.

- [ ] **Step 5: Commit the session**

```bash
git add src/aiconfigurator/sdk/resolution/session.py tests/unit/sdk/resolution/test_session.py
git commit -m "feat: add callback-local resolution session"
```

### Task 4: Resolution-aware operations and composites

**Files:**
- Modify: `src/aiconfigurator/sdk/operations/base.py` (`Operation` contract)
- Modify: `src/aiconfigurator/sdk/operations/overlap.py` (`FallbackOp` and `OverlapOp`)
- Test: `tests/unit/sdk/resolution/test_operations.py`

- [ ] **Step 1: Write tests for overlay precedence, literal curated hits, exact misses, and recursive composite discovery**

Use a fake table-backed operation whose ordinary `query` deliberately returns an interpolated SILICON value for an off-grid shape, whose `measurement_request` returns a fixed request, whose `curated_exact_result` returns a result only for a literal row, and whose `performance_from_record` applies its scale. Assert:

```python
assert float(op.query_with_resolution(db, session=session, x=8)) == pytest.approx(0.25)
assert executor.calls == 0  # overlay hit bypasses curated lookup
```

Also assert a literal curated row returns without collection, the populated-table off-grid shape records a miss despite ordinary `query()` succeeding through interpolation, and an operation with neither a literal row nor a request records `MISSING_ADAPTER`. The resolution path must never call ordinary interpolating `query()` as its exact-evidence predicate.

For `OverlapOp`, put one missing fake op in each group and assert that both keys enter the same `MissSet`. For current-upstream `FallbackOp`, cover both branches: an adapter-capable primary records its exact miss and does not execute fallback ops; a primary with no adapter checks `curated_exact_result()` and, when no literal row exists, executes the fallback operations without adding `missing_adapter` to the session. Assert no code references the removed sticky `_primary_unavailable` state.

- [ ] **Step 2: Run and verify missing-method failures**

Run: `pytest -m unit tests/unit/sdk/resolution/test_operations.py -v`

Expected: `AttributeError` for `query_with_resolution`.

- [ ] **Step 3: Add the base operation hooks**

```python
def measurement_request(self, database, protocol, **kwargs):
    return None

def curated_exact_result(self, database, **kwargs):
    """Return a literal compatible row, or None. Never interpolate or fall back."""
    return None

def performance_from_record(self, record, **kwargs):
    return record.performance_result(scale_factor=self._scale_factor)

def query_with_resolution(self, database, *, session=None, **kwargs):
    if session is None:
        return self.query(database, **kwargs)
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.perf_database import _get_configured_database_view

    exact_database = _get_configured_database_view(
        database,
        common.DatabaseMode.SILICON,
        getattr(database, "transfer_policy", None),
    )
    request = self.measurement_request(exact_database, session.protocol, **kwargs)
    if request is not None:
        record = session.lookup(request.key)
        if record is not None:
            return self.performance_from_record(record, **kwargs)
    curated = self.curated_exact_result(exact_database, **kwargs)
    if curated is not None:
        return curated
    if request is None:
        session.record_missing_adapter(
            self._name,
            RuntimeError("operation has no literal exact row or lazy adapter for this query"),
        )
    else:
        session.record_miss(request, self._name)
    return PerformanceResult(0.0, energy=0.0, source="unresolved")
```

Import resolution types only under `TYPE_CHECKING` where possible; keep `query()` unchanged. Never assign `_default_database_mode` directly and never call `set_default_database_mode()` on the caller's object. Add a HYBRID-mode test whose empirical/interpolated fallback would otherwise succeed and assert measure-on-miss uses a SILICON configured view plus `curated_exact_result`, forces an off-grid exact miss into the `MissSet`, leaves the caller's database/view and caches unchanged, and is race-safe across two sessions sharing the same root template. This is the guard against mixed evidence, interpolation masquerading as a hit, and mode/cache corruption.

- [ ] **Step 4: Override composites to recurse**

Implement `OverlapOp.query_with_resolution` with the same sum/max/energy logic as `query`, replacing each child call with `child.query_with_resolution(database, session=session, **kwargs)`. Even after one child returns the zero unresolved sentinel, continue both groups so the `MissSet` is complete.

Implement `FallbackOp.query_with_resolution` as follows:

```python
def query_with_resolution(self, database, *, session=None, **kwargs):
    if session is None:
        return self.query(database, **kwargs)
    from aiconfigurator.sdk.perf_database import _get_configured_database_view

    primary_database = _get_configured_database_view(
        database,
        common.DatabaseMode.SILICON,
        getattr(database, "transfer_policy", None),
    )
    primary_request = self._primary.measurement_request(
        primary_database,
        session.protocol,
        **kwargs,
    )
    if primary_request is not None:
        record = session.lookup(primary_request.key)
        if record is not None:
            return self._primary.performance_from_record(record, **kwargs)

    primary_curated = self._primary.curated_exact_result(primary_database, **kwargs)
    if primary_curated is not None:
        return primary_curated

    if primary_request is not None:
        session.record_miss(primary_request, self._primary._name)
        return PerformanceResult(0.0, energy=0.0, source="unresolved")

    total = PerformanceResult(0.0, energy=0.0, source="empirical")
    for op in self._fallback:
        total += op.query_with_resolution(database, session=session, **kwargs)
    return total
```

This mirrors current-upstream fallback intent while replacing its ordinary interpolating probe with literal exact evidence. A primary with neither a literal exact row nor an adapter proceeds directly to fallback children without recording `missing_adapter`. An adapter-capable primary remains preferred and may become an overlay hit after collection. Preserve current logging around the primary attempt and preserve `query()` byte-for-behavior when no session is supplied.

- [ ] **Step 5: Run operation tests**

Run: `pytest -m unit tests/unit/sdk/resolution/test_operations.py -v`

Expected: all operation and composite tests pass.

- [ ] **Step 6: Commit the operation hooks**

```bash
git add src/aiconfigurator/sdk/operations/base.py src/aiconfigurator/sdk/operations/overlap.py tests/unit/sdk/resolution/test_operations.py
git commit -m "feat: discover lazy misses across operation composites"
```

### Task 5: BaseBackend and InferenceSession one-requery integration

**Files:**
- Modify: `src/aiconfigurator/sdk/backends/base_backend.py` (static phase walks and Rust-engine gate)
- Modify: `src/aiconfigurator/sdk/inference_session.py` (`run_static*` SDK boundary)
- Modify: `tests/unit/sdk/backends/test_base_backend.py`
- Modify: `tests/unit/sdk/test_inference_session.py`

- [ ] **Step 1: Add failing backend tests**

Add a model with two fake missing operations sharing one key. Call `inference_session.run_static(runtime_config, "static", resolution_session=session)` and assert:

- the backend op walk executes twice;
- the executor receives one unique request;
- the returned summary contains the overlay value twice, once per consumer;
- calling without `resolution_session` preserves the existing exception/fallback behavior;
- `run_static_latency_only` follows the same lifecycle.
- with a `RuntimeConfig` for which `should_use_rust_engine_step` is true, no-session calls still invoke the existing Rust estimator exactly once and never touch `query_with_resolution`;
- the same config with a session bypasses the Rust estimator, performs the Python discovery/requery walk, and leaves subsequent no-session fast-path calls unchanged.

- [ ] **Step 2: Run the focused tests**

Run: `pytest -m unit tests/unit/sdk/backends/test_base_backend.py tests/unit/sdk/test_inference_session.py -v`

Expected: failures reporting the unexpected `resolution_session` keyword.

- [ ] **Step 3: Thread the optional session through the operation walks**

Add keyword-only `resolution_session: ResolutionSession | None = None` to `run_static` and `run_static_latency_only`, and thread it through `_run_static_breakdown`, `_run_encoder_phase`, `_run_context_phase`, and `_run_generation_phase`. Gate the existing Rust branch as `resolution_session is None and should_use_rust_engine_step(runtime_config)`: the compiled engine cannot discover a complete `MissSet` or read the mutable overlay in V1, while the ordinary path must remain byte-for-behavior fast. Replace each direct query with its phase-equivalent resolving call. Encoder:

```python
result = op.query_with_resolution(
    database,
    session=resolution_session,
    x=x,
    batch_size=eff_batch,
    beam_width=1,
    s=eff_s,
    prefix=0,
    model_name=getattr(model, "model_name", ""),
)
```

Context:

```python
result = op.query_with_resolution(
    database,
    session=resolution_session,
    x=x,
    batch_size=batch_size,
    beam_width=1,
    s=effective_isl,
    prefix=prefix,
    seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
)
```

Generation:

```python
result = op.query_with_resolution(
    database,
    session=resolution_session,
    x=batch_size * beam_width,
    batch_size=batch_size,
    beam_width=beam_width,
    s=isl + i + 1,
    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
)
```

Refactor the current bodies into local `run_once()` closures at the public `run_static` and `run_static_latency_only` boundaries. With no session, return `run_once()` directly. With a session, return `resolution_session.execute_callback(run_once)`. Do not wrap `_run_static_breakdown` or individual operations: `run_static` currently has its own encoder/context/generation flow, while `run_static_latency_only` owns the breakdown flow. Wrapping each public call exactly once guarantees one collection batch and one complete requery without nesting callback state.

- [ ] **Step 4: Expose the session from InferenceSession**

Add the same optional keyword-only argument to `InferenceSession.run_static` and `run_static_latency_only`, forwarding it to the backend. Do not change defaults or positional argument order.

- [ ] **Step 5: Run focused and full unit tests**

Run: `pytest -m unit tests/unit/sdk/backends/test_base_backend.py tests/unit/sdk/test_inference_session.py tests/unit/sdk/resolution -v`

Expected: all focused tests pass.

Run: `pytest -m unit`

Expected: the existing unit suite passes, aside from already documented environment-specific TTY/Rust-network exclusions.

- [ ] **Step 6: Commit SDK integration**

```bash
git add src/aiconfigurator/sdk/backends/base_backend.py src/aiconfigurator/sdk/inference_session.py tests/unit/sdk/backends/test_base_backend.py tests/unit/sdk/test_inference_session.py
git commit -m "feat: resolve exact misses around one AIC callback"
```

### Task 6: Observe-only mode and resolution report serialization

**Files:**
- Modify: `src/aiconfigurator/sdk/resolution/session.py`
- Modify: `src/aiconfigurator/sdk/resolution/types.py`
- Test: `tests/unit/sdk/resolution/test_session.py`

- [ ] **Step 1: Add failing report tests**

Assert that `session.report.to_dict()` contains overlay hits, unique and consumer misses, accepted/rejected records, elapsed collection seconds, and unresolved reasons. Add `policy="observe_only"` and assert it emits the `MissSet` report but raises no executor call.

- [ ] **Step 2: Implement JSON-safe reporting and observe-only policy**

Use the existing `ResolutionPolicy` enum. `PURE` never constructs a session at call sites; `OBSERVE_ONLY` records keys and raises a structured `ResolutionFailed` after discovery; `MEASURE_ON_MISS` executes the lifecycle above. Implement JSON-safe reporting directly on `ResolutionReport`:

```python
def to_dict(self) -> dict[str, object]:
    return {
        "overlay_hits": self.overlay_hits,
        "unique_misses": self.unique_misses,
        "consumer_misses": self.consumer_misses,
        "accepted_records": self.accepted_records,
        "rejected_records": self.rejected_records,
        "collection_seconds": self.collection_seconds,
        "unresolved": [
            {
                "code": reason.code.value,
                "operation": reason.operation,
                "detail": reason.detail,
            }
            for reason in self.unresolved
        ],
    }
```

- [ ] **Step 3: Run resolution and regression tests**

Run: `pytest -m unit tests/unit/sdk/resolution tests/unit/sdk/backends/test_base_backend.py -v`

Expected: all tests pass.

- [ ] **Step 4: Commit report support**

```bash
git add src/aiconfigurator/sdk/resolution tests/unit/sdk/resolution
git commit -m "feat: report lazy resolution evidence and coverage"
```

## Completion check

Run:

```bash
git diff --check
pytest -m unit tests/unit/sdk/resolution tests/unit/sdk/backends/test_base_backend.py tests/unit/sdk/test_inference_session.py -v
```

Expected: no diff errors and all focused tests pass. The subsystem is complete when pure calls are unchanged, fake-executor cold calls perform one deduplicated resolution batch, and repeated/reopened-overlay calls perform zero executor work.
