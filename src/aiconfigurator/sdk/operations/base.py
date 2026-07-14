# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base class and shared infrastructure for the operations package.

This module defines the ``Operation`` ABC plus two pieces of shared
infrastructure that future op classes will rely on:

- **Class-level ``_data_cache``** — each Operation subclass that owns CSV data
  overrides this in its own class. Keyed by ``(system_path, db_mode)`` so the
  same op type can serve multiple databases in one process.
- **``_load_data_call_count`` instrumentation** — used by tests to assert
  which op classes actually loaded data during a model run. The expected set
  for Minimax M2.5 NVFP4 is the canonical lazy-load success assertion
  (see ``~/forks/sdk-refactor-regression/tests/test_load_data_counts.py``).
- **``supported_quant_modes`` classmethod** — placeholder API used by
  ``inference_session`` post-Phase-4 to build the support-matrix warning.
  Default returns the empty set; ops with quant-mode-keyed CSVs override.

``clear_all_op_caches()`` is a module-level utility that walks every
``Operation`` subclass and clears both its data cache and any LRU on
``query``. Exported from the ``aiconfigurator.sdk.operations`` package — same
function powers a pytest ``autouse`` fixture and serves as a manual eviction
lever for long-running webapps.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase
    from aiconfigurator.sdk.resolution.fallback import FallbackRecord
    from aiconfigurator.sdk.resolution.session import HybridFallbackValue, ResolutionSession
    from aiconfigurator.sdk.resolution.types import MeasurementProtocol, MeasurementRecord, MeasurementRequest

logger = logging.getLogger(__name__)


def _read_perf_rows(perf_file: str) -> list[dict[str, object]]:
    if perf_file.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Loading parquet perf data requires the 'pyarrow' package. "
                "Install aiconfigurator with its declared runtime dependencies."
            ) from exc
        return [
            {key: "" if value is None else value for key, value in row.items()}
            for row in pq.read_table(perf_file).to_pylist()
        ]

    with open(perf_file, encoding="utf-8", newline="") as f:
        return [{key: "" if value is None else value for key, value in row.items()} for row in csv.DictReader(f)]


def _resolve_perf_data_path(perf_file: str) -> str:
    if os.path.exists(perf_file):
        return perf_file
    stem, suffix = os.path.splitext(perf_file)
    if suffix.lower() == ".parquet":
        legacy_file = f"{stem}.txt"
        if os.path.exists(legacy_file):
            return legacy_file
    return perf_file


def _read_filtered_rows(file_or_sources):
    """Read perf rows from one or more sources. Used by every ``load_*_data``
    in this package.

    Accepts:
      - A single path string: yields all rows. Returns ``None`` if the file is
        missing, an empty list if it exists but has no rows. Preserves the
        legacy distinction the per-op ``load_*`` functions rely on.
      - An iterable of ``(path, kernel_source_filter)`` tuples: yields rows
        from each source in order; missing files are skipped; rows are
        filtered by ``kernel_source`` when a filter is provided. Returns
        ``None`` only if **every** path is missing.

    The order of the returned list mirrors the order of the input sources, so
    when the per-row loaders skip on key conflict, the earliest source wins on
    every coordinate — same first-wins semantic the shared-layer loader needs
    without a separate merge step.

    Lives here (not in ``perf_database``) so the per-op-module loaders can
    import it without a circular dependency on ``perf_database`` at module
    load time.
    """
    if isinstance(file_or_sources, str):
        path = _resolve_perf_data_path(file_or_sources)
        if not os.path.exists(path):
            return None
        return _read_perf_rows(path)

    rows: list[dict] = []
    any_exists = False
    for path, ks_filter in file_or_sources:
        path = _resolve_perf_data_path(path)
        if not os.path.exists(path):
            continue
        any_exists = True
        for row in _read_perf_rows(path):
            if ks_filter is None or row.get("kernel_source") in ks_filter:
                rows.append(
                    {
                        **row,
                        "__aic_source_path": path,
                        "__aic_source_backend": os.path.basename(os.path.dirname(os.path.dirname(path))),
                        "__aic_source_version": os.path.basename(os.path.dirname(path)),
                    }
                )
    return rows if any_exists else None


class Operation:
    """
    Base operation class.

    Note: query() returns PerformanceResult (float-like) instead of plain float.
    The class behaves as a float for backward compatibility while carrying
    energy data and a ``source`` tag ("silicon" / "empirical" / "mixed").
    """

    # Subclasses that own CSV data override this. Keyed by (system_path, db_mode).
    _data_cache: ClassVar[dict] = {}

    # Test/observability counter. Each subclass's load_data() calls
    # Operation._record_load(cls) after a successful parse (NOT on cache hit).
    _load_data_call_count: ClassVar[dict[type, int]] = defaultdict(int)

    # Context-parallel opt-in. Subclasses set True after auditing how they
    # respond to ``seq_split``. Constructing an op with ``seq_split > 1`` on a
    # class that has NOT opted in raises -- protects against a new op silently
    # mis-modeling CP. Token-major ops (GEMM/Embedding/ElementWise/NCCL/AR/P2P)
    # divide their per-rank token count ``x`` by ``self._seq_split`` in query().
    _CP_AWARE: ClassVar[bool] = False

    # Composite operations opt in when their query_with_resolution() method
    # owns a complete descendant walk. Merely overriding that method (for
    # instrumentation or leaf-specific behavior) does not imply ownership.
    _OWNS_RESOLUTION_WALK: ClassVar[bool] = False

    # Explicitly reviewed analytical operations bypass exact evidence lookup in
    # resolving mode. Shape-dependent operations override
    # ``is_resolution_deterministic`` instead of setting this class capability.
    _RESOLUTION_DETERMINISTIC: ClassVar[bool] = False

    # Static namespace used by capability preflight. Runtime query tracing
    # remains authoritative for the exact MeasurementRequest and shape.
    _RESOLUTION_NAMESPACE: ClassVar[str | None] = None

    def __init__(self, name: str, scale_factor: float, *, seq_split: int = 1) -> None:
        if seq_split > 1 and not self._CP_AWARE:
            raise NotImplementedError(
                f"{type(self).__name__} has not been audited for context parallelism "
                f"(seq_split={seq_split}). Set ``_CP_AWARE = True`` on the class after "
                f"verifying query() divides its token-count input by self._seq_split "
                f"(or is handled CP-style-specifically at the model construction site)."
            )
        self._name = name
        self._scale_factor = scale_factor
        # Sequence-axis shard factor (= cp_size under context parallelism). Token-
        # major ops divide ``x`` by this in query(); default 1 means no shard.
        self._seq_split: int = seq_split

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Return latency (scaled by ``scale_factor``) plus energy/source data."""
        raise NotImplementedError

    def normalize_perf_query(self, **kwargs: object) -> Mapping[str, object]:
        """Return the canonical physical query consumed by exact lookup paths.

        Operations with aliases, sharding, or shape transforms override this
        method and call it from their ordinary ``query`` implementation. The
        identity default preserves existing operations during migration.
        """
        return dict(kwargs)

    def is_resolution_deterministic(self, **kwargs: object) -> bool:
        """Whether this exact invocation is a reviewed analytical result."""
        del kwargs
        return self._RESOLUTION_DETERMINISTIC

    def resolution_capabilities(self):
        """Classify this reachable op without executing or forecasting it."""
        from aiconfigurator.collector.preflight import OperationCapability, OperationKind

        operation = type(self).__name__
        if self._RESOLUTION_NAMESPACE is not None:
            kind = OperationKind.MEASURED
            namespace = self._RESOLUTION_NAMESPACE
        elif self.is_resolution_deterministic():
            kind = OperationKind.DETERMINISTIC
            namespace = None
        elif self._OWNS_RESOLUTION_WALK:
            kind = OperationKind.COMPOSITION_ONLY
            namespace = None
        else:
            kind = OperationKind.UNSUPPORTED
            namespace = None
        return (OperationCapability(operation, kind, namespace),)

    def _normalize_for_resolution(self, **kwargs: object) -> Mapping[str, object]:
        normalized_query = self.normalize_perf_query(**kwargs)
        if not isinstance(normalized_query, Mapping):
            raise TypeError(
                f"{type(self).__name__}.normalize_perf_query() must return a Mapping, "
                f"got {type(normalized_query).__name__}"
            )
        return normalized_query

    def measurement_request(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        **kwargs,
    ) -> MeasurementRequest | None:
        """Build one exact lazy-measurement request, or report no adapter."""
        return None

    def _measurement_request_from_normalized(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> MeasurementRequest | None:
        """Dispatch a normalized lookup without changing legacy hook kwargs."""
        del normalized_query
        return self.measurement_request(database, protocol, **kwargs)

    def curated_exact_result(self, database: PerfDatabase, **kwargs) -> PerformanceResult | None:
        """Return one final, already-scaled literal curated row, or ``None``; never interpolate."""
        return None

    def _curated_exact_result_from_normalized(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult | None:
        """Dispatch an exact probe without changing legacy hook kwargs."""
        del normalized_query
        return self.curated_exact_result(database, **kwargs)

    def provisional_result(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult:
        """Return the ordinary approximate value used only for dependency discovery.

        Covered operations may override this seam to consume the already-normalized
        mapping without repeating normalization. The compatibility default preserves
        the existing ordinary query path while operations migrate. A measured-only
        operation may have no approximate row; that expected miss becomes a zero
        discovery placeholder and remains protected by the session taint.
        """
        del normalized_query
        from aiconfigurator.sdk.perf_database import _MISSING_SILICON_DATA_EXCEPTIONS

        try:
            return self.query(database, **kwargs)
        except _MISSING_SILICON_DATA_EXCEPTIONS:
            return PerformanceResult(0.0, energy=0.0, source="unresolved")

    def performance_from_record(self, record: MeasurementRecord, **kwargs) -> PerformanceResult:
        """Convert validated overlay evidence using this operation's scaling."""
        return record.performance_result(scale_factor=self._scale_factor)

    def hybrid_fallback_value(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> HybridFallbackValue:
        """Return one unscaled physical HYBRID value for durable per-key fallback."""

        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.perf_database import _get_configured_database_view
        from aiconfigurator.sdk.resolution.session import HybridFallbackValue

        if not math.isfinite(self._scale_factor) or self._scale_factor <= 0:
            raise ValueError("HYBRID fallback requires a finite positive operation scale factor")
        hybrid_database = _get_configured_database_view(
            database,
            common.DatabaseMode.HYBRID,
            getattr(database, "transfer_policy", None),
        )
        result = self.provisional_result(
            hybrid_database,
            normalized_query=normalized_query,
            **kwargs,
        )
        return HybridFallbackValue(
            latency_ms=float(result) / self._scale_factor,
            source=result.source,
            provenance=dict(result.provenance),
        )

    def performance_from_fallback(self, record: FallbackRecord, **kwargs) -> PerformanceResult:
        """Apply consumer scaling once to a durable unscaled HYBRID fallback."""

        del kwargs
        return PerformanceResult(
            record.latency_ms * self._scale_factor,
            energy=0.0,
            source="hybrid",
            provenance={
                "fallback_identity": record.identity_digest,
                "fallback_path": str(record.path),
                "hybrid_source": record.hybrid_source,
                "prediction_revision": record.prediction_revision,
                "measurement_failure": {
                    "code": record.measurement_failure.code.value,
                    "operation": record.measurement_failure.operation,
                    "detail": record.measurement_failure.detail,
                },
            },
        )

    def query_with_resolution(
        self,
        database: PerfDatabase,
        *,
        session: ResolutionSession | None = None,
        **kwargs,
    ) -> PerformanceResult:
        """Query exact evidence and record a lazy miss when a session is supplied."""
        if session is None or self.is_resolution_deterministic(**kwargs):
            return self.query(database, **kwargs)

        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.perf_database import (
            _MISSING_SILICON_DATA_EXCEPTIONS,
            _get_configured_database_view,
        )

        exact_database = _get_configured_database_view(
            database,
            common.DatabaseMode.SILICON,
            getattr(database, "transfer_policy", None),
        )
        normalized_query = self._normalize_for_resolution(**kwargs)
        request = self._measurement_request_from_normalized(
            exact_database,
            session.protocol,
            normalized_query=normalized_query,
            **kwargs,
        )
        binding_error: Exception | None = None
        if request is not None:
            try:
                request = session.bind_request(request)
            except Exception as error:
                binding_error = error
            else:
                record = session.lookup(request.key, request.protocol, consumer=self._name)
                if record is not None:
                    return self.performance_from_record(record, **kwargs)

        curated = self._curated_exact_result_from_normalized(
            exact_database,
            normalized_query=normalized_query,
            **kwargs,
        )
        if curated is not None:
            if request is not None:
                curated_evidence = {
                    "latency_ms": float(curated),
                    "energy_wms": float(curated.energy),
                    "provenance": dict(curated.provenance),
                }
                try:
                    session.record_curated_hit(request.key, self._name, evidence_link=curated_evidence)
                except (TypeError, ValueError):
                    # Evidence reporting must never change the float-compatible
                    # curated query contract. Retain the exact value and mark an
                    # opaque/non-finite provenance payload as unavailable.
                    session.record_curated_hit(
                        request.key,
                        self._name,
                        evidence_link={
                            "latency_ms": float(curated),
                            "energy_wms": float(curated.energy),
                            "provenance": {"unavailable": "not_json_safe"},
                        },
                    )
            return curated

        if request is not None and binding_error is None:
            fallback = session.lookup_fallback(request.key, self._name)
            if fallback is not None:
                return self.performance_from_fallback(fallback, **kwargs)

        if request is None or binding_error is not None:
            if binding_error is not None:
                session.record_binding_error(self._name, binding_error)
            else:
                session.record_missing_adapter(
                    self._name,
                    RuntimeError("operation has no literal exact row or lazy adapter for this query"),
                )
        else:
            normalized_snapshot = dict(normalized_query)
            kwargs_snapshot = dict(kwargs)

            def resolve_hybrid() -> HybridFallbackValue:
                return self.hybrid_fallback_value(
                    database,
                    normalized_query=normalized_snapshot,
                    **kwargs_snapshot,
                )

            session.record_miss(
                request,
                self._name,
                hybrid_resolver=resolve_hybrid,
            )
        session.mark_tainted(self._name)
        try:
            return self.provisional_result(
                database,
                normalized_query=normalized_query,
                **kwargs,
            )
        except _MISSING_SILICON_DATA_EXCEPTIONS:
            return PerformanceResult(0.0, energy=0.0, source="unresolved")

    def get_weights(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Subclasses with CSV data override; default no-op for
        ops like ``ElementWise`` that compute analytically from system spec.

        The full ``database`` is passed (not just ``system_path``/``system_spec``)
        so subclasses can derive their own cache key plus reuse PerfDatabase
        helpers like ``_build_op_sources`` for HYBRID-mode source discovery."""
        return None

    @classmethod
    def clear_cache(cls):
        """Clear this op's data cache and any LRU on ``query``. Subclasses
        with their own ``_data_cache`` override the class attribute; if a
        subclass never declared one, fall back to evicting the shared
        ``Operation._data_cache`` so ``clear_all_op_caches()`` doesn't
        silently skip it."""
        cache = cls.__dict__.get("_data_cache")
        if cache is None:
            cache = Operation._data_cache
        cache.clear()
        # query may be wrapped in functools.lru_cache — clear if present.
        query = cls.__dict__.get("query")
        if query is not None and hasattr(query, "cache_clear"):
            query.cache_clear()

    @classmethod
    def supported_quant_modes(cls, database: PerfDatabase) -> set:
        """Return the quant modes for which this op has CSV data on the
        given database. Default empty — ops with quant-mode-keyed data
        override. Used by ``_update_support_matrix`` (moves to
        ``inference_session`` in ISSUE-16).

        Takes the full ``database`` for symmetry with ``load_data``."""
        return set()

    @classmethod
    def _record_load(cls):
        """Subclasses call this from load_data() after a successful parse,
        NOT on a cache hit. The instrumentation lets tests assert which op
        classes loaded for a given model run."""
        Operation._load_data_call_count[cls] += 1


def _all_operation_subclasses(root: type = Operation) -> set[type]:
    """Recursively collect every Operation subclass currently imported."""
    seen: set[type] = set()
    stack: list[type] = [root]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def clear_all_op_caches() -> None:
    """Walk every imported Operation subclass and call its ``clear_cache()``.

    Used by:
    - production callers (long-running webapps) that need a manual eviction
      lever; the per-op ``_data_cache`` is process-wide and never auto-evicts
    - test helpers that need a fully clean slate (the conftest autouse
      fixture clears only the counter, not data caches — clearing the
      caches would force a fresh-disk reload mid-suite)

    Also clears empirical utilization grids and the shared instrumentation
    counter. Util grids are derived from per-op data, so retaining them after
    their source caches are evicted can mix an old custom ``systems_root`` or
    shared-layer view into newly loaded data.

    Note: this does NOT clear the ``@functools.lru_cache`` on the
    ``PerfDatabase.query_*`` wrappers — those caches live on each database
    instance and must be cleared separately via
    ``database.clear_runtime_caches()`` if callers also want to invalidate
    interpolated/extrapolated query results."""
    for cls in _all_operation_subclasses():
        cls.clear_cache()
    # Import lazily to avoid a base <-> util_empirical module cycle at import
    # time. This is part of the same eviction contract as the per-op caches.
    from aiconfigurator.sdk.operations import util_empirical

    util_empirical.clear_grid_cache()
    Operation._load_data_call_count.clear()


def warm_all_op_data(database: PerfDatabase) -> None:
    """Eagerly call ``load_data`` on every ``Operation`` subclass against
    ``database``.

    The lazy-load contract (lazy per-op data ownership) defers per-op CSV reads until the
    first query (or the first read of ``database.supported_quant_mode``
    for the op's key). Diagnostic tooling that walks every op's instance
    attribute directly — notebooks, sanity-check scripts, support-matrix
    dumpers — wants the legacy "everything loaded" semantics; this
    helper restores them in one call.

    Idempotent: every ``load_data`` is cache-key gated, so calling this
    repeatedly is cheap. Op classes that don't own CSV data inherit the
    base ``Operation.load_data`` no-op and are walked without effect.

    Production callers that read ``database.supported_quant_mode[<key>]``
    or call ``database.query_<op>(...)`` should NOT use this — those
    paths trigger the lazy load on the ops they actually need, which is
    the whole point of lazy per-op data ownership."""
    for cls in _all_operation_subclasses():
        cls.load_data(database)
