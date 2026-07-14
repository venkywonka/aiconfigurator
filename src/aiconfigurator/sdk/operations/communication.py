# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Communication ops: NCCL + CustomAllReduce + P2P (ISSUE-07 / AIC-541).

- ``CustomAllReduce`` owns ``custom_allreduce_perf.txt`` (CSV) — keyed by
  ``(quant_mode, tp_size, strategy)``. ``PerfDatabase.query_custom_allreduce``
  delegates here. No SOL clamp, no extrapolation in the legacy
  ``_correct_data`` / ``__init__`` path.

- ``NCCL`` owns ``nccl_perf.txt`` AND the optional oneCCL fallback table.
  ``PerfDatabase.query_nccl`` delegates here. The oneCCL fallback is loaded
  alongside NCCL data because ``query_nccl`` picks between them at query
  time (XPU systems load oneCCL when NCCL is empty).

- ``P2P`` has no silicon table — latency is computed analytically from
  ``inter_node_bw`` + ``p2p_latency``. The base ``Operation.load_data``
  no-op default applies; ``_query_p2p_table`` is factored out for
  parity with the other ops.

Cache key matches every other migrated op: ``(systems_root, system,
backend, version, enable_shared_layer)``. ``_build_op_sources`` early-
exits for ``nccl`` / ``oneccl`` (framework-agnostic dirs, no shared-layer
inheritance), so HYBRID mode doesn't union sibling rows for those.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows
from aiconfigurator.sdk.perf_namespace import perf_namespace
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase
    from aiconfigurator.sdk.resolution.session import HybridFallbackValue

logger = logging.getLogger(__name__)

SGLANG_CUSTOM_ALLREDUCE_MAX_BYTES = 8 * 1024 * 1024


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM and Attention.

    TODO: hoist to ``operations/base.py`` once Phase 3 lands and there
    are 4-5 op families duplicating this helper. Two callers (GEMM,
    Attention) was below the abstraction threshold; with Communication
    + DSA + MLA + Mamba + DSV4 coming, the threshold is now crossed.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class CustomAllReduce(Operation):
    """
    Custom AllReduce operation with power tracking.

    Owns ``_data_cache`` for the custom_allreduce CSV table.
    """

    _data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank AR payload)
    _V1_2_RUNTIME_VERSIONS: ClassVar[dict[str, str]] = {
        "cuda": "13.0",
        "model_profile": "dsv4-v1.2",
        "sglang": "0.5.10",
    }
    _V1_2_PROFILE_COMPATIBILITY: ClassVar[dict[str, object]] = {
        "model_artifact": "sgl-project/DeepSeek-V4-Flash-FP8",
        "serving_mode": "aggregated",
        "tp_size": 4,
        "attention_dp_size": 1,
        "cp_size": 1,
        "pp_size": 1,
        "moe_tp_size": 1,
        "moe_ep_size": 4,
        "nextn": 0,
    }

    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int, *, seq_split: int = 1) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._h = h
        self._tp_size = tp_size
        self._weights = 0.0

    def is_resolution_deterministic(self, **kwargs: object) -> bool:
        """TP1 is a reviewed no-op and needs no measurement evidence."""

        del kwargs
        return self._tp_size == 1

    def resolution_capabilities(self):
        from aiconfigurator.collector.preflight import OperationCapability, OperationKind

        if self.is_resolution_deterministic():
            return (OperationCapability(type(self).__name__, OperationKind.DETERMINISTIC),)
        return (
            OperationCapability(
                type(self).__name__,
                OperationKind.MEASURED,
                perf_namespace("custom_allreduce_perf.txt"),
            ),
            OperationCapability(
                "NCCL",
                OperationKind.MEASURED,
                perf_namespace("nccl_perf.txt"),
            ),
        )

    @staticmethod
    def _uses_nccl_fallback(
        database: PerfDatabase,
        normalized_query: Mapping[str, object],
    ) -> bool:
        """Match SGLang's runtime dispatch when CustomAllReduce is too large."""

        return (
            database.backend == "sglang"
            and int(normalized_query["elements"]) * common.CommQuantMode.half.value.memory
            > SGLANG_CUSTOM_ALLREDUCE_MAX_BYTES
        )

    def _nccl_fallback(
        self,
        quant_mode: common.CommQuantMode = common.CommQuantMode.bfloat16,
    ) -> NCCL:
        return NCCL(
            self._name,
            self._scale_factor,
            "all_reduce",
            self._h,
            self._tp_size,
            quant_mode,
            seq_split=self._seq_split,
        )

    @staticmethod
    def _nccl_query(
        normalized_query: Mapping[str, object],
        quant_mode: common.CommQuantMode = common.CommQuantMode.bfloat16,
    ) -> Mapping[str, object]:
        return {
            "nccl_dtype": quant_mode.name,
            "operation": "all_reduce",
            "num_gpus": int(normalized_query["world_size"]),
            "message_size": int(normalized_query["elements"]),
        }

    def _static_oversized_result(
        self,
        database: PerfDatabase,
        normalized_query: Mapping[str, object],
    ) -> PerformanceResult:
        """Approximate physical BF16 NCCL with legacy same-width FP16 data."""

        approximate_query = self._nccl_query(normalized_query, common.CommQuantMode.half)
        fallback = self._nccl_fallback(common.CommQuantMode.half)
        if not fallback._curated_nccl_version_matches(database):
            return PerformanceResult(0.0, energy=0.0, source="unresolved")
        try:
            return fallback._query_from_normalized(
                database,
                approximate_query,
            )
        except (PerfDataNotAvailableError, EmpiricalNotImplementedError):
            return self._query_from_normalized(database, normalized_query)

    def normalize_perf_query(self, **kwargs: object) -> Mapping[str, object]:
        """Normalize the exact physical collective used by prediction."""

        x = kwargs.get("x")
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError("CustomAllReduce token count must be an integer")
        if x <= 0:
            raise ValueError("CustomAllReduce token count must be positive")
        elements = (-(-x // self._seq_split)) * self._h
        return {
            "dtype": common.CommQuantMode.half.name,
            "operation": "all_reduce",
            "world_size": self._tp_size,
            "elements": elements,
        }

    @staticmethod
    def _measurement_environment(database: PerfDatabase) -> MeasurementEnvironment:
        environment = getattr(database, "measurement_environment", None)
        if not isinstance(environment, MeasurementEnvironment):
            raise TypeError("CustomAllReduce lazy collection requires a bound MeasurementEnvironment")
        expected = (database.system, database.backend)
        actual = (environment.system, environment.backend)
        if actual != expected:
            raise ValueError("database measurement environment does not match system/backend")
        return environment

    def measurement_request(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        **kwargs,
    ) -> MeasurementRequest | None:
        normalized = self.normalize_perf_query(**kwargs)
        return self._measurement_request_from_normalized(
            database,
            protocol,
            normalized_query=normalized,
            **kwargs,
        )

    def _measurement_request_from_normalized(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> MeasurementRequest | None:
        if self._uses_nccl_fallback(database, normalized_query):
            return self._nccl_fallback()._measurement_request_from_normalized(
                database,
                protocol,
                normalized_query=self._nccl_query(normalized_query),
                **kwargs,
            )
        del kwargs
        if int(normalized_query["world_size"]) <= 1:
            return None
        query = dict(normalized_query)
        environment = self._measurement_environment(database)
        namespace = perf_namespace("custom_allreduce_perf.txt")
        return MeasurementRequest(
            op_id=self._name,
            key=PerfKey.build(namespace, query, environment),
            query=query,
            environment=environment,
            semantic_descriptor={
                "operation": "all_reduce",
                "implementation": "sglang_custom_allreduce",
                "mode": "graph",
            },
            protocol=protocol,
        )

    def curated_exact_result(self, database: PerfDatabase, **kwargs) -> PerformanceResult | None:
        normalized = self.normalize_perf_query(**kwargs)
        return self._curated_exact_result_from_normalized(
            database,
            normalized_query=normalized,
            **kwargs,
        )

    def _is_v1_2_curated_compatible(self, database: PerfDatabase) -> bool:
        environment = getattr(database, "measurement_environment", None)
        return bool(
            isinstance(environment, MeasurementEnvironment)
            and environment.system == "gb200"
            and environment.backend == "sglang"
            and environment.backend_version == "0.5.10"
            and " ".join(environment.gpu_class.split()).casefold() == "nvidia gb200"
            and isinstance(environment.topology_schema, str)
            and bool(environment.topology_schema.strip())
            and isinstance(environment.topology_fingerprint, str)
            and bool(environment.topology_fingerprint.strip())
            and all(
                environment.runtime_versions.get(name) == version
                for name, version in self._V1_2_RUNTIME_VERSIONS.items()
            )
            and dict(environment.profile_compatibility or {}) == self._V1_2_PROFILE_COMPATIBILITY
        )

    def _curated_exact_result_from_normalized(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult | None:
        if self._uses_nccl_fallback(database, normalized_query):
            return self._nccl_fallback()._curated_exact_result_from_normalized(
                database,
                normalized_query=self._nccl_query(normalized_query),
                **kwargs,
            )
        del kwargs
        if not self._is_v1_2_curated_compatible(database):
            return None
        try:
            dtype = common.CommQuantMode[str(normalized_query["dtype"])]
            if normalized_query["operation"] != "all_reduce":
                return None
            world_size = int(normalized_query["world_size"])
            elements = int(normalized_query["elements"])
        except (KeyError, TypeError, ValueError):
            return None
        self.load_data(database)
        wrapper = getattr(database, "_custom_allreduce_data", None)
        if wrapper is None:
            return None
        value: object = wrapper.data if hasattr(wrapper, "data") else wrapper
        try:
            for component in (dtype, world_size, "AUTO", elements):
                if not isinstance(value, Mapping):
                    return None
                value = value[component]
        except KeyError:
            return None
        if isinstance(value, Mapping):
            try:
                latency = float(value["latency"])
                energy = float(value.get("energy", 0.0))
            except (KeyError, TypeError, ValueError):
                return None
        else:
            try:
                latency = float(value)
            except (TypeError, ValueError):
                return None
            energy = 0.0
        return PerformanceResult(
            latency * self._scale_factor,
            energy=energy * self._scale_factor,
            source="curated_exact",
            provenance={"curated_path": getattr(wrapper, "filepath", None)},
        )

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads custom_allreduce CSV, binds
        ``database._custom_allreduce_data``."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.custom_allreduce.value)
            sources = database._build_op_sources(PerfDataFilename.custom_allreduce, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_custom_allreduce_data(sources), PerfDataFilename.custom_allreduce, primary_path
            )
            cls._record_load()

        if "_custom_allreduce_data" not in database.__dict__:
            database._custom_allreduce_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_custom_allreduce)
    # ------------------------------------------------------------------

    @classmethod
    def _query_custom_allreduce_table(
        cls,
        database: PerfDatabase,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query custom_allreduce table. Verbatim port of the legacy body."""
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        def get_sol(quant_mode: common.CommQuantMode, tp_size: int, size: int) -> tuple[float, float, float]:
            if tp_size == 1:
                return 0, 0, 0
            p2p_bw = database._get_p2p_bandwidth(tp_size)
            # assume all are ring allreduce, ignore constant latency
            # (~1us for hopper, ~2us for two-die blackwell). assume bfloat16.
            sol_time = 2 * size * 2 / tp_size * (tp_size - 1) / p2p_bw
            return sol_time * 1000, 0, 0

        def get_empirical(quant_mode: common.CommQuantMode, tp_size: int, size: int) -> float:
            # Data-calibrated: util = SOL/measured read from the collected
            # custom_allreduce curve (smooth in message size), reconstructed with
            # this query's SOL. Falls back to the legacy 1/0.8 constant when the
            # slice has no data. SOL uses the real tp_size; the util grid is built
            # from the effective (node-capped) tp slice, so SOL ratio carries any
            # multi-node bandwidth scaling.
            sol_q = get_sol(quant_mode, tp_size, size)[0]
            if tp_size <= 1 or sol_q <= 0:
                return sol_q
            eff = min(tp_size, database.system_spec["node"]["num_gpus_per_node"])

            def _slice():
                cls.load_data(database)
                dw = database._custom_allreduce_data
                dw.raise_if_not_loaded()
                return util_empirical.require_data_slice(dw, quant_mode, eff, "AUTO")

            grid = util_empirical.grid_for(
                ("custom_allreduce", database.system, database.backend, database.version, quant_mode.value.name, eff),
                _slice,
                lambda c: get_sol(quant_mode, eff, int(c[0]))[0],
                depth=1,
            )
            # Compatibility exception: rank-count overflow still borrows the
            # node-boundary utilization even when XSHAPE is disabled, because
            # rejecting it would remove support for common multi-node configs.
            # TODO(#1260): define a strict policy where an unmeasured TP count
            # fails accurately without regressing default model coverage.
            prov = "xshape" if eff != tp_size else "empirical"
            latency, _ = util_empirical.estimate(sol_q, (float(size),), grid, provenance=prov)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(quant_mode, tp_size, size)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(quant_mode, tp_size, size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(quant_mode, tp_size, size)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._custom_allreduce_data

        def get_silicon():
            if tp_size == 1:
                return PerformanceResult(0.0, energy=0.0, source="empirical")
            if database.system_spec["node"]["num_gpus_per_node"] == 72 and tp_size > 4:
                # on GB200, we only have custom all reduce for up to tp4.
                return database.query_nccl(quant_mode, tp_size, "all_reduce", size)

            data_wrapper.raise_if_not_loaded()

            # The loader returns a 4-deep defaultdict, so chained indexing silently
            # synthesizes empty dicts for missing (quant_mode, tp_size, strategy)
            # combinations. Validate explicitly so upstream callers see a structured
            # PerfDataNotAvailableError instead of an internal AssertionError from
            # _nearest_1d_point_helper when the CSV has no rows for this bucket.
            effective_tp = min(tp_size, database.system_spec["node"]["num_gpus_per_node"])
            by_tp = data_wrapper.get(quant_mode, {})
            strategy_dict = by_tp.get(effective_tp, {})
            comm_dict = strategy_dict.get("AUTO", {})
            if not comm_dict:
                raise PerfDataNotAvailableError(
                    f"No custom_allreduce silicon data for quant_mode={quant_mode.value.name}, "
                    f"tp_size={effective_tp} (requested tp_size={tp_size}). "
                    f"Available tp_sizes for this quant_mode: {sorted(by_tp.keys())}. "
                    "Consider using HYBRID mode, or supply custom_allreduce_perf.txt rows "
                    "covering this tp_size."
                )
            size_left, size_right = interpolation.nearest_1d_point_helper(
                size, list(comm_dict.keys()), inner_only=False
            )
            result = interpolation.interp_1d(
                [size_left, size_right], [comm_dict[size_left], comm_dict[size_right]], size
            )

            if isinstance(result, dict):
                lat = result["latency"]
                energy = result.get("energy", 0.0)
            else:
                lat = result
                energy = 0.0

            if tp_size > database.system_spec["node"]["num_gpus_per_node"]:
                base_bw = database._get_p2p_bandwidth(database.system_spec["node"]["num_gpus_per_node"])
                target_bw = database._get_p2p_bandwidth(tp_size)
                scale_factor = (
                    (tp_size - 1)
                    / tp_size
                    * database.system_spec["node"]["num_gpus_per_node"]
                    / (database.system_spec["node"]["num_gpus_per_node"] - 1)
                    * base_bw
                    / target_bw
                )
                lat = lat * scale_factor
                energy = energy * scale_factor

            return database._interp_pr(lat, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(quant_mode, tp_size, size),
            database_mode=database_mode,
            error_msg=f"Failed to query custom allreduce data for {quant_mode=}, {tp_size=}, {size=}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query custom allreduce latency with power data."""
        if self._tp_size == 1:
            # No-op short-circuit: tp_size=1 has no allreduce. Tag as
            # ``empirical`` rather than letting the constructor default to
            # ``silicon`` so EMPIRICAL/SOL modes don't get a spurious
            # silicon leakage in the breakdown report.
            return PerformanceResult(0.0, 0.0, source="empirical")
        normalized = self.normalize_perf_query(**kwargs)
        if self._uses_nccl_fallback(database, normalized):
            self._nccl_fallback(common.CommQuantMode.half)._require_curated_nccl_version_match(database)
            return self._static_oversized_result(database, normalized)
        return self._query_from_normalized(database, normalized)

    def provisional_result(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult:
        del kwargs
        if self._uses_nccl_fallback(database, normalized_query):
            return self._static_oversized_result(database, normalized_query)
        return self._query_from_normalized(database, normalized_query)

    def hybrid_fallback_value(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> HybridFallbackValue:
        if self._uses_nccl_fallback(database, normalized_query):
            fallback = self._nccl_fallback(common.CommQuantMode.half)
            try:
                return fallback.hybrid_fallback_value(
                    database,
                    normalized_query=self._nccl_query(normalized_query, common.CommQuantMode.half),
                    **kwargs,
                )
            except (PerfDataNotAvailableError, EmpiricalNotImplementedError):
                pass
        return super().hybrid_fallback_value(
            database,
            normalized_query=normalized_query,
            **kwargs,
        )

    def _query_from_normalized(
        self,
        database: PerfDatabase,
        normalized_query: Mapping[str, object],
    ) -> PerformanceResult:
        dtype = common.CommQuantMode[str(normalized_query["dtype"])]
        world_size = int(normalized_query["world_size"])
        operation = str(normalized_query["operation"])
        elements = int(normalized_query["elements"])
        if operation != "all_reduce":
            raise ValueError(f"unsupported CustomAllReduce operation {operation!r}")

        result = database.query_custom_allreduce(dtype, world_size, elements)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class NCCL(Operation):
    """
    NCCL collective communication operation with power tracking.

    Owns ``_data_cache`` for the NCCL CSV table plus ``_oneccl_data_cache``
    for the optional oneCCL fallback (loaded together because
    ``query_nccl`` picks between them at query time when NCCL data is
    empty on XPU systems).
    """

    _data_cache: ClassVar[dict] = {}
    _oneccl_data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank payload)

    def __init__(
        self,
        name: str,
        scale_factor: float,
        nccl_op: str,
        num_elements_per_token: int,
        num_gpus: int,
        comm_quant_mode: common.CommQuantMode,
        *,
        seq_split: int = 1,
    ) -> None:
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError("NCCL num_gpus must be positive")
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads NCCL CSV + the optional oneCCL fallback,
        binds ``database._nccl_data`` and ``database._oneccl_data``."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            # NCCL data lives under ``systems_data_root/nccl/<nccl_version>/``,
            # NOT under ``backend/version/``. Per ``_build_op_sources`` early-
            # exit, NCCL ops never inherit shared-layer sibling rows.
            nccl_data_dir = os.path.join(system_data_root, "nccl", database.system_spec["misc"]["nccl_version"])
            nccl_primary = os.path.join(nccl_data_dir, PerfDataFilename.nccl.value)
            nccl_sources = database._build_op_sources(PerfDataFilename.nccl, nccl_primary, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_nccl_data(nccl_sources), PerfDataFilename.nccl, nccl_primary)

            # oneCCL fallback (XPU systems). Only loaded when system_spec
            # declares an ``oneccl_version`` under ``misc``.
            oneccl_version = database.system_spec.get("misc", {}).get("oneccl_version")
            if oneccl_version:
                oneccl_data_dir = os.path.join(system_data_root, "oneccl", oneccl_version)
                oneccl_primary = os.path.join(oneccl_data_dir, PerfDataFilename.oneccl.value)
                oneccl_sources = database._build_op_sources(PerfDataFilename.oneccl, oneccl_primary, system_data_root)
                cls._oneccl_data_cache[key] = LoadedOpData(
                    load_nccl_data(oneccl_sources), PerfDataFilename.oneccl, oneccl_primary
                )
            else:
                cls._oneccl_data_cache[key] = None

            cls._record_load()

        if "_nccl_data" not in database.__dict__:
            database._nccl_data = cls._data_cache[key]
        if "_oneccl_data" not in database.__dict__:
            database._oneccl_data = cls._oneccl_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._oneccl_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_nccl)
    # ------------------------------------------------------------------

    @classmethod
    def _query_nccl_table(
        cls,
        database: PerfDatabase,
        dtype: common.CommQuantMode,
        num_gpus: int,
        operation: str,
        message_size: int,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query NCCL table. Verbatim port of the legacy body."""

        def require_nccl_bucket(node: object, description: str) -> Mapping:
            if not isinstance(node, Mapping):
                raise TypeError(
                    f"Malformed NCCL performance data for {description}: expected a mapping, got {type(node).__name__}."
                )
            if not node:
                raise PerfDataNotAvailableError(f"No NCCL performance data for {description}.")
            return node

        def get_sol(
            dtype: common.CommQuantMode, num_gpus: int, operation: str, message_size: int
        ) -> tuple[float, float, float]:
            sol_time = 0.0
            p2p_bw = database._get_p2p_bandwidth(num_gpus)

            if operation == "all_gather" or operation == "alltoall" or operation == "reduce_scatter":
                sol_time = dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            elif operation == "all_reduce":
                sol_time = 2 * dtype.value.memory * message_size * (num_gpus - 1) / num_gpus / p2p_bw * 1000
            return sol_time, 0, sol_time

        def get_empirical(dtype: common.CommQuantMode, num_gpus: int, operation: str, message_size: int) -> float:
            # Data-calibrated: util = SOL/measured read from the collected NCCL
            # curve for this (dtype, operation, num_gpus) slice, reconstructed with
            # this query's SOL. Falls back to the legacy 1/0.8 constant when no
            # data. The grid is built from the available (capped) num_gpus slice;
            # SOL uses the real num_gpus so the SOL ratio carries scaling beyond
            # the largest collected num_gpus.
            sol_q = get_sol(dtype, num_gpus, operation, message_size)[0]
            if num_gpus <= 1 or sol_q <= 0:
                return sol_q  # no communication for a single rank -> 0, not a data gap
            cls.load_data(database)
            src = database._nccl_data
            if not src.loaded and database._oneccl_data is not None and database._oneccl_data.loaded:
                src = database._oneccl_data
            if not src.loaded:
                raise EmpiricalNotImplementedError(
                    f"No NCCL data to estimate {operation} ({dtype.value.name}, num_gpus={num_gpus})."
                )
            try:
                by_op = require_nccl_bucket(
                    util_empirical.require_data_slice(src, dtype, operation),
                    f"dtype={dtype.value.name}, operation={operation!r}",
                )
                eff = min(num_gpus, max(by_op.keys()))
            except PerfDataNotAvailableError as exc:
                raise EmpiricalNotImplementedError(
                    f"No NCCL data for operation {operation!r} ({dtype.value.name}, num_gpus={num_gpus})."
                ) from exc

            def _slice():
                return util_empirical.require_data_slice(by_op, eff)

            grid = util_empirical.grid_for(
                ("nccl", database.system, database.backend, database.version, dtype.value.name, operation, eff),
                _slice,
                lambda c: get_sol(dtype, eff, operation, int(c[0]))[0],
                depth=1,
            )
            # Compatibility exception: rank-count overflow still borrows the
            # largest collected utilization even when XSHAPE is disabled,
            # preserving the existing NCCL boundary-correction coverage.
            # TODO(#1260): define a strict policy where an unmeasured GPU count
            # fails accurately without regressing default model coverage.
            prov = "xshape" if eff != num_gpus else "empirical"
            latency, _ = util_empirical.estimate(sol_q, (float(message_size),), grid, provenance=prov)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(dtype, num_gpus, operation, message_size)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(dtype, num_gpus, operation, message_size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(
                get_empirical(dtype, num_gpus, operation, message_size), energy=0.0, source="empirical"
            )

        cls.load_data(database)

        def get_silicon():
            if num_gpus == 1:
                return PerformanceResult(0.0, energy=0.0, source="empirical")

            # Use oneCCL data as fallback when NCCL data is not available (e.g. XPU systems)
            nccl_source = database._nccl_data
            if not nccl_source.loaded and database._oneccl_data is not None and database._oneccl_data.loaded:
                nccl_source = database._oneccl_data
            nccl_source.raise_if_not_loaded()

            by_num_gpus = require_nccl_bucket(
                util_empirical.require_data_slice(nccl_source, dtype, operation),
                f"dtype={dtype.value.name}, operation={operation!r}",
            )
            max_num_gpus = max(by_num_gpus.keys())
            effective_num_gpus = min(num_gpus, max_num_gpus)
            nccl_dict = require_nccl_bucket(
                util_empirical.require_data_slice(by_num_gpus, effective_num_gpus),
                f"dtype={dtype.value.name}, operation={operation!r}, num_gpus={effective_num_gpus}",
            )
            size_left, size_right = interpolation.nearest_1d_point_helper(
                message_size,
                list(nccl_dict.keys()),
                inner_only=False,
            )
            result = interpolation.interp_1d(
                [size_left, size_right],
                [nccl_dict[size_left], nccl_dict[size_right]],
                message_size,
            )

            if isinstance(result, dict):
                lat = result["latency"]
                energy = result.get("energy", 0.0)
            else:
                lat = result
                energy = 0.0

            if num_gpus > max_num_gpus:  # need to do some correction
                logger.debug(f"nccl num_gpus {num_gpus} > max_num_gpus {max_num_gpus}, need to do some correction")
                max_num_gpus_bw = database._get_p2p_bandwidth(max_num_gpus)
                num_gpus_bw = database._get_p2p_bandwidth(num_gpus)
                scale_factor = max_num_gpus_bw / num_gpus_bw
                scaling_formula = (num_gpus - 1) / num_gpus * max_num_gpus / (max_num_gpus - 1) * scale_factor
                lat = lat * scaling_formula
                energy = energy * scaling_formula

            return database._interp_pr(lat, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(dtype, num_gpus, operation, message_size),
            database_mode=database_mode,
            error_msg=f"Failed to query nccl data for {dtype=}, {num_gpus=}, {operation=}, {message_size=}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def normalize_perf_query(self, **kwargs: object) -> Mapping[str, object]:
        x = kwargs.get("x")
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError("NCCL x must be an integer")
        message_size = (-(-x // self._seq_split)) * self._num_elements_per_token
        return {
            "nccl_dtype": self._comm_quant_mode.name,
            "operation": self._nccl_op,
            "num_gpus": self._num_gpus,
            "message_size": message_size,
        }

    @staticmethod
    def _measurement_environment(database: PerfDatabase) -> MeasurementEnvironment:
        environment = getattr(database, "measurement_environment", None)
        if not isinstance(environment, MeasurementEnvironment):
            raise TypeError("NCCL lazy collection requires a bound MeasurementEnvironment")
        return environment

    def measurement_request(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        **kwargs,
    ) -> MeasurementRequest | None:
        normalized = self.normalize_perf_query(**kwargs)
        return self._measurement_request_from_normalized(
            database,
            protocol,
            normalized_query=normalized,
            **kwargs,
        )

    def _measurement_request_from_normalized(
        self,
        database: PerfDatabase,
        protocol: MeasurementProtocol,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> MeasurementRequest | None:
        del kwargs
        if int(normalized_query["num_gpus"]) <= 1:
            return None
        query = dict(normalized_query)
        environment = self._measurement_environment(database)
        namespace = perf_namespace("nccl_perf.txt")
        return MeasurementRequest(
            op_id=self._name,
            key=PerfKey.build(namespace, query, environment),
            query=query,
            environment=environment,
            semantic_descriptor={"tensor_generator": "normal-v1", "seed": 0},
            protocol=protocol,
        )

    def curated_exact_result(self, database: PerfDatabase, **kwargs) -> PerformanceResult | None:
        normalized = self.normalize_perf_query(**kwargs)
        return self._curated_exact_result_from_normalized(
            database,
            normalized_query=normalized,
            **kwargs,
        )

    @staticmethod
    def _curated_nccl_version_matches(database: PerfDatabase) -> bool:
        environment = getattr(database, "measurement_environment", None)
        if not isinstance(environment, MeasurementEnvironment):
            return True
        requested = environment.runtime_versions.get("nccl")
        if not isinstance(requested, str) or not requested.strip():
            return False
        configured = database.system_spec.get("misc", {}).get("nccl_version")
        return isinstance(configured, str) and configured.strip() == requested.strip()

    @classmethod
    def _require_curated_nccl_version_match(cls, database: PerfDatabase) -> None:
        if not cls._curated_nccl_version_matches(database):
            raise ValueError("NCCL runtime version does not match the configured fallback table")

    def _curated_exact_result_from_normalized(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult | None:
        del kwargs
        try:
            dtype = common.CommQuantMode[str(normalized_query["nccl_dtype"])]
            operation = str(normalized_query["operation"])
            num_gpus = int(normalized_query["num_gpus"])
            message_size = int(normalized_query["message_size"])
        except (KeyError, TypeError, ValueError):
            return None
        if num_gpus == 1:
            return PerformanceResult(0.0, energy=0.0, source="curated_exact")
        if not self._curated_nccl_version_matches(database):
            return None
        self.load_data(database)
        wrapper = database._nccl_data
        by_operation = wrapper.data.get(dtype)
        by_world = by_operation.get(operation) if isinstance(by_operation, Mapping) else None
        by_size = by_world.get(num_gpus) if isinstance(by_world, Mapping) else None
        result = by_size.get(message_size) if isinstance(by_size, Mapping) else None
        if result is None:
            return None
        if isinstance(result, Mapping):
            latency = float(result["latency"])
            energy = float(result.get("energy", 0.0))
        else:
            latency = float(result)
            energy = 0.0
        return PerformanceResult(
            latency * self._scale_factor,
            energy=energy * self._scale_factor,
            source="curated_exact",
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        normalized = self.normalize_perf_query(**kwargs)
        if int(normalized["num_gpus"]) <= 1:
            return PerformanceResult(0.0, energy=0.0, source="empirical")
        self._require_curated_nccl_version_match(database)
        return self._query_from_normalized(database, normalized)

    def provisional_result(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> PerformanceResult:
        del kwargs
        if int(normalized_query["num_gpus"]) <= 1:
            return PerformanceResult(0.0, energy=0.0, source="empirical")
        if not self._curated_nccl_version_matches(database):
            return PerformanceResult(0.0, energy=0.0, source="unresolved")
        return self._query_from_normalized(database, normalized_query)

    def hybrid_fallback_value(
        self,
        database: PerfDatabase,
        *,
        normalized_query: Mapping[str, object],
        **kwargs,
    ) -> HybridFallbackValue:
        self._require_curated_nccl_version_match(database)
        return super().hybrid_fallback_value(
            database,
            normalized_query=normalized_query,
            **kwargs,
        )

    def _query_from_normalized(
        self,
        database: PerfDatabase,
        normalized_query: Mapping[str, object],
    ) -> PerformanceResult:
        dtype = common.CommQuantMode[str(normalized_query["nccl_dtype"])]
        num_gpus = int(normalized_query["num_gpus"])
        operation = str(normalized_query["operation"])
        message_size = int(normalized_query["message_size"])
        result = database.query_nccl(dtype, num_gpus, operation, message_size)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P (point-to-point) communication operation with power tracking.

    Purely analytical — no silicon table. The base ``Operation.load_data``
    no-op default handles the missing CSV; ``_query_p2p_table`` is factored
    out only for parity with the other migrated ops.
    """

    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank payload)

    def is_resolution_deterministic(self, **kwargs: object) -> bool:
        """PP1 is the frozen profile's reviewed deterministic no-op."""
        del kwargs
        return self._pp_size == 1

    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int, *, seq_split: int = 1) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_p2p)
    # ------------------------------------------------------------------

    @classmethod
    def _query_p2p_table(
        cls,
        database: PerfDatabase,
        message_bytes: int,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query P2P latency analytically. Verbatim port of the legacy body."""

        def get_sol(message_bytes: int) -> tuple[float, float, float]:
            # TODO, use intra_node_bw if num_gpus < num_gpus_per_node
            sol_time = message_bytes / database.system_spec["node"]["inter_node_bw"] * 1000
            return sol_time, 0, sol_time

        def get_empirical(message_bytes: int) -> float:
            return (
                message_bytes / database.system_spec["node"]["inter_node_bw"]
                + database.system_spec["node"]["p2p_latency"]
            ) * 1000

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(message_bytes)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(message_bytes)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(message_bytes), energy=0.0, source="empirical")
        # No silicon table for P2P — even SILICON/HYBRID modes use the
        # empirical formula here, so tag the source accordingly.
        return PerformanceResult(get_empirical(message_bytes), energy=0.0, source="empirical")

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query P2P latency with power data."""
        if self._pp_size == 1:
            # No-op short-circuit: pp_size=1 has no P2P transfer. See note on
            # CustomAllReduce.query for source-tag rationale.
            return PerformanceResult(0.0, 0.0, source="empirical")

        size = (-(-kwargs.get("x") // self._seq_split)) * self._h  # CP: ceil = busiest rank
        p2p_bytes = size * 2

        result = database.query_p2p(p2p_bytes)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_custom_allreduce_data(custom_allreduce_file):
    """
    Load the custom allreduce data with power support (backward compatible).

    Supports multiple data formats:
    - TRTLLM: kernel_source="TRTLLM", last column="implementation"
    - vLLM/SGLang: kernel_source="*_graph" or "*_eager", last column="backend"

    For vLLM/SGLang with both graph and eager modes, only graph mode data is kept
    (better performance for decode phase).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(custom_allreduce_file)
    if rows is None:
        logger.debug(f"Custom allreduce data file {custom_allreduce_file} not found.")
        return None
    custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (custom_allreduce) - power will default to 0.0")

    if isinstance(custom_allreduce_file, str):
        is_b60 = "b60" in custom_allreduce_file
    else:
        is_b60 = any("b60" in path for path, _ in custom_allreduce_file)

    for row in rows:
        # Check kernel_source to filter graph vs eager mode (for vLLM/SGLang)
        kernel_source = row.get("kernel_source", "")
        backend = row.get("backend", "")

        # For vLLM/SGLang format: only keep graph mode data (skip eager mode)
        # kernel_source patterns: "vLLM_custom_graph", "SGLang_CustomAllReduce_graph", etc.
        # backend patterns: "vllm_graph", "sglang_graph", etc.
        if (kernel_source.endswith("_eager") or backend.endswith("_eager")) and not is_b60:
            continue  # Skip eager mode, use graph mode only

        dtype, tp_size, message_size, latency = (
            row["allreduce_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["latency"],
        )
        allreduce_strategy = "AUTO"
        message_size = int(message_size)
        latency = float(latency)
        tp_size = int(tp_size)
        dtype = common.CommQuantMode.half  # TODO

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size]
            logger.debug(
                f"value conflict in custom allreduce data: {dtype} {tp_size} {allreduce_strategy} {message_size}"
            )
        except KeyError:
            # Store all three values
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return custom_allreduce_data


def load_nccl_data(nccl_file):
    """
    Load the nccl data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(nccl_file)
    if rows is None:
        logger.debug(f"NCCL data file {nccl_file} not found.")
        return None
    nccl_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (nccl) - power will default to 0.0")

    for row in rows:
        dtype, num_gpus, message_size, op_name, latency = (
            row["nccl_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["op_name"],
            row["latency"],
        )
        message_size = int(message_size)
        latency = float(latency)
        num_gpus = int(num_gpus)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        dtype = common.CommQuantMode[dtype]
        try:
            # Check for conflict
            nccl_data[dtype][op_name][num_gpus][message_size]
            logger.debug(f"value conflict in nccl data: {dtype} {op_name} {num_gpus} {message_size}")
        except KeyError:
            # Store all three values
            nccl_data[dtype][op_name][num_gpus][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return nccl_data
