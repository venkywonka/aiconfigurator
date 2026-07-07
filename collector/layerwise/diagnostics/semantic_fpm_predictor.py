#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed AIC prediction orchestration for normalized semantic FPM bins."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

from collector.layerwise.diagnostics.semantic_fpm_insights import (
    AicPredictionRecord,
    AicQueryShape,
    AxisLookup,
    OperationLookup,
    SemanticBinPredictor,
    SemanticBinQuery,
    SemanticShape,
    build_semantic_query,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    ContractError,
    SemanticPopulation,
)

LOOKUP_POLICY_VERSION = "conservative-v1"
ARTIFACT_PROXY_LOOKUP_POLICY_VERSION = "artifact-proxy-v1"
SUPPORTED_LOOKUP_POLICIES = frozenset({LOOKUP_POLICY_VERSION, ARTIFACT_PROXY_LOOKUP_POLICY_VERSION})
EXPECTED_UNAVAILABLE_REASONS = frozenset({"missing_surface", "out_of_cap", "unsupported_phase"})
PREDICTOR_VERSION = "aiconfigurator-semantic-bin-v1"
COMPONENT_CLASSIFIER_VERSION = "exact-registry-v1"

_AIC_COMPONENT_REGISTRY = {
    "context_layerwise": "compute",
    "generation_layerwise": "compute",
    "context_tp_allreduce": "communication",
    "generation_tp_allreduce": "communication",
    "generation_tp_allreduce_rms": "communication",
}


def aic_component_class(operation_name: str) -> str | None:
    """Return the exact dense-Qwen component class for one AIC operation."""

    return _AIC_COMPONENT_REGISTRY.get(operation_name)


@dataclass(frozen=True)
class SemanticBinPrediction:
    """Prediction attached to one measured clean semantic bin."""

    bin_id: str
    query: SemanticBinQuery
    record: AicPredictionRecord


@dataclass(frozen=True)
class LayerwiseSurfacePoint:
    """One collected scheduler point after exact configuration filtering."""

    phase: str
    batch_size: int
    new_tokens: int
    past_kv: int
    row_content_hash: str
    latency_ms: float
    detail_json: str

    def __post_init__(self) -> None:
        if self.phase not in {"context", "decode"}:
            raise ValueError(f"unsupported layerwise phase {self.phase!r}")
        if self.batch_size <= 0 or self.new_tokens <= 0 or self.past_kv < 0:
            raise ValueError(f"invalid layerwise surface point {self!r}")
        if not self.row_content_hash:
            raise ValueError("row_content_hash must be non-empty")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")
        try:
            detail = json.loads(self.detail_json)
        except json.JSONDecodeError as exc:
            raise ValueError("detail_json must be valid JSON") from exc
        if not isinstance(detail, dict) or not detail:
            raise ValueError("detail_json must be a non-empty JSON object")
        detail_latency = detail.get("latency")
        if not isinstance(detail_latency, int | float) or isinstance(detail_latency, bool):
            raise TypeError("detail_json latency must be numeric")
        if not math.isclose(float(detail_latency), self.latency_ms, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("detail_json latency disagrees with latency_ms")


@dataclass(frozen=True)
class ConservativeSurfaceSelection:
    """Result of scheduler-grid selection without invoking AIC."""

    status: str
    reason: str
    lookup_policy: str
    requested_shape: AicQueryShape
    evaluated_shape: AicQueryShape | None
    point: LayerwiseSurfacePoint | None
    lookup_surface_id: str | None
    scheduler_surface_content_hash: str | None
    axis_lookups: tuple[AxisLookup, ...]
    match_type: str | None


@dataclass(frozen=True)
class RawAicStep:
    """Unclassified operation inventory returned by one repository AIC call."""

    operations: tuple[tuple[str, float], ...]
    sources: tuple[tuple[str, str], ...]
    operation_lookups: tuple[OperationLookup, ...]


class AicStepRunner(Protocol):
    """Repository-dependent execution hidden behind the pure predictor policy."""

    api_version: str

    def predict(
        self,
        *,
        phase: str,
        shape: AicQueryShape,
        point: LayerwiseSurfacePoint,
    ) -> RawAicStep: ...


class AicSurfaceUnavailableError(RuntimeError):
    """The selected scheduler point lacks an introspectable lower-level surface."""


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nearest_bounds(values: tuple[int, ...], requested: int) -> tuple[int, int]:
    lower_values = tuple(value for value in values if value <= requested)
    upper_values = tuple(value for value in values if value >= requested)
    lower = max(lower_values) if lower_values else values[0]
    upper = min(upper_values) if upper_values else values[-1]
    return lower, upper


def _axis_lookup(
    axis: str,
    requested: int,
    evaluated: int,
    *,
    lower: int | None = None,
    upper: int | None = None,
    mode: str | None = None,
) -> AxisLookup:
    lower = evaluated if lower is None else lower
    upper = evaluated if upper is None else upper
    if lower == upper:
        weight = 0.0
    else:
        weight = (requested - lower) / (upper - lower)
    return AxisLookup(
        axis=axis,
        requested=requested,
        evaluated=evaluated,
        lower=lower,
        upper=upper,
        weight=weight,
        delta=evaluated - requested,
        mode=mode or ("exact" if requested == evaluated else "nearest"),
    )


class ConservativeSurfaceIndex:
    """Exact scheduler-surface index with bounded nearest-KV wiggle room."""

    def __init__(
        self,
        *,
        points: tuple[LayerwiseSurfacePoint, ...],
        surface_provenance: str,
        lookup_policy: str = LOOKUP_POLICY_VERSION,
        phase_axis_lookups: tuple[tuple[str, tuple[AxisLookup, ...]], ...] = (),
    ):
        if not surface_provenance:
            raise ValueError("surface_provenance must be non-empty")
        if lookup_policy not in SUPPORTED_LOOKUP_POLICIES:
            raise ValueError(f"unsupported lookup policy {lookup_policy!r}")
        phase_lookup_map = dict(phase_axis_lookups)
        if len(phase_lookup_map) != len(phase_axis_lookups):
            raise ValueError("duplicate phase axis lookup inventory")
        if set(phase_lookup_map) - {"context", "decode", "mixed"}:
            raise ValueError("phase axis lookups contain an unsupported phase")
        identities = [(point.phase, point.batch_size, point.new_tokens, point.past_kv) for point in points]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate layerwise scheduler surface point")
        self._points = tuple(
            sorted(
                points,
                key=lambda point: (point.phase, point.batch_size, point.new_tokens, point.past_kv),
            )
        )
        self._surface_provenance = surface_provenance
        self._lookup_policy = lookup_policy
        self._phase_axis_lookups = phase_lookup_map

    @property
    def lookup_policy(self) -> str:
        return self._lookup_policy

    def _unavailable(self, query: SemanticBinQuery, reason: str) -> ConservativeSurfaceSelection:
        requested_shape = AicQueryShape.from_query(query)
        lookups = list(self._phase_axis_lookups.get(query.phase, ()))
        present_axes = {lookup.axis for lookup in lookups}
        lookups.extend(
            AxisLookup(
                axis=axis,
                requested=requested,
                evaluated=None,
                lower=None,
                upper=None,
                weight=None,
                delta=None,
                mode="missing",
            )
            for axis, requested in requested_shape.items()
            if axis not in present_axes
        )
        return ConservativeSurfaceSelection(
            status="unavailable",
            reason=reason,
            lookup_policy=self._lookup_policy,
            requested_shape=requested_shape,
            evaluated_shape=None,
            point=None,
            lookup_surface_id=None,
            scheduler_surface_content_hash=None,
            axis_lookups=tuple(lookups),
            match_type=None,
        )

    def select(self, query: SemanticBinQuery) -> ConservativeSurfaceSelection:
        requested_shape = AicQueryShape.from_query(query)
        if query.phase == "mixed":
            return self._unavailable(query, "unsupported_phase")
        if query.phase == "context":
            batch_size = query.ctx_requests
            new_tokens_total = query.query_ctx_new_total
            requested_kv = query.ctx_kv_per_request
        elif query.phase == "decode":
            batch_size = query.decode_requests
            new_tokens_total = batch_size
            requested_kv = query.decode_kv_per_request
        else:
            return self._unavailable(query, "unsupported_phase")
        if requested_kv is None:
            return self._unavailable(query, "missing_surface")

        surface = tuple(
            point
            for point in self._points
            if point.phase == query.phase
            and point.batch_size == batch_size
            and point.new_tokens * point.batch_size == new_tokens_total
        )
        if not surface:
            return self._unavailable(query, "missing_surface")
        kv_values = tuple(sorted(point.past_kv for point in surface))
        if query.phase == "context" and requested_kv == 0:
            if 0 not in kv_values:
                return self._unavailable(query, "missing_surface")
            evaluated_kv = 0
        else:
            evaluated_kv = min(kv_values, key=lambda value: (abs(value - requested_kv), value))
            cap = max(1024, requested_kv // 2)
            if abs(evaluated_kv - requested_kv) > cap:
                return self._unavailable(query, "out_of_cap")
        point = next(point for point in surface if point.past_kv == evaluated_kv)

        if query.phase == "context":
            evaluated_shape = AicQueryShape(
                ctx_requests=batch_size,
                decode_requests=0,
                ctx_new_total=new_tokens_total,
                ctx_kv_total=batch_size * evaluated_kv,
                decode_kv=0,
            )
            kv_axis = "ctx_kv_total"
            requested_axis_kv = requested_shape.ctx_kv_total
            evaluated_axis_kv = evaluated_shape.ctx_kv_total
            lower, upper = _nearest_bounds(kv_values, requested_kv)
            lower *= batch_size
            upper *= batch_size
        else:
            evaluated_shape = AicQueryShape(
                ctx_requests=0,
                decode_requests=batch_size,
                ctx_new_total=0,
                ctx_kv_total=0,
                decode_kv=evaluated_kv,
            )
            kv_axis = "decode_kv"
            requested_axis_kv = requested_shape.decode_kv
            evaluated_axis_kv = evaluated_shape.decode_kv
            lower, upper = _nearest_bounds(kv_values, requested_kv)

        lookups = list(self._phase_axis_lookups.get(query.phase, ()))
        for axis, requested in requested_shape.items():
            evaluated = dict(evaluated_shape.items())[axis]
            if axis == kv_axis:
                lookups.append(
                    _axis_lookup(
                        axis,
                        requested_axis_kv,
                        evaluated_axis_kv,
                        lower=lower,
                        upper=upper,
                    )
                )
            else:
                lookups.append(_axis_lookup(axis, requested, evaluated))

        surface_identity = {
            "batch_size": batch_size,
            "new_tokens_total": new_tokens_total,
            "phase": query.phase,
            "provenance": self._surface_provenance,
        }
        surface_content = [
            {
                "past_kv": candidate.past_kv,
                "row_content_hash": candidate.row_content_hash,
            }
            for candidate in surface
        ]
        return ConservativeSurfaceSelection(
            status="ok",
            reason="selected",
            lookup_policy=self._lookup_policy,
            requested_shape=requested_shape,
            evaluated_shape=evaluated_shape,
            point=point,
            lookup_surface_id=_canonical_hash(surface_identity),
            scheduler_surface_content_hash=_canonical_hash(surface_content),
            axis_lookups=tuple(lookups),
            match_type="exact" if requested_kv == evaluated_kv else "nearest",
        )


def _empty_inventory_hash() -> str:
    return _canonical_hash([])


def _unavailable_prediction(
    query: SemanticBinQuery,
    selection: ConservativeSurfaceSelection,
    *,
    reason: str,
    runner: AicStepRunner,
    configuration_provenance: str,
) -> AicPredictionRecord:
    return AicPredictionRecord(
        configuration_fingerprint=query.configuration_fingerprint,
        concurrency=query.concurrency,
        phase=query.phase,
        semantic_key=query.semantic_key,
        status="unavailable",
        reason=reason,
        requested_shape=selection.requested_shape,
        evaluated_shape=selection.evaluated_shape,
        lookup_policy=selection.lookup_policy,
        lookup_surface_id=selection.lookup_surface_id,
        scheduler_surface_content_hash=selection.scheduler_surface_content_hash,
        axis_lookups=selection.axis_lookups,
        total_ms=None,
        total_basis=None,
        compute_ms=None,
        communication_ms=None,
        other_ms=None,
        component_sum_ms=None,
        source=None,
        match_type=selection.match_type,
        predictor_version=PREDICTOR_VERSION,
        api_version=runner.api_version,
        component_classifier_version=COMPONENT_CLASSIFIER_VERSION,
        classified_operation_count=0,
        unclassified_operation_count=0,
        operation_inventory=(),
        operation_inventory_hash=_empty_inventory_hash(),
        operation_values=(),
        operation_lookups=(),
        configuration_provenance=configuration_provenance,
    )


class AiconfiguratorSemanticBinPredictor:
    """Reference semantic-bin adapter over a pinned AIC scheduler surface."""

    def __init__(
        self,
        *,
        surface_index: ConservativeSurfaceIndex,
        runner: AicStepRunner,
        configuration_provenance: str,
        expected_configuration_fingerprint: str | None = None,
    ):
        if not configuration_provenance:
            raise ValueError("configuration_provenance must be non-empty")
        self._surface_index = surface_index
        self._runner = runner
        self._configuration_provenance = configuration_provenance
        self._expected_configuration_fingerprint = expected_configuration_fingerprint

    def predict(self, query: SemanticBinQuery) -> AicPredictionRecord:
        if (
            self._expected_configuration_fingerprint is not None
            and query.configuration_fingerprint != self._expected_configuration_fingerprint
        ):
            raise ContractError(
                process_code="configuration_mismatch",
                detail=(
                    "predictor configuration fingerprint "
                    f"{self._expected_configuration_fingerprint!r} does not match query "
                    f"{query.configuration_fingerprint!r}"
                ),
            )
        selection = self._surface_index.select(query)
        if selection.status != "ok":
            return _unavailable_prediction(
                query,
                selection,
                reason=selection.reason,
                runner=self._runner,
                configuration_provenance=self._configuration_provenance,
            )
        if selection.evaluated_shape is None or selection.point is None:
            raise RuntimeError("successful scheduler selection lacks an evaluated point")

        try:
            raw = self._runner.predict(
                phase=query.phase,
                shape=selection.evaluated_shape,
                point=selection.point,
            )
        except AicSurfaceUnavailableError:
            return _unavailable_prediction(
                query,
                selection,
                reason="missing_surface",
                runner=self._runner,
                configuration_provenance=self._configuration_provenance,
            )
        operation_names = tuple(name for name, _ in raw.operations)
        if not operation_names or len(operation_names) != len(set(operation_names)):
            raise RuntimeError("AIC operation inventory is empty or contains duplicate names")
        source_map = dict(raw.sources)
        if len(source_map) != len(raw.sources) or set(source_map) != set(operation_names):
            raise RuntimeError("AIC source inventory does not match operation inventory")
        if any(not math.isfinite(value) for _, value in raw.operations):
            raise RuntimeError("AIC operation inventory contains a non-finite value")

        classified = []
        unclassified = []
        component_values = {"compute": 0.0, "communication": 0.0, "other": 0.0}
        inventory = []
        for name, value in raw.operations:
            component = aic_component_class(name)
            inventory.append((name, component or "unclassified"))
            if component is None:
                unclassified.append(name)
            else:
                classified.append(name)
                component_values[component] += value
        total_ms = sum(value for _, value in raw.operations)
        if not math.isfinite(total_ms) or total_ms < 0.0:
            raise RuntimeError(f"AIC total must be finite and non-negative, got {total_ms!r}")

        positive_comm_names = {
            name for name, value in raw.operations if aic_component_class(name) == "communication" and value != 0.0
        }
        lookup_consumers = {consumer for lookup in raw.operation_lookups for consumer in lookup.consumer_operations}
        if positive_comm_names != lookup_consumers:
            return _unavailable_prediction(
                query,
                selection,
                reason="missing_surface",
                runner=self._runner,
                configuration_provenance=self._configuration_provenance,
            )

        if unclassified:
            compute_ms = communication_ms = other_ms = component_sum_ms = None
        else:
            compute_ms = component_values["compute"]
            communication_ms = component_values["communication"]
            other_ms = component_values["other"]
            component_sum_ms = compute_ms + communication_ms + other_ms
        active_names = {name for name, value in raw.operations if value != 0.0}
        missing_active_sources = sorted(name for name in active_names if not source_map[name])
        if missing_active_sources:
            raise RuntimeError("active AIC operations lack source provenance: " + ", ".join(missing_active_sources))
        active_sources = {source_map[name] for name in active_names}
        if not active_sources:
            source = "n/a"
        elif len(active_sources) == 1:
            source = next(iter(active_sources))
        else:
            source = "mixed"

        return AicPredictionRecord(
            configuration_fingerprint=query.configuration_fingerprint,
            concurrency=query.concurrency,
            phase=query.phase,
            semantic_key=query.semantic_key,
            status="ok",
            reason="predicted",
            requested_shape=selection.requested_shape,
            evaluated_shape=selection.evaluated_shape,
            lookup_policy=self._surface_index.lookup_policy,
            lookup_surface_id=selection.lookup_surface_id,
            scheduler_surface_content_hash=selection.scheduler_surface_content_hash,
            axis_lookups=selection.axis_lookups,
            total_ms=total_ms,
            total_basis="operation_sum",
            compute_ms=compute_ms,
            communication_ms=communication_ms,
            other_ms=other_ms,
            component_sum_ms=component_sum_ms,
            source=source,
            match_type=selection.match_type,
            predictor_version=PREDICTOR_VERSION,
            api_version=self._runner.api_version,
            component_classifier_version=COMPONENT_CLASSIFIER_VERSION,
            classified_operation_count=len(classified),
            unclassified_operation_count=len(unclassified),
            operation_inventory=tuple(sorted(inventory)),
            operation_inventory_hash=_canonical_hash(sorted(inventory)),
            operation_values=tuple(sorted((name, value, source_map[name]) for name, value in raw.operations)),
            operation_lookups=raw.operation_lookups,
            configuration_provenance=self._configuration_provenance,
        )


def _query_for_bin(bin_) -> SemanticBinQuery:
    ctx_requests, decode_requests, ctx_new, ctx_kv, decode_kv = bin_.semantic_key
    shape = SemanticShape(
        ctx_requests=ctx_requests,
        decode_requests=decode_requests,
        ctx_new_tokens=ctx_requests * (ctx_new or 0),
        ctx_kv_tokens=ctx_requests * (ctx_kv or 0),
        decode_kv_tokens=decode_requests * (decode_kv or 0),
    )
    return build_semantic_query(
        configuration_fingerprint=bin_.configuration_fingerprint,
        concurrency=bin_.concurrency,
        shape=shape,
        source_phase=bin_.phase,
    )


def _raise_contract(process_code: str, detail: str) -> None:
    raise ContractError(process_code=process_code, detail=detail)


def _validate_axis_inventory(record: AicPredictionRecord, *, required_lookup_policy: str) -> None:
    seen = set()
    for lookup in record.axis_lookups:
        if lookup.axis in seen:
            _raise_contract("predictor_error", f"duplicate axis lookup {lookup.axis!r}")
        seen.add(lookup.axis)
        if lookup.evaluated is None:
            if (
                lookup.mode != "missing"
                or lookup.lower is not None
                or lookup.upper is not None
                or lookup.weight is not None
                or lookup.delta is not None
            ):
                _raise_contract("predictor_error", f"incoherent missing axis lookup {lookup.axis!r}")
            continue
        if lookup.lower is None or lookup.upper is None or lookup.weight is None or lookup.delta is None:
            _raise_contract("predictor_error", f"incomplete axis lookup {lookup.axis!r}")
        if lookup.lower > lookup.upper or not math.isfinite(lookup.weight):
            _raise_contract("predictor_error", f"invalid bounds/weight for axis {lookup.axis!r}")
        if lookup.delta != lookup.evaluated - lookup.requested:
            _raise_contract("predictor_error", f"invalid delta for axis {lookup.axis!r}")
        if lookup.mode == "exact":
            if not (lookup.requested == lookup.evaluated == lookup.lower == lookup.upper and lookup.weight == 0.0):
                _raise_contract("predictor_error", f"incoherent exact axis lookup {lookup.axis!r}")
        elif lookup.mode == "nearest":
            if not (
                lookup.evaluated in {lookup.lower, lookup.upper}
                and (lookup.lower == lookup.upper or lookup.lower <= lookup.requested <= lookup.upper)
            ):
                _raise_contract("predictor_error", f"incoherent nearest axis lookup {lookup.axis!r}")
            expected_weight = (
                0.0
                if lookup.lower == lookup.upper
                else (lookup.requested - lookup.lower) / (lookup.upper - lookup.lower)
            )
            if not math.isclose(lookup.weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
                _raise_contract("predictor_error", f"invalid nearest weight for axis {lookup.axis!r}")
        elif lookup.mode == "proxy":
            if lookup.lower != lookup.evaluated or lookup.upper != lookup.evaluated or lookup.weight != 0.0:
                _raise_contract("predictor_error", f"incoherent proxy axis lookup {lookup.axis!r}")
        else:
            _raise_contract("predictor_error", f"unsupported axis lookup mode {lookup.mode!r}")

    if record.status in {"ok", "unavailable"}:
        requested = dict(record.requested_shape.items())
        shape_axes = set(requested)
        config_axis = (
            "max_num_seqs"
            if record.phase == "decode"
            else "max_num_batched_tokens"
            if record.phase == "context"
            else None
        )
        expected_axes = shape_axes | ({config_axis} if config_axis is not None else set())
        if seen != expected_axes:
            _raise_contract(
                "predictor_error",
                f"prediction axis inventory {sorted(seen)!r} does not match required {sorted(expected_axes)!r}",
            )
        by_axis = {lookup.axis: lookup for lookup in record.axis_lookups}
        if record.status == "unavailable" and record.evaluated_shape is None:
            if config_axis is not None:
                allowed_config_modes = {"missing", "exact"}
                if required_lookup_policy == ARTIFACT_PROXY_LOOKUP_POLICY_VERSION and record.phase == "decode":
                    allowed_config_modes = {"missing", "proxy"}
                if by_axis[config_axis].mode not in allowed_config_modes:
                    _raise_contract(
                        "predictor_error",
                        f"unavailable configuration axis {config_axis!r} has invalid lookup mode",
                    )
            for axis in shape_axes:
                lookup = by_axis[axis]
                if lookup.requested != requested[axis] or lookup.mode != "missing":
                    _raise_contract(
                        "predictor_error",
                        f"unavailable scheduler axis {axis!r} must preserve its request as missing",
                    )
            return

    if record.status == "ok" and record.evaluated_shape is None:
        _raise_contract("predictor_error", "successful prediction lacks evaluated scheduler shape")
    if record.evaluated_shape is not None:
        evaluated = dict(record.evaluated_shape.items())
        for axis in shape_axes:
            lookup = by_axis[axis]
            if lookup.requested != requested[axis] or lookup.evaluated != evaluated[axis]:
                _raise_contract("predictor_error", f"axis lookup {axis!r} disagrees with scheduler shapes")
            kv_axis = "decode_kv" if record.phase == "decode" else "ctx_kv_total"
            if axis != kv_axis and lookup.mode != "exact":
                _raise_contract("predictor_error", f"scheduler axis {axis!r} must match exactly")
        kv_axis = "decode_kv" if record.phase == "decode" else "ctx_kv_total"
        kv_lookup = by_axis[kv_axis]
        if kv_lookup.mode == "nearest":
            if kv_lookup.evaluated is None or kv_lookup.lower is None or kv_lookup.upper is None:
                _raise_contract("predictor_error", "nearest KV lookup is incomplete")
            nearest = min(
                (kv_lookup.lower, kv_lookup.upper),
                key=lambda value: (abs(value - kv_lookup.requested), value),
            )
            if kv_lookup.evaluated != nearest:
                _raise_contract("predictor_error", "KV lookup violates nearest/lower-tie policy")
            if record.phase == "context":
                count = record.requested_shape.ctx_requests
                if count <= 0 or kv_lookup.requested % count or kv_lookup.evaluated % count:
                    _raise_contract("predictor_error", "context KV lookup is not request-uniform")
                requested_kv = kv_lookup.requested // count
                evaluated_kv = kv_lookup.evaluated // count
            else:
                requested_kv = kv_lookup.requested
                evaluated_kv = kv_lookup.evaluated
            if abs(evaluated_kv - requested_kv) > max(1024, requested_kv // 2):
                _raise_contract("predictor_error", "KV lookup exceeds conservative-v1 snap cap")
        assert config_axis is not None
        config_lookup = by_axis[config_axis]
        expected_config_mode = (
            "proxy"
            if required_lookup_policy == ARTIFACT_PROXY_LOOKUP_POLICY_VERSION and record.phase == "decode"
            else "exact"
        )
        if config_lookup.mode != expected_config_mode:
            _raise_contract("predictor_error", f"configuration axis {config_axis!r} has invalid lookup mode")


def _validate_operation_inventory(record: AicPredictionRecord) -> dict[str, tuple[float, str]]:
    if tuple(sorted(record.operation_inventory)) != record.operation_inventory:
        _raise_contract("predictor_error", "AIC operation inventory is not canonically ordered")
    names = [name for name, _ in record.operation_inventory]
    if len(names) != len(set(names)):
        _raise_contract("predictor_error", "AIC operation inventory contains duplicate names")
    expected_inventory = tuple((name, aic_component_class(name) or "unclassified") for name in names)
    if record.operation_inventory != expected_inventory:
        _raise_contract("predictor_error", "AIC operation inventory disagrees with component registry")
    classified = sum(component != "unclassified" for _, component in record.operation_inventory)
    unclassified = len(record.operation_inventory) - classified
    if record.classified_operation_count != classified or record.unclassified_operation_count != unclassified:
        _raise_contract("predictor_error", "AIC operation counts disagree with inventory")
    if record.operation_inventory_hash != _canonical_hash(record.operation_inventory):
        _raise_contract("predictor_error", "AIC operation inventory hash mismatch")

    if tuple(sorted(record.operation_values)) != record.operation_values:
        _raise_contract("predictor_error", "AIC operation values are not canonically ordered")
    values_by_name = {name: (value, source) for name, value, source in record.operation_values}
    if len(values_by_name) != len(record.operation_values) or set(values_by_name) != set(names):
        _raise_contract("predictor_error", "AIC operation values disagree with inventory")
    if any(not math.isfinite(value) for value, _ in values_by_name.values()):
        _raise_contract("predictor_error", "AIC operation values contain non-finite values")
    if any(value != 0.0 and not source for value, source in values_by_name.values()):
        _raise_contract("predictor_error", "active AIC operation value lacks source provenance")

    communication_names = {name for name, component in record.operation_inventory if component == "communication"}
    consumers = []
    for lookup in record.operation_lookups:
        if not lookup.surface_content_hash or len(lookup.surface_content_hash) != 64:
            _raise_contract("predictor_error", "invalid operation surface content hash")
        try:
            topology = json.loads(lookup.topology)
        except json.JSONDecodeError as exc:
            raise ContractError(process_code="predictor_error", detail="invalid operation topology JSON") from exc
        if (
            not isinstance(topology, dict)
            or not topology
            or json.dumps(
                topology,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            != lookup.topology
        ):
            _raise_contract("predictor_error", "invalid operation topology")
        if lookup.lower > lookup.upper or not (lookup.lower <= lookup.requested <= lookup.upper):
            _raise_contract("predictor_error", "operation interpolation is outside collected bounds")
        if not math.isfinite(lookup.weight):
            _raise_contract("predictor_error", "operation interpolation weight is non-finite")
        expected_weight = (
            0.0 if lookup.lower == lookup.upper else (lookup.requested - lookup.lower) / (lookup.upper - lookup.lower)
        )
        expected_mode = "exact" if lookup.lower == lookup.upper else "interpolate"
        if lookup.mode != expected_mode or not math.isclose(
            lookup.weight,
            expected_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            _raise_contract("predictor_error", "incoherent operation interpolation provenance")
        if not lookup.consumer_operations:
            _raise_contract("predictor_error", "operation lookup has no consumer operation")
        consumers.extend(lookup.consumer_operations)
    if len(consumers) != len(set(consumers)) or set(consumers) != communication_names:
        _raise_contract("predictor_error", "operation lookup consumers do not match communication inventory")
    return values_by_name


def _validate_prediction(
    query: SemanticBinQuery,
    record: AicPredictionRecord,
    *,
    required_lookup_policy: str,
) -> None:
    if not isinstance(record, AicPredictionRecord):
        _raise_contract("predictor_error", f"predictor returned {type(record).__name__}, expected AicPredictionRecord")
    identity = (
        record.configuration_fingerprint,
        record.concurrency,
        record.phase,
        record.semantic_key,
    )
    expected_identity = (
        query.configuration_fingerprint,
        query.concurrency,
        query.phase,
        query.semantic_key,
    )
    if identity != expected_identity or record.requested_shape != AicQueryShape.from_query(query):
        _raise_contract(
            "predictor_call_identity_mismatch",
            f"predictor returned identity {identity!r} for query {expected_identity!r}",
        )
    if required_lookup_policy not in SUPPORTED_LOOKUP_POLICIES:
        raise ValueError(f"unsupported required lookup policy {required_lookup_policy!r}")
    if record.lookup_policy != required_lookup_policy:
        _raise_contract(
            "predictor_error",
            f"lookup policy {record.lookup_policy!r} does not match required {required_lookup_policy!r}",
        )
    if required_lookup_policy == LOOKUP_POLICY_VERSION and any(
        lookup.mode == "proxy" for lookup in record.axis_lookups
    ):
        _raise_contract("predictor_error", "conservative-v1 record contains a proxy axis lookup")
    if (
        not record.predictor_version
        or not record.api_version
        or record.component_classifier_version != COMPONENT_CLASSIFIER_VERSION
        or not record.configuration_provenance
    ):
        _raise_contract("predictor_error", "prediction provenance/version fields are incomplete")
    try:
        configuration_provenance = json.loads(record.configuration_provenance)
    except json.JSONDecodeError as exc:
        raise ContractError(process_code="predictor_error", detail="invalid configuration provenance JSON") from exc
    if not isinstance(configuration_provenance, dict):
        _raise_contract("predictor_error", "configuration provenance must be a JSON object")
    _validate_axis_inventory(record, required_lookup_policy=required_lookup_policy)
    values_by_name = _validate_operation_inventory(record)
    if record.status == "unavailable":
        if record.reason not in EXPECTED_UNAVAILABLE_REASONS:
            _raise_contract("predictor_error", f"unexpected unavailable reason {record.reason!r}")
        if record.total_basis is not None or record.source is not None:
            _raise_contract("predictor_error", "unavailable prediction contains result provenance")
        if record.evaluated_shape is None:
            if (
                record.match_type is not None
                or record.lookup_surface_id is not None
                or record.scheduler_surface_content_hash is not None
            ):
                _raise_contract(
                    "predictor_error",
                    "pre-selection unavailable prediction contains selected-surface provenance",
                )
        else:
            if not record.lookup_surface_id or not record.scheduler_surface_content_hash:
                _raise_contract(
                    "predictor_error",
                    "post-selection unavailable prediction lacks selected-surface provenance",
                )
            kv_axis = "decode_kv" if record.phase == "decode" else "ctx_kv_total"
            kv_lookup = next(lookup for lookup in record.axis_lookups if lookup.axis == kv_axis)
            expected_match_type = "nearest" if kv_lookup.mode == "nearest" else "exact"
            if record.match_type != expected_match_type:
                _raise_contract(
                    "predictor_error",
                    "post-selection unavailable match type disagrees with KV lookup mode",
                )
        numeric = (
            record.total_ms,
            record.compute_ms,
            record.communication_ms,
            record.other_ms,
            record.component_sum_ms,
        )
        if any(value is not None for value in numeric):
            _raise_contract("predictor_error", "unavailable prediction contains numeric results")
        if (
            record.operation_inventory
            or record.operation_values
            or record.operation_lookups
            or record.classified_operation_count
            or record.unclassified_operation_count
            or record.operation_inventory_hash != _empty_inventory_hash()
        ):
            _raise_contract("predictor_error", "unavailable prediction contains operation results")
        return
    if record.status != "ok" or record.reason != "predicted":
        _raise_contract("predictor_error", f"unexpected predictor status {record.status!r}/{record.reason!r}")
    if record.evaluated_shape is None or record.total_basis not in {"direct", "operation_sum"}:
        _raise_contract("predictor_error", "successful prediction lacks evaluated shape or total basis")
    if not record.lookup_surface_id or not record.scheduler_surface_content_hash:
        _raise_contract("predictor_error", "successful prediction lacks scheduler-surface provenance")
    if not record.source:
        _raise_contract("predictor_error", "successful prediction lacks source provenance")
    kv_axis = "decode_kv" if record.phase == "decode" else "ctx_kv_total"
    kv_lookup = next(lookup for lookup in record.axis_lookups if lookup.axis == kv_axis)
    expected_match_type = "nearest" if kv_lookup.mode == "nearest" else "exact"
    if record.match_type != expected_match_type:
        _raise_contract(
            "predictor_error",
            f"prediction match type {record.match_type!r} disagrees with KV lookup mode",
        )
    if any(value < 0 for _, value in record.evaluated_shape.items()):
        _raise_contract("predictor_error", "evaluated scheduler shape contains a negative value")
    numeric_values = [record.total_ms]
    numeric_values.extend(
        value
        for value in (record.compute_ms, record.communication_ms, record.other_ms, record.component_sum_ms)
        if value is not None
    )
    if any(value is None or not math.isfinite(value) for value in numeric_values):
        _raise_contract("predictor_error", "prediction contains missing or non-finite numeric results")
    components = (record.compute_ms, record.communication_ms, record.other_ms, record.component_sum_ms)
    if any(value is None for value in components) and any(value is not None for value in components):
        _raise_contract("predictor_error", "prediction components must be all present or all unavailable")
    if all(value is not None for value in components):
        assert record.compute_ms is not None
        assert record.communication_ms is not None
        assert record.other_ms is not None
        assert record.component_sum_ms is not None
        expected_sum = record.compute_ms + record.communication_ms + record.other_ms
        component_tolerance = max(1e-6, 1e-6 * abs(record.component_sum_ms))
        if not math.isclose(record.component_sum_ms, expected_sum, rel_tol=0.0, abs_tol=component_tolerance):
            _raise_contract("predictor_error", "AIC component sum does not close")
        total_tolerance = max(1e-6, 1e-6 * abs(record.total_ms or 0.0))
        if not math.isclose(
            record.total_ms or 0.0,
            record.component_sum_ms,
            rel_tol=0.0,
            abs_tol=total_tolerance,
        ):
            _raise_contract("predictor_error", "AIC total does not close against signed components")
        if record.communication_ms != 0.0 and not record.operation_lookups:
            _raise_contract("predictor_error", "positive AIC communication lacks operation lookup provenance")
        operation_components = {"compute": 0.0, "communication": 0.0, "other": 0.0}
        component_by_name = dict(record.operation_inventory)
        for name, (value, _source) in values_by_name.items():
            operation_components[component_by_name[name]] += value
        for component, actual in (
            ("compute", record.compute_ms),
            ("communication", record.communication_ms),
            ("other", record.other_ms),
        ):
            tolerance = max(1e-6, 1e-6 * abs(actual))
            if not math.isclose(
                operation_components[component],
                actual,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                _raise_contract(
                    "predictor_error",
                    f"AIC {component} operation values do not close against component",
                )
    operation_total = sum(value for value, _ in values_by_name.values())
    total_tolerance = max(1e-6, 1e-6 * abs(record.total_ms or 0.0))
    if not math.isclose(
        operation_total,
        record.total_ms or 0.0,
        rel_tol=0.0,
        abs_tol=total_tolerance,
    ):
        _raise_contract("predictor_error", "AIC operation values do not close against total")


def predict_clean_bins(
    population: SemanticPopulation,
    predictor: SemanticBinPredictor,
    *,
    required_lookup_policy: str = LOOKUP_POLICY_VERSION,
) -> tuple[SemanticBinPrediction, ...]:
    """Call ``predictor`` exactly once for each measured clean semantic bin."""

    records = []
    for bin_ in population.bins:
        if not bin_.clean_samples:
            continue
        query = _query_for_bin(bin_)
        try:
            record = predictor.predict(query)
        except Exception as exc:  # Every unexpected adapter failure aborts the contract.
            if isinstance(exc, ContractError):
                raise
            raise ContractError(
                process_code="predictor_error",
                detail=f"predictor failed for bin {bin_.bin_id}: {exc}",
            ) from exc
        _validate_prediction(query, record, required_lookup_policy=required_lookup_policy)
        records.append(SemanticBinPrediction(bin_id=bin_.bin_id, query=query, record=record))
    return tuple(records)
