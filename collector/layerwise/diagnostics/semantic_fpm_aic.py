#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-backed surfaces and execution for semantic FPM AIC prediction."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from collector.layerwise.diagnostics.semantic_fpm_insights import AicQueryShape, AxisLookup, OperationLookup
from collector.layerwise.diagnostics.semantic_fpm_predictor import (
    ARTIFACT_PROXY_LOOKUP_POLICY_VERSION,
    LOOKUP_POLICY_VERSION,
    AiconfiguratorSemanticBinPredictor,
    AicSurfaceUnavailableError,
    ConservativeSurfaceIndex,
    LayerwiseSurfacePoint,
    RawAicStep,
    aic_component_class,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import ContractError

_REQUIRED_LAYERWISE_COLUMNS = frozenset(
    {
        "framework",
        "framework_version",
        "system",
        "model",
        "attn_tp",
        "moe_tp",
        "ep",
        "num_slots",
        "gemm_quant",
        "moe_quant",
        "attn_quant",
        "kv_quant",
        "phase",
        "batch_size",
        "new_tokens",
        "past_kv",
        "layer_type",
        "layer_index",
        "measured_layer_count",
        "layer_multiplier",
        "latency_ms",
        "rms_latency_ms",
        "rms_kernel_count",
        "includes_moe",
        "moe_weight_mode",
        "latency_source",
        "physical_gpus",
        "max_num_seqs",
        "max_num_batched_tokens",
        "vllm_config_hash",
    }
)
_BACKEND = "vllm"
_MOE_TP_SIZE = 1
_EP_SIZE = 1
_UNVERIFIED_PARITY_FIELDS = (
    "chunked_prefill",
    "model_revision",
    "prefix_caching",
    "runtime_flags",
)
_SEMANTIC_EXECUTION_PATHS = (
    "collector/layerwise/common/parse_nsys_step_sweep.py",
    "collector/layerwise/diagnostics/aic_fpm_gap.py",
    "collector/layerwise/diagnostics/semantic_fpm_aic.py",
    "collector/layerwise/diagnostics/semantic_fpm_contract.py",
    "collector/layerwise/diagnostics/semantic_fpm_insights.py",
    "collector/layerwise/diagnostics/semantic_fpm_nsys.py",
    "collector/layerwise/diagnostics/semantic_fpm_predictor.py",
    "collector/layerwise/diagnostics/semantic_fpm_reduction.py",
    "collector/layerwise/diagnostics/semantic_fpm_stage1.py",
    "collector/layerwise/reproduce_layerwise_fpm.sh",
    "src/aiconfigurator/sdk/backends/vllm_backend.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bounded_operation_lookup(
    *,
    operation: str,
    topology: dict[str, object],
    requested: int,
    surface: dict[int, object],
) -> OperationLookup:
    """Return in-range linear interpolation provenance for one operation surface."""

    values = tuple(sorted(int(value) for value in surface))
    if not values or requested < values[0] or requested > values[-1]:
        raise AicSurfaceUnavailableError(
            f"{operation} request {requested} is outside collected range "
            f"{values[0] if values else None}..{values[-1] if values else None}"
        )
    if requested in surface:
        lower = upper = requested
        weight = 0.0
        mode = "exact"
    else:
        lower = max(value for value in values if value < requested)
        upper = min(value for value in values if value > requested)
        weight = (requested - lower) / (upper - lower)
        mode = "interpolate"
    canonical_surface = [{"coordinate": coordinate, "value": surface[coordinate]} for coordinate in values]
    return OperationLookup(
        operation=operation,
        topology=_canonical_json(topology),
        requested=requested,
        lower=lower,
        upper=upper,
        weight=weight,
        mode=mode,
        surface_content_hash=_sha256_bytes(_canonical_json(canonical_surface).encode("utf-8")),
    )


def _parse_int(row: dict[str, str], field: str, row_number: int) -> int:
    try:
        numeric = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layerwise row {row_number}: {field} must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"layerwise row {row_number}: {field} must be an integer")
    return int(numeric)


def _parse_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layerwise row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"layerwise row {row_number}: {field} must be finite")
    return value


@dataclass(frozen=True)
class RepositoryAicConfig:
    """Pinned repository/runtime identity used to filter raw layerwise rows."""

    repo_root: Path
    repo_commit: str
    configuration_fingerprint: str
    parity_record: str
    layerwise_csv: Path
    comm_version: str
    context_vllm_config_hash: str
    decode_vllm_config_hash: str
    artifact_proxy_decode_max_num_seqs: int | None = None
    verify_repository: bool = True
    _parity: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.repo_commit:
            raise ValueError("repository commit must be non-empty")
        if not isinstance(self.verify_repository, bool):
            raise TypeError("verify_repository must be a boolean")
        try:
            parsed_parity = json.loads(self.parity_record)
        except json.JSONDecodeError as exc:
            raise ValueError("parity_record must be canonical JSON") from exc
        if not isinstance(parsed_parity, dict) or _canonical_json(parsed_parity) != self.parity_record:
            raise ValueError("parity_record must be a canonical JSON object")
        object.__setattr__(self, "_parity", parsed_parity)
        actual_fingerprint = _sha256_bytes(self.parity_record.encode("utf-8"))
        if actual_fingerprint != self.configuration_fingerprint:
            raise ValueError(
                f"configuration fingerprint mismatch: expected {actual_fingerprint}, "
                f"got {self.configuration_fingerprint}"
            )
        required_fields = {
            "attention_dp_size",
            "attention_quant",
            "backend",
            "backend_version",
            "chunked_prefill",
            "dp_size",
            "ep_size",
            "gemm_quant",
            "gpu_count",
            "kv_cache_dtype",
            "kv_cache_quant",
            "max_num_batched_tokens",
            "max_num_seqs",
            "model",
            "model_revision",
            "moe_quant",
            "numerical_dtype",
            "pp_size",
            "prefix_caching",
            "runtime_flags",
            "schema_version",
            "system",
            "tp_size",
        }
        missing = sorted(required_fields - set(parsed_parity))
        if missing:
            raise ValueError(f"parity_record is missing required fields: {', '.join(missing)}")
        unexpected = sorted(set(parsed_parity) - required_fields)
        if unexpected:
            raise ValueError(f"parity_record has unexpected fields: {', '.join(unexpected)}")
        if parsed_parity["schema_version"] != "aic-runtime-parity/v1":
            raise ValueError("unsupported parity_record schema_version")
        if parsed_parity["backend"] != _BACKEND:
            raise ValueError(f"semantic predictor requires backend={_BACKEND!r}")
        for parity_field in (
            "attention_dp_size",
            "dp_size",
            "ep_size",
            "gpu_count",
            "max_num_batched_tokens",
            "max_num_seqs",
            "pp_size",
            "tp_size",
        ):
            value = parsed_parity[parity_field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"parity_record {parity_field} must be a positive integer")
        fixed = {
            "attention_dp_size": 1,
            "attention_quant": "bf16",
            "dp_size": 1,
            "ep_size": _EP_SIZE,
            "gemm_quant": "bf16",
            "kv_cache_dtype": "bf16",
            "kv_cache_quant": "bf16",
            "moe_quant": "bf16",
            "numerical_dtype": "bf16",
            "pp_size": 1,
        }
        for parity_field, expected in fixed.items():
            if parsed_parity[parity_field] != expected:
                raise ValueError(f"semantic predictor requires parity_record {parity_field}={expected!r}")
        if not isinstance(parsed_parity["prefix_caching"], bool) or not isinstance(
            parsed_parity["chunked_prefill"], bool
        ):
            raise TypeError("parity_record prefix_caching and chunked_prefill must be booleans")
        if not isinstance(parsed_parity["runtime_flags"], dict):
            raise TypeError("parity_record runtime_flags must be an object")
        for parity_field in ("backend_version", "model", "model_revision", "system"):
            if not isinstance(parsed_parity[parity_field], str) or not parsed_parity[parity_field]:
                raise ValueError(f"parity_record {parity_field} must be a non-empty string")
        if parsed_parity["gpu_count"] != parsed_parity["tp_size"]:
            raise ValueError("dense Qwen predictor requires gpu_count == tp_size")
        proxy_max_num_seqs = self.artifact_proxy_decode_max_num_seqs
        if proxy_max_num_seqs is not None and (
            isinstance(proxy_max_num_seqs, bool) or not isinstance(proxy_max_num_seqs, int) or proxy_max_num_seqs <= 0
        ):
            raise ValueError("artifact_proxy_decode_max_num_seqs must be a positive integer")
        for field_name, value in (
            ("comm_version", self.comm_version),
            ("context_vllm_config_hash", self.context_vllm_config_hash),
            ("decode_vllm_config_hash", self.decode_vllm_config_hash),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if not self.verify_repository and not self.is_artifact_proxy:
            raise ValueError("repository verification may only be disabled for artifact-proxy-v1")

    @property
    def parity(self) -> dict[str, Any]:
        return dict(self._parity)

    @property
    def model(self) -> str:
        return str(self._parity["model"])

    @property
    def system(self) -> str:
        return str(self._parity["system"])

    @property
    def framework_version(self) -> str:
        return str(self._parity["backend_version"])

    @property
    def tp_size(self) -> int:
        return int(self._parity["tp_size"])

    @property
    def runtime_max_num_batched_tokens(self) -> int:
        return int(self._parity["max_num_batched_tokens"])

    @property
    def runtime_max_num_seqs(self) -> int:
        return int(self._parity["max_num_seqs"])

    @property
    def is_artifact_proxy(self) -> bool:
        return self.artifact_proxy_decode_max_num_seqs not in (None, self.runtime_max_num_seqs)


@dataclass(frozen=True)
class RepositorySurface:
    """Filtered layerwise index plus its canonical configuration provenance."""

    index: ConservativeSurfaceIndex
    configuration_provenance: str


def _base_row_matches(row: dict[str, str], config: RepositoryAicConfig) -> bool:
    expected = {
        "framework": "vLLM",
        "framework_version": config.framework_version,
        "system": config.system,
        "model": config.model,
        "attn_tp": str(config.tp_size),
        "moe_tp": str(_MOE_TP_SIZE),
        "ep": str(_EP_SIZE),
        "num_slots": "",
        "gemm_quant": "bf16",
        "moe_quant": "bf16",
        "attn_quant": "bf16",
        "kv_quant": "bf16",
        "layer_type": "dense",
        "includes_moe": "False",
        "moe_weight_mode": "dense",
        "physical_gpus": "1",
    }
    return all(str(row.get(field, "")) == value for field, value in expected.items())


def _phase_row_matches(row: dict[str, str], config: RepositoryAicConfig) -> tuple[str, bool]:
    phase = str(row.get("phase", "")).lower()
    if phase == "ctx":
        matches = (
            row.get("max_num_seqs", "") == ""
            and row.get("max_num_batched_tokens", "") == str(config.runtime_max_num_batched_tokens)
            and row.get("vllm_config_hash", "") == config.context_vllm_config_hash
            and row.get("latency_source", "") == "schedule_to_update"
        )
        return "context", matches
    if phase == "gen":
        target_mns = (
            config.artifact_proxy_decode_max_num_seqs
            if config.artifact_proxy_decode_max_num_seqs is not None
            else config.runtime_max_num_seqs
        )
        matches = (
            row.get("max_num_seqs", "") == str(target_mns)
            and row.get("max_num_batched_tokens", "") == ""
            and row.get("vllm_config_hash", "") == config.decode_vllm_config_hash
            and row.get("latency_source", "") == "execute_model_gpu"
        )
        return "decode", matches
    return phase, False


def _detail_json(row: dict[str, str], row_number: int) -> str:
    detail = {
        "latency": _parse_float(row, "latency_ms", row_number),
        "energy": 0.0,
        "rms_latency": _parse_float(row, "rms_latency_ms", row_number),
        "rms_kernel_count": _parse_float(row, "rms_kernel_count", row_number),
        "includes_moe": row["includes_moe"].lower() == "true",
        "layer_type": row["layer_type"],
        "layer_index": _parse_float(row, "layer_index", row_number),
        "measured_layer_count": _parse_float(row, "measured_layer_count", row_number),
        "layer_multiplier": _parse_float(row, "layer_multiplier", row_number),
        "physical_gpus": _parse_float(row, "physical_gpus", row_number),
        "latency_source": row["latency_source"],
        "moe_weight_mode": row["moe_weight_mode"],
        "vllm_config_hash": row["vllm_config_hash"],
    }
    if row.get("max_num_seqs", ""):
        detail["max_num_seqs"] = _parse_float(row, "max_num_seqs", row_number)
    if row.get("max_num_batched_tokens", ""):
        detail["max_num_batched_tokens"] = _parse_float(row, "max_num_batched_tokens", row_number)
    return _canonical_json(detail)


def _config_axis_lookup(*, axis: str, requested: int, evaluated: int | None) -> AxisLookup:
    return AxisLookup(
        axis=axis,
        requested=requested,
        evaluated=evaluated,
        lower=evaluated,
        upper=evaluated,
        weight=0.0 if evaluated is not None else None,
        delta=evaluated - requested if evaluated is not None else None,
        mode="missing" if evaluated is None else ("exact" if evaluated == requested else "proxy"),
    )


def build_repository_surface(config: RepositoryAicConfig) -> RepositorySurface:
    """Filter a raw layerwise CSV before exposing any scheduler candidates."""

    layerwise_path = Path(config.layerwise_csv)
    raw_bytes = layerwise_path.read_bytes()
    points = []
    matched_phase_counts = {"context": 0, "decode": 0}
    with layerwise_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(_REQUIRED_LAYERWISE_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"layerwise CSV is missing required columns: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if not _base_row_matches(row, config):
                continue
            phase, phase_matches = _phase_row_matches(row, config)
            if not config.verify_repository and phase != "decode":
                continue
            if not phase_matches:
                continue
            batch_size = _parse_int(row, "batch_size", row_number)
            new_tokens = _parse_int(row, "new_tokens", row_number)
            past_kv = _parse_int(row, "past_kv", row_number)
            latency_ms = _parse_float(row, "latency_ms", row_number)
            if batch_size <= 0 or new_tokens <= 0 or past_kv < 0 or latency_ms <= 0.0:
                raise ValueError(f"layerwise row {row_number} has an invalid scheduler shape")
            canonical_row = _canonical_json(row)
            points.append(
                LayerwiseSurfacePoint(
                    phase=phase,
                    batch_size=batch_size,
                    new_tokens=new_tokens,
                    past_kv=past_kv,
                    row_content_hash=_sha256_bytes(canonical_row.encode("utf-8")),
                    latency_ms=latency_ms,
                    detail_json=_detail_json(row, row_number),
                )
            )
            matched_phase_counts[phase] += 1

    context_evaluated = config.runtime_max_num_batched_tokens
    if matched_phase_counts["context"] == 0:
        context_evaluated = None
    decode_evaluated = (
        config.artifact_proxy_decode_max_num_seqs
        if config.artifact_proxy_decode_max_num_seqs is not None
        else config.runtime_max_num_seqs
    )
    if matched_phase_counts["decode"] == 0:
        decode_evaluated = None
    policy = ARTIFACT_PROXY_LOOKUP_POLICY_VERSION if config.is_artifact_proxy else LOOKUP_POLICY_VERSION
    phase_axis_lookups = (
        (
            "context",
            (
                _config_axis_lookup(
                    axis="max_num_batched_tokens",
                    requested=config.runtime_max_num_batched_tokens,
                    evaluated=context_evaluated,
                ),
            ),
        ),
        (
            "decode",
            (
                _config_axis_lookup(
                    axis="max_num_seqs",
                    requested=config.runtime_max_num_seqs,
                    evaluated=decode_evaluated,
                ),
            ),
        ),
    )
    provenance = {
        "backend": _BACKEND,
        "comm_version": config.comm_version,
        "configuration_fingerprint": config.configuration_fingerprint,
        "context_vllm_config_hash": config.context_vllm_config_hash,
        "decode_vllm_config_hash": config.decode_vllm_config_hash,
        "ep_size": _EP_SIZE,
        "framework_version": config.framework_version,
        "layerwise_csv_sha256": _sha256_bytes(raw_bytes),
        "lookup_policy": policy,
        "matched_phase_counts": matched_phase_counts,
        "model": config.model,
        "moe_tp_size": _MOE_TP_SIZE,
        "repo_commit": config.repo_commit,
        "parity_record": config.parity,
        "parity_field_binding": {
            "bound_by_predictor": sorted(set(config.parity) - set(_UNVERIFIED_PARITY_FIELDS)),
            "provenance_only_unverified": list(_UNVERIFIED_PARITY_FIELDS),
            "surface_row_filters": [
                "context_vllm_config_hash",
                "decode_vllm_config_hash",
            ],
        },
        "runtime_max_num_batched_tokens": config.runtime_max_num_batched_tokens,
        "runtime_max_num_seqs": config.runtime_max_num_seqs,
        "selected_context_max_num_batched_tokens": context_evaluated,
        "selected_decode_max_num_seqs": decode_evaluated,
        "system": config.system,
        "tp_size": config.tp_size,
    }
    configuration_provenance = _canonical_json(provenance)
    return RepositorySurface(
        index=ConservativeSurfaceIndex(
            points=tuple(points),
            surface_provenance=configuration_provenance,
            lookup_policy=policy,
            phase_axis_lookups=phase_axis_lookups,
        ),
        configuration_provenance=configuration_provenance,
    )


def _quant_name(quant_mode: object) -> str:
    value = getattr(quant_mode, "value", None)
    return str(getattr(value, "name", None) or getattr(quant_mode, "name", None) or quant_mode)


class _ExactSchedulerDatabase:
    """Proxy that permits exactly the preselected scheduler leaf and traces comm."""

    def __init__(self, inner: object, config: RepositoryAicConfig):
        self._inner = inner
        self._config = config
        self._active_phase: str | None = None
        self._active_shape: AicQueryShape | None = None
        self._active_point: LayerwiseSurfacePoint | None = None
        self._scheduler_query_count = 0
        self._operation_lookups = []

    @property
    def layerwise(self) -> object:
        return self._inner.layerwise

    @property
    def system_spec(self) -> dict[str, Any]:
        return self._inner.system_spec

    @property
    def backend(self) -> str:
        return str(self._inner.backend)

    @property
    def system(self) -> str:
        return str(self._inner.system)

    @property
    def version(self) -> str:
        return str(self._inner.version)

    def begin(self, *, phase: str, shape: AicQueryShape, point: LayerwiseSurfacePoint) -> None:
        if self._active_phase is not None:
            raise RuntimeError("nested exact scheduler prediction")
        self._active_phase = phase
        self._active_shape = shape
        self._active_point = point
        self._scheduler_query_count = 0
        self._operation_lookups = []

    def abort(self) -> None:
        self._active_phase = None
        self._active_shape = None
        self._active_point = None
        self._scheduler_query_count = 0
        self._operation_lookups = []

    def finish(self) -> tuple[OperationLookup, ...]:
        if self._scheduler_query_count != 1:
            raise RuntimeError(f"AIC issued {self._scheduler_query_count} scheduler queries; expected exactly one")
        lookups = tuple(self._operation_lookups)
        self.abort()
        return lookups

    def query_layerwise_detail(
        self,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
        *,
        moe_weight_mode: str | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, Any]:
        normalized_moe_weight_mode = moe_weight_mode or "dense"
        if normalized_moe_weight_mode != "dense":
            raise RuntimeError(f"AIC attempted unsupported moe_weight_mode={moe_weight_mode!r}")
        shape = self._active_shape
        point = self._active_point
        if shape is None or point is None or self._active_phase is None:
            raise RuntimeError("scheduler query made outside an active prediction")
        if self._active_phase == "context":
            expected = (
                self._config.model.lower(),
                "CTX",
                self._config.tp_size,
                shape.ctx_requests,
                shape.ctx_new_total // shape.ctx_requests,
                shape.ctx_kv_total // shape.ctx_requests,
                self._config.runtime_max_num_batched_tokens,
                None,
                _MOE_TP_SIZE,
                _EP_SIZE,
            )
        elif self._active_phase == "decode":
            expected = (
                self._config.model.lower(),
                "GEN",
                self._config.tp_size,
                shape.decode_requests,
                shape.decode_kv,
                0,
                None,
                self._config.runtime_max_num_seqs,
                _MOE_TP_SIZE,
                _EP_SIZE,
            )
        else:
            raise RuntimeError(f"unsupported active phase {self._active_phase!r}")
        actual = (
            model.lower(),
            phase.upper(),
            int(tp_size),
            int(batch_size),
            int(seq_len),
            int(seq_len_kv_cache),
            int(max_num_batched_tokens) if max_num_batched_tokens is not None else None,
            int(max_num_seqs) if max_num_seqs is not None else None,
            int(moe_tp_size) if moe_tp_size is not None else 1,
            int(moe_ep_size) if moe_ep_size is not None else 1,
        )
        if actual != expected:
            raise RuntimeError(f"AIC attempted scheduler fallback {actual!r}; selected leaf is {expected!r}")
        if self._scheduler_query_count:
            raise RuntimeError("AIC attempted more than one scheduler-grid lookup")
        detail = json.loads(point.detail_json)
        if not math.isclose(float(detail["latency"]), float(point.latency_ms), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("selected layerwise row latency disagrees with its detail payload")
        detail["query_seq_len_q"] = float(seq_len)
        detail["query_seq_len_kv_cache"] = float(seq_len_kv_cache)
        self._scheduler_query_count += 1
        return detail

    def _real_database(self) -> object:
        return self._inner.real_database

    def query_custom_allreduce(
        self,
        quant_mode: object,
        tp_size: int,
        size: int,
        database_mode: object | None = None,
        execution_mode: str | None = None,
    ) -> object:
        real_database = self._real_database()
        from aiconfigurator.sdk.operations.communication import CustomAllReduce

        CustomAllReduce.load_data(real_database)
        data = getattr(real_database, "_custom_allreduce_data", None)
        if data is None or not getattr(data, "loaded", False):
            raise AicSurfaceUnavailableError("custom allreduce data is not introspectable")
        effective_tp = min(int(tp_size), int(real_database.system_spec["node"]["num_gpus_per_node"]))
        by_tp = data.get(quant_mode, {})
        strategy_dict = by_tp.get(effective_tp, {})
        requested_strategy = (execution_mode or "AUTO").upper()
        selected_strategy = requested_strategy
        surface = strategy_dict.get(selected_strategy, {})
        if not surface and selected_strategy != "AUTO":
            selected_strategy = "AUTO"
            surface = strategy_dict.get(selected_strategy, {})
        if not surface:
            raise AicSurfaceUnavailableError("custom allreduce surface is missing")
        lookup = bounded_operation_lookup(
            operation="custom_allreduce",
            topology={
                "effective_tp": effective_tp,
                "quant_mode": _quant_name(quant_mode),
                "requested_strategy": requested_strategy,
                "selected_strategy": selected_strategy,
                "tp_size": int(tp_size),
            },
            requested=int(size),
            surface=surface,
        )
        result = self._inner.query_custom_allreduce(
            quant_mode,
            tp_size,
            size,
            database_mode=database_mode,
            execution_mode=execution_mode,
        )
        self._operation_lookups.append(lookup)
        return result

    def query_allreduce_rms(
        self,
        quant_mode: object,
        tp_size: int,
        size: int,
        hidden_size: int,
        fusion_pattern: str = "allreduce_residual_rms",
        database_mode: object | None = None,
    ) -> object:
        real_database = self._real_database()
        from aiconfigurator.sdk.operations.communication import AllReduceRMS

        AllReduceRMS.load_data(real_database)
        data = getattr(real_database, "_allreduce_rms_data", None)
        if data is None or not getattr(data, "loaded", False):
            raise AicSurfaceUnavailableError("allreduce RMS data is not introspectable")
        effective_tp = min(int(tp_size), int(real_database.system_spec["node"]["num_gpus_per_node"]))
        by_hidden = data.get(quant_mode, {}).get(effective_tp, {}).get(fusion_pattern, {})
        if not by_hidden:
            raise AicSurfaceUnavailableError("allreduce RMS surface is missing")
        selected_hidden = min(by_hidden, key=lambda candidate: (abs(candidate - hidden_size), candidate))
        surface = by_hidden[selected_hidden]
        lookup = bounded_operation_lookup(
            operation="allreduce_rms",
            topology={
                "effective_tp": effective_tp,
                "fusion_pattern": fusion_pattern,
                "quant_mode": _quant_name(quant_mode),
                "requested_hidden_size": int(hidden_size),
                "selected_hidden_size": int(selected_hidden),
                "tp_size": int(tp_size),
            },
            requested=int(size),
            surface=surface,
        )
        result = self._inner.query_allreduce_rms(
            quant_mode,
            tp_size,
            size,
            hidden_size,
            fusion_pattern=fusion_pattern,
            database_mode=database_mode,
        )
        self._operation_lookups.append(lookup)
        return result


class RepositoryAicStepRunner:
    """Execute one exact semantic scheduler point through the in-tree vLLM backend."""

    def __init__(
        self,
        *,
        config: RepositoryAicConfig,
        backend: object,
        model: object,
        database: _ExactSchedulerDatabase,
        runtime_config: object,
        vllm_backend_module: object,
        use_fused_allreduce_rms: bool,
    ):
        self.api_version = config.repo_commit
        self._config = config
        self._backend = backend
        self._model = model
        self._database = database
        self._runtime_config = runtime_config
        self._vllm_backend_module = vllm_backend_module
        self._use_fused_allreduce_rms = use_fused_allreduce_rms

    def predict(
        self,
        *,
        phase: str,
        shape: AicQueryShape,
        point: LayerwiseSurfacePoint,
    ) -> RawAicStep:
        previous_flags = (
            self._vllm_backend_module._USE_LAYERWISE,
            self._vllm_backend_module._DECODE_COMPUTE_BATCH_CAL,
            self._vllm_backend_module._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
        )
        self._vllm_backend_module._USE_LAYERWISE = True
        self._vllm_backend_module._DECODE_COMPUTE_BATCH_CAL = 0.0
        self._vllm_backend_module._LAYERWISE_USE_FUSED_ALLREDUCE_RMS = self._use_fused_allreduce_rms
        try:
            self._database.begin(phase=phase, shape=shape, point=point)
            try:
                if phase == "context":
                    latency, _, sources = self._backend._get_context_step_latency(
                        self._model,
                        self._database,
                        self._runtime_config,
                        ctx_tokens=shape.ctx_new_total,
                        ctx_kv_tokens=shape.ctx_kv_total,
                        ctx_requests=shape.ctx_requests,
                    )
                elif phase == "decode":
                    latency, _, sources = self._backend._get_decode_step_latency(
                        self._model,
                        self._database,
                        self._runtime_config,
                        batch_size=shape.decode_requests,
                        past_kv=shape.decode_kv,
                    )
                else:
                    raise RuntimeError(f"repository runner does not support phase {phase!r}")
                operation_lookups = self._database.finish()
            except Exception:
                self._database.abort()
                raise
        finally:
            (
                self._vllm_backend_module._USE_LAYERWISE,
                self._vllm_backend_module._DECODE_COMPUTE_BATCH_CAL,
                self._vllm_backend_module._LAYERWISE_USE_FUSED_ALLREDUCE_RMS,
            ) = previous_flags
        operations = tuple(sorted((str(name), float(value)) for name, value in latency.items()))
        communication_operations = tuple(
            name for name, value in operations if value != 0.0 and aic_component_class(name) == "communication"
        )
        if len(operation_lookups) != len(communication_operations):
            raise AicSurfaceUnavailableError(
                "lower-level operation lookup count does not match positive communication inventory"
            )
        operation_lookups = tuple(
            replace(lookup, consumer_operations=(consumer,))
            for lookup, consumer in zip(operation_lookups, communication_operations, strict=True)
        )
        source_inventory = tuple((name, str(sources.get(name, ""))) for name, _ in operations)
        return RawAicStep(
            operations=operations,
            sources=source_inventory,
            operation_lookups=operation_lookups,
        )


def _repository_file_hashes(config: RepositoryAicConfig) -> dict[str, str | None]:
    systems_root = Path(config.repo_root) / "src/aiconfigurator/systems"
    candidates = {
        "allreduce_rms": systems_root
        / "data"
        / config.system
        / _BACKEND
        / config.comm_version
        / "allreduce_rms_perf.parquet",
        "custom_allreduce": systems_root
        / "data"
        / config.system
        / _BACKEND
        / config.comm_version
        / "custom_allreduce_perf.parquet",
        "operation_manifest": systems_root / "op_kernel_source_manifest.yaml",
        "system": systems_root / f"{config.system}.yaml",
    }
    return {role: _sha256_bytes(path.read_bytes()) if path.is_file() else None for role, path in candidates.items()}


def _verify_repository_checkout(config: RepositoryAicConfig) -> None:
    if not config.verify_repository:
        return
    repo_root = Path(config.repo_root).resolve()

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", "-C", str(repo_root), *args),
            check=False,
            capture_output=True,
            text=True,
        )

    head = _git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != config.repo_commit:
        raise ContractError(
            process_code="repository_commit_mismatch",
            detail=(f"repository HEAD {head.stdout.strip()!r} does not match declared commit {config.repo_commit!r}"),
        )
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = _git(*args)
        if result.returncode != 0:
            raise ContractError(
                process_code="repository_commit_mismatch",
                detail="repository has tracked changes outside the declared commit",
            )
    for relative_path in _SEMANTIC_EXECUTION_PATHS:
        tracked = _git("cat-file", "-e", f"{config.repo_commit}:{relative_path}")
        if tracked.returncode != 0:
            raise ContractError(
                process_code="repository_commit_mismatch",
                detail=(
                    f"execution-relevant path {relative_path!r} is absent from declared commit {config.repo_commit}"
                ),
            )


def _verify_import_root(config: RepositoryAicConfig, api: dict[str, object], gap: object) -> None:
    if not config.verify_repository:
        return
    repo_root = Path(config.repo_root).resolve()
    module_paths = {"aic_fpm_gap": Path(gap.__file__).resolve()}
    for name, value in api.items():
        module = inspect.getmodule(value)
        raw_path = getattr(value, "__file__", None) or getattr(module, "__file__", None)
        if raw_path is not None:
            module_paths[name] = Path(raw_path).resolve()
    for name, path in module_paths.items():
        if not path.is_relative_to(repo_root):
            raise ContractError(
                process_code="repository_commit_mismatch",
                detail=f"imported {name} from {path}, outside declared repository {repo_root}",
            )
        relative_path = path.relative_to(repo_root).as_posix()
        tracked = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "cat-file",
                "-e",
                f"{config.repo_commit}:{relative_path}",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise ContractError(
                process_code="repository_commit_mismatch",
                detail=(
                    f"imported {name} from {relative_path!r}, which is absent from declared commit {config.repo_commit}"
                ),
            )


def build_repository_predictor(config: RepositoryAicConfig) -> AiconfiguratorSemanticBinPredictor:
    """Build the strict/proxy surface and its exact in-tree AIC runner."""

    _verify_repository_checkout(config)
    surface = build_repository_surface(config)
    from collector.layerwise.diagnostics import aic_fpm_gap as gap

    api = gap._import_repo(Path(config.repo_root))
    _verify_import_root(config, api, gap)
    database_file_hashes = _repository_file_hashes(config)
    previous_model_name = gap.MODEL_NAME
    gap.MODEL_NAME = config.model
    try:
        backend = api["VLLMBackend"]()
        model, database, error = gap.build_model_and_db(
            "layerwise",
            True,
            None,
            config.framework_version,
            config.tp_size,
            system=config.system,
            backend=_BACKEND,
            comm_version=config.comm_version,
            systems_root=str(Path(config.repo_root) / "src/aiconfigurator/systems"),
            layerwise_csv=str(config.layerwise_csv),
            api=api,
        )
    finally:
        gap.MODEL_NAME = previous_model_name
    if error or model is None or database is None:
        raise RuntimeError(f"failed to construct AIC model/database: {error}")
    runtime_config = api["RuntimeConfig"](
        vllm_max_num_batched_tokens=config.runtime_max_num_batched_tokens,
        vllm_max_num_seqs=config.runtime_max_num_seqs,
    )
    runner = RepositoryAicStepRunner(
        config=config,
        backend=backend,
        model=model,
        database=_ExactSchedulerDatabase(database, config),
        runtime_config=runtime_config,
        vllm_backend_module=api["vllm_backend"],
        use_fused_allreduce_rms=database_file_hashes["allreduce_rms"] is not None,
    )
    provenance = json.loads(surface.configuration_provenance)
    provenance["database_file_hashes"] = database_file_hashes
    provenance["layerwise_use_fused_allreduce_rms"] = database_file_hashes["allreduce_rms"] is not None
    provenance["repository_verified"] = config.verify_repository
    return AiconfiguratorSemanticBinPredictor(
        surface_index=surface.index,
        runner=runner,
        configuration_provenance=_canonical_json(provenance),
        expected_configuration_fingerprint=config.configuration_fingerprint,
    )
