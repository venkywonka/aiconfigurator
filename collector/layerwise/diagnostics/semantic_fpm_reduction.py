#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV normalization and cross-run population construction for semantic FPM insights."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from collector.layerwise.diagnostics.semantic_fpm_insights import (
    SCHEMA_VERSION,
    SemanticKey,
    SemanticShape,
    ShapeValidationError,
    canonical_semantic_key,
    stable_bin_id,
)

FPM_REQUIRED_COLUMNS = (
    "phase",
    "workload_segment",
    "counter_id",
    "worker_id",
    "dp_rank",
    "ctx_tokens",
    "ctx_requests",
    "ctx_kv_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "latency_ms",
)


class ContractError(RuntimeError):
    """A source row or normalized population violates the v1 contract."""

    def __init__(self, *, process_code: str, detail: str):
        super().__init__(detail)
        self.process_code = process_code


@dataclass(frozen=True)
class FpmSample:
    """One normalized source observation, retained regardless of analytic eligibility."""

    lane: str
    concurrency: int
    sample_id: str
    phase: str
    workload_segment: str
    counter_id: int
    worker_id: str
    dp_rank: int
    shape: SemanticShape
    wall_ms: float
    measured: bool | None = None
    analytic_eligible: bool | None = None
    eligibility_reason: str | None = None

    @property
    def semantic_key(self) -> SemanticKey:
        return self.shape.semantic_key

    @property
    def serialized_semantic_key(self) -> str:
        return canonical_semantic_key(self.semantic_key)

    @property
    def raw_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.shape.ctx_requests,
            self.shape.decode_requests,
            self.shape.ctx_new_tokens,
            self.shape.ctx_kv_tokens,
            self.shape.decode_kv_tokens,
        )


@dataclass(frozen=True)
class SemanticPopulationBin:
    """Lane-preserving observations for one normalized semantic bin."""

    bin_id: str
    configuration_fingerprint: str
    concurrency: int
    phase: str
    semantic_key: SemanticKey
    clean_samples: tuple[FpmSample, ...]
    profiled_samples: tuple[FpmSample, ...]

    @property
    def serialized_semantic_key(self) -> str:
        return canonical_semantic_key(self.semantic_key)

    @property
    def n_clean(self) -> int:
        return len(self.clean_samples)

    @property
    def n_profiled(self) -> int:
        return len(self.profiled_samples)

    @property
    def all_clean(self) -> bool:
        return bool(self.clean_samples)

    @property
    def nsight_shared(self) -> bool:
        return bool(self.clean_samples and self.profiled_samples)

    @property
    def clean_only(self) -> bool:
        return bool(self.clean_samples and not self.profiled_samples)

    @property
    def profiled_only(self) -> bool:
        return bool(self.profiled_samples and not self.clean_samples)

    @property
    def clean_raw_shape_cardinality(self) -> int:
        return len({sample.raw_shape for sample in self.clean_samples})

    @property
    def profiled_raw_shape_cardinality(self) -> int:
        return len({sample.raw_shape for sample in self.profiled_samples})


@dataclass(frozen=True)
class SemanticPopulation:
    """Every source sample plus the measured, non-idle union of lane bin sets."""

    schema_version: str
    configuration_fingerprint: str
    measured_segments: frozenset[str]
    samples: tuple[FpmSample, ...]
    bins: tuple[SemanticPopulationBin, ...]
    excluded_reason_counts: dict[str, int]


def _parse_nonnegative_int(row: dict[str, str], column: str, row_number: int) -> int:
    try:
        value = int(row[column])
    except (TypeError, ValueError) as exc:
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"row {row_number}: {column} must be an integer, got {row.get(column)!r}",
        ) from exc
    if value < 0:
        raise ContractError(
            process_code="invalid_shape",
            detail=f"row {row_number}: {column} must be non-negative, got {value}",
        )
    return value


def _sample_id(
    *, lane: str, concurrency: int, workload_segment: str, worker_id: str, dp_rank: int, counter_id: int
) -> str:
    identity = {
        "concurrency": concurrency,
        "counter_id": counter_id,
        "dp_rank": dp_rank,
        "lane": lane,
        "schema_version": SCHEMA_VERSION,
        "worker_id": worker_id,
        "workload_segment": workload_segment,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_fpm_phase_csv(path: str | Path, *, lane: str, concurrency: int) -> list[FpmSample]:
    """Normalize every row of one clean or profiled ``fpm_metrics_phase.csv``."""

    if lane not in {"clean", "profiled"}:
        raise ValueError(f"lane must be 'clean' or 'profiled', got {lane!r}")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError(f"concurrency must be a positive integer, got {concurrency!r}")
    source = Path(path)
    if not source.is_file():
        raise ContractError(process_code="schema_missing", detail=f"FPM phase CSV does not exist: {source}")

    samples: list[FpmSample] = []
    seen_ids: set[str] = set()
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(set(FPM_REQUIRED_COLUMNS) - fieldnames)
        if missing:
            raise ContractError(
                process_code="schema_missing",
                detail=f"FPM phase CSV is missing required columns: {', '.join(missing)}",
            )
        for row_number, row in enumerate(reader, start=2):
            counter_id = _parse_nonnegative_int(row, "counter_id", row_number)
            dp_rank = _parse_nonnegative_int(row, "dp_rank", row_number)
            shape = SemanticShape(
                ctx_requests=_parse_nonnegative_int(row, "ctx_requests", row_number),
                decode_requests=_parse_nonnegative_int(row, "decode_requests", row_number),
                ctx_new_tokens=_parse_nonnegative_int(row, "ctx_tokens", row_number),
                ctx_kv_tokens=_parse_nonnegative_int(row, "ctx_kv_tokens", row_number),
                decode_kv_tokens=_parse_nonnegative_int(row, "decode_kv_tokens", row_number),
            )
            source_phase = row["phase"]
            try:
                phase = shape.validate(source_phase=source_phase)
            except ShapeValidationError as exc:
                raise ContractError(process_code=exc.process_code, detail=f"row {row_number}: {exc}") from exc

            try:
                wall_ms = float(row["latency_ms"])
            except (TypeError, ValueError) as exc:
                raise ContractError(
                    process_code="invalid_wall_time",
                    detail=f"row {row_number}: invalid latency_ms {row.get('latency_ms')!r}",
                ) from exc
            if not math.isfinite(wall_ms) or wall_ms <= 0:
                raise ContractError(
                    process_code="invalid_wall_time",
                    detail=f"row {row_number}: latency_ms must be finite and positive, got {wall_ms!r}",
                )

            worker_id = row["worker_id"]
            workload_segment = row["workload_segment"]
            if not worker_id or not workload_segment:
                raise ContractError(
                    process_code="schema_incompatible",
                    detail=f"row {row_number}: worker_id and workload_segment must be non-empty",
                )
            sample_id = _sample_id(
                lane=lane,
                concurrency=concurrency,
                workload_segment=workload_segment,
                worker_id=worker_id,
                dp_rank=dp_rank,
                counter_id=counter_id,
            )
            if sample_id in seen_ids:
                raise ContractError(
                    process_code="duplicate_identity",
                    detail=f"row {row_number}: duplicate source identity for sample {sample_id}",
                )
            seen_ids.add(sample_id)
            samples.append(
                FpmSample(
                    lane=lane,
                    concurrency=concurrency,
                    sample_id=sample_id,
                    phase=phase,
                    workload_segment=workload_segment,
                    counter_id=counter_id,
                    worker_id=worker_id,
                    dp_rank=dp_rank,
                    shape=shape,
                    wall_ms=wall_ms,
                )
            )
    return samples


def build_semantic_population(
    samples: Iterable[FpmSample],
    *,
    configuration_fingerprint: str,
    measured_segments: frozenset[str],
) -> SemanticPopulation:
    """Build exact cohort/phase bin sets without cross-run row pairing."""

    if not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be non-empty")
    if not measured_segments or any(not segment for segment in measured_segments):
        raise ValueError("measured_segments must be an explicit non-empty set")
    source_samples = tuple(samples)
    if len({sample.sample_id for sample in source_samples}) != len(source_samples):
        raise ContractError(process_code="duplicate_identity", detail="combined sample IDs are not unique")

    excluded: Counter[str] = Counter()
    classified_samples: list[FpmSample] = []
    grouped: dict[tuple[int, str, SemanticKey], dict[str, list[FpmSample]]] = defaultdict(
        lambda: {"clean": [], "profiled": []}
    )
    for sample in source_samples:
        if sample.lane not in {"clean", "profiled"}:
            raise ContractError(process_code="schema_incompatible", detail=f"unknown sample lane {sample.lane!r}")
        if sample.workload_segment not in measured_segments:
            excluded["non_measured_segment"] += 1
            classified_samples.append(
                replace(
                    sample,
                    measured=False,
                    analytic_eligible=False,
                    eligibility_reason="non_measured_segment",
                )
            )
            continue
        if sample.phase == "idle":
            excluded["idle_step"] += 1
            classified_samples.append(
                replace(sample, measured=True, analytic_eligible=False, eligibility_reason="idle_step")
            )
            continue
        classified = replace(sample, measured=True, analytic_eligible=True, eligibility_reason=None)
        classified_samples.append(classified)
        grouped[(classified.concurrency, classified.phase, classified.semantic_key)][classified.lane].append(classified)

    def source_order(sample: FpmSample) -> tuple[int, str, str, str, int, int, str]:
        return (
            sample.concurrency,
            sample.lane,
            sample.workload_segment,
            sample.worker_id,
            sample.dp_rank,
            sample.counter_id,
            sample.sample_id,
        )

    all_samples = tuple(sorted(classified_samples, key=source_order))

    phase_order = {"context": 0, "decode": 1, "mixed": 2}
    bins = []
    for (concurrency, phase, semantic_key), lanes in sorted(
        grouped.items(), key=lambda item: (item[0][0], phase_order[item[0][1]], item[0][2])
    ):
        bins.append(
            SemanticPopulationBin(
                bin_id=stable_bin_id(
                    configuration_fingerprint=configuration_fingerprint,
                    concurrency=concurrency,
                    phase=phase,
                    semantic_key=semantic_key,
                ),
                configuration_fingerprint=configuration_fingerprint,
                concurrency=concurrency,
                phase=phase,
                semantic_key=semantic_key,
                clean_samples=tuple(sorted(lanes["clean"], key=source_order)),
                profiled_samples=tuple(sorted(lanes["profiled"], key=source_order)),
            )
        )
    return SemanticPopulation(
        schema_version=SCHEMA_VERSION,
        configuration_fingerprint=configuration_fingerprint,
        measured_segments=measured_segments,
        samples=all_samples,
        bins=tuple(bins),
        excluded_reason_counts=dict(sorted(excluded.items())),
    )


def support_class(count: int) -> str:
    """Return the v1 descriptive support class for a non-empty sample set."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"support count must be a positive integer, got {count!r}")
    if count == 1:
        return "singleton"
    if count <= 4:
        return "sparse"
    return "repeated"


def _linear_quantile(sorted_values: list[float], probability: float) -> float:
    rank = probability * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def descriptive_stats(values: Iterable[float]) -> dict[str, int | float | str | None]:
    """Return the deterministic v1 point and descriptive support band."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("descriptive statistics require at least one value")
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("descriptive statistics require finite values")
    count = len(ordered)
    repeated = count >= 5
    return {
        "count": count,
        "support_class": support_class(count),
        "minimum": ordered[0],
        "q1": _linear_quantile(ordered, 0.25) if repeated else None,
        "median": _linear_quantile(ordered, 0.5),
        "q3": _linear_quantile(ordered, 0.75) if repeated else None,
        "maximum": ordered[-1],
    }
