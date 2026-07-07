#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mandatory attribute-stage tail for the semantic FPM source contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from collector.layerwise.diagnostics.semantic_fpm_aic import RepositoryAicConfig
from collector.layerwise.diagnostics.semantic_fpm_contract import (
    ContractBundle,
    ContractSourceMetadata,
    reduce_and_write_semantic_contract,
)
from collector.layerwise.diagnostics.semantic_fpm_nsys import (
    KERNEL_CLASSIFIER_VERSION,
    build_cohort_nsight_contract,
)
from collector.layerwise.diagnostics.semantic_fpm_predictor import (
    ARTIFACT_PROXY_LOOKUP_POLICY_VERSION,
    LOOKUP_POLICY_VERSION,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    ContractError,
    build_semantic_population,
    load_fpm_phase_csv,
)


@dataclass(frozen=True)
class CohortInput:
    """One clean/profiled/Nsight concurrency cohort."""

    concurrency: int
    clean_run: Path
    profiled_run: Path
    sqlite_path: Path

    @classmethod
    def parse(cls, value: str) -> CohortInput:
        parts = value.split(":", maxsplit=3)
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("cohort must be CONCURRENCY:CLEAN_RUN:PROFILED_RUN:NSYS_SQLITE")
        try:
            concurrency = int(parts[0])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("cohort concurrency must be an integer") from exc
        if concurrency <= 0:
            raise argparse.ArgumentTypeError("cohort concurrency must be positive")
        return cls(
            concurrency=concurrency,
            clean_run=Path(parts[1]),
            profiled_run=Path(parts[2]),
            sqlite_path=Path(parts[3]),
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_effective_config(run_dir: Path) -> dict[str, object]:
    path = run_dir / "effective_vllm_config.json"
    if not path.is_file():
        raise ContractError(
            process_code="schema_missing",
            detail=f"effective vLLM config is missing: {path}",
        )
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"cannot parse effective vLLM config {path}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"effective vLLM config must be an object: {path}",
        )
    return value


def _uniform_effective_config(cohorts: tuple[CohortInput, ...]) -> dict[str, object]:
    records = []
    for cohort in cohorts:
        records.append(_load_effective_config(cohort.clean_run))
        records.append(_load_effective_config(cohort.profiled_run))
    canonical = {_canonical_json(record) for record in records}
    if len(canonical) != 1:
        raise ContractError(
            process_code="configuration_mismatch",
            detail="clean/profiled effective vLLM configurations are not identical across cohorts",
        )
    return records[0]


def _required_config_value(config: dict[str, object], field: str, expected_type: type):
    value = config.get(field)
    valid = (
        isinstance(value, bool)
        if expected_type is bool
        else (not isinstance(value, bool) and isinstance(value, expected_type))
    )
    if not valid:
        raise ContractError(
            process_code="configuration_mismatch",
            detail=f"effective vLLM config {field!r} has invalid value {value!r}",
        )
    return value


def _layerwise_config_hashes(
    layerwise_csv: Path,
    *,
    model: str,
    system: str,
    backend_version: str,
    tp_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> tuple[str, str]:
    if not layerwise_csv.is_file():
        raise ContractError(
            process_code="schema_missing",
            detail=f"layerwise CSV is missing: {layerwise_csv}",
        )
    context_hashes = set()
    decode_hashes = set()
    with layerwise_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("framework") != "vLLM"
                or row.get("framework_version") != backend_version
                or row.get("system") != system
                or row.get("model") != model
                or row.get("attn_tp") != str(tp_size)
                or row.get("moe_tp") != "1"
                or row.get("ep") != "1"
            ):
                continue
            phase = row.get("phase")
            if (
                phase == "ctx"
                and row.get("max_num_seqs", "") == ""
                and row.get("max_num_batched_tokens") == str(max_num_batched_tokens)
                and row.get("latency_source") == "schedule_to_update"
            ):
                context_hashes.add(row.get("vllm_config_hash", ""))
            elif (
                phase == "gen"
                and row.get("max_num_seqs") == str(max_num_seqs)
                and row.get("latency_source") == "execute_model_gpu"
            ):
                decode_hashes.add(row.get("vllm_config_hash", ""))
    if len(context_hashes) != 1 or len(decode_hashes) != 1 or "" in context_hashes | decode_hashes:
        raise ContractError(
            process_code="configuration_mismatch",
            detail=(
                "layerwise context/decode config hashes are not unique for the exact runtime "
                f"surface: context={sorted(context_hashes)!r}, decode={sorted(decode_hashes)!r}"
            ),
        )
    return next(iter(context_hashes)), next(iter(decode_hashes))


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(
            process_code="repository_commit_mismatch",
            detail=f"cannot resolve repository HEAD for {repo_root}: {exc}",
        ) from exc


def _validate_shipping_population(population) -> None:
    measured_clean = [sample for sample in population.samples if sample.lane == "clean" and sample.measured]
    measured_profiled = [sample for sample in population.samples if sample.lane == "profiled" and sample.measured]
    shared_bins = [bin_ for bin_ in population.bins if bin_.nsight_shared]
    if not measured_clean or not measured_profiled or not shared_bins:
        raise ContractError(
            process_code="configuration_mismatch",
            detail=(
                "shipping Stage 1 requires nonempty measured clean/profiled lanes and "
                "at least one shared semantic bin; verify both lanes used the same workload segment"
            ),
        )


def _build_parity_record(
    *,
    config: dict[str, object],
    model: str,
    model_revision: str,
    system: str,
    chunked_prefill: bool,
    ep_size: int,
    gemm_quant: str,
    attention_quant: str,
    kv_cache_quant: str,
    moe_quant: str,
) -> tuple[str, int, int, int, str]:
    backend_version = str(_required_config_value(config, "vllm_version", str))
    tp_size = int(_required_config_value(config, "parallel_config.tensor_parallel_size", int))
    dp_size = int(_required_config_value(config, "parallel_config.data_parallel_size", int))
    pp_size = int(_required_config_value(config, "parallel_config.pipeline_parallel_size", int))
    max_num_seqs = int(_required_config_value(config, "scheduler_config.max_num_seqs", int))
    max_num_batched_tokens = int(_required_config_value(config, "scheduler_config.max_num_batched_tokens", int))
    prefix_caching = _required_config_value(config, "cache_config.enable_prefix_caching", bool)
    parity = {
        "attention_dp_size": 1,
        "attention_quant": attention_quant,
        "backend": "vllm",
        "backend_version": backend_version,
        "chunked_prefill": chunked_prefill,
        "dp_size": dp_size,
        "ep_size": ep_size,
        "gemm_quant": gemm_quant,
        "gpu_count": tp_size,
        "kv_cache_dtype": kv_cache_quant,
        "kv_cache_quant": kv_cache_quant,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "model": model,
        "model_revision": model_revision,
        "moe_quant": moe_quant,
        "numerical_dtype": gemm_quant,
        "pp_size": pp_size,
        "prefix_caching": prefix_caching,
        "runtime_flags": config,
        "schema_version": "aic-runtime-parity/v1",
        "system": system,
        "tp_size": tp_size,
    }
    return (
        _canonical_json(parity),
        max_num_seqs,
        max_num_batched_tokens,
        tp_size,
        backend_version,
    )


def run_stage1(
    *,
    cohorts: tuple[CohortInput, ...],
    output_dir: Path,
    aic_repo: Path,
    auto_collector_commit: str,
    layerwise_csv: Path,
    model: str,
    model_revision: str,
    system: str,
    comm_version: str,
    chunked_prefill: bool,
    ep_size: int,
    gemm_quant: str,
    attention_quant: str,
    kv_cache_quant: str,
    moe_quant: str,
    job_id: int,
    pipeline_id: int,
    measured_segments: frozenset[str],
    artifact_proxy_decode_max_num_seqs: int | None = None,
    verify_repository: bool = True,
) -> ContractBundle:
    """Run the complete trace-adjacent reducer over all requested cohorts."""

    ordered_cohorts = tuple(sorted(cohorts, key=lambda cohort: cohort.concurrency))
    concurrencies = tuple(cohort.concurrency for cohort in ordered_cohorts)
    if not ordered_cohorts or len(concurrencies) != len(set(concurrencies)):
        raise ContractError(
            process_code="concurrency_mismatch",
            detail="Stage 1 requires a non-empty unique concurrency cohort set",
        )
    effective_config = _uniform_effective_config(ordered_cohorts)
    parity_record, max_num_seqs, max_num_batched_tokens, tp_size, backend_version = _build_parity_record(
        config=effective_config,
        model=model,
        model_revision=model_revision,
        system=system,
        chunked_prefill=chunked_prefill,
        ep_size=ep_size,
        gemm_quant=gemm_quant,
        attention_quant=attention_quant,
        kv_cache_quant=kv_cache_quant,
        moe_quant=moe_quant,
    )
    context_hash, decode_hash = _layerwise_config_hashes(
        layerwise_csv,
        model=model,
        system=system,
        backend_version=backend_version,
        tp_size=tp_size,
        max_num_seqs=artifact_proxy_decode_max_num_seqs or max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    aic_commit = _git_head(aic_repo)
    configuration_fingerprint = _sha256(parity_record)
    repository_config = RepositoryAicConfig(
        repo_root=aic_repo,
        repo_commit=aic_commit,
        configuration_fingerprint=configuration_fingerprint,
        parity_record=parity_record,
        layerwise_csv=layerwise_csv,
        comm_version=comm_version,
        context_vllm_config_hash=context_hash,
        decode_vllm_config_hash=decode_hash,
        artifact_proxy_decode_max_num_seqs=artifact_proxy_decode_max_num_seqs,
        verify_repository=verify_repository,
    )

    source_samples = []
    for cohort in ordered_cohorts:
        source_samples.extend(
            load_fpm_phase_csv(
                cohort.clean_run / "fpm_metrics_phase.csv",
                lane="clean",
                concurrency=cohort.concurrency,
            )
        )
        source_samples.extend(
            load_fpm_phase_csv(
                cohort.profiled_run / "fpm_metrics_phase.csv",
                lane="profiled",
                concurrency=cohort.concurrency,
            )
        )
    population = build_semantic_population(
        source_samples,
        configuration_fingerprint=configuration_fingerprint,
        measured_segments=measured_segments,
    )
    _validate_shipping_population(population)
    alignments = []
    compositions = {}
    for cohort in ordered_cohorts:
        profiled = tuple(
            sample
            for sample in population.samples
            if sample.lane == "profiled" and sample.measured and sample.concurrency == cohort.concurrency
        )
        alignment, cohort_compositions = build_cohort_nsight_contract(
            concurrency=cohort.concurrency,
            profiled_samples=profiled,
            sqlite_path=cohort.sqlite_path,
            expected_rank_count=tp_size,
        )
        alignments.append(alignment)
        overlap = set(compositions) & set(cohort_compositions)
        if overlap:
            raise ContractError(
                process_code="duplicate_identity",
                detail=f"profiled sample identities repeat across cohorts: {sorted(overlap)!r}",
            )
        compositions.update(cohort_compositions)

    source_metadata = ContractSourceMetadata(
        job_id=job_id,
        pipeline_id=pipeline_id,
        model=model,
        system=system,
        backend="vllm",
        backend_version=backend_version,
        nsight_kernel_classifier_version=KERNEL_CLASSIFIER_VERSION,
        alignments=tuple(alignments),
    )
    return reduce_and_write_semantic_contract(
        population=population,
        repository_config=repository_config,
        compositions=compositions,
        output_dir=output_dir,
        aiconfigurator_commit=aic_commit,
        auto_collector_commit=auto_collector_commit,
        source_metadata=source_metadata,
        required_lookup_policy=(
            ARTIFACT_PROXY_LOOKUP_POLICY_VERSION
            if artifact_proxy_decode_max_num_seqs is not None
            else LOOKUP_POLICY_VERSION
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, type=CohortInput.parse)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--aic-repo", required=True, type=Path)
    parser.add_argument("--auto-collector-commit", required=True)
    parser.add_argument("--layerwise-csv", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--comm-version", required=True)
    parser.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--gemm-quant", default="bf16")
    parser.add_argument("--attention-quant", default="bf16")
    parser.add_argument("--kv-cache-quant", default="bf16")
    parser.add_argument("--moe-quant", default="bf16")
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--pipeline-id", type=int, required=True)
    parser.add_argument("--measured-segment", action="append")
    parser.add_argument("--artifact-proxy-decode-max-num-seqs", type=int)
    parser.add_argument("--skip-repository-verification", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = run_stage1(
        cohorts=tuple(args.cohort),
        output_dir=args.output_dir,
        aic_repo=args.aic_repo,
        auto_collector_commit=args.auto_collector_commit,
        layerwise_csv=args.layerwise_csv,
        model=args.model,
        model_revision=args.model_revision,
        system=args.system,
        comm_version=args.comm_version,
        chunked_prefill=args.chunked_prefill,
        ep_size=args.ep_size,
        gemm_quant=args.gemm_quant,
        attention_quant=args.attention_quant,
        kv_cache_quant=args.kv_cache_quant,
        moe_quant=args.moe_quant,
        job_id=args.job_id,
        pipeline_id=args.pipeline_id,
        measured_segments=frozenset(args.measured_segment or ("real",)),
        artifact_proxy_decode_max_num_seqs=args.artifact_proxy_decode_max_num_seqs,
        verify_repository=not args.skip_repository_verification,
    )
    print(
        _canonical_json(
            {
                "bins_csv": str(bundle.bins_csv),
                "manifest_json": str(bundle.manifest_json),
                "samples_csv": str(bundle.samples_csv),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
