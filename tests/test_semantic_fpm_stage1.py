# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from collector.layerwise.diagnostics.semantic_fpm_insights import SemanticShape
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    ContractError,
    FpmSample,
    build_semantic_population,
)
from collector.layerwise.diagnostics.semantic_fpm_stage1 import (
    CohortInput,
    _build_parity_record,
    _layerwise_config_hashes,
    _uniform_effective_config,
    _validate_shipping_population,
)


def _effective_config() -> dict[str, object]:
    return {
        "cache_config.enable_prefix_caching": False,
        "parallel_config.data_parallel_size": 1,
        "parallel_config.pipeline_parallel_size": 1,
        "parallel_config.tensor_parallel_size": 8,
        "scheduler_config.max_num_batched_tokens": 40960,
        "scheduler_config.max_num_seqs": 256,
        "vllm_version": "0.20.1",
    }


def test_parity_record_is_canonical_and_uses_typed_runtime_config():
    record, max_sequences, max_tokens, tp_size, backend_version = _build_parity_record(
        config=_effective_config(),
        model="Qwen/Qwen3-32B",
        model_revision="main",
        system="h100_sxm",
        chunked_prefill=True,
        ep_size=1,
        gemm_quant="bf16",
        attention_quant="bf16",
        kv_cache_quant="bf16",
        moe_quant="bf16",
    )

    assert record.startswith('{"attention_dp_size":1,')
    assert '"chunked_prefill":true' in record
    assert (max_sequences, max_tokens, tp_size, backend_version) == (
        256,
        40960,
        8,
        "0.20.1",
    )


def test_layerwise_hash_selection_requires_exact_runtime_axes(tmp_path):
    layerwise = tmp_path / "layerwise.csv"
    header = (
        "framework,framework_version,system,model,attn_tp,moe_tp,ep,phase,"
        "latency_source,max_num_seqs,max_num_batched_tokens,vllm_config_hash\n"
    )
    layerwise.write_text(
        header + "vLLM,0.20.1,h100_sxm,Qwen/Qwen3-32B,8,1,1,ctx,"
        "schedule_to_update,,40960,ctx-hash\n" + "vLLM,0.20.1,h100_sxm,Qwen/Qwen3-32B,8,1,1,gen,"
        "execute_model_gpu,256,,gen-hash\n"
    )

    assert _layerwise_config_hashes(
        layerwise,
        model="Qwen/Qwen3-32B",
        system="h100_sxm",
        backend_version="0.20.1",
        tp_size=8,
        max_num_seqs=256,
        max_num_batched_tokens=40960,
    ) == ("ctx-hash", "gen-hash")


def test_clean_and_profiled_effective_configs_must_match(tmp_path):
    clean = tmp_path / "clean"
    profiled = tmp_path / "profiled"
    clean.mkdir()
    profiled.mkdir()
    (clean / "effective_vllm_config.json").write_text('{"scheduler_config.max_num_seqs":256}')
    (profiled / "effective_vllm_config.json").write_text('{"scheduler_config.max_num_seqs":128}')
    cohort = CohortInput(16, clean, profiled, tmp_path / "trace.sqlite")

    with pytest.raises(ContractError) as exc_info:
        _uniform_effective_config((cohort,))

    assert exc_info.value.process_code == "configuration_mismatch"


def test_attribute_driver_invokes_stage1_after_per_cohort_gate_and_before_tail_done():
    script = Path(__file__).parents[1] / "collector" / "layerwise" / "reproduce_layerwise_fpm.sh"
    source = script.read_text()
    gate = source.index("python3 -m collector.layerwise.diagnostics.assert_attribution_valid")
    invocation = source.index("python3 -m collector.layerwise.diagnostics.semantic_fpm_stage1")
    tail_done = source.index('mark_done "$semantic_unit"', invocation)

    assert gate < invocation < tail_done
    assert 'LW_LATENCY_SOURCE="${LW_LATENCY_SOURCE:-auto}"' in source
    assert "--gen-max-num-seqs ${FPM_MAX_NUM_SEQS}" in source
    assert '[[ "$ATTRIBUTE_REAL_WORKLOAD" == "0" ]] && measured_segment="sweep"' in source
    assert '--measured-segment "$measured_segment"' in source
    assert 'attribute) log "== STAGE attribute =="; stage_attribute;;' in source
    assert "--skip-repository-verification" not in source
    assert "--artifact-proxy-decode-max-num-seqs" not in source
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_clean_and_profiled_shell_lanes_share_real_or_static_workload_arguments():
    script = Path(__file__).parents[1] / "collector" / "layerwise" / "reproduce_layerwise_fpm.sh"
    command = f"""
source {script!s}
ATTRIBUTE_REAL_WORKLOAD=0
build_fpm_workload_args 32 16
printf 'static:%s\n' "${{FPM_WORKLOAD_ARGS[*]}}"
ATTRIBUTE_REAL_WORKLOAD=1
build_fpm_workload_args 32 16
printf 'real:%s\n' "${{FPM_WORKLOAD_ARGS[*]}}"
"""
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True, text=True)

    assert "static:--no-real-workload --decode-batches" in result.stdout
    assert "real:--real-workload --real-workload-requests 32 --real-workload-concurrency 16" in result.stdout


def test_stage1_rejects_cross_segment_lanes_with_no_shared_measured_bin():
    clean_shape = SemanticShape(0, 1, 0, 0, 100)
    profiled_shape = SemanticShape(0, 1, 0, 0, 200)
    samples = (
        FpmSample(
            lane="clean",
            concurrency=16,
            sample_id="clean-real",
            phase="decode",
            workload_segment="real",
            counter_id=1,
            worker_id="clean",
            dp_rank=0,
            shape=clean_shape,
            wall_ms=1.0,
        ),
        FpmSample(
            lane="profiled",
            concurrency=16,
            sample_id="profiled-sweep",
            phase="decode",
            workload_segment="sweep",
            counter_id=2,
            worker_id="profiled",
            dp_rank=0,
            shape=profiled_shape,
            wall_ms=1.0,
        ),
    )
    population = build_semantic_population(
        samples,
        configuration_fingerprint="config",
        measured_segments=frozenset({"sweep"}),
    )

    with pytest.raises(ContractError) as exc_info:
        _validate_shipping_population(population)

    assert exc_info.value.process_code == "configuration_mismatch"
