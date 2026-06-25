# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task 8: scheduler spec for co-scheduled (mixed) work units.

A mixed work unit must:
  * force ``enable_layer_patch=True`` (strict-subset target_layers => patched depth),
  * auto-derive ``max_num_batched_tokens`` >= prefill_tokens + decode_requests (P+B),
  * set ``max_num_seqs`` >= decode_requests + 1 (B+1).

Stdlib-only: vLLM/torch are never imported; the one engine-config probe
(`_get_vllm_deployment_max_num_batched_tokens`) is mocked.
"""

import argparse
import tempfile
from pathlib import Path
from unittest import mock

from collector.layerwise.vllm import scheduler as sched_mod
from collector.layerwise.vllm.data import DataPoint, RepresentativeLayer, WorkUnit

# Mixed cell under test: P=2048 prefill tokens, B=64 decode requests, K=4096 past.
_P = 2048
_B = 64
_K = 4096


def _mixed_unit(tmp_path):
    dp = DataPoint(
        "mixed", 0, 0, 0,
        prefill_tokens=_P, decode_requests=_B, decode_past_kv=_K,
    )
    representative = RepresentativeLayer(
        layer_index=0,
        layer_type="dense",
        measured_layer_count=2,
        layer_multiplier=32,
        target_layers=(0, 1),
    )
    return WorkUnit(
        work_unit_id="wu_mixed",
        model_dir=str(tmp_path / "model"),
        row_base={
            "framework": "vLLM",
            "framework_version": "test",
            "system": "gpu",
            "model": "Qwen/Qwen3-32B",
            "attn_tp": 1,
            "moe_tp": 1,
            "ep": 1,
            "num_slots": "",
            "gemm_quant": "bf16",
            "moe_quant": "bf16",
            "attn_quant": "bf16",
            "kv_quant": "bf16",
        },
        representative=representative,
        target_layers=[0, 1],
        datapoints=[dp],
        model_layer_count=64,
        max_model_len=8192,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        gpu_memory_utilization=0.9,
    )


def _args(tmp_path):
    return argparse.Namespace(
        work_dir=str(tmp_path / "work"),
        output=str(tmp_path / "out.csv"),
        gpus="0",
        max_workers=None,
        timeout=30,
        nsys_capture="none",
        extra_vllm_arg=[],
        latency_source="span",
        moe_decode_gpu_batch_threshold=8,
        ctx_warmup_runs=0,
        ctx_measured_runs=1,
        ctx_repeat_aggregation="median",
        gen_warmup_runs=0,
        gen_measured_runs=1,
        gen_repeat_aggregation="median",
        live_step_driver=False,
        live_step_gen_min_past_kv=8192,
        live_step_gen_min_batch_size=256,
        live_step_gen_max_workers=0,
        prompt_seed=None,
        rollup=r"layers\.(\d+)\.(self_attn|mlp)",
        rank_reduce="sum",
    )


def _make_spec(tmp_path, deployment_value):
    unit = _mixed_unit(tmp_path)
    scheduler = sched_mod.Scheduler(_args(tmp_path), [unit])
    with mock.patch.object(
        sched_mod,
        "_get_vllm_deployment_max_num_batched_tokens",
        return_value=deployment_value,
    ):
        return scheduler._make_spec(unit, unit.datapoints, attempt_id=0)


def test_mixed_spec_forces_layer_patch():
    with tempfile.TemporaryDirectory() as td:
        spec = _make_spec(Path(td), deployment_value=8192)
        assert spec["enable_layer_patch"] is True


def test_mixed_spec_keeps_max_model_len():
    with tempfile.TemporaryDirectory() as td:
        spec = _make_spec(Path(td), deployment_value=8192)
        assert spec["max_model_len"] == 8192


def test_mixed_spec_max_num_batched_tokens_floors_to_p_plus_b():
    # Deployment value below the co-schedule floor must be raised to >= P+B.
    with tempfile.TemporaryDirectory() as td:
        spec = _make_spec(Path(td), deployment_value=512)
        assert spec["max_num_batched_tokens"] >= _P + _B


def test_mixed_spec_max_num_batched_tokens_keeps_larger_deployment():
    # A deployment value already above the floor is preserved.
    with tempfile.TemporaryDirectory() as td:
        spec = _make_spec(Path(td), deployment_value=16384)
        assert spec["max_num_batched_tokens"] >= _P + _B
        assert spec["max_num_batched_tokens"] == 16384


def test_mixed_spec_max_num_seqs_at_least_b_plus_one():
    with tempfile.TemporaryDirectory() as td:
        spec = _make_spec(Path(td), deployment_value=8192)
        assert spec["max_num_seqs"] >= _B + 1
