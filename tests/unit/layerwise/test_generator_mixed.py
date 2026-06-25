# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-side tests for mixed (overlapping prefill+decode) work units.

These tests must run with the stdlib only: ``datapoint_generator`` is pure
Python (no vLLM/torch import at module load), and a local config dir keeps the
HF download path out of the test entirely.
"""

import argparse
import json
from pathlib import Path

from collector.layerwise.vllm.datapoint_generator import build_work_units


# Dense, 64-layer config shaped like Qwen3-32B for the generator's purposes:
# no expert keys (dense), no ``layer_types`` list (so layer-type expansion is a
# no-op and the target_layer_count truncation branch is taken).
_DENSE_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "hidden_size": 5120,
    "intermediate_size": 27648,
    "max_position_embeddings": 40960,
}


def _write_model_dir(tmp_path: Path) -> str:
    model_dir = tmp_path / "qwen3-32b"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_DENSE_CONFIG))
    return str(model_dir)


def _mixed_args(model_dir: str, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model=model_dir,
        work_dir=str(tmp_path / "profiles"),
        config_cache_dir=str(tmp_path / "config_cache"),
        no_config_cache=False,
        system="H100",
        framework_version="0.20.1",
        tp_sizes="1",
        moe_tp=1,
        effective_ep_size=1,
        num_slots=0,
        moe_noop=False,
        target_layer_count=2,
        target_layers=None,
        target_layer_config_depth=None,
        phases="mixed",
        mixed_specs=[(2048, 64, 4096)],
        ctx_new_tokens="1",
        ctx_past_kv="0",
        ctx_batch_sizes="1",
        no_filter_model_max_len=False,
        gen_batch_sizes="1",
        gen_past_kv="1",
        # Set explicitly so build_work_units never spawns the vLLM deployment
        # helper subprocess for the ctx reference datapoint.
        max_num_batched_tokens=8192,
        max_num_seqs=128,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        gen_driver="prefix_cache",
        live_step_driver=False,
        live_step_gen_min_past_kv=8192,
        live_step_gen_min_batch_size=256,
        latency_source="span",
        gemm_quant="fp8",
        moe_quant="fp8",
        attn_quant="fp8",
        kv_quant="fp8",
        moe_real_router=False,
        physical_tp=False,
        allow_multi_gpu_diagnostic=False,
        physical_tp_real_weights=False,
        default_gen_target_layer_count=False,
    )


def test_build_mixed_work_unit_small_depth(tmp_path):
    model_dir = _write_model_dir(tmp_path)
    units = build_work_units(_mixed_args(model_dir, tmp_path))

    mixed_units = [u for u in units if any(dp.phase == "mixed" for dp in u.datapoints)]
    assert mixed_units, "expected at least one work unit carrying a mixed datapoint"
    u = mixed_units[0]

    # Patched/resident depth is the small N; the full model count stays 64 so
    # target_layers is a strict subset (=> needs_layer_patch True downstream).
    assert u.patched_num_hidden_layers == 2
    assert u.model_layer_count == 64
    assert set(u.target_layers) < set(range(u.model_layer_count))

    dp = next(dp for dp in u.datapoints if dp.phase == "mixed")
    assert (dp.prefill_tokens, dp.decode_requests, dp.decode_past_kv) == (2048, 64, 4096)


def test_mixed_work_unit_carries_three_reference_datapoints(tmp_path):
    model_dir = _write_model_dir(tmp_path)
    units = build_work_units(_mixed_args(model_dir, tmp_path))
    u = next(u for u in units if any(dp.phase == "mixed" for dp in u.datapoints))

    shapes = {dp.shape_key for dp in u.datapoints}
    # mixed cell + the three reference shapes on the SAME work unit
    assert "mixed:P2048:B64:K4096" in shapes
    assert "ctx:bs1:new2048:past0" in shapes  # prefill-only reference
    assert "gen:bs64:new1:past4096" in shapes  # decode-at-K reference
    assert "gen:bs64:new1:past1" in shapes  # decode-at-1 reference


def test_mixed_unit_survives_max_model_len_filter_for_k4096(tmp_path):
    model_dir = _write_model_dir(tmp_path)
    units = build_work_units(_mixed_args(model_dir, tmp_path))
    u = next(u for u in units if any(dp.phase == "mixed" for dp in u.datapoints))
    # P(2048)+K(4096) = 6144 < max_model_len(8192): the mixed datapoint must
    # not be dropped by the model-max-length filter.
    assert any(dp.phase == "mixed" for dp in u.datapoints)


def test_mixed_unit_target_layers_strict_subset_triggers_patch(tmp_path):
    model_dir = _write_model_dir(tmp_path)
    units = build_work_units(_mixed_args(model_dir, tmp_path))
    u = next(u for u in units if any(dp.phase == "mixed" for dp in u.datapoints))
    assert u.uses_full_layer_depth() is False
    assert u.needs_layer_patch(enable_layerwise_nvtx_tracing=False) is True
