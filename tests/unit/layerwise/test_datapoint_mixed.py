# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collector.layerwise.vllm.data import (
    DataPoint,
    RepresentativeLayer,
    WorkUnit,
)


def test_pure_ctx_still_constructs_from_old_raw():
    raw = {"phase": "ctx", "batch_size": 1, "new_tokens": 128, "past_kv": 0}
    dp = DataPoint(**raw)
    assert dp.shape_key == "ctx:bs1:new128:past0"


def test_mixed_datapoint_shape_key():
    dp = DataPoint("mixed", 0, 0, 0, prefill_tokens=2048, decode_requests=64, decode_past_kv=4096)
    assert dp.shape_key == "mixed:P2048:B64:K4096"
    assert dp.datapoint_id("wu1") == "wu1:mixed:P2048:B64:K4096"


def _make_mixed_work_unit():
    """Build a WorkUnit carrying a mixed DataPoint with a strict-subset
    ``target_layers`` over a model that has more layers than are kept."""

    dp = DataPoint(
        "mixed", 0, 0, 0,
        prefill_tokens=2048, decode_requests=64, decode_past_kv=4096,
    )
    representative = RepresentativeLayer(
        layer_index=0,
        layer_type="dense",
        measured_layer_count=2,
        layer_multiplier=32,
        target_layers=(0, 1),
    )
    return WorkUnit(
        work_unit_id="wu1",
        model_dir="/tmp/model",
        row_base={"model": "Qwen/Qwen3-32B"},
        representative=representative,
        target_layers=[0, 1],
        datapoints=[dp],
        model_layer_count=64,
    )


def test_mixed_work_unit_needs_layer_patch():
    unit = _make_mixed_work_unit()
    # strict subset of range(model_layer_count) => not full depth => patch needed
    assert set(unit.target_layers) < set(range(unit.model_layer_count))
    assert unit.uses_full_layer_depth() is False
    assert unit.needs_layer_patch(enable_layerwise_nvtx_tracing=False) is True


def test_mixed_manifest_rows_emit_pbk_fields():
    unit = _make_mixed_work_unit()
    rows = unit.manifest_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["prefill_tokens"] == 2048
    assert row["decode_requests"] == 64
    assert row["decode_past_kv"] == 4096
    # pure-phase columns remain present and defaulted for the mixed datapoint
    assert row["phase"] == "mixed"
    assert row["batch_size"] == 0
    assert row["new_tokens"] == 0
    assert row["past_kv"] == 0
