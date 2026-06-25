# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests pinning the main CSV schema (Task 3, side-artifact decision).

Per the U6 decision, mixed-cell M/C/D values live in ``mixed_steps.jsonl`` (Task 7),
NOT the main CSV. These tests guarantee the main CSV is untouched for pure phases and
tolerant of a ``phase="mixed"`` value.
"""

from __future__ import annotations

import csv
from pathlib import Path

from collector.layerwise.vllm.results import (
    CSV_COLUMNS,
    _append_success_row,
    _write_csv_header_if_needed,
)


# The exact, frozen 30-column order. Any change here is a backward-compat break.
_EXPECTED_COLUMNS = [
    "framework", "framework_version", "system", "model", "attn_tp", "moe_tp", "ep",
    "num_slots", "gemm_quant", "moe_quant", "attn_quant", "kv_quant", "phase",
    "batch_size", "new_tokens", "past_kv", "layer_type", "layer_index",
    "measured_layer_count", "layer_multiplier", "latency_ms", "rms_latency_ms",
    "rms_kernel_count", "includes_moe", "moe_weight_mode", "latency_source",
    "physical_gpus", "max_num_seqs", "max_num_batched_tokens", "vllm_config_hash",
]


def test_csv_columns_pinned_30_column_order():
    """The leading 30 columns must stay byte-identical (backward compatibility)."""
    assert len(CSV_COLUMNS) == 30
    assert CSV_COLUMNS == _EXPECTED_COLUMNS


def test_pure_ctx_row_byte_identical(tmp_path: Path):
    """A pure ctx row writes exactly the historical 30-column line."""
    path = tmp_path / "out.csv"
    _write_csv_header_if_needed(path)
    row = {
        "framework": "vllm", "framework_version": "0.20.1", "system": "h100_pcie",
        "model": "Qwen/Qwen3-32B", "attn_tp": 1, "moe_tp": 1, "ep": 1,
        "num_slots": 1, "gemm_quant": "fp8", "moe_quant": "fp8", "attn_quant": "fp8",
        "kv_quant": "fp8", "phase": "ctx", "batch_size": 1, "new_tokens": 128,
        "past_kv": 0, "layer_type": "attention", "layer_index": 0,
        "measured_layer_count": 64, "layer_multiplier": 1.0, "latency_ms": 1.23,
        "rms_latency_ms": 0.01, "rms_kernel_count": 5, "includes_moe": False,
        "moe_weight_mode": "", "latency_source": "measured", "physical_gpus": 1,
        "max_num_seqs": 256, "max_num_batched_tokens": 8192, "vllm_config_hash": "abc",
    }
    _append_success_row(path, row)

    text = path.read_text()
    header, data_line = text.splitlines()[0], text.splitlines()[1]
    assert header == ",".join(_EXPECTED_COLUMNS)
    # 30 columns -> 29 commas in the data line
    assert data_line.count(",") == 29


def test_mixed_phase_row_produces_valid_30_column_line(tmp_path: Path):
    """A row dict with phase='mixed' + standard keys writes a valid 30-column line.

    Extra mixed-only keys (prefill_tokens/decode_requests/decode_past_kv) must be
    silently dropped by the writer; the CSV stays at 30 columns.
    """
    path = tmp_path / "out.csv"
    _write_csv_header_if_needed(path)
    row = {
        "framework": "vllm", "framework_version": "0.20.1", "system": "h100_pcie",
        "model": "Qwen/Qwen3-32B", "attn_tp": 1, "moe_tp": 1, "ep": 1,
        "num_slots": 1, "gemm_quant": "fp8", "moe_quant": "fp8", "attn_quant": "fp8",
        "kv_quant": "fp8", "phase": "mixed", "batch_size": 0, "new_tokens": 0,
        "past_kv": 0, "layer_type": "attention", "layer_index": 0,
        "measured_layer_count": 2, "layer_multiplier": 1.0, "latency_ms": 4.56,
        "rms_latency_ms": 0.02, "rms_kernel_count": 7, "includes_moe": False,
        "moe_weight_mode": "", "latency_source": "measured", "physical_gpus": 1,
        "max_num_seqs": 65, "max_num_batched_tokens": 8192, "vllm_config_hash": "def",
        # mixed-only side-artifact keys that must NOT leak into the CSV:
        "prefill_tokens": 2048, "decode_requests": 64, "decode_past_kv": 4096,
    }
    _append_success_row(path, row)

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    record = rows[0]
    assert list(record.keys()) == _EXPECTED_COLUMNS
    assert record["phase"] == "mixed"
    # mixed-only keys are absent from the written CSV
    assert "prefill_tokens" not in record
    assert "decode_requests" not in record
    assert "decode_past_kv" not in record
