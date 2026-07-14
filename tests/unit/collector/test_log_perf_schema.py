# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

import pytest

from collector.helper import convert_perf_csv_to_parquet, log_perf

_BASE_METADATA = {
    "framework": "SGLang",
    "version": "0.5.10rc0",
    "device": "NVIDIA GB200",
    "op_name": "gemm",
    "kernel_source": "deepgemm",
}


def _log_canonical_row(path, row):
    log_perf(
        item_list=[row],
        framework=_BASE_METADATA["framework"],
        version=_BASE_METADATA["version"],
        device_name=_BASE_METADATA["device"],
        op_name=_BASE_METADATA["op_name"],
        kernel_source=_BASE_METADATA["kernel_source"],
        perf_filename=str(path),
    )


def test_log_perf_writes_canonical_metadata_columns_exactly_once(tmp_path) -> None:
    output = tmp_path / "gemm_perf.txt"
    _log_canonical_row(
        output,
        {
            **_BASE_METADATA,
            "gemm_dtype": "fp8_block",
            "m": 258,
            "n": 4096,
            "k": 4096,
            "latency": 0.021,
        },
    )

    with output.open(newline="") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    assert len(header) == len(set(header))
    assert header[:5] == list(_BASE_METADATA)
    assert rows[1][header.index("framework")] == "SGLang"
    assert rows[1][header.index("m")] == "258"


def test_log_perf_rejects_conflicting_canonical_metadata_before_writing(tmp_path) -> None:
    output = tmp_path / "gemm_perf.txt"

    with pytest.raises(ValueError, match=r"conflicting canonical perf metadata.*framework"):
        _log_canonical_row(
            output,
            {
                **_BASE_METADATA,
                "framework": "TensorRT-LLM",
                "gemm_dtype": "fp8_block",
                "m": 258,
                "n": 4096,
                "k": 4096,
                "latency": 0.021,
            },
        )

    assert not output.exists()
    assert not output.with_name(f"{output.name}.lock").exists()


def test_canonical_metadata_survives_csv_parquet_database_round_trip(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    from aiconfigurator.sdk.operations.base import _read_perf_rows

    output = tmp_path / "gemm_perf.txt"
    _log_canonical_row(
        output,
        {
            **_BASE_METADATA,
            "gemm_dtype": "fp8_block",
            "m": 258,
            "n": 4096,
            "k": 4096,
            "latency": 0.021,
        },
    )

    parquet = convert_perf_csv_to_parquet(output)
    rows = _read_perf_rows(str(parquet))
    assert len(rows) == 1
    assert {key: rows[0][key] for key in _BASE_METADATA} == _BASE_METADATA
    assert rows[0]["m"] == 258
