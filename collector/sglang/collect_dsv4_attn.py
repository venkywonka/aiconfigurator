# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline DSv4 attention sweep backed by the installable exact runner."""

from __future__ import annotations

import argparse
import os
import sys

from aiconfigurator.collector.sglang.dsv4_attn import (
    close_dsv4_attn_runtime,
    open_dsv4_attn_runtime,
    run_dsv4_attn_case,
)
from aiconfigurator.sdk.resolution.types import MeasurementProtocol
from collector.case_generator import (
    _DSV4_MODULE_BATCH_SIZES as _BATCH_SIZES,
)
from collector.case_generator import (
    _DSV4_MODULE_PAST_KV_LIST as _PREFIX_LENGTHS,
)
from collector.case_generator import (
    _DSV4_MODULE_SEQ_LENGTHS as _SEQ_LENGTHS,
)
from collector.case_generator import (
    _DSV4_MODULE_TP_SIZES as _TP_SIZES,
)
from collector.case_generator import (
    DSV4_ATTN_KINDS as ATTN_KINDS,
)
from collector.case_generator import (
    _dsv4_module_filter_pairs as _filter_pairs,
)
from collector.case_generator import (
    _dsv4_module_is_valid_shape as _is_valid_shape,
)
from collector.case_generator import (
    get_dsv4_csa_context_test_cases as _get_dsv4_csa_context_test_cases_impl,
)
from collector.case_generator import (
    get_dsv4_csa_generation_test_cases as _get_dsv4_csa_generation_test_cases_impl,
)
from collector.case_generator import (
    get_dsv4_hca_context_test_cases as _get_dsv4_hca_context_test_cases_impl,
)
from collector.case_generator import (
    get_dsv4_hca_generation_test_cases as _get_dsv4_hca_generation_test_cases_impl,
)
from collector.helper import log_perf

NATIVE_HEADS = 64
CLI_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
ATTN_KIND_TO_COMPRESS_RATIO = {"csa": 4, "hca": 128}


def _expand_grid() -> tuple[list[int], list[int]]:
    return list(_BATCH_SIZES), list(_SEQ_LENGTHS)


def get_dsv4_csa_context_test_cases():
    return _get_dsv4_csa_context_test_cases_impl()


def get_dsv4_csa_generation_test_cases():
    return _get_dsv4_csa_generation_test_cases_impl()


def get_dsv4_hca_context_test_cases():
    return _get_dsv4_hca_context_test_cases_impl()


def get_dsv4_hca_generation_test_cases():
    return _get_dsv4_hca_generation_test_cases_impl()


get_dsv4_flash_csa_context_test_cases = get_dsv4_csa_context_test_cases
get_dsv4_flash_csa_generation_test_cases = get_dsv4_csa_generation_test_cases
get_dsv4_flash_hca_context_test_cases = get_dsv4_hca_context_test_cases
get_dsv4_flash_hca_generation_test_cases = get_dsv4_hca_generation_test_cases


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=5,
        samples=20,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )


def _log_measurement(raw, *, perf_filename: str, op_name: str) -> None:
    log_perf(
        item_list=[dict(raw.perf_row)],
        framework=str(raw.provenance["framework"]),
        version=str(raw.provenance["framework_version"]),
        device_name=str(raw.provenance["device"]),
        op_name=op_name,
        kernel_source=str(raw.provenance["kernel_source"]),
        perf_filename=perf_filename,
        power_stats=dict(raw.power_stats) if raw.power_stats is not None else None,
    )


def run_dsv4_attn_worker(
    seq_len: int,
    batch_size: int,
    tp_size: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    model_path: str,
    attn_kind: str,
    attention_backend: str | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> list[dict[str, object]]:
    """Expand one offline grouped task into exact canonical runner calls."""

    del seq_len, attention_backend
    if attn_kind not in ATTN_KINDS:
        raise ValueError(f"unknown attn_kind={attn_kind}; expected one of {ATTN_KINDS}")
    if tp_size not in _TP_SIZES:
        raise ValueError(f"unsupported tp_size={tp_size}; expected one of {_TP_SIZES}")
    if NATIVE_HEADS % tp_size:
        raise ValueError(f"tp_size={tp_size} does not divide padded heads={NATIVE_HEADS}")

    mode = "context" if "context" in os.path.basename(perf_filename) else "generation"
    sequence_lengths = [value for value in _SEQ_LENGTHS if "--smoke" not in sys.argv or value in (1, 128)]
    prefixes = list(_PREFIX_LENGTHS) if mode == "context" else [0]
    if mode == "context" and "--smoke" in sys.argv:
        prefixes = [value for value in prefixes if value in (0, 512)]
    valid_lengths = {sl for bs, sl in _filter_pairs(mode, [batch_size], sequence_lengths) if bs == batch_size}
    protocol = _protocol()
    rows: list[dict[str, object]] = []
    shapes: list[dict[str, int | None]] = []
    for prefix in prefixes:
        for current_seq_len in sorted(valid_lengths, reverse=True):
            if not _is_valid_shape(mode, batch_size, current_seq_len, prefix):
                continue
            shapes.append(
                {"isl": current_seq_len, "prefix": prefix, "s_total": None}
                if mode == "context"
                else {"isl": None, "prefix": None, "s_total": current_seq_len + 1}
            )
    if not shapes:
        return rows

    runtime = open_dsv4_attn_runtime(
        mode=mode,
        attn_kind=attn_kind,
        tp_size=tp_size,
        canonical_num_heads=NATIVE_HEADS // tp_size,
        num_heads=NATIVE_HEADS,
        compress_ratio=ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
        batch_size=batch_size,
        mla_dtype=compute_dtype,
        kv_cache_dtype=kv_cache_dtype,
        gemm_type=gemm_type,
        case_shapes=shapes,
        device=device,
        model_path=model_path,
    )
    try:
        for shape in shapes:
            raw = run_dsv4_attn_case(
                mode=mode,
                attn_kind=attn_kind,
                tp_size=tp_size,
                canonical_num_heads=NATIVE_HEADS // tp_size,
                num_heads=NATIVE_HEADS,
                compress_ratio=ATTN_KIND_TO_COMPRESS_RATIO[attn_kind],
                batch_size=batch_size,
                mla_dtype=compute_dtype,
                kv_cache_dtype=kv_cache_dtype,
                gemm_type=gemm_type,
                protocol=protocol,
                device=device,
                model_path=model_path,
                runtime=runtime,
                **shape,
            )
            _log_measurement(
                raw,
                perf_filename=perf_filename,
                op_name=f"dsv4_{attn_kind}_{mode}_module",
            )
            rows.append(dict(raw.perf_row))
    finally:
        close_dsv4_attn_runtime(runtime)
    return rows


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DSv4 SGLang attention-module latency.")
    parser.add_argument("--mode", choices=("context", "generation"), required=True)
    parser.add_argument("--attn-kind", choices=ATTN_KINDS, default="csa")
    parser.add_argument("--model-path", default=CLI_DEFAULT_MODEL)
    parser.add_argument("--tp-sizes", default=",".join(str(value) for value in _TP_SIZES))
    parser.add_argument("--gemm-type", choices=("bfloat16", "fp8_block"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=os.getcwd())
    args = parser.parse_args()

    filename = os.path.join(
        args.output_path,
        f"dsv4_{args.attn_kind}_{args.mode}_module_perf.txt",
    )
    for tp_size in _parse_int_list(args.tp_sizes):
        for batch_size in _BATCH_SIZES:
            run_dsv4_attn_worker(
                0,
                batch_size,
                tp_size,
                "fp8",
                "bfloat16",
                args.gemm_type,
                args.model_path,
                args.attn_kind,
                perf_filename=filename,
                device=args.device,
            )


__all__ = [
    "ATTN_KINDS",
    "_BATCH_SIZES",
    "_SEQ_LENGTHS",
    "_TP_SIZES",
    "_filter_pairs",
    "get_dsv4_csa_context_test_cases",
    "get_dsv4_csa_generation_test_cases",
    "get_dsv4_flash_csa_context_test_cases",
    "get_dsv4_flash_csa_generation_test_cases",
    "get_dsv4_flash_hca_context_test_cases",
    "get_dsv4_flash_hca_generation_test_cases",
    "get_dsv4_hca_context_test_cases",
    "get_dsv4_hca_generation_test_cases",
    "run_dsv4_attn_worker",
]


if __name__ == "__main__":
    main()
