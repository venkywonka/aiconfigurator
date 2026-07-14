# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline NCCL sweep backed by the installable exact-case runner."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager

from aiconfigurator.collector.network.nccl import close_nccl_worker, get_nccl_test_cases, run_nccl_case
from aiconfigurator.sdk.resolution.types import MeasurementProtocol
from collector.helper import log_perf


@contextmanager
def _offline_gpu_visibility(num_gpus: int) -> Iterator[None]:
    """Bind this one-shot offline process to exactly the requested GPU slice."""

    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    available = tuple(value.strip() for value in (original or "").split(",") if value.strip())
    if original is not None and len(available) < num_gpus:
        raise RuntimeError(
            "offline NCCL sweep does not have enough visible GPUs: "
            f"visible={len(available)}, requested={num_gpus}"
        )
    selected = (
        available[:num_gpus] if original is not None else tuple(str(index) for index in range(num_gpus))
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def nccl_benchmark(
    dtype: str,
    nccl_op: str = "all_gather",
    test_range: str = "10,10000000,1000",
    num_gpus: int = 8,
    measure_power: bool = False,
) -> None:
    """Expand the offline range, execute canonical cases, and persist their rows."""

    if measure_power:
        raise ValueError("persistent NCCL collection does not support power sampling")
    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=5,
        samples=10,
        statistic="median",
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )
    with _offline_gpu_visibility(num_gpus):
        try:
            for case in get_nccl_test_cases(dtype, nccl_op, test_range, num_gpus):
                raw = run_nccl_case(*case, protocol=protocol)
                log_perf(
                    item_list=[dict(raw.perf_row)],
                    framework=str(raw.provenance["framework"]),
                    version=str(raw.provenance["framework_version"]),
                    device_name=str(raw.provenance["device"]),
                    op_name=nccl_op,
                    kernel_source=str(raw.provenance["kernel_source"]),
                    perf_filename="nccl_perf.txt",
                    power_stats=None,
                )
        finally:
            close_nccl_worker()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--nccl_op",
        "-NCCL",
        default="all_gather",
        choices=["all_gather", "alltoall", "reduce_scatter", "all_reduce"],
        help="NCCL OP: all_gather, alltoall, reduce_scatter, all_reduce",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        default="half",
        choices=["half", "bfloat16", "int8"],
        help="NCCL OP data type",
    )
    parser.add_argument(
        "--range",
        "-r",
        default="512,536870913,2",
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--num_gpus", "-n", default=8, type=int)
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Enable power monitoring during NCCL benchmark execution",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Retained CLI compatibility; nccl-tests already runs long enough for power sampling",
    )
    args = parser.parse_args()
    nccl_benchmark(args.dtype, args.nccl_op, args.range, args.num_gpus, args.measure_power)


__all__ = ["nccl_benchmark"]
