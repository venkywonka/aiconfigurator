# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline DeepSeek-V4 mHC sweep backed by the installable exact runner."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from aiconfigurator.collector.sglang.mhc import close_mhc_runtime, open_mhc_runtime, run_mhc_case
from aiconfigurator.sdk.resolution.types import MeasurementProtocol
from collector.case_generator import get_common_mhc_test_cases
from collector.helper import log_perf
from collector.registry_types import PerfFile

DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
PERF_FILENAME = PerfFile.MHC_MODULE.value


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _profile(model_path: str, op: str):
    cases = get_common_mhc_test_cases()
    for case in cases:
        if case.model_name == model_path and case.phase == op:
            return case
    for case in cases:
        if case.phase == op:
            return case
    raise RuntimeError(f"no mHC case profile is registered for op={op!r}")


def _default_num_tokens(model_path: str) -> list[int]:
    return list(_profile(model_path, "pre").num_tokens_list)


def get_mhc_module_test_cases() -> list[dict]:
    """Return one grouped task for every distinct offline model shape and phase."""

    cases: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for case in get_common_mhc_test_cases():
        key = (case.phase, case.hidden_size, case.hc_mult)
        if key in seen:
            continue
        seen.add(key)
        model_id = case.model_name.replace("/", "_")
        cases.append(
            {
                "id": f"mhc_{case.phase}_hs{case.hidden_size}_hcm{case.hc_mult}_{model_id}",
                "params": [case.phase, case.model_name],
            }
        )
    return cases


def _resolve_perf_path(output_path: str | None, filename: str | None) -> str:
    filename = filename or PERF_FILENAME
    if not output_path:
        return filename
    if output_path.endswith(".txt"):
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, filename)


def _log_measurement(raw, *, op: str, output_path: str | None, perf_filename: str | None) -> None:
    log_perf(
        item_list=[dict(raw.perf_row)],
        framework=str(raw.provenance["framework"]),
        version=str(raw.provenance["framework_version"]),
        device_name=str(raw.provenance["device"]),
        op_name=op,
        kernel_source=str(raw.provenance["kernel_source"]),
        perf_filename=_resolve_perf_path(output_path, perf_filename),
        power_stats=dict(raw.power_stats) if raw.power_stats is not None else None,
    )


def run_mhc_module(
    *,
    ops: Sequence[str],
    num_tokens_cases: Sequence[int] | None = None,
    model_path: str = DEFAULT_MODEL,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda:0",
    output_path: str | None = None,
    mem_fraction_static: float = 0.5,
    perf_filename: str | None = None,
) -> list[dict[str, float]]:
    """Run an offline token sweep through the canonical exact-case function."""

    if num_iterations < 3:
        raise ValueError("num_iterations must be at least 3")
    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=num_warmup,
        samples=num_iterations,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-mhc-v1",
    )
    results: list[dict[str, float]] = []
    runtime = open_mhc_runtime(
        model_path=model_path,
        device=device,
        mem_fraction_static=mem_fraction_static,
    )
    try:
        for op in ops:
            profile = _profile(model_path, op)
            token_cases = [int(value) for value in (num_tokens_cases or profile.num_tokens_list)]
            for num_tokens in token_cases:
                raw = run_mhc_case(
                    op,
                    num_tokens,
                    int(profile.hidden_size),
                    int(profile.hc_mult),
                    20,
                    "bfloat16",
                    protocol=protocol,
                    device=device,
                    model_path=model_path,
                    mem_fraction_static=mem_fraction_static,
                    runtime=runtime,
                )
                _log_measurement(raw, op=op, output_path=output_path, perf_filename=perf_filename)
                results.append(
                    {
                        "op": op,
                        "num_tokens": num_tokens,
                        "mean_ms": raw.latency_ms,
                        "n": len(raw.samples_ms),
                        "used_cuda_graph": bool(raw.provenance.get("used_cuda_graph", False)),
                        "throttled": bool(raw.provenance.get("throttled", False)),
                    }
                )
    finally:
        close_mhc_runtime(runtime)
    return results


def run_mhc_module_worker(
    op: str,
    model_path: str | None = None,
    *,
    perf_filename: str,
    device: str = "cuda:0",
) -> None:
    model_path = model_path or os.environ.get("COLLECTOR_MODEL_PATH") or DEFAULT_MODEL
    run_mhc_module(
        ops=[op],
        model_path=model_path,
        device=device,
        output_path=os.path.dirname(perf_filename) or os.getcwd(),
        perf_filename=os.path.basename(perf_filename),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DeepSeek-V4 mHC pre/post latency on SGLang.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--op", choices=["pre", "post", "all"], default="all")
    parser.add_argument("--num-tokens", default=None)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    args = parser.parse_args()
    run_mhc_module(
        ops=["pre", "post"] if args.op == "all" else [args.op],
        num_tokens_cases=_parse_int_list(args.num_tokens) if args.num_tokens else None,
        model_path=args.model_path,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=args.device,
        output_path=args.output_path,
        mem_fraction_static=args.mem_fraction_static,
    )


if __name__ == "__main__":
    main()


__all__ = ["get_mhc_module_test_cases", "run_mhc_module", "run_mhc_module_worker"]
