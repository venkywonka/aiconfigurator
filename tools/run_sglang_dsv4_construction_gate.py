#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the R25-P construction-only gate for DSv4 compressed attention.

This gate is deliberately narrower than the cold/warm/reopen lifecycle runner:
it proves the exact SGLang compressed-backend construction path on real GB200
hardware, but it does not time kernels, write the performance DB, or persist any
measurement records. R25-F is only supposed to run after this receipt exists.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConstructionGateError(RuntimeError):
    """The R25-P construction-only contract was violated."""


@dataclass(frozen=True, slots=True)
class PreparationRoute:
    """One terminal DSv4 attention route that must construct before R25-F."""

    route_id: str
    mode: str
    attn_kind: str
    compress_ratio: int
    attention_backend: str = "compressed"


R25_PREPARATION_ROUTES = (
    PreparationRoute("context-csa", "context", "csa", 4),
    PreparationRoute("context-hca", "context", "hca", 128),
    PreparationRoute("generation-csa", "generation", "csa", 4),
    PreparationRoute("generation-hca", "generation", "hca", 128),
)

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_ARCHITECTURE = "DeepseekV4ForCausalLM"
_TP_SIZE = 4
_CANONICAL_NUM_HEADS = 16
_PADDED_NUM_HEADS = 64


def validate_constructed_backend(
    runner: object,
    *,
    backend_type: type,
    indexer_type: type,
    compressor_type: type,
) -> dict[str, object]:
    """Validate that an already-constructed runner selected compressed DSv4."""

    server_args = getattr(runner, "server_args", None)
    attention_backend = getattr(server_args, "attention_backend", None)
    if attention_backend != "compressed":
        raise ConstructionGateError(f"constructed runner selected attention_backend={attention_backend!r}")
    attn_backend = getattr(runner, "attn_backend", None)
    if not isinstance(attn_backend, backend_type):
        raise ConstructionGateError(
            f"constructed attention backend is {type(attn_backend).__name__}, expected {backend_type.__name__}"
        )
    if not isinstance(attn_backend, indexer_type):
        raise ConstructionGateError(f"constructed backend is not a {indexer_type.__name__}")
    if not isinstance(attn_backend, compressor_type):
        raise ConstructionGateError(f"constructed backend is not a {compressor_type.__name__}")
    return {
        "attention_backend": attention_backend,
        "backend_class": type(attn_backend).__name__,
        "mro": [cls.__name__ for cls in type(attn_backend).__mro__],
    }


def validate_runtime_backend_surface() -> dict[str, object]:
    """Validate compressed factory registration and class inheritance."""

    from sglang.srt.layers.attention import attention_registry
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.indexer import C4IndexerBackend
    from sglang.srt.layers.attention.deepseek_v4_backend_radix import DeepseekV4BackendRadix

    if "dsv4" in attention_registry.ATTENTION_BACKENDS:
        raise ConstructionGateError("runtime registry unexpectedly exposes forbidden dsv4 alias")
    factory = attention_registry.ATTENTION_BACKENDS.get("compressed")
    if factory is None:
        raise ConstructionGateError("runtime registry does not expose compressed backend")
    if getattr(factory, "__name__", None) != "create_compressed_backend":
        raise ConstructionGateError("compressed registry factory identity mismatch")
    if not issubclass(DeepseekV4BackendRadix, C4IndexerBackend):
        raise ConstructionGateError("DeepseekV4BackendRadix does not inherit C4IndexerBackend")
    if not issubclass(DeepseekV4BackendRadix, CompressorBackend):
        raise ConstructionGateError("DeepseekV4BackendRadix does not inherit CompressorBackend")
    return {
        "registered_backend": "compressed",
        "dsv4_alias_present": False,
        "factory": factory.__name__,
        "backend_class": DeepseekV4BackendRadix.__name__,
        "mro": [cls.__name__ for cls in DeepseekV4BackendRadix.__mro__],
    }


def validate_visible_gb200_inventory(*, required_gpus: int) -> dict[str, object]:
    """Require the representative multi-GPU GB200 node before construction."""

    from aiconfigurator.collector.hardware import discover_hardware

    inventory = discover_hardware()
    gb200_devices = tuple(device for device in inventory.devices if "gb200" in device.name.casefold())
    if len(gb200_devices) < required_gpus:
        raise ConstructionGateError(
            f"R25-P requires at least {required_gpus} visible GB200 GPUs, found {len(gb200_devices)}"
        )
    return {
        "schema_revision": inventory.schema_revision,
        "topology_fingerprint": inventory.topology_fingerprint,
        "required_visible_gpus": required_gpus,
        "visible_gpu_count": len(inventory.devices),
        "visible_gb200_count": len(gb200_devices),
        "gpu_ids": [device.index for device in inventory.devices],
        "gb200_gpu_ids": [device.index for device in gb200_devices],
        "device_uuids": [device.uuid for device in inventory.devices],
        "device_names": [device.name for device in inventory.devices],
    }


def _route_case_kwargs(route: PreparationRoute, *, device: str) -> dict[str, object]:
    common: dict[str, object] = {
        "mode": route.mode,
        "attn_kind": route.attn_kind,
        "tp_size": _TP_SIZE,
        "canonical_num_heads": _CANONICAL_NUM_HEADS,
        "num_heads": _PADDED_NUM_HEADS,
        "compress_ratio": route.compress_ratio,
        "batch_size": 1,
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
        "device": device,
        "model_path": _MODEL_ARTIFACT,
    }
    if route.mode == "context":
        return {**common, "isl": 128, "prefix": 64, "s_total": None}
    return {**common, "isl": None, "prefix": None, "s_total": 128}


def construct_route(route: PreparationRoute, *, device: str) -> dict[str, object]:
    """Construct one route without timing or DB persistence."""

    from aiconfigurator.collector.sglang.dsv4_attn import _prepare_dsv4_attn_case

    prepared = _prepare_dsv4_attn_case(**_route_case_kwargs(route, device=device))
    try:
        if prepared.mode != route.mode or prepared.attn_kind != route.attn_kind:
            raise ConstructionGateError("prepared route identity mismatch")
        if prepared.compress_ratio != route.compress_ratio:
            raise ConstructionGateError("prepared route compress ratio mismatch")
        if prepared.model_artifact != _MODEL_ARTIFACT or prepared.architecture != _ARCHITECTURE:
            raise ConstructionGateError("prepared model identity mismatch")
        if (
            prepared.tp_size != _TP_SIZE
            or prepared.canonical_num_heads != _CANONICAL_NUM_HEADS
            or prepared.padded_num_heads != _PADDED_NUM_HEADS
            or prepared.mla_dtype != "bfloat16"
            or prepared.kv_cache_dtype != "fp8"
            or prepared.gemm_type != "fp8_block"
        ):
            raise ConstructionGateError("prepared route left the frozen DSv4 envelope")
        return {
            "route_id": route.route_id,
            "mode": prepared.mode,
            "attn_kind": prepared.attn_kind,
            "compress_ratio": prepared.compress_ratio,
            "attention_backend": route.attention_backend,
            "framework_version": prepared.framework_version,
            "device_name": prepared.device_name,
            "model_artifact": prepared.model_artifact,
            "architecture": prepared.architecture,
            "tp_size": prepared.tp_size,
            "canonical_num_heads": prepared.canonical_num_heads,
            "padded_num_heads": prepared.padded_num_heads,
            "mla_dtype": prepared.mla_dtype,
            "kv_cache_dtype": prepared.kv_cache_dtype,
            "gemm_type": prepared.gemm_type,
            "model_weight_generator": prepared.model_weight_generator,
            "model_weight_std": prepared.model_weight_std,
            "model_weight_seed": prepared.model_weight_seed,
        }
    finally:
        if prepared.cleanup_func is not None:
            prepared.cleanup_func()


def run_construction_gate(
    *,
    device: str = "cuda:0",
    required_gpus: int = 4,
    route_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run R25-P and return a receipt that explicitly excludes measurement."""

    selected = set(route_ids or [route.route_id for route in R25_PREPARATION_ROUTES])
    unknown = selected - {route.route_id for route in R25_PREPARATION_ROUTES}
    if unknown:
        raise ConstructionGateError(f"unknown R25-P route ids: {sorted(unknown)!r}")
    routes = tuple(route for route in R25_PREPARATION_ROUTES if route.route_id in selected)
    if not routes:
        raise ConstructionGateError("R25-P requires at least one construction route")

    started = time.time()
    inventory = validate_visible_gb200_inventory(required_gpus=required_gpus)
    backend = validate_runtime_backend_surface()
    constructed = [construct_route(route, device=device) for route in routes]
    return {
        "schema": "aic-r25-construction-gate-receipt-v1",
        "status": "pass",
        "stage": "R25-P",
        "kernel_source": "compressed_flashmla",
        "tuning_revision": "sglang-dsv4-attn-v1",
        "measurement_performed": False,
        "timing_performed": False,
        "database_access_performed": False,
        "constructed_route_count": len(constructed),
        "constructed_routes": constructed,
        "runtime_backend": backend,
        "inventory": inventory,
        "device_argument": device,
        "host": platform.node(),
        "elapsed_seconds": round(time.time() - started, 6),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--route-id", action="append", dest="route_ids")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = run_construction_gate(
        device=args.device,
        required_gpus=args.required_gpus,
        route_ids=args.route_ids,
    )
    _write_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "R25_PREPARATION_ROUTES",
    "ConstructionGateError",
    "PreparationRoute",
    "construct_route",
    "run_construction_gate",
    "validate_constructed_backend",
    "validate_runtime_backend_surface",
    "validate_visible_gb200_inventory",
]
