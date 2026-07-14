# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold/warm/reopen contracts for the bounded DSv4 attention GPU gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.run_dsv4_attention_lifecycle as lifecycle_runner
from aiconfigurator.collector.executor import WorkerReply
from aiconfigurator.collector.types import (
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    canonical_topology_fingerprint,
)
from tools.run_dsv4_attention_lifecycle import ATTENTION_ROUTE_MATRIX, run_attention_lifecycle

pytestmark = pytest.mark.unit


def _inventory(
    count: int = 1,
    *,
    names: tuple[str, ...] | None = None,
) -> HardwareInventory:
    device_names = names or ("NVIDIA GB200",) * count
    devices = tuple(
        GpuDevice(
            index=index,
            uuid=f"GPU-gb200-test-{index}",
            name=device_names[index],
            pci_bus_id=f"00000000:{index:02X}:00.0",
        )
        for index in range(count)
    )
    links = {(left, right): "SYS" for left in range(count) for right in range(count) if left != right}
    capabilities = dict.fromkeys(links, False)
    fingerprint = canonical_topology_fingerprint(
        "test-gb200-inventory-v1",
        devices,
        links,
        capabilities,
        capabilities,
    )
    return HardwareInventory(
        schema_revision="test-gb200-inventory-v1",
        devices=devices,
        links=links,
        p2p_read=capabilities,
        p2p_write=capabilities,
        fabric_domains={},
        topology_fingerprint=fingerprint,
        evidence=HardwareDiscoveryEvidence(
            raw_gpu_query="synthetic GB200 query",
            raw_topology="synthetic GPU topology",
            raw_p2p_read="synthetic GPU reads",
            raw_p2p_write="synthetic GPU writes",
        ),
    )


def _raw_result(case: dict[str, object], protocol) -> dict[str, object]:
    route_order = {
        ("context", "csa"): 1.0,
        ("context", "hca"): 2.0,
        ("generation", "csa"): 3.0,
        ("generation", "hca"): 4.0,
    }
    latency_ms = route_order[(str(case["mode"]), str(case["attn_kind"]))]
    return {
        "latency_ms": latency_ms,
        "energy_wms": latency_ms * 100.0,
        "samples_ms": (latency_ms - 0.1, latency_ms, latency_ms + 0.1),
        "statistic": protocol.statistic,
        "protocol_digest": protocol.digest,
        "perf_row": {
            "model": "sgl-project/DeepSeek-V4-Flash-FP8",
            "architecture": "DeepseekV4ForCausalLM",
            "mla_dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "gemm_type": "fp8_block",
            "num_heads": 64,
            "batch_size": 1,
            "isl": case["isl"] if case["mode"] == "context" else 1,
            "tp_size": 4,
            "step": case["prefix"] if case["mode"] == "context" else int(case["s_total"]) - 1,
            "compress_ratio": case["compress_ratio"],
            "latency": latency_ms,
        },
        "provenance": {
            "framework": "SGLang",
            "framework_version": "0.5.10",
            "kernel_source": "compressed_flashmla",
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
            "throttled": False,
            "model_artifact": "sgl-project/DeepSeek-V4-Flash-FP8",
            "full_module": True,
            "mode": case["mode"],
            "attn_kind": case["attn_kind"],
            "tp_simulation": "single-gpu-tp4",
            "canonical_num_heads": 16,
            "padded_num_heads": 64,
            "tensor_generator": "normal-v1",
            "seed": 0,
            "model_weight_generator": "proper-normal-v1",
            "model_weight_std": 0.05,
            "model_weight_seed": 1234,
        },
    }


class _FakeChannel:
    def __init__(self, commands: list[object], bootstrap) -> None:
        self._commands = commands
        self._bootstrap = bootstrap
        self._pending: list[WorkerReply] = []
        self._closed = False

    def send(self, command) -> None:
        self._commands.append(command)
        case = json.loads(command.payload)
        self._pending.append(
            WorkerReply(
                invocation_id=command.invocation_id,
                request_digest=command.request_digest,
                raw_result={
                    **_raw_result(case, command.protocol),
                    "_worker_binding": {
                        "cuda_visible_devices": ",".join(self._bootstrap.device_uuids),
                        "device_uuids": list(self._bootstrap.device_uuids),
                        "local_ordinals": list(self._bootstrap.local_ordinals),
                        "topology_fingerprint": self._bootstrap.topology_fingerprint,
                    },
                },
            )
        )

    def recv(self) -> WorkerReply:
        return self._pending.pop(0)

    def is_alive(self) -> bool:
        return not self._closed

    def close(self) -> None:
        self._closed = True

    def join(self) -> None:
        return None

    @property
    def pending(self) -> bool:
        return bool(self._pending)


class _FakeWorkerRuntime:
    def __init__(self) -> None:
        self.commands: list[object] = []
        self.bootstraps: list[object] = []

    @property
    def command_count(self) -> int:
        return len(self.commands)

    def __call__(self, bootstrap) -> _FakeChannel:
        self.bootstraps.append(bootstrap)
        return _FakeChannel(self.commands, bootstrap)

    @staticmethod
    def wait_ready(channels, timeout_seconds):
        del timeout_seconds
        return tuple(channel for channel in channels if channel.pending)


def test_route_matrix_is_the_exact_bounded_four_route_envelope() -> None:
    assert [
        (
            route.route_id,
            route.mode,
            route.attn_kind,
            route.compress_ratio,
            route.namespace,
            route.runtime_kwargs,
        )
        for route in ATTENTION_ROUTE_MATRIX
    ] == [
        (
            "context-csa",
            "context",
            "csa",
            4,
            "dsv4_csa_context_module_perf.txt/v1",
            {
                "x": 128,
                "batch_size": 1,
                "beam_width": 1,
                "s": 128,
                "prefix": 64,
                "model_name": "sgl-project/DeepSeek-V4-Flash-FP8",
                "seq_imbalance_correction_scale": 1.0,
            },
        ),
        (
            "context-hca",
            "context",
            "hca",
            128,
            "dsv4_hca_context_module_perf.txt/v1",
            {
                "x": 128,
                "batch_size": 1,
                "beam_width": 1,
                "s": 128,
                "prefix": 64,
                "model_name": "sgl-project/DeepSeek-V4-Flash-FP8",
                "seq_imbalance_correction_scale": 1.0,
            },
        ),
        (
            "generation-csa",
            "generation",
            "csa",
            4,
            "dsv4_csa_generation_module_perf.txt/v1",
            {
                "x": 1,
                "batch_size": 1,
                "beam_width": 1,
                "s": 128,
                "prefix": 64,
                "model_name": "sgl-project/DeepSeek-V4-Flash-FP8",
                "gen_seq_imbalance_correction_scale": 1.0,
            },
        ),
        (
            "generation-hca",
            "generation",
            "hca",
            128,
            "dsv4_hca_generation_module_perf.txt/v1",
            {
                "x": 1,
                "batch_size": 1,
                "beam_width": 1,
                "s": 128,
                "prefix": 64,
                "model_name": "sgl-project/DeepSeek-V4-Flash-FP8",
                "gen_seq_imbalance_correction_scale": 1.0,
            },
        ),
    ]


def test_cold_warm_reopen_lifecycle_measures_each_unique_key_once_and_pure_is_gpu_free(
    tmp_path: Path,
) -> None:
    runtime = _FakeWorkerRuntime()
    overlay_path = tmp_path / "dsv4-attention-lifecycle.sqlite"

    report = run_attention_lifecycle(
        overlay_path=overlay_path,
        inventory=_inventory(),
        worker_factory=runtime,
        wait_ready=runtime.wait_ready,
        command_count=lambda: runtime.command_count,
        max_wall_seconds=10.0,
    )

    assert report.route_ids == tuple(route.route_id for route in ATTENTION_ROUTE_MATRIX)
    assert report.namespaces == tuple(route.namespace for route in ATTENTION_ROUTE_MATRIX)
    assert len(set(report.key_digests)) == 4
    assert report.pure.command_count == 0
    assert report.pure.additional_commands == 0
    assert report.pure.sources == ("silicon",) * 4
    assert report.cold.command_count == 4
    assert report.cold.additional_commands == 4
    assert report.cold.sources == ("overlay",) * 4
    assert report.cold_unique_misses == 4
    assert report.cold_accepted_records == 4
    assert report.warm.command_count == 4
    assert report.warm.additional_commands == 0
    assert report.reopened.command_count == 4
    assert report.reopened.additional_commands == 0
    assert report.reopened_unique_misses == 0
    assert report.cold.results == report.warm.results == report.reopened.results
    assert len(report.records) == 4
    assert all(record.status == "valid" for record in report.records)
    assert all(record.latency_ms > 0 for record in report.records)
    assert all(len(record.samples_ms) == 3 for record in report.records)
    assert {record.namespace for record in report.records} == set(report.namespaces)
    assert overlay_path.is_file()

    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["cold"]["additional_commands"] == 4
    assert payload["warm"]["additional_commands"] == 0
    assert payload["reopened"]["additional_commands"] == 0
    assert all(record["provenance"]["framework_version"] == "0.5.10" for record in payload["records"])


def test_four_gpu_lifecycle_reports_disjoint_one_gpu_leases_and_worker_binding(
    tmp_path: Path,
) -> None:
    runtime = _FakeWorkerRuntime()
    inventory = _inventory(4)

    report = run_attention_lifecycle(
        overlay_path=tmp_path / "dsv4-attention-four-gpu.sqlite",
        inventory=inventory,
        worker_factory=runtime,
        wait_ready=runtime.wait_ready,
        command_count=lambda: runtime.command_count,
        max_wall_seconds=10.0,
    )

    assert report.inventory.gpu_ids == (0, 1, 2, 3)
    assert report.inventory.device_uuids == tuple(device.uuid for device in inventory.devices)
    assert report.inventory.topology_fingerprint == inventory.topology_fingerprint
    assert report.resource_contract_gpu_count == 1
    assert report.cold.additional_commands == 4
    assert report.warm.additional_commands == 0
    assert report.reopened.additional_commands == 0
    assert len(runtime.bootstraps) == 4
    assert {bootstrap.device_uuids for bootstrap in runtime.bootstraps} == {
        (device.uuid,) for device in inventory.devices
    }
    assert {lease.assigned_gpu_ids for lease in report.leases} == {(0,), (1,), (2,), (3,)}
    assert all(lease.contract_gpu_count == 1 for lease in report.leases)
    assert all(lease.worker_visible_device_uuids == lease.assigned_device_uuids for lease in report.leases)
    assert all(lease.worker_cuda_visible_devices == lease.assigned_device_uuids[0] for lease in report.leases)
    assert all(lease.worker_local_ordinals == (0,) for lease in report.leases)
    assert all(lease.worker_topology_fingerprint == inventory.topology_fingerprint for lease in report.leases)
    assert all(len(lease.remaining_gpu_ids) == 3 for lease in report.leases)
    assert all(set(lease.remaining_gpu_ids).isdisjoint(lease.assigned_gpu_ids) for lease in report.leases)


def test_single_gpu_worker_execution_evidence_proves_four_case_reuse() -> None:
    bootstrap = type("Bootstrap", (), {"device_uuids": ("GPU-gb200-test-0",)})()
    runtime = type(
        "Runtime",
        (),
        {
            "bootstraps": (bootstrap,),
            "command_count": 4,
            "command_counts": (4,),
        },
    )()

    evidence = lifecycle_runner._single_worker_reuse_evidence(
        runtime,
        expected_commands=4,
    )

    assert evidence.bootstrap_count == 1
    assert evidence.command_count == 4
    assert evidence.commands_per_worker == (4,)
    assert evidence.bootstrap_device_uuids == (("GPU-gb200-test-0",),)
    assert evidence.reused_worker_count == 1


def _patch_gpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    *,
    sglang_version: str = "0.5.10",
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return len(names)

        @staticmethod
        def get_device_name(index: int) -> str:
            return names[index]

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"cuda": "13.0"})()

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(lifecycle_runner.importlib.metadata, "version", lambda _name: sglang_version)
    monkeypatch.setattr(
        lifecycle_runner.importlib,
        "import_module",
        lambda name: FakeTorch() if name == "torch" else object(),
    )


def test_scheduler_gpu_runtime_accepts_four_homogeneous_gb200s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gpu_runtime(monkeypatch, ("NVIDIA GB200",) * 4)

    evidence = lifecycle_runner.validate_scheduler_gpu_runtime(required_gpu_count=4)

    assert evidence["gpu_count"] == 4
    assert evidence["required_gpu_count"] == 4
    assert evidence["gpu_names"] == ("NVIDIA GB200",) * 4


def test_scheduler_gpu_runtime_preserves_coherent_release_candidate_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gpu_runtime(
        monkeypatch,
        ("NVIDIA GB200",) * 4,
        sglang_version="0.5.10rc0",
    )

    evidence = lifecycle_runner.validate_scheduler_gpu_runtime(required_gpu_count=4)
    environment = lifecycle_runner._environment(
        _inventory(4),
        backend_version=str(evidence["sglang_version"]),
    )

    assert evidence["sglang_version"] == "0.5.10rc0"
    assert environment.backend_version == "0.5.10rc0"
    assert environment.runtime_versions["sglang"] == "0.5.10rc0"


def test_scheduler_gpu_runtime_rejects_mixed_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gpu_runtime(
        monkeypatch,
        ("NVIDIA GB200", "NVIDIA GB200", "NVIDIA H100 80GB HBM3", "NVIDIA GB200"),
    )

    with pytest.raises(RuntimeError, match=r"all visible GPUs.*NVIDIA GB200"):
        lifecycle_runner.validate_scheduler_gpu_runtime(required_gpu_count=4)


def test_local_gpu_runtime_retains_exactly_one_visible_gpu_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gpu_runtime(monkeypatch, ("NVIDIA GB200",) * 4)

    with pytest.raises(RuntimeError, match="exactly one visible CUDA GPU"):
        lifecycle_runner.validate_local_gpu_runtime()
