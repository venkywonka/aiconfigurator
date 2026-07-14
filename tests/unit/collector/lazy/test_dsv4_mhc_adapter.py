# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 exact one-GPU mHC adapter and runner contracts."""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.sglang import mhc, mhc_adapter
from aiconfigurator.collector.sglang.registry import MHC_LAZY_SPEC, SGLANG_LAZY_REGISTRY
from aiconfigurator.collector.types import FabricRequirement, RawMeasurement, ResourceContract
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
    ProtocolMismatchError,
)

pytestmark = pytest.mark.unit

_NAMESPACE = f"{PerfFile.MHC_MODULE}/v1"
_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
_PROFILE_COMPATIBILITY = {
    "model_artifact": _MODEL_ARTIFACT,
    "serving_mode": "aggregated",
    "tp_size": 4,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}
_SEMANTIC_DESCRIPTOR = {
    "num_sites": 2,
    "tensor_generator": "normal-v1",
    "seed": 0,
}


def _environment(**overrides: Any) -> MeasurementEnvironment:
    values: dict[str, Any] = {
        "system": "gb200",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "gpu_class": "NVIDIA GB200",
        "runtime_versions": _RUNTIME_VERSIONS,
        "topology_schema": "nvidia-smi-v1",
        "topology_fingerprint": "gb200-nvlink4",
        "profile_compatibility": _PROFILE_COMPATIBILITY,
    }
    values.update(overrides)
    return MeasurementEnvironment(**values)


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-mhc-v1",
    )


def _query(op: str = "pre", **overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "op": op,
        "num_tokens": 19,
        "hidden_size": 4096,
        "hc_mult": 4,
        "sinkhorn_iters": 20,
        "quant_mode": "bfloat16",
    }
    query.update(overrides)
    return query


def _request(
    op: str = "pre",
    *,
    query: dict[str, object] | None = None,
    environment: MeasurementEnvironment | None = None,
    op_id: str | None = None,
    semantic_descriptor: dict[str, object] | None = None,
) -> MeasurementRequest:
    query = query or _query(op)
    environment = environment or _environment()
    return MeasurementRequest(
        op_id=op_id or f"context_mhc_{op}",
        key=PerfKey.build(_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor=semantic_descriptor or _SEMANTIC_DESCRIPTOR,
        protocol=_protocol(),
    )


def test_mhc_rejects_malformed_sampling_protocol_as_protocol_mismatch() -> None:
    request = replace(_request(), protocol=replace(_protocol(), samples=2))

    with pytest.raises(ProtocolMismatchError, match="samples must be at least three"):
        mhc_adapter.mhc_request_to_case(request)


def _raw_result(request: MeasurementRequest) -> dict[str, object]:
    latency_ms = 1.25
    return {
        "latency_ms": latency_ms,
        "energy_wms": 125.0,
        "samples_ms": (1.2, latency_ms, 1.3),
        "statistic": request.protocol.statistic,
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "architecture": "DeepseekV4ForCausalLM",
            "op_name": request.query["op"],
            "num_tokens": request.query["num_tokens"],
            "num_sites": 2,
            "hc_mult": request.query["hc_mult"],
            "hidden_size": request.query["hidden_size"],
            "sinkhorn_iters": request.query["sinkhorn_iters"],
            "quant_mode": request.query["quant_mode"],
            "latency": latency_ms,
        },
        "provenance": {
            "framework": "SGLang",
            "framework_version": "0.5.10",
            "kernel_source": "sglang_mhc",
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
            "throttled": False,
            "model_artifact": _MODEL_ARTIFACT,
            "full_module": True,
            "num_sites": 2,
            "tensor_generator": "normal-v1",
            "seed": 0,
        },
    }


def test_packaged_and_source_registries_share_one_mhc_lazy_route() -> None:
    from collector.sglang.registry import REGISTRY as SOURCE_REGISTRY

    source_entry = next(entry for entry in SOURCE_REGISTRY if entry.op == "mhc_module")
    routes = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )

    assert source_entry.lazy is MHC_LAZY_SPEC
    assert len(routes) == 1
    assert routes[0].lazy is MHC_LAZY_SPEC
    assert routes[0].collector_module == "aiconfigurator.collector.sglang.mhc"


@pytest.mark.parametrize("op", ["pre", "post"])
def test_canonical_mhc_request_round_trips_through_case_and_emitted_row(op: str) -> None:
    request = _request(op)
    expected_query = _query(op)

    assert request.query == expected_query
    assert request.key == PerfKey.build(_NAMESPACE, expected_query, request.environment)

    case = mhc_adapter.mhc_request_to_case(request)
    assert case == expected_query
    assert mhc_adapter.mhc_resource_for_request(request, case) == ResourceContract(
        gpu_count=1,
        fabric=FabricRequirement.NONE,
    )

    record = mhc_adapter.mhc_result_to_record(request, case, _raw_result(request))
    emitted_query = {
        "op": record.perf_row["op_name"],
        "num_tokens": record.perf_row["num_tokens"],
        "hidden_size": record.perf_row["hidden_size"],
        "hc_mult": record.perf_row["hc_mult"],
        "sinkhorn_iters": record.perf_row["sinkhorn_iters"],
        "quant_mode": record.perf_row["quant_mode"],
    }
    assert record.key == request.key == PerfKey.build(_NAMESPACE, emitted_query, request.environment)


def test_mhc_consumer_and_scale_multiplicity_stay_out_of_physical_identity() -> None:
    layer_scaled = _request("pre", op_id="context_mhc_pre_x43")
    single_consumer = _request("pre", op_id="another_consumer_x1")

    assert layer_scaled.op_id != single_consumer.op_id
    assert layer_scaled.key == single_consumer.key
    assert set(layer_scaled.query) == {
        "op",
        "num_tokens",
        "hidden_size",
        "hc_mult",
        "sinkhorn_iters",
        "quant_mode",
    }
    assert "scale_factor" not in layer_scaled.query
    assert "num_layers" not in layer_scaled.query


def test_mhc_capability_allows_unrelated_runtime_inventory_entries() -> None:
    environment = replace(
        _environment(),
        runtime_versions={**_RUNTIME_VERSIONS, "tensorrt_llm": "1.3.0rc10"},
    )
    request = _request(environment=environment)

    assert mhc_adapter.mhc_request_to_case(request) == _query()


@pytest.mark.parametrize("sglang_version", ["0.5.10", "0.5.10rc0"])
def test_mhc_preserves_exact_supported_sglang_runtime_identity(
    sglang_version: str,
) -> None:
    environment = replace(
        _environment(),
        backend_version=sglang_version,
        runtime_versions={**_RUNTIME_VERSIONS, "sglang": sglang_version},
    )
    request = _request(environment=environment)
    raw_result = _raw_result(request)
    raw_result["provenance"]["framework_version"] = sglang_version

    case = mhc_adapter.mhc_request_to_case(request)
    record = mhc_adapter.mhc_result_to_record(request, case, raw_result)

    assert record.key == request.key
    assert request.environment.backend_version == sglang_version
    assert record.provenance["framework_version"] == sglang_version


@pytest.mark.parametrize(
    ("mismatch", "request_factory"),
    [
        ("system", lambda: _request(environment=replace(_environment(), system="h100_sxm"))),
        ("gpu_class", lambda: _request(environment=replace(_environment(), gpu_class="NVIDIA H100"))),
        (
            "backend_version",
            lambda: _request(
                environment=replace(
                    _environment(),
                    backend_version="0.5.11",
                    runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.11"},
                )
            ),
        ),
        (
            "cuda",
            lambda: _request(
                environment=replace(
                    _environment(),
                    runtime_versions={**_RUNTIME_VERSIONS, "cuda": "12.9"},
                )
            ),
        ),
        (
            "model_profile",
            lambda: _request(
                environment=replace(
                    _environment(),
                    runtime_versions={**_RUNTIME_VERSIONS, "model_profile": "other-profile"},
                )
            ),
        ),
        (
            "profile_compatibility",
            lambda: _request(
                environment=replace(
                    _environment(),
                    profile_compatibility={**_PROFILE_COMPATIBILITY, "tp_size": 8},
                )
            ),
        ),
        ("semantic_descriptor", lambda: _request(semantic_descriptor={**_SEMANTIC_DESCRIPTOR, "num_sites": 1})),
        ("op", lambda: _request(query=_query("both"))),
        ("shape", lambda: _request(query=_query(hidden_size=7168))),
        ("hc_mult", lambda: _request(query=_query(hc_mult=2))),
        ("sinkhorn_iters", lambda: _request(query=_query(sinkhorn_iters=10))),
        ("quantization", lambda: _request(query=_query(quant_mode="fp8"))),
        ("query_fields", lambda: _request(query={**_query(), "scale_factor": 43})),
    ],
)
def test_mhc_capability_mismatch_fails_before_resource_acquisition(
    mismatch: str,
    request_factory,
) -> None:
    request = request_factory()
    resource_calls: list[object] = []

    def _resource(request, case):
        resource_calls.append((request, case))
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    with pytest.raises(ValueError, match=r"capability|descriptor|fields|profile|version|bfloat16"):
        case = mhc_adapter.mhc_request_to_case(request)
        _resource(request, case)
    assert resource_calls == [], mismatch


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["perf_row"].__setitem__("num_sites", 1), "full-module|site"),
        (lambda raw: raw["perf_row"].__setitem__("quant_mode", "fp8"), "identity"),
        (lambda raw: raw["provenance"].__setitem__("used_cuda_graph", False), "CUDA Graph"),
        (lambda raw: raw["provenance"].__setitem__("model_artifact", "other/model"), "artifact"),
        (lambda raw: raw["provenance"].__setitem__("framework_version", "0.5.11"), "version"),
        (lambda raw: raw["provenance"].__setitem__("full_module", False), "full-module"),
    ],
)
def test_mhc_result_validation_fails_closed(mutation, message: str) -> None:
    request = _request()
    case = mhc_adapter.mhc_request_to_case(request)
    raw = _raw_result(request)
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        mhc_adapter.mhc_result_to_record(request, case, raw)


def test_mhc_helpers_preserve_bfloat16_two_site_pre_and_post_semantics() -> None:
    calls: list[tuple[object, ...]] = []

    class _NoGrad:
        def __enter__(self):
            calls.append(("no_grad_enter",))

        def __exit__(self, *args):
            calls.append(("no_grad_exit",))

    class _Generator:
        def __init__(self, *, device) -> None:
            calls.append(("generator", device))

        def manual_seed(self, seed: int):
            calls.append(("seed", seed))
            return self

    bfloat16 = object()

    def _randn(*shape, **kwargs):
        calls.append(("randn", shape, kwargs))
        return "residual"

    torch_module = SimpleNamespace(
        bfloat16=bfloat16,
        Generator=_Generator,
        randn=_randn,
        no_grad=lambda: _NoGrad(),
        cuda=SimpleNamespace(synchronize=lambda: calls.append(("synchronize",))),
    )

    class _Layer:
        hc_mult = 4
        config = SimpleNamespace(hidden_size=4096)
        hc_attn_fn = "attn_fn"
        hc_attn_scale = "attn_scale"
        hc_attn_base = "attn_base"
        hc_ffn_fn = "ffn_fn"
        hc_ffn_scale = "ffn_scale"
        hc_ffn_base = "ffn_base"

        def hc_pre(self, residual, *args):
            calls.append(("pre", residual, *args))
            return (f"x:{args[0]}", f"post:{args[0]}", f"comb:{args[0]}")

        def hc_post(self, x, residual, post, comb):
            calls.append(("post", x, residual, post, comb))
            return x

    layer = _Layer()
    residual = mhc._make_residual(layer, 19, "cuda:3", torch_module=torch_module)
    assert residual == "residual"
    randn_call = next(call for call in calls if call[0] == "randn")
    assert randn_call[1] == (19, 4, 4096)
    assert randn_call[2]["dtype"] is bfloat16
    assert randn_call[2]["device"] == "cuda:3"
    assert ("seed", 0) in calls

    calls.clear()
    pre_kernel = mhc._make_kernel(layer, "pre", residual, torch_module=torch_module)
    assert pre_kernel() == [("x:attn_fn", "post:attn_fn", "comb:attn_fn"), ("x:ffn_fn", "post:ffn_fn", "comb:ffn_fn")]
    assert [call[2:] for call in calls if call[0] == "pre"] == [
        ("attn_fn", "attn_scale", "attn_base"),
        ("ffn_fn", "ffn_scale", "ffn_base"),
    ]

    calls.clear()
    post_kernel = mhc._make_kernel(layer, "post", residual, torch_module=torch_module)
    assert ("synchronize",) in calls
    calls.clear()
    assert post_kernel() == ["x:attn_fn", "x:ffn_fn"]
    assert len([call for call in calls if call[0] == "post"]) == 2


def test_exact_mhc_runner_returns_one_raw_full_module_row_without_logging(monkeypatch, tmp_path) -> None:
    protocol = _protocol()
    calls: list[tuple[object, ...]] = []

    def _prepare(op, num_tokens, hidden_size, hc_mult, sinkhorn_iters, quant_mode, device, model_path):
        calls.append(("prepare", op, num_tokens, hidden_size, hc_mult, sinkhorn_iters, quant_mode, device, model_path))
        return mhc.PreparedMhcCase(
            kernel_func=lambda: calls.append(("kernel",)),
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
            architecture="DeepseekV4ForCausalLM",
            model_artifact=_MODEL_ARTIFACT,
            num_sites=2,
            hidden_size=4096,
            hc_mult=4,
            sinkhorn_iters=20,
            quant_mode="bfloat16",
            cleanup=lambda: calls.append(("cleanup",)),
        )

    @contextmanager
    def _benchmark(**kwargs):
        calls.append(("benchmark", kwargs))
        yield {
            "latency_ms": 1.25,
            "samples_ms": (1.2, 1.25, 1.3),
            "power_stats": {"power": 100.0, "power_limit": 1200.0},
            "throttled": False,
            "used_cuda_graph": True,
            "num_runs_executed": 3,
        }

    monkeypatch.setattr(mhc, "_prepare_mhc_case", _prepare)
    monkeypatch.setattr(mhc, "benchmark_with_power", _benchmark)
    monkeypatch.chdir(tmp_path)

    result = mhc.run_mhc_case("post", 19, 4096, 4, 20, "bfloat16", protocol=protocol)

    assert isinstance(result, RawMeasurement)
    assert result.latency_ms == 1.25
    assert result.samples_ms == (1.2, 1.25, 1.3)
    assert result.energy_wms == 125.0
    assert result.protocol_digest == protocol.digest
    assert result.perf_row == {
        "architecture": "DeepseekV4ForCausalLM",
        "op_name": "post",
        "num_tokens": 19,
        "num_sites": 2,
        "hc_mult": 4,
        "hidden_size": 4096,
        "sinkhorn_iters": 20,
        "quant_mode": "bfloat16",
        "latency": 1.25,
    }
    assert result.provenance == {
        "framework": "SGLang",
        "framework_version": "0.5.10",
        "kernel_source": "sglang_mhc",
        "device": "NVIDIA GB200",
        "used_cuda_graph": True,
        "throttled": False,
        "model_artifact": _MODEL_ARTIFACT,
        "full_module": True,
        "num_sites": 2,
        "tensor_generator": "normal-v1",
        "seed": 0,
    }
    assert calls[0] == ("prepare", "post", 19, 4096, 4, 20, "bfloat16", "cuda:0", _MODEL_ARTIFACT)
    benchmark_kwargs = next(call[1] for call in calls if call[0] == "benchmark")
    assert benchmark_kwargs["num_warmups"] == 2
    assert benchmark_kwargs["num_runs"] == 3
    assert benchmark_kwargs["repeat_n"] == 1
    assert benchmark_kwargs["return_samples"] is True
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert benchmark_kwargs["use_cuda_graph"] is True
    assert calls[-1] == ("cleanup",)
    assert not tuple(tmp_path.iterdir())


def test_mhc_runner_preserves_benchmark_failure_when_cleanup_also_fails(monkeypatch) -> None:
    def _prepare(*_args, **_kwargs):
        def _cleanup() -> None:
            raise RuntimeError("cleanup boom")

        return mhc.PreparedMhcCase(
            kernel_func=lambda: None,
            framework_version="0.5.10rc0",
            device_name="NVIDIA GB200",
            device=object(),
            architecture="DeepseekV4ForCausalLM",
            model_artifact=_MODEL_ARTIFACT,
            num_sites=2,
            hidden_size=4096,
            hc_mult=4,
            sinkhorn_iters=20,
            quant_mode="bfloat16",
            cleanup=_cleanup,
        )

    @contextmanager
    def _benchmark(**_kwargs):
        raise RuntimeError("benchmark boom")
        yield

    monkeypatch.setattr(mhc, "_prepare_mhc_case", _prepare)
    monkeypatch.setattr(mhc, "benchmark_with_power", _benchmark)

    with pytest.raises(RuntimeError, match="benchmark boom"):
        mhc.run_mhc_case("pre", 19, 4096, 4, 20, "bfloat16", protocol=_protocol())


def test_mhc_patched_model_dirs_are_unique_and_removed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        mhc,
        "_read_model_config",
        lambda _model_id: {"expert_dtype": "fp8", "num_hidden_layers": 8},
    )
    real_mkdtemp = mhc.tempfile.mkdtemp
    monkeypatch.setattr(
        mhc.tempfile,
        "mkdtemp",
        lambda **kwargs: real_mkdtemp(dir=tmp_path, **kwargs),
    )

    first = Path(mhc._patched_model_dir(_MODEL_ARTIFACT))
    second = Path(mhc._patched_model_dir(_MODEL_ARTIFACT))
    assert first != second
    assert first.is_dir() and second.is_dir()

    mhc._cleanup_temporary_model_dirs()
    assert not first.exists()
    assert not second.exists()


def test_mhc_temp_cleanup_attempts_remaining_dirs_after_one_oserror(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "blocked"
    removable = tmp_path / "removable"
    blocked.mkdir()
    removable.mkdir()

    class OrderedDirs(list[Path]):
        def discard(self, item: Path) -> None:
            if item in self:
                self.remove(item)

    tracked = OrderedDirs([blocked, removable])
    calls: list[Path] = []

    def remove_tree(path: Path) -> None:
        calls.append(path)
        if path == blocked:
            raise PermissionError("blocked temp directory")
        path.rmdir()

    monkeypatch.setattr(mhc, "_TEMPORARY_MODEL_DIRS", tracked)
    monkeypatch.setattr(mhc.shutil, "rmtree", remove_tree)

    with pytest.raises(PermissionError, match="blocked temp directory"):
        mhc._cleanup_temporary_model_dirs()

    assert calls == [blocked, removable]
    assert tracked == [blocked]
    assert blocked.is_dir()
    assert not removable.exists()


def test_mhc_runner_is_exact_only_import_light_and_has_no_offline_output_api(monkeypatch) -> None:
    assert mhc.get_mhc_test_cases() == ()
    assert {"output_path", "perf_filename", "num_tokens_cases", "ops"}.isdisjoint(
        inspect.signature(mhc.run_mhc_case).parameters
    )
    source = Path(mhc.__file__).read_text()
    assert "log_perf" not in source
    assert "from collector" not in source
    assert "import collector" not in source

    module_name = "aiconfigurator.collector.sglang.mhc"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name == "sglang" or name.startswith("sglang."):
            raise AssertionError(f"heavy import at module import time: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    importlib.import_module(module_name)
