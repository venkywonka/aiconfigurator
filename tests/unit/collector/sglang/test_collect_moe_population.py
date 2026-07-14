# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import itertools
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "moe_runtime.py"
PACKAGED_RUNTIME_PATH = REPO_ROOT / "src" / "aiconfigurator" / "collector" / "sglang" / "_moe_runtime.py"
OFFLINE_WRAPPER_PATH = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"


def _load_functions(*names: str, namespace: dict | None = None, source_path: Path = RUNTIME_SOURCE_PATH) -> dict:
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


def _gptoss_case(*, tp: int, ep: int):
    return SimpleNamespace(
        num_tokens_list=[128],
        hidden_size=2880,
        inter_size=2880,
        topk=4,
        num_experts=128,
        tp=tp,
        ep=ep,
        model_name="openai/gpt-oss-120b",
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GptOssForCausalLM",
    )


def _populate_gptoss_cases(cases):
    loaded = _load_functions(
        "get_moe_test_cases",
        namespace={
            "itertools": itertools,
            "get_sm_version": lambda: 100,
            "get_common_moe_test_cases": lambda: cases,
            "moe_model_allows_quantization": (lambda _backend, _model, mode: mode == "w4a8_mxfp4_mxfp8"),
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "_SM120_NEMOTRON_NVFP4_MODELS": set(),
        },
        source_path=OFFLINE_WRAPPER_PATH,
    )
    return loaded["get_moe_test_cases"]()


@pytest.mark.parametrize(("tp", "ep"), [(4, 8), (32, 1), (32, 8)])
def test_gptoss_mxfp4_population_retains_tp_and_ep_buckets(tp, ep):
    populated = _populate_gptoss_cases([_gptoss_case(tp=tp, ep=ep)])

    assert len(populated) == 1
    assert populated[0][0] == "w4a8_mxfp4_mxfp8"
    assert populated[0][6:8] == [tp, ep]
    assert populated[0][-1] is None


def test_offline_case_enumeration_does_not_import_the_sglang_runtime_backend():
    source = OFFLINE_WRAPPER_PATH.read_text()

    assert "aiconfigurator.collector.sglang._moe_runtime" not in source


def test_legacy_moe_runtime_is_source_only():
    assert RUNTIME_SOURCE_PATH.is_file()
    assert not PACKAGED_RUNTIME_PATH.exists()


def test_legacy_moe_runtime_has_no_import_time_server_args_mutation():
    tree = ast.parse(RUNTIME_SOURCE_PATH.read_text(), filename=str(RUNTIME_SOURCE_PATH))
    mutations = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        mutations.extend(
            child
            for child in ast.walk(node)
            if isinstance(child, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Attribute) and target.attr == "_global_server_args"
                for target in (
                    child.targets if isinstance(child, ast.Assign) else (child.target,)
                )
            )
        )

    assert mutations == []


@pytest.mark.parametrize("fail_during_patch", [False, True])
def test_legacy_server_args_patch_is_scoped_and_restores_unset_global(fail_during_patch):
    server_args_module = SimpleNamespace(_global_server_args=None)
    loaded = _load_functions(
        "_temporary_legacy_server_args",
        namespace={
            "contextmanager": contextmanager,
            "MagicMock": MagicMock,
            "_server_args_module": server_args_module,
        },
    )

    expectation = pytest.raises(RuntimeError, match="benchmark failed") if fail_during_patch else nullcontext()
    with expectation, loaded["_temporary_legacy_server_args"]():
        mock = server_args_module._global_server_args
        assert isinstance(mock, MagicMock)
        assert mock.enable_fused_moe_sum_all_reduce is False
        if fail_during_patch:
            raise RuntimeError("benchmark failed")

    assert server_args_module._global_server_args is None


def test_legacy_run_wrapper_scopes_server_args_around_entire_quantized_case():
    events = []

    @contextmanager
    def scoped_server_args():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    loaded = _load_functions(
        "run_moe_torch",
        namespace={
            "_temporary_legacy_server_args": scoped_server_args,
            "_run_moe_torch_impl": lambda *args, **kwargs: (args, kwargs),
        },
    )

    args = ("nvfp4", 128, 4096, 11008, 8, 256, 1, 4, "model")
    result = loaded["run_moe_torch"](*args, device="cuda:0")

    assert result == (args, {"device": "cuda:0"})
    assert events == ["enter", "exit"]


def test_rank_local_workloads_cycle_independently_of_requested_sample_count():
    source = RUNTIME_SOURCE_PATH.read_text()

    assert "num_iters = len(workloads)" not in source
    assert "workloads[i % num_iters]" not in source
    assert "masked_m_list[i % num_iters]" not in source


@pytest.mark.parametrize("fail_during_patch", [False, True])
def test_legacy_reduction_patch_is_scoped_and_restores_native_sglang(fail_during_patch):
    calls = []

    def native_reduction(*_args):
        calls.append("native")

    fused_moe_module = SimpleNamespace(moe_sum_reduce_torch_compile=native_reduction)
    fake_torch = SimpleNamespace(
        sum=lambda _x, *, dim, out: calls.append(("sum", dim, out)),
    )
    loaded = _load_functions(
        "_temporary_eager_moe_sum_reduce",
        namespace={
            "contextmanager": contextmanager,
            "_fmoe_mod": fused_moe_module,
            "torch": fake_torch,
        },
    )

    expectation = pytest.raises(RuntimeError, match="benchmark failed") if fail_during_patch else nullcontext()
    with expectation, loaded["_temporary_eager_moe_sum_reduce"]():
        patched = fused_moe_module.moe_sum_reduce_torch_compile
        assert patched is not native_reduction
        output = SimpleNamespace(mul_=lambda scale: calls.append(("mul", scale)))
        patched("input", output, 1.5)
        if fail_during_patch:
            raise RuntimeError("benchmark failed")

    assert fused_moe_module.moe_sum_reduce_torch_compile is native_reduction
    fused_moe_module.moe_sum_reduce_torch_compile()
    assert calls[-1] == "native"


@pytest.mark.parametrize("fail_during_benchmark", [False, True])
def test_mxfp4_parallel_patch_covers_benchmark_and_restores_helpers(fail_during_benchmark):
    def original_helper(name):
        def helper(*_args, **_kwargs):
            return name

        return helper

    moe_layer = SimpleNamespace(
        get_tp_group=original_helper("layer_tp_group"),
        is_allocation_symmetric=original_helper("layer_symmetric"),
        get_moe_expert_parallel_world_size=original_helper("layer_ep_world"),
        get_moe_expert_parallel_rank=original_helper("layer_ep_rank"),
        get_moe_tensor_parallel_world_size=original_helper("layer_tp_world"),
        get_moe_tensor_parallel_rank=original_helper("layer_tp_rank"),
        create_kt_config_from_server_args=original_helper("layer_kt_config"),
    )
    standard_dispatch = SimpleNamespace(
        get_tp_group=original_helper("dispatch_tp_group"),
        is_allocation_symmetric=original_helper("dispatch_symmetric"),
        get_moe_expert_parallel_world_size=original_helper("dispatch_ep_world"),
        get_moe_expert_parallel_rank=original_helper("dispatch_ep_rank"),
    )
    mxfp4 = SimpleNamespace(
        get_tp_group=original_helper("mxfp4_tp_group"),
        is_allocation_symmetric=original_helper("mxfp4_symmetric"),
    )
    modules = (moe_layer, standard_dispatch, mxfp4)
    originals = [(module, name, value) for module in modules for name, value in vars(module).items()]

    def benchmark_config(*_args, **_kwargs):
        assert moe_layer.get_moe_expert_parallel_world_size() == 8
        assert moe_layer.get_moe_expert_parallel_rank() == 0
        assert moe_layer.get_moe_tensor_parallel_world_size() == 4
        assert moe_layer.get_moe_tensor_parallel_rank() == 0
        assert moe_layer.create_kt_config_from_server_args(object(), 0) is None
        assert standard_dispatch.get_moe_expert_parallel_world_size() == 8
        assert standard_dispatch.get_moe_expert_parallel_rank() == 0
        for module in modules:
            assert module.get_tp_group() is None
            assert not module.is_allocation_symmetric()
        if fail_during_benchmark:
            raise RuntimeError("benchmark failed")
        return 1.25, {"power": 100.0}

    fake_torch = SimpleNamespace(dtype=object, cuda=SimpleNamespace(manual_seed_all=lambda _seed: None))
    loaded = _load_functions(
        "_patch_mxfp4_single_process_parallel",
        "benchmark",
        namespace={
            "contextmanager": contextmanager,
            "nullcontext": nullcontext,
            "torch": fake_torch,
            "_moe_layer_mod": moe_layer,
            "_std_dispatch_mod": standard_dispatch,
            "_mxfp4_mod": mxfp4,
            "_HAS_SGLANG_MXFP4": True,
            "_HAS_MARLIN_MOE": False,
            "benchmark_config": benchmark_config,
        },
    )
    benchmark = loaded["benchmark"]
    kwargs = {
        "num_tokens": 128,
        "num_experts": 8,
        "shard_intermediate_size": 512,
        "hidden_size": 256,
        "topk": 2,
        "dtype": object(),
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_mxfp4_w4a16": True,
        "moe_tp_size": 4,
        "moe_ep_size": 8,
    }

    if fail_during_benchmark:
        with pytest.raises(RuntimeError, match="benchmark failed"):
            benchmark(**kwargs)
    else:
        assert benchmark(**kwargs) == (1.25, {"power": 100.0})

    for module, name, original in originals:
        assert getattr(module, name) is original
