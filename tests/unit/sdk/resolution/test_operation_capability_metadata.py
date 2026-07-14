from __future__ import annotations

from types import ModuleType

import pytest

from aiconfigurator.collector.adapters import ResolvedLazyAdapter
from aiconfigurator.collector.preflight import OperationKind
from aiconfigurator.collector.registry_types import OpEntry, PerfFile
from aiconfigurator.collector.sglang.registry import (
    CUSTOM_ALLREDUCE_LAZY_SPEC,
    DSV4_CSA_CONTEXT_LAZY_SPEC,
    DSV4_CSA_GENERATION_LAZY_SPEC,
    DSV4_HCA_CONTEXT_LAZY_SPEC,
    DSV4_HCA_GENERATION_LAZY_SPEC,
    GEMM_LAZY_SPEC,
    MHC_LAZY_SPEC,
    MOE_LAZY_SPEC,
)
from aiconfigurator.collector.types import (
    FabricRequirement,
    LazyOpEntry,
    ResourceContract,
)
from aiconfigurator.sdk import config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.perf_namespace import perf_namespace
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"


@pytest.fixture(scope="module")
def dsv4_profile_model():
    model_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        attention_dp_size=1,
        cp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        nextn=0,
        workload_distribution="power_law",
        moe_backend=None,
    )
    return get_model(_MODEL_ARTIFACT, model_config, backend_name="sglang")


def _reachable_capabilities(model):
    return tuple(
        capability
        for operation in (*model.context_ops, *model.generation_ops)
        for capability in operation.resolution_capabilities()
    )


def test_frozen_dsv4_profile_classifies_every_reachable_operation(dsv4_profile_model) -> None:
    capabilities = _reachable_capabilities(dsv4_profile_model)

    assert capabilities
    assert not [capability for capability in capabilities if capability.kind is OperationKind.UNSUPPORTED]
    assert {capability.kind for capability in capabilities} == {
        OperationKind.MEASURED,
        OperationKind.DETERMINISTIC,
        OperationKind.COMPOSITION_ONLY,
    }
    assert {capability.namespace for capability in capabilities if capability.kind is OperationKind.MEASURED} == {
        perf_namespace(str(PerfFile.GEMM)),
        perf_namespace(str(PerfFile.MHC_MODULE)),
        perf_namespace(str(PerfFile.MOE)),
        perf_namespace(str(PerfFile.CUSTOM_ALLREDUCE)),
        perf_namespace(str(PerfFile.DSV4_CSA_CONTEXT_MODULE)),
        perf_namespace(str(PerfFile.DSV4_HCA_CONTEXT_MODULE)),
        perf_namespace(str(PerfFile.DSV4_CSA_GENERATION_MODULE)),
        perf_namespace(str(PerfFile.DSV4_HCA_GENERATION_MODULE)),
    }


def test_capability_walk_is_static_and_does_not_query_operations(
    dsv4_profile_model, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query(*args, **kwargs):
        raise AssertionError("capability preflight must not execute an operation query")

    monkeypatch.setattr(Operation, "query_with_resolution", unexpected_query)

    capabilities = _reachable_capabilities(dsv4_profile_model)

    assert capabilities


def test_unknown_operation_fails_closed_as_unsupported() -> None:
    operation = Operation("unknown", 1.0)

    (capability,) = operation.resolution_capabilities()

    assert capability.operation == "Operation"
    assert capability.kind is OperationKind.UNSUPPORTED
    assert capability.namespace is None


@pytest.mark.parametrize(
    ("lazy_spec", "expected"),
    (
        (GEMM_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (MHC_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (MOE_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (
            CUSTOM_ALLREDUCE_LAZY_SPEC,
            ResourceContract(
                4,
                FabricRequirement.NVLINK,
                reserve_fabric_domain=True,
            ),
        ),
        (DSV4_CSA_CONTEXT_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (DSV4_HCA_CONTEXT_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (DSV4_CSA_GENERATION_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
        (DSV4_HCA_GENERATION_LAZY_SPEC, ResourceContract(1, FabricRequirement.NONE)),
    ),
)
def test_frozen_routes_declare_shape_independent_preflight_resources(lazy_spec, expected: ResourceContract) -> None:
    assert lazy_spec.preflight_resource == expected


def test_runtime_resource_contract_cannot_drift_from_static_preflight() -> None:
    namespace = perf_namespace(str(PerfFile.GEMM))
    declared = ResourceContract(
        4,
        FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    )
    lazy = LazyOpEntry(
        namespace=namespace,
        run_module="aiconfigurator.collector.fake_runner",
        run_func="run_case",
        adapter_module="aiconfigurator.collector.fake_adapter",
        case_func="request_to_case",
        result_func="result_to_record",
        resource_func="resource_for_request",
        protocol_revision="cuda-event-v1",
        timer="cuda_event",
        tuning_revision="fake-v1",
        preflight_resource=declared,
    )
    route = ResolvedLazyAdapter(
        entry=OpEntry(
            op="gemm",
            module="aiconfigurator.collector.fake_runner",
            get_func="get_cases",
            run_func="run_case",
            perf_filename=PerfFile.GEMM,
            lazy=lazy,
        ),
        lazy=lazy,
        collector_module="aiconfigurator.collector.fake_runner",
        backend="sglang",
        backend_version="0.5.10",
        adapter_module=ModuleType("aiconfigurator.collector.fake_adapter"),
        _case_func=lambda request: dict(request.query),
        _resource_func=lambda request, case: ResourceContract(1, FabricRequirement.NONE),
        _result_func=lambda request, case, result: result,
    )
    environment = MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions={"sglang": "0.5.10"},
    )
    query = {"m": 128, "n": 256, "k": 64}
    protocol = MeasurementProtocol(
        revision="cuda-event-v1",
        warmups=2,
        samples=3,
        timer="cuda_event",
        tuning_revision="fake-v1",
    )
    request = MeasurementRequest(
        op_id="gemm",
        key=PerfKey.build(namespace, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={},
        protocol=protocol,
    )

    with pytest.raises(ValueError, match="preflight resource contract"):
        route.prepare(request)
