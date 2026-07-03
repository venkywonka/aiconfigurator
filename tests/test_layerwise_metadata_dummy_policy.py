from __future__ import annotations

import collections.abc
import importlib
import inspect
import logging
import types
import typing
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Mapping

import pytest


def _pipeline_policy_module() -> ModuleType:
    return importlib.import_module("collector.layerwise.common.pipeline_policy")


def _metadata_dummy_environment(model_dir: Path) -> dict[str, str]:
    return {
        "AIC_MODEL_MODE": "metadata_dummy",
        "AIC_MODEL_METADATA_DIR": str(model_dir),
    }


def _create_metadata_bundle(model_dir: Path) -> dict[str, str]:
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    return _metadata_dummy_environment(model_dir)


def _assert_policy_error(
    module: ModuleType,
    environ: Mapping[str, str],
    reason: str,
) -> BaseException:
    with pytest.raises(module.MetadataDummyPolicyError) as exc_info:
        module.validate_metadata_dummy_environment(environ)

    error = exc_info.value
    message = str(error)
    assert message.strip()
    assert error.reason == reason
    return error


def test_valid_metadata_bundle_returns_immutable_offline_dummy_policy(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    parent = tmp_path / "parent"
    parent.mkdir()
    bundle = parent / "bundle"
    _create_metadata_bundle(bundle)
    detour = parent / "detour"
    detour.mkdir()
    environ = _metadata_dummy_environment(detour / ".." / "bundle")

    policy = module.validate_metadata_dummy_environment(environ)

    assert isinstance(policy, module.MetadataDummyPolicy)
    assert policy.model_dir == bundle.resolve()
    assert policy.load_format == "dummy"
    assert policy.shape_source == "synthetic"
    assert policy.offline is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_dir", Path("other-bundle")),
        ("load_format", "auto"),
        ("shape_source", "scaled_dataset"),
        ("offline", False),
    ],
)
def test_metadata_dummy_policy_fields_are_immutable(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    module = _pipeline_policy_module()
    policy = module.validate_metadata_dummy_environment(
        _create_metadata_bundle(tmp_path / "bundle")
    )

    with pytest.raises(AttributeError):
        setattr(policy, field, replacement)


def test_metadata_dummy_accepts_an_immutable_environment_mapping(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    environ = MappingProxyType(_create_metadata_bundle(bundle))

    policy = module.validate_metadata_dummy_environment(environ)

    assert isinstance(policy, module.MetadataDummyPolicy)
    assert policy.model_dir == bundle.resolve()


def test_validator_exposes_the_approved_typed_api() -> None:
    module = _pipeline_policy_module()
    validator = module.validate_metadata_dummy_environment

    signature = inspect.signature(validator)
    hints = typing.get_type_hints(validator)

    bound = signature.bind(environ={})
    assert bound.arguments["environ"] == {}
    environ_hint = hints["environ"]
    assert typing.get_origin(environ_hint) in {typing.Mapping, collections.abc.Mapping}
    assert typing.get_args(environ_hint) == (str, str)
    return_hint = hints["return"]
    assert typing.get_origin(return_hint) in {typing.Union, types.UnionType}
    assert set(typing.get_args(return_hint)) == {module.MetadataDummyPolicy, type(None)}


def test_unset_model_mode_is_a_noop_for_existing_environment() -> None:
    module = _pipeline_policy_module()
    environ = {
        "AIC_MODEL_METADATA_DIR": "Qwen/Qwen3-32B",
        "AIC_LOAD_FORMAT": "auto",
        "HF_TOKEN": "existing-public-mode-token",
    }

    assert module.validate_metadata_dummy_environment(environ=environ) is None


def test_metadata_dummy_rejects_missing_bundle_environment_variable() -> None:
    module = _pipeline_policy_module()

    _assert_policy_error(
        module,
        {"AIC_MODEL_MODE": "metadata_dummy"},
        "missing_metadata_dir",
    )


def test_metadata_dummy_rejects_nonexistent_bundle_path(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    missing = tmp_path / "missing"

    _assert_policy_error(
        module,
        _metadata_dummy_environment(missing),
        "invalid_metadata_dir",
    )


def test_metadata_dummy_rejects_non_directory_bundle_path(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    bundle_file = tmp_path / "bundle.json"
    bundle_file.write_text("{}")

    _assert_policy_error(
        module,
        _metadata_dummy_environment(bundle_file),
        "invalid_metadata_dir",
    )


def test_metadata_dummy_rejects_symlinked_bundle_root(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    target = tmp_path / "target"
    _create_metadata_bundle(target)
    bundle_link = tmp_path / "bundle-link"
    bundle_link.symlink_to(target, target_is_directory=True)

    _assert_policy_error(
        module,
        _metadata_dummy_environment(bundle_link),
        "invalid_metadata_dir",
    )


def test_metadata_dummy_rejects_bundle_without_config_json(tmp_path: Path) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    _assert_policy_error(
        module,
        _metadata_dummy_environment(bundle),
        "missing_config",
    )


def test_metadata_dummy_rejects_remote_repository_id() -> None:
    module = _pipeline_policy_module()

    _assert_policy_error(
        module,
        _metadata_dummy_environment(Path("Qwen/Qwen3-32B")),
        "invalid_metadata_dir",
    )


@pytest.mark.parametrize(
    "weight_name",
    [
        "weights.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "model-00001-of-00001",
    ],
)
def test_metadata_dummy_rejects_weight_like_bundle_contents(tmp_path: Path, weight_name: str) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    environ = _create_metadata_bundle(bundle)
    (bundle / weight_name).write_text("must not be loaded")

    _assert_policy_error(module, environ, "forbidden_model_file")


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("MOE_REAL_ROUTER", "1"),
        ("PHYSICAL_TP_REAL_WEIGHTS", "1"),
        ("FPM_REAL_WORKLOAD_SHAPE_SOURCE", "scaled_dataset"),
        ("AIC_LOAD_FORMAT", "auto"),
    ],
)
def test_metadata_dummy_rejects_real_data_overrides(tmp_path: Path, variable: str, value: str) -> None:
    module = _pipeline_policy_module()
    environ = _create_metadata_bundle(tmp_path / "bundle")
    environ[variable] = value

    _assert_policy_error(module, environ, "forbidden_override")


@pytest.mark.parametrize("variable", ["HF_TOKEN", "HF_TOKEN_FILE"])
def test_metadata_dummy_rejects_hf_secrets_without_echoing_values(
    tmp_path: Path,
    variable: str,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = _pipeline_policy_module()
    environ = _create_metadata_bundle(tmp_path / "bundle")
    sentinel = f"never-echo-{variable.lower()}-sentinel"
    environ[variable] = sentinel

    caplog.set_level(logging.NOTSET)
    error = _assert_policy_error(module, environ, "forbidden_credential")
    captured = capfd.readouterr()

    assert sentinel not in str(error)
    assert sentinel not in repr(error)
    assert all(sentinel not in str(item) for item in error.args)
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert sentinel not in caplog.text
    assert all(sentinel not in record.getMessage() for record in caplog.records)
