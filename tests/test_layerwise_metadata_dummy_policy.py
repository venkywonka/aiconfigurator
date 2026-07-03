from __future__ import annotations

import argparse
import collections.abc
import importlib
import inspect
import logging
import os
import stat
import subprocess
import sys
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

from collector.layerwise.fpm.datapoint_generator import FpmCase
from collector.layerwise.fpm.docker import build_collect_command
from collector.layerwise.vllm import datapoint_generator as vllm_datapoints
from collector.layerwise.vllm import engine as vllm_engine

ROOT = Path(__file__).resolve().parents[1]
OUTER_DRIVER = ROOT / "collector/layerwise/reproduce_layerwise_fpm.sh"
INNER_FPM_DRIVER = ROOT / "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"
_POLICY_ENV_KEYS = {
    "AIC_MODEL_MODE",
    "AIC_MODEL_METADATA_DIR",
    "AIC_LOAD_FORMAT",
    "LOAD_FORMAT",
    "MOE_REAL_ROUTER",
    "PHYSICAL_TP_REAL_WEIGHTS",
    "FPM_REAL_WORKLOAD_SHAPE_SOURCE",
    "FPM_SHAPE_SOURCE",
    "REAL_WORKLOAD_SHAPE_SOURCE",
    "HF_TOKEN",
    "HF_TOKEN_FILE",
}


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


def _clean_subprocess_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _POLICY_ENV_KEYS}


def _outer_driver_shell_function(name: str) -> str:
    script = OUTER_DRIVER.read_text()
    start = script.index(f"{name}() {{")
    end = script.index("\n}\n", start) + 3
    return script[start:end]


def _run_outer_driver(
    tmp_path: Path,
    bundle: Path,
    *,
    stage: str = "layerwise",
    overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    out_root = tmp_path / "outer-run"
    env = {
        **_clean_subprocess_environment(),
        "AIC_REPO": str(ROOT),
        "AIC_MODEL_MODE": "metadata_dummy",
        "AIC_MODEL_METADATA_DIR": str(bundle),
        "DRY_RUN": "1",
        "FORCE": "1",
        "MODEL_SLUG": "metadata",
        "OUT_ROOT": str(out_root),
        "PARETO_CONCURRENCY": "1",
        "PREFLIGHT_IMAGE_CHECK": "0",
        "STAGES": stage,
    }
    if overrides:
        env.update(overrides)
    if stage == "attribute":
        sqlite_dir = out_root / "fpm/metadata/c1/attribute/nsys"
        sqlite_dir.mkdir(parents=True)
        (sqlite_dir / "dry-run.sqlite").touch()
    return subprocess.run(
        ["bash", str(OUTER_DRIVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_inner_driver(
    tmp_path: Path,
    bundle: Path,
    *,
    args: list[str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **_clean_subprocess_environment(),
        "AIC_MODEL_MODE": "metadata_dummy",
        "AIC_MODEL_METADATA_DIR": str(bundle),
        "DRY_RUN": "1",
        "MODEL": str(bundle),
        "REAL_WORKLOAD": "1",
        "RUN_DIR": str(tmp_path / "inner-run"),
        "SKIP_REQUESTS": "1",
        "TP_SIZE": "1",
    }
    if overrides:
        env.update(overrides)
    return subprocess.run(
        ["bash", str(INNER_FPM_DRIVER), *(args or [])],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _enable_metadata_mode(monkeypatch: pytest.MonkeyPatch, bundle: Path) -> None:
    for key in _POLICY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AIC_MODEL_MODE", "metadata_dummy")
    monkeypatch.setenv("AIC_MODEL_METADATA_DIR", str(bundle))


def _fpm_args(model: str, *, extra_vllm_arg: list[str] | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        phases="context,decode,mixed",
        contexts="128",
        context_repeats="1",
        decode_batches="1",
        decode_osl="8",
        decode_repeats=1,
        include_sweep=False,
        real_workload=True,
        real_workload_requests=4,
        real_workload_concurrency=1,
        real_workload_dataset="OpenAssistant/oasst1",
        real_workload_shape_source="scaled_dataset",
        real_workload_isl_min=128,
        real_workload_isl_max=1024,
        real_workload_isl_mean=512,
        real_workload_osl_min=8,
        real_workload_osl_max=64,
        real_workload_osl_mean=32,
        request_allow_failures=0,
        prompt_token_mode="safe_ascii",
        image="image",
        warmup_requests=None,
        gpus=None,
        keep_running=False,
        dry_run=True,
        extra_vllm_arg=extra_vllm_arg or [],
        expected_vllm_version=None,
        allow_version_mismatch=False,
        nsys_profile_worker=False,
        nsys_full_worker=False,
        nsys_cuda_profiler_window=None,
    )


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
    policy = module.validate_metadata_dummy_environment(_create_metadata_bundle(tmp_path / "bundle"))

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


@pytest.mark.parametrize(
    "violation",
    [
        "missing_bundle",
        "mismatched_model",
        "non_dummy_load_format",
        "real_router",
        "physical_real_weights",
        "hf_token",
    ],
)
def test_outer_driver_rejects_metadata_policy_violations_before_preflight(
    tmp_path: Path,
    violation: str,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    overrides: dict[str, str] = {}
    sentinel = "never-echo-outer-driver-secret"
    if violation == "missing_bundle":
        overrides["AIC_MODEL_METADATA_DIR"] = str(tmp_path / "missing")
    elif violation == "mismatched_model":
        other = tmp_path / "other-bundle"
        _create_metadata_bundle(other)
        overrides["MODEL"] = str(other)
    elif violation == "non_dummy_load_format":
        overrides["LOAD_FORMAT"] = "auto"
    elif violation == "real_router":
        overrides["MOE_REAL_ROUTER"] = "1"
    elif violation == "physical_real_weights":
        overrides["PHYSICAL_TP_REAL_WEIGHTS"] = "1"
    elif violation == "hf_token":
        overrides["HF_TOKEN"] = sentinel

    result = _run_outer_driver(tmp_path, bundle, overrides=overrides)
    transcript = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Preflight" not in transcript
    assert "docker run" not in transcript
    assert "python3 -m collector.layerwise" not in transcript
    assert sentinel not in transcript


@pytest.mark.parametrize("stage", ["layerwise", "fpm", "attribute"])
def test_outer_driver_metadata_mode_transcript_is_local_offline_dummy_and_secret_free(
    tmp_path: Path,
    stage: str,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)

    result = _run_outer_driver(tmp_path, bundle, stage=stage)
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert "HF_HUB_OFFLINE=1" in transcript
    assert "TRANSFORMERS_OFFLINE=1" in transcript
    assert "HF_DATASETS_OFFLINE=1" in transcript
    assert "--load-format=dummy" in transcript or "--load-format dummy" in transcript
    if stage == "layerwise":
        assert f"{bundle.resolve()}:/aic-model-metadata:ro" in transcript
        assert '--model "/aic-model-metadata"' in transcript
    else:
        assert f"--model {bundle.resolve()}" in transcript
        assert "--real-workload-shape-source synthetic" in transcript
    assert "HF_TOKEN=" not in transcript
    assert "HF_TOKEN_FILE" not in transcript
    assert "hf.token" not in transcript


def test_outer_layerwise_metadata_mode_uses_private_job_cache_and_shape_policy_env(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    ordinary_hf_home = tmp_path / "ordinary-hf-home"
    ordinary_hf_home.mkdir()
    sentinel = "never-expose-outer-hf-home-secret"
    (ordinary_hf_home / "token").write_text(sentinel)
    (ordinary_hf_home / "config.json").write_text(sentinel)

    result = _run_outer_driver(
        tmp_path,
        bundle,
        stage="layerwise",
        overrides={"HF_HOME": str(ordinary_hf_home)},
    )
    transcript = result.stdout + result.stderr
    job_caches = list((tmp_path / "outer-run").glob("metadata-hf-cache.*"))

    assert result.returncode == 0, transcript
    assert "FPM_REAL_WORKLOAD_SHAPE_SOURCE=synthetic" in transcript
    assert str(ordinary_hf_home) not in transcript
    assert sentinel not in transcript
    assert len(job_caches) == 1
    job_cache = job_caches[0]
    assert f"{job_cache}:/hf-cache" in transcript
    assert job_cache.is_dir()
    assert stat.S_IMODE(job_cache.stat().st_mode) == 0o700
    assert not (job_cache / "token").exists()
    assert not (job_cache / "config.json").exists()


def test_outer_fpm_metadata_mode_preserves_whitespace_in_bundle_path_when_executing(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "metadata bundle with spaces"
    _create_metadata_bundle(bundle)
    capture = tmp_path / "captured-model-dir.txt"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    python_wrapper = fake_bin / "python3"
    python_wrapper.write_text(
        f"""#!{sys.executable}
import os
from pathlib import Path
import sys

REAL_PYTHON = {sys.executable!r}
if sys.argv[1:3] == [\"-m\", \"collector.layerwise.fpm.collect\"]:
    actual = os.environ.get(\"AIC_MODEL_METADATA_DIR\", \"\")
    expected = os.environ[\"EXPECTED_METADATA_MODEL_DIR\"]
    if actual != expected:
        print(f\"metadata path mismatch: {{actual!r}} != {{expected!r}}\", file=sys.stderr)
        raise SystemExit(71)
    Path(os.environ[\"METADATA_MODEL_CAPTURE\"]).write_text(actual)
    raise SystemExit(0)
os.execv(REAL_PYTHON, [REAL_PYTHON, *sys.argv[1:]])
"""
    )
    python_wrapper.chmod(0o755)

    result = _run_outer_driver(
        tmp_path,
        bundle,
        stage="fpm",
        overrides={
            "DRY_RUN": "0",
            "EXPECTED_METADATA_MODEL_DIR": str(bundle.resolve()),
            "METADATA_MODEL_CAPTURE": str(capture),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        },
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert capture.read_text() == str(bundle.resolve())


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="requires deterministic ENOSPC sink")
def test_outer_run_env_propagates_log_failure_in_checked_context(tmp_path: Path) -> None:
    run_env = _outer_driver_shell_function("run_env")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -euo pipefail
DRY_RUN=0
C_DIM=
C_OFF=
die() {{ printf 'die: %s\\n' "$*" >&2; exit 1; }}
{run_env}
if run_env /dev/full -- bash -c 'printf payload'; then
  observed_rc=0
else
  observed_rc=$?
fi
printf '\\nobserved_rc=%s\\n' "$observed_rc"
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "observed_rc=1" in result.stdout


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="requires deterministic ENOSPC sink")
def test_outer_run_propagates_log_failure_before_profile_can_mark_done(tmp_path: Path) -> None:
    run = _outer_driver_shell_function("run")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -euo pipefail
DRY_RUN=0
C_DIM=
C_OFF=
{run}
if run /dev/full bash -c 'printf payload'; then
  observed_rc=0
else
  observed_rc=$?
fi
printf '\\nobserved_rc=%s\\n' "$observed_rc"
""",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "observed_rc=1" in result.stdout


def test_inner_fpm_driver_metadata_mode_stages_local_model_and_forces_offline_dummy(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)

    result = _run_inner_driver(tmp_path, bundle)
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert (tmp_path / "inner-run/model/config.json").is_file()
    assert "--model /work/model" in transcript
    assert "HF_HUB_OFFLINE=1" in transcript
    assert "TRANSFORMERS_OFFLINE=1" in transcript
    assert "HF_DATASETS_OFFLINE=1" in transcript
    assert "--load-format=dummy" in transcript or "--load-format dummy" in transcript
    assert "shape_source=synthetic" in transcript
    assert "HF_TOKEN=" not in transcript
    assert "HF_TOKEN_FILE" not in transcript
    assert "hf.token" not in transcript


def test_inner_fpm_metadata_mode_pins_synthetic_shape_seed_without_outer_driver(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)

    result = _run_inner_driver(
        tmp_path,
        bundle,
        overrides={
            "REAL_WORKLOAD_REQUESTS": "1",
            "SKIP_REQUESTS": "0",
            "WARMUP_REQUESTS": "0",
        },
    )
    transcript = result.stdout + result.stderr

    assert result.returncode == 0, transcript
    assert " --real-workload-shape-source synthetic --real-workload-isl-min " in transcript
    # The real-workload segment adds its stable 2e9 offset to the metadata-mode base seed 0.
    assert " --prompt-token-seed 2000000000 --real-workload " in transcript


def test_inner_fpm_metadata_mode_uses_private_job_cache_for_worker_and_requests(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    ordinary_hf_home = tmp_path / "ordinary-hf-home"
    ordinary_hf_home.mkdir()
    sentinel = "never-expose-inner-hf-home-secret"
    (ordinary_hf_home / "token").write_text(sentinel)
    (ordinary_hf_home / "config.json").write_text(sentinel)

    result = _run_inner_driver(
        tmp_path,
        bundle,
        overrides={
            "HF_HOME": str(ordinary_hf_home),
            "REAL_WORKLOAD_REQUESTS": "1",
            "REAL_WORKLOAD_CONCURRENCY": "1",
            "SKIP_REQUESTS": "0",
            "WARMUP_REQUESTS": "0",
        },
    )
    transcript = result.stdout + result.stderr
    job_caches = list((tmp_path / "inner-run").glob("metadata-hf-cache.*"))

    assert result.returncode == 0, transcript
    assert str(ordinary_hf_home) not in transcript
    assert sentinel not in transcript
    assert len(job_caches) == 1
    job_cache = job_caches[0]
    assert transcript.count(f"{job_cache}:/work/hf-home") >= 2
    assert job_cache.is_dir()
    assert stat.S_IMODE(job_cache.stat().st_mode) == 0o700
    assert not (job_cache / "token").exists()
    assert not (job_cache / "config.json").exists()


@pytest.mark.parametrize(
    "worker_args",
    [
        ["--", "--load-format=auto"],
        ["--", "--load-format", "auto"],
        ["--", "--load-format=dummy", "--load-format=auto"],
    ],
)
def test_inner_fpm_driver_metadata_mode_rejects_non_dummy_worker_args_before_launch(
    tmp_path: Path,
    worker_args: list[str],
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)

    result = _run_inner_driver(tmp_path, bundle, args=worker_args)
    transcript = result.stdout + result.stderr

    assert result.returncode != 0
    assert "Starting frontend container" not in transcript
    assert "docker run" not in transcript


def test_inner_fpm_driver_rejects_secret_without_echo_before_launch(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    sentinel = "never-echo-inner-driver-secret"

    result = _run_inner_driver(tmp_path, bundle, overrides={"HF_TOKEN": sentinel})
    transcript = result.stdout + result.stderr

    assert result.returncode != 0
    assert "docker run" not in transcript
    assert sentinel not in transcript


def test_real_weight_resolution_fails_before_snapshot_download_in_metadata_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)
    fake_hub = ModuleType("huggingface_hub")

    def fail_snapshot_download(model: str) -> str:
        pytest.fail(f"snapshot_download must not be called for {model}")

    fake_hub.snapshot_download = fail_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(module.MetadataDummyPolicyError) as exc_info:
        vllm_datapoints._resolve_real_weight_model_dir("remote/repository")

    assert exc_info.value.reason == "forbidden_override"


@pytest.mark.parametrize(
    ("moe_real_router", "physical_tp_real_weights", "physical_tp"),
    [(True, False, False), (False, True, True)],
)
def test_real_weight_paths_fail_before_model_config_resolution_in_metadata_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    moe_real_router: bool,
    physical_tp_real_weights: bool,
    physical_tp: bool,
) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)
    monkeypatch.setattr(
        vllm_datapoints,
        "_load_original_config",
        lambda model: pytest.fail(f"model config resolution reached for {model}"),
    )
    args = SimpleNamespace(
        model=str(bundle),
        ctx_new_tokens="1",
        ctx_past_kv="0",
        ctx_batch_sizes="1",
        gen_batch_sizes="1",
        gen_past_kv="1",
        tp_sizes="1",
        max_num_seqs=None,
        max_num_batched_tokens=None,
        max_model_len=None,
        gpu_memory_utilization=0.9,
        physical_tp=physical_tp,
        allow_multi_gpu_diagnostic=physical_tp,
        moe_real_router=moe_real_router,
        physical_tp_real_weights=physical_tp_real_weights,
    )

    with pytest.raises(module.MetadataDummyPolicyError) as exc_info:
        vllm_datapoints.build_work_units(args)

    assert exc_info.value.reason == "forbidden_override"


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--load-format=auto"],
        ["--load-format", "auto"],
        ["--load-format=dummy", "--load-format", "auto"],
    ],
)
def test_vllm_engine_rejects_non_dummy_load_format_in_metadata_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)

    with pytest.raises(module.MetadataDummyPolicyError) as exc_info:
        vllm_engine._engine_tokens(
            model_dir=str(bundle),
            datapoints=[],
            extra_vllm_args=extra_args,
        )

    assert exc_info.value.reason == "forbidden_override"


def test_vllm_engine_metadata_mode_forces_a_single_dummy_load_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)

    tokens = vllm_engine._engine_tokens(
        model_dir=str(bundle),
        datapoints=[],
        extra_vllm_args=["--load-format", "dummy"],
    )

    load_tokens = [token for token in tokens if token == "--load-format" or token.startswith("--load-format=")]
    assert load_tokens == ["--load-format=dummy"]


def test_vllm_engine_metadata_mode_accepts_config_derived_from_validated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    derived = tmp_path / "derived-config"
    _create_metadata_bundle(derived)
    _enable_metadata_mode(monkeypatch, bundle)

    tokens = vllm_engine._engine_tokens(
        model_dir=str(derived),
        datapoints=[],
        extra_vllm_args=[],
    )

    assert tokens[tokens.index("--model") + 1] == str(derived)
    assert tokens[-1] == "--load-format=dummy"


def test_vllm_engine_public_mode_keeps_explicit_load_format_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _POLICY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    tokens = vllm_engine._engine_tokens(
        model_dir="remote/repository",
        datapoints=[],
        extra_vllm_args=["--load-format", "auto"],
    )

    assert tokens[-2:] == ["--load-format", "auto"]
    assert "--load-format=dummy" not in tokens


@pytest.mark.parametrize(
    "script",
    [
        ROOT / "collector/layerwise/vllm/datapoint_generator.py",
        ROOT / "collector/layerwise/vllm/engine.py",
    ],
)
def test_metadata_enforcement_preserves_direct_script_import_compatibility(script: Path) -> None:
    env = _clean_subprocess_environment()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_fpm_command_metadata_mode_forces_local_offline_dummy_synthetic_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)

    command = build_collect_command(
        _fpm_args(str(bundle)),
        FpmCase(tp_size=1, ep_size=1, decode_past_kv=1024),
        tmp_path / "run",
    )

    assert command.argv[command.argv.index("--model") + 1] == str(bundle.resolve())
    assert command.argv[command.argv.index("--real-workload-shape-source") + 1] == "synthetic"
    passthrough = command.argv[command.argv.index("--") + 1 :]
    assert passthrough == ["--load-format=dummy"]
    assert command.env == {
        "AIC_LOAD_FORMAT": "dummy",
        "AIC_MODEL_METADATA_DIR": str(bundle.resolve()),
        "AIC_MODEL_MODE": "metadata_dummy",
        "FPM_REAL_WORKLOAD_SHAPE_SOURCE": "synthetic",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


@pytest.mark.parametrize(
    "extra_args",
    [["--load-format=auto"], ["--load-format", "auto"]],
)
def test_fpm_command_metadata_mode_rejects_explicit_non_dummy_load_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
) -> None:
    module = _pipeline_policy_module()
    bundle = tmp_path / "bundle"
    _create_metadata_bundle(bundle)
    _enable_metadata_mode(monkeypatch, bundle)

    with pytest.raises(module.MetadataDummyPolicyError) as exc_info:
        build_collect_command(
            _fpm_args(str(bundle), extra_vllm_arg=extra_args),
            FpmCase(tp_size=1, ep_size=1, decode_past_kv=1024),
            tmp_path / "run",
        )

    assert exc_info.value.reason == "forbidden_override"
