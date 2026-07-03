# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTER_DRIVER = ROOT / "collector/layerwise/reproduce_layerwise_fpm.sh"
INNER_DRIVER = ROOT / "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"
MODULE_NAME = "collector.layerwise.common.nsight_arch"


def _nsight_arch():
    return importlib.import_module(MODULE_NAME)


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("x86_64", ("target-linux-x64", "host-linux-x64")),
        ("aarch64", ("target-linux-sbsa-armv8", "host-linux-armv8")),
    ],
)
def test_nsight_layout_has_exact_supported_mappings(
    architecture: str,
    expected: tuple[str, str],
) -> None:
    assert _nsight_arch().nsight_layout(architecture) == expected


def test_nsight_layout_rejects_unsupported_architecture() -> None:
    module = _nsight_arch()

    with pytest.raises(module.NsightArchitectureError, match=r"unsupported.*ppc64le"):
        module.nsight_layout("ppc64le")


def test_registry_architecture_must_match_allocated_uname() -> None:
    module = _nsight_arch()

    with pytest.raises(
        module.NsightArchitectureError,
        match=r"expected.*aarch64.*observed.*x86_64",
    ):
        module.validate_nsight_architecture("aarch64", "x86_64")


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ("target", r"target.*missing"),
        ("importer", r"importer.*missing"),
        ("binary", r"nsys.*missing"),
    ],
)
def test_nsight_install_requires_arch_specific_directories_and_binary(
    tmp_path: Path,
    missing: str,
    reason: str,
) -> None:
    module = _nsight_arch()
    target_name, importer_name = module.nsight_layout("aarch64")
    target = tmp_path / target_name
    importer = tmp_path / importer_name
    binary = tmp_path / "bin/nsys"
    if missing != "target":
        target.mkdir(parents=True)
    if missing != "importer":
        importer.mkdir(parents=True)
    if missing != "binary":
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/usr/bin/env bash\nexit 0\n")
        binary.chmod(0o755)

    with pytest.raises(module.NsightInstallationError, match=reason):
        module.validate_nsight_install(tmp_path, "aarch64")


def _fake_uname(tmp_path: Path, architecture: str) -> Path:
    fake_bin = tmp_path / f"fake-bin-{architecture}"
    fake_bin.mkdir()
    uname = fake_bin / "uname"
    uname.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' {architecture!r}\n")
    uname.chmod(0o755)
    return fake_bin


def _outer_dry_run(
    tmp_path: Path,
    *,
    stage: str,
    expected_architecture: str,
    observed_architecture: str,
    latency_source: str | None = None,
    runtime: str = "enroot",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_uname(tmp_path, observed_architecture)
    env = {
        **os.environ,
        "AIC_NODE_ARCH": expected_architecture,
        "DOCKER_BIN": str(tmp_path / "missing-docker"),
        "DRY_RUN": "1",
        "ENROOT_BIN": str(tmp_path / "missing-enroot"),
        "ENROOT_IMAGE_PATH": str(tmp_path / "missing-image.sqsh"),
        "MODEL": "Qwen/Qwen3-0.6B",
        "NSYS_ROOT": str(tmp_path / "nsight-root"),
        "OUT_ROOT": str(tmp_path / f"output-{stage}"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "RUNTIME": runtime,
        "SETSID_BIN": str(tmp_path / "missing-setsid"),
        "STAGES": stage,
    }
    if latency_source is not None:
        env["LW_LATENCY_SOURCE"] = latency_source
    if extra_env is not None:
        env.update(extra_env)
    env.pop("AIC_MODEL_MODE", None)
    return subprocess.run(
        ["bash", str(OUTER_DRIVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


@pytest.mark.parametrize("runtime", ("docker", "enroot"))
def test_arm_nsys_profiled_layerwise_uses_arm_host_and_container_layout(
    tmp_path: Path,
    runtime: str,
) -> None:
    result = _outer_dry_run(
        tmp_path,
        stage="layerwise",
        expected_architecture="aarch64",
        observed_architecture="aarch64",
        latency_source="gpu",
        runtime=runtime,
    )
    output = _combined_output(result)
    host_root = tmp_path / "nsight-root"
    container_root = "/opt/nvidia/nsight-systems/2026.3.1"

    assert result.returncode == 0, output
    assert "target-linux-sbsa-armv8" in output
    assert "host-linux-armv8" in output
    assert f"{host_root}/bin:{container_root}/bin" in output
    assert (f'export PATH="{container_root}/bin:{container_root}/target-linux-sbsa-armv8:$PATH"') in output
    assert f"{container_root}/bin/nsys --version" in output
    assert "target-linux-x64" not in output
    assert "host-linux-x64" not in output


@pytest.mark.parametrize(
    ("stage", "latency_source"),
    [("layerwise", "gpu"), ("attribute", None)],
)
def test_nsys_profiled_stages_reject_registry_uname_mismatch(
    tmp_path: Path,
    stage: str,
    latency_source: str | None,
) -> None:
    result = _outer_dry_run(
        tmp_path,
        stage=stage,
        expected_architecture="aarch64",
        observed_architecture="x86_64",
        latency_source=latency_source,
    )

    assert result.returncode != 0
    assert "expected architecture 'aarch64' but observed 'x86_64'" in _combined_output(result)


def test_plain_layerwise_does_not_require_nsight_architecture(tmp_path: Path) -> None:
    result = _outer_dry_run(
        tmp_path,
        stage="layerwise",
        expected_architecture="aarch64",
        observed_architecture="x86_64",
    )
    output = _combined_output(result)

    assert result.returncode == 0, output
    assert "target-linux-" not in output
    assert "host-linux-" not in output
    assert "nsys --version" not in output


def test_plain_fpm_does_not_require_nsight_architecture(tmp_path: Path) -> None:
    result = _outer_dry_run(
        tmp_path,
        stage="fpm",
        expected_architecture="aarch64",
        observed_architecture="x86_64",
    )

    assert result.returncode == 0, _combined_output(result)


@pytest.mark.parametrize("allow_mismatch", (False, True))
def test_attribute_version_mismatch_flags_reach_inner_collector(
    tmp_path: Path,
    allow_mismatch: bool,
) -> None:
    version = "9.9.9"
    result = _outer_dry_run(
        tmp_path,
        stage="attribute",
        expected_architecture="x86_64",
        observed_architecture="x86_64",
        extra_env={
            "ALLOW_VERSION_MISMATCH": "1" if allow_mismatch else "0",
            "VLLM_VERSION": version,
        },
    )
    collect_command = next(
        line for line in _combined_output(result).splitlines() if "python3 -m collector.layerwise.fpm.collect" in line
    )

    if allow_mismatch:
        assert "--allow-version-mismatch" in collect_command
        assert f"--expected-vllm-version {version}" in collect_command
    else:
        assert "--allow-version-mismatch" not in collect_command
        assert "--expected-vllm-version" not in collect_command


def test_normal_attribute_header_only_decomposition_cannot_mark_done(tmp_path: Path) -> None:
    source = OUTER_DRIVER.read_text()
    gate_start = source.index("python3 -m collector.layerwise.diagnostics.assert_attribution_valid")
    mark_done = source.index('mark_done "$unit"', gate_start)
    assert "--allow-empty-decomposition" not in source[gate_start:mark_done]

    database = tmp_path / "profile.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)")
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1)")
        connection.execute("CREATE TABLE NVTX_EVENTS (text TEXT)")
        connection.execute("INSERT INTO NVTX_EVENTS VALUES ('bench_step::1')")
    decomposition = tmp_path / "decomposition.csv"
    decomposition.write_text("shape,gpu_compute_ms\n")
    marker = tmp_path / "attribute.done"
    gate = shlex.join(
        [
            sys.executable,
            "-m",
            "collector.layerwise.diagnostics.assert_attribution_valid",
            "--sqlite",
            str(database),
            "--decomposition",
            str(decomposition),
        ]
    )
    result = subprocess.run(
        ["bash", "-c", f"{gate} && touch {shlex.quote(str(marker))}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "no data rows" in result.stderr
    assert not marker.exists()


def _inner_dry_run(
    tmp_path: Path,
    *,
    profile_worker: bool,
    expected_architecture: str,
    observed_architecture: str,
    include_host_dir: bool = True,
) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_uname(tmp_path, observed_architecture)
    nsight_root = tmp_path / "inner-nsight-root"
    env = {
        **os.environ,
        "AIC_NODE_ARCH": expected_architecture,
        "DRY_RUN": "1",
        "MODEL": "Qwen/Qwen3-0.6B",
        "NSYS_BIN": str(nsight_root / "bin/nsys"),
        "NSYS_HOST_DIR": str(nsight_root),
        "NSYS_PROFILE_WORKER": "1" if profile_worker else "0",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REAL_WORKLOAD": "0",
        "RUN_DIR": str(tmp_path / "inner-output"),
        "SKIP_REQUESTS": "1",
        "TP_SIZE": "1",
    }
    if not include_host_dir:
        env.pop("NSYS_HOST_DIR")
        nsys_binary = nsight_root / "bin/nsys"
        nsys_binary.parent.mkdir(parents=True)
        nsys_binary.write_text("#!/usr/bin/env bash\nexit 0\n")
        nsys_binary.chmod(0o755)
    return subprocess.run(
        ["bash", str(INNER_DRIVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_direct_profiled_fpm_rejects_registry_uname_mismatch(tmp_path: Path) -> None:
    result = _inner_dry_run(
        tmp_path,
        profile_worker=True,
        expected_architecture="aarch64",
        observed_architecture="x86_64",
    )

    assert result.returncode != 0
    assert "expected architecture 'aarch64' but observed 'x86_64'" in _combined_output(result)


def test_direct_arm_profiled_fpm_mounts_paired_architecture_paths(tmp_path: Path) -> None:
    result = _inner_dry_run(
        tmp_path,
        profile_worker=True,
        expected_architecture="aarch64",
        observed_architecture="aarch64",
    )
    output = _combined_output(result)
    host_root = tmp_path / "inner-nsight-root"
    container_root = "/opt/nvidia/nsight-systems"

    assert result.returncode == 0, output
    assert (f"{host_root}/target-linux-sbsa-armv8:{container_root}/target-linux-sbsa-armv8:ro") in output
    assert (f"{host_root}/host-linux-armv8:{container_root}/host-linux-armv8:ro") in output
    assert f"{host_root}/bin:{container_root}/bin:ro" in output
    assert 'export PATH="${nsys_bin_dir}:${nsys_target}:${PATH}"' in output
    assert 'if [ -n "${LD_LIBRARY_PATH:-}" ]' in output
    assert 'export LD_LIBRARY_PATH="${nsys_target}:${nsys_importer}:${LD_LIBRARY_PATH}"' in output
    assert 'export LD_LIBRARY_PATH="${nsys_target}:${nsys_importer}"' in output
    assert f"{container_root}/bin/nsys profile" in output


def test_direct_plain_fpm_does_not_require_nsight_architecture(tmp_path: Path) -> None:
    result = _inner_dry_run(
        tmp_path,
        profile_worker=False,
        expected_architecture="aarch64",
        observed_architecture="x86_64",
    )

    assert result.returncode == 0, _combined_output(result)


def test_direct_profiled_fpm_derives_root_before_architecture_paths(tmp_path: Path) -> None:
    result = _inner_dry_run(
        tmp_path,
        profile_worker=True,
        expected_architecture="aarch64",
        observed_architecture="aarch64",
        include_host_dir=False,
    )
    output = _combined_output(result)
    host_root = tmp_path / "inner-nsight-root"

    assert result.returncode == 0, output
    assert f"{host_root}/target-linux-sbsa-armv8:" in output
    assert f"{host_root}/host-linux-armv8:" in output


def test_executable_shell_paths_do_not_hard_code_x86_nsight_layout() -> None:
    for path in (OUTER_DRIVER, INNER_DRIVER):
        executable_lines = [
            line.split("#", 1)[0] for line in path.read_text().splitlines() if not line.lstrip().startswith("#")
        ]
        executable_source = "\n".join(executable_lines)
        assert "target-linux-x64" not in executable_source, path
        assert "host-linux-x64" not in executable_source, path


def test_outer_resolver_invokes_staged_script_without_package_imports() -> None:
    source = OUTER_DRIVER.read_text()
    resolver = source[source.index("resolve_nsight_architecture()") : source.index("# run <logfile>")]

    assert 'python3 "$SCRIPT_DIR/common/nsight_arch.py"' in resolver
    assert "python3 -m collector.layerwise.common.nsight_arch" not in resolver
    assert "PYTHONPATH=" not in resolver


def test_profiled_loader_paths_never_add_current_directory() -> None:
    outer = OUTER_DRIVER.read_text()
    outer_preamble = outer[
        outer.index("nsys_preamble=$(cat") : outer.index("layerwise_nsys_args=(--nsys-capture full)")
    ]
    inner = INNER_DRIVER.read_text()
    inner_prefix = inner[inner.index("NSYS_COMMAND_PREFIX=(") : inner.index('case "${REQUEST_ENDPOINT}"')]

    assert 'if [[ -n "\\${LD_LIBRARY_PATH:-}" ]]' in outer_preamble
    assert ':\\${LD_LIBRARY_PATH:-}"' not in outer_preamble
    assert 'if [ -n "${LD_LIBRARY_PATH:-}" ]' in inner_prefix
    assert ':${LD_LIBRARY_PATH:-}"' not in inner_prefix


@pytest.mark.parametrize("inherited", (None, "/inherited/lib"))
def test_inner_loader_prefix_executes_under_posix_sh(inherited: str | None) -> None:
    source = INNER_DRIVER.read_text()
    match = re.search(
        r"NSYS_COMMAND_PREFIX=\(\s*sh -c '(?P<body>.*?)' _ ",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    env = os.environ.copy()
    if inherited is None:
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["LD_LIBRARY_PATH"] = inherited
    expected = "/nsys/target:/nsys/importer"
    if inherited is not None:
        expected = f"{expected}:{inherited}"

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            match.group("body"),
            "_",
            "/nsys/bin",
            "/nsys/target",
            "/nsys/importer",
            "/bin/sh",
            "-c",
            'printf "%s" "$LD_LIBRARY_PATH"',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == expected
