# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import spica
import spica.config as config_mod
from spica import SmartSearchConfig

PUBLIC_RESOLUTION_FIELDS = {
    "policy",
    "on_measurement_failure",
    "overlay_path",
    "fallback_cache_dir",
    "max_new_keys",
    "max_wall_seconds",
    "max_block_seconds",
    "force_remeasure",
}


def _search_config(**overrides):
    payload = {
        "search_space": {"model_name": "m", "hardware_sku": "gb200"},
        "workload": {"trace_path": "/tmp/v13-schema-trace.jsonl"},
    }
    if "aic_resolution" in overrides:
        payload["search_space"].update(backend=["sglang"], deployment_mode=["agg"])
    payload.update(overrides)
    return SmartSearchConfig.model_validate(payload)


def test_aic_resolution_is_the_only_public_schema_and_absence_is_pure():
    assert hasattr(config_mod, "AicResolutionConfig")
    assert hasattr(config_mod, "AicResolutionPolicy")
    assert hasattr(config_mod, "MeasurementFailurePolicy")
    assert not hasattr(spica, "LazyCollectionConfig")
    assert not hasattr(spica, "LazyCollectionPolicy")
    assert set(config_mod.AicResolutionConfig.model_fields) == PUBLIC_RESOLUTION_FIELDS

    pure = _search_config()
    assert pure.aic_resolution is None
    assert pure.measurement_gpu_groups is None
    assert "lazy_collection" not in type(pure).model_fields


@pytest.mark.parametrize("also_supplies_aic_resolution", [False, True])
def test_lazy_collection_is_targeted_hard_rejection_without_alias(tmp_path, also_supplies_aic_resolution):
    overrides = {}
    if also_supplies_aic_resolution:
        overrides["aic_resolution"] = {
            "policy": "measure_on_miss",
            "overlay_path": tmp_path / "canonical.sqlite",
        }
    with pytest.raises(ValidationError) as caught:
        _search_config(
            lazy_collection={
                "policy": "measure_on_miss",
                "overlay_path": tmp_path / "legacy.sqlite",
            },
            **overrides,
        )

    message = str(caught.value)
    assert "lazy_collection" in message
    assert "removed" in message
    assert "aic_resolution" in message


def test_aic_resolution_defaults_are_restart_stable_and_runtime_payload_matches_rust(tmp_path):
    overlay = (tmp_path / "evidence.sqlite").resolve()
    payload = {
        "sweep": {"parallel_evals": 2},
        "aic_resolution": {
            "policy": "measure_on_miss",
            "overlay_path": overlay,
        },
        "measurement_gpu_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
    }
    first = _search_config(**payload)
    reopened = _search_config(**payload)

    resolution = first.aic_resolution
    assert resolution is not None
    assert resolution.policy.value == "measure_on_miss"
    assert resolution.on_measurement_failure.value == "error"
    assert resolution.overlay_path == overlay
    assert resolution.fallback_cache_dir == Path(f"{overlay}.live-fallbacks")
    assert resolution.max_new_keys == 256
    assert resolution.max_wall_seconds == 3600.0
    assert resolution.max_block_seconds is None
    assert resolution.force_remeasure is False
    assert reopened.aic_resolution == resolution

    pool = first.measurement_gpu_groups
    assert pool is not None
    lease = pool.lease(1)
    assert lease.gpu_ids == (4, 5, 6, 7)
    assert resolution.runtime_payload(lease) == {
        "policy": "measure_on_miss",
        "on_measurement_failure": "error",
        "overlay_path": str(overlay),
        "fallback_cache_dir": f"{overlay}.live-fallbacks",
        "max_new_keys": 256,
        "max_wall_seconds": 3600.0,
        "force_remeasure": False,
        "gpu_ids": [4, 5, 6, 7],
    }


def test_aic_resolution_lexically_normalizes_paths_without_filesystem_access():
    resolution = config_mod.AicResolutionConfig.model_validate(
        {
            "policy": "measure_on_miss",
            "overlay_path": "/tmp/aic-v13/../evidence.sqlite",
            "fallback_cache_dir": "/tmp/aic-v13-cache/../fallbacks",
        }
    )

    assert resolution.runtime_payload() == {
        "policy": "measure_on_miss",
        "on_measurement_failure": "error",
        "overlay_path": "/tmp/evidence.sqlite",
        "fallback_cache_dir": "/tmp/fallbacks",
        "max_new_keys": 256,
        "max_wall_seconds": 3600.0,
        "force_remeasure": False,
    }


def test_aic_resolution_collapses_double_leading_root_to_match_rust():
    resolution = config_mod.AicResolutionConfig.model_validate(
        {
            "policy": "measure_on_miss",
            "overlay_path": "//nfs/share/aic-v13/../perf.sqlite",
        }
    )

    payload = resolution.runtime_payload()
    assert payload["overlay_path"] == "/nfs/share/perf.sqlite"
    assert payload["fallback_cache_dir"] == "/nfs/share/perf.sqlite.live-fallbacks"


@pytest.mark.parametrize(
    ("search_space", "expected"),
    [
        (
            {"model_name": "m", "hardware_sku": "gb200", "backend": ["trtllm"], "deployment_mode": ["agg"]},
            r"aic_resolution.*backend.*sglang",
        ),
        (
            {"model_name": "m", "hardware_sku": "gb200", "backend": ["sglang"], "deployment_mode": ["disagg"]},
            r"aic_resolution.*deployment_mode.*agg",
        ),
    ],
)
def test_aic_resolution_rejects_unsupported_spica_candidate_profiles(tmp_path, search_space, expected):
    with pytest.raises(ValidationError, match=expected):
        _search_config(
            search_space=search_space,
            aic_resolution={
                "policy": "measure_on_miss",
                "overlay_path": tmp_path / "evidence.sqlite",
            },
        )


def test_explicit_hybrid_deadline_force_and_fallback_fields_serialize_exactly(tmp_path):
    overlay = (tmp_path / "evidence.sqlite").resolve()
    fallback = (tmp_path / "fallbacks").resolve()
    resolution = config_mod.AicResolutionConfig.model_validate(
        {
            "policy": "measure_on_miss",
            "on_measurement_failure": "hybrid",
            "overlay_path": overlay,
            "fallback_cache_dir": fallback,
            "max_new_keys": 17,
            "max_wall_seconds": 45.0,
            "max_block_seconds": 3.5,
            "force_remeasure": True,
        }
    )

    assert resolution.runtime_payload() == {
        "policy": "measure_on_miss",
        "on_measurement_failure": "hybrid",
        "overlay_path": str(overlay),
        "fallback_cache_dir": str(fallback),
        "max_new_keys": 17,
        "max_wall_seconds": 45.0,
        "max_block_seconds": 3.5,
        "force_remeasure": True,
    }


@pytest.mark.parametrize(
    ("resolution", "expected"),
    [
        ({"policy": "observe_only", "on_measurement_failure": "hybrid"}, "observe_only"),
        ({"policy": "measure_on_miss", "fallback_cache_dir": "relative"}, "fallback_cache_dir"),
        ({"policy": "measure_on_miss", "max_new_keys": 0}, "max_new_keys"),
        ({"policy": "measure_on_miss", "max_wall_seconds": 0.0}, "max_wall_seconds"),
        ({"policy": "measure_on_miss", "max_wall_seconds": float("inf")}, "max_wall_seconds"),
        ({"policy": "measure_on_miss", "max_block_seconds": 0.0}, "max_block_seconds"),
        ({"policy": "measure_on_miss", "max_block_seconds": float("inf")}, "max_block_seconds"),
    ],
)
def test_aic_resolution_rejects_invalid_policy_paths_and_budgets(tmp_path, resolution, expected):
    with pytest.raises(ValidationError, match=expected):
        _search_config(
            aic_resolution={
                "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
                **resolution,
            }
        )


@pytest.mark.parametrize(
    "groups",
    [[], [[]], [[-1]], [[0, 0]], [[0, 1], [1, 2]]],
)
def test_measurement_resource_pool_rejects_empty_overlapping_or_invalid_groups(tmp_path, groups):
    with pytest.raises(ValidationError, match="measurement_gpu_groups"):
        _search_config(
            aic_resolution={
                "policy": "measure_on_miss",
                "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
            },
            measurement_gpu_groups=groups,
        )


def test_measurement_pool_and_concrete_lease_cannot_enter_public_policy(tmp_path):
    overlay = (tmp_path / "evidence.sqlite").resolve()
    for forbidden in (
        {"gpu_ids": [0, 1, 2, 3]},
        {"measurement_gpu_groups": [[0, 1, 2, 3]]},
    ):
        with pytest.raises(ValidationError):
            _search_config(
                aic_resolution={
                    "policy": "measure_on_miss",
                    "overlay_path": overlay,
                    **forbidden,
                }
            )

    with pytest.raises(ValidationError, match=r"measurement_gpu_groups.*1 < 2"):
        _search_config(
            sweep={"parallel_evals": 4, "candidates_per_round": 2},
            aic_resolution={"policy": "measure_on_miss", "overlay_path": overlay},
            measurement_gpu_groups=[[0, 1, 2, 3]],
        )
