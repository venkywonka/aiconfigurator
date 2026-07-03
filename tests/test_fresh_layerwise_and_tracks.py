# tests/test_fresh_layerwise_and_tracks.py
"""Covers two changes to the layerwise<->FPM gap path:
  - aic_fpm_gap.make_tracks no longer emits the layerwise_cal0066 calibration lane.
  - aic_fpm_attribute._main accepts --layerwise-csv and threads it into build_model_and_db
    (so the nsys decomposition grades against the run's FRESH layerwise.csv, not shipped data),
    falling back to the shipped systems-data path when the flag is omitted.
"""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.unit

# aic_fpm_attribute pulls in the step marker (torch) transitively via the diagnostics package;
# stub torch like the sibling tests so the pure-CLI path is importable without a GPU.
os.environ.setdefault("LAYERWISE_STEP_MARKER", "0")
sys.modules.setdefault("torch", mock.Mock())
sys.modules.setdefault("torch.cuda", mock.Mock())
sys.modules.setdefault("torch.cuda.nvtx", mock.Mock())


def test_make_tracks_drops_calibration_lane():
    import collector.layerwise.diagnostics.aic_fpm_gap as G

    names = [t[0] for t in G.make_tracks("0.20.1", "0.19.0")]
    assert "layerwise_cal0066" not in names
    # the five real tracks survive, in order
    assert names == ["layerwise", "opwise_silicon", "empirical", "hybrid", "sol"]
    # the headline layerwise track still forces cal off (decode_cal_override == 0.0)
    layerwise = next(t for t in G.make_tracks("0.20.1", "0.19.0") if t[0] == "layerwise")
    assert layerwise[-1] == 0.0


class _StopAfterBuildError(Exception):
    """Raised by the fake build_model_and_db to halt _main right after the call we assert on."""


def _run_main_capturing_layerwise_csv(argv):
    """Invoke aic_fpm_attribute._main with the real aic_fpm_gap (G) but its repo-touching
    surface patched: _import_repo returns a fake api, DEFAULT_REPO_ROOT is pinned, and
    build_model_and_db records the layerwise_csv kwarg then aborts before any real SDK/FPM
    work. _main does `import ... aic_fpm_gap as G` internally, so we patch attributes on the
    actual module object rather than swapping sys.modules. Returns the recorded path string."""
    import collector.layerwise.diagnostics.aic_fpm_attribute as A
    import collector.layerwise.diagnostics.aic_fpm_gap as G

    captured = {}

    def fake_build(*args, **kwargs):
        captured["layerwise_csv"] = kwargs.get("layerwise_csv")
        raise _StopAfterBuildError

    fake_api = {
        "VLLMBackend": mock.Mock(),
        "vllm_backend": mock.Mock(),
        "RuntimeConfig": mock.Mock(),
    }

    with mock.patch.object(G, "DEFAULT_REPO_ROOT", "/repo"), \
            mock.patch.object(G, "_import_repo", return_value=fake_api), \
            mock.patch.object(
                G,
                "resolve_and_verify_runtime_config",
                return_value={
                    "tp": 8,
                    "vllm_max_num_batched_tokens": 40960,
                    "vllm_max_num_seqs": 256,
                    "config_source": "test",
                    "config_mismatch": False,
                    "config_path": "/run/effective_vllm_config.json",
                },
            ), \
            mock.patch.object(G, "build_model_and_db", side_effect=fake_build), pytest.raises(_StopAfterBuildError):
        A._main(argv)
    return captured["layerwise_csv"]


def test_attribute_uses_explicit_layerwise_csv():
    fresh = "/out/layerwise/Qwen-Qwen3-32B/layerwise.csv"
    got = _run_main_capturing_layerwise_csv([
        "--sqlite", "x.sqlite", "--fpm-run", "/run", "--model", "Qwen/Qwen3-32B",
        "--system", "h100_sxm", "--tp", "8", "--out", "/out/decomposition.csv",
        "--layerwise-csv", fresh,
    ])
    assert got == fresh


def test_attribute_falls_back_to_shipped_when_flag_omitted():
    got = _run_main_capturing_layerwise_csv([
        "--sqlite", "x.sqlite", "--fpm-run", "/run", "--model", "Qwen/Qwen3-32B",
        "--system", "h100_sxm", "--tp", "8", "--out", "/out/decomposition.csv",
    ])
    # default resolves to the shipped systems-data layerwise_perf.csv under DEFAULT_REPO_ROOT
    assert got == str(
        Path("/repo") / "src/aiconfigurator/systems/data/h100_sxm/vllm/0.20.1/layerwise_perf.csv"
    )
