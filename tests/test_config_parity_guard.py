# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hard config-parity guard tests (closes the AIC-1205 / task-#31 landmine).

AIC-vs-FPM gap/attribution numbers are meaningless if the AIC prediction is computed
at a config that differs from the config the FPM ground truth was collected under.
Task #31 proved a ~5%->~24% error swing (5x) purely from a config mismatch
(predicted TP=8/mnbt=2048/mns=128 vs the run's real TP=4/mnbt=40960/mns=256).

The current `_read_runtime_config` has THREE holes these tests pin closed:
1. Silent fallback to hard-coded mnbt=2048/mns=128 on ANY read failure (warning only).
2. TP never read from the effective config (relies on a hand-passed --tp/TP_VALUES).
3. `effective_config: null` (a real DLC vllm_metadata.json shape) can "read" as success.

The guard `resolve_and_verify_runtime_config(subdir, requested_tp, ...)` reads all three
knobs (tensor_parallel_size, max_num_batched_tokens, max_num_seqs) from the effective
config, cross-checks TP against the caller, FAILS LOUD (ParityError) on the holes, stamps
config provenance into the returned row, and offers a scoped escape hatch that DOWNGRADES a
hard-fail to a loud warning + stamp (default: fail-loud).
"""
from __future__ import annotations

import json
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.layerwise.diagnostics import aic_fpm_gap

# The real, populated GOOD effective config harvested from an aws-dfw GB200 c1 run.
# It carries the flattened dotted keys: parallel_config.tensor_parallel_size=4,
# scheduler_config.max_num_batched_tokens=40960, scheduler_config.max_num_seqs=256.
GOLDEN_GOOD = Path(
    "/home/gvenkatarama/scratch/agent-slop/aic-auto-collector/awsdfw-gb200-pipeline"
    "/harvested/fpm/Qwen-Qwen3-32B/c1/attribute/effective_vllm_config.json"
)


def _write_effective_vllm_config(subdir: Path, *, tp, mnbt, mns) -> None:
    """Write a real-shaped effective_vllm_config.json (flattened dotted keys)."""
    (subdir / "effective_vllm_config.json").write_text(
        json.dumps(
            {
                "parallel_config.tensor_parallel_size": tp,
                "scheduler_config.max_num_batched_tokens": mnbt,
                "scheduler_config.max_num_seqs": mns,
                "vllm_version": "0.20.1",
            }
        )
    )


def _write_vllm_metadata(subdir: Path, *, effective_config) -> None:
    """Write a vllm_metadata.json with the given effective_config (may be None)."""
    (subdir / "vllm_metadata.json").write_text(
        json.dumps(
            {
                "artifact_kind": "fpm",
                "effective_config": effective_config,
                "requested": {
                    # DLC null case had a WRONG requested TP=8 relative to the golden run.
                    "deployment_config": {
                        "tensor_parallel_size": 8,
                        "max_num_batched_tokens": 2048,
                        "max_num_seqs": 256,
                    }
                },
            }
        )
    )


class ParityGuardHoleTests(unittest.TestCase):
    """Each test pins one of the three holes closed (fail-loud, not silent-fallback)."""

    # ---- HOLE #1: silent 2048/128 fallback on a missing config -------------------
    def test_missing_config_file_fails_loud(self):
        """No effective config anywhere -> ParityError, NOT the silent 2048/128 fallback."""
        with self.assertRaises(aic_fpm_gap.ParityError):
            aic_fpm_gap.resolve_and_verify_runtime_config(Path("/nonexistent/subdir"), requested_tp=8)

    def test_missing_config_does_not_return_2048_128(self):
        """Regression: the missing-config path must never yield the bogus 2048/128 constants."""
        import tempfile

        with tempfile.TemporaryDirectory() as td, self.assertRaises(aic_fpm_gap.ParityError):
            aic_fpm_gap.resolve_and_verify_runtime_config(Path(td), requested_tp=8)

    # ---- HOLE #3: effective_config: null (real DLC vllm_metadata.json) -----------
    def test_null_effective_config_fails_loud(self):
        """A present vllm_metadata.json with effective_config: null must fail loud."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_vllm_metadata(subdir, effective_config=None)
            with self.assertRaises(aic_fpm_gap.ParityError):
                aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=8)

    def test_empty_effective_vllm_config_fails_loud(self):
        """An empty ({}) effective_vllm_config.json must fail loud (no knobs to read)."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            (subdir / "effective_vllm_config.json").write_text("{}")
            with self.assertRaises(aic_fpm_gap.ParityError):
                aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=8)

    # ---- missing individual knob -------------------------------------------------
    def test_missing_knob_fails_loud(self):
        """A config present but with NO max_num_batched_tokens anywhere -> fail loud."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            (subdir / "effective_vllm_config.json").write_text(
                json.dumps(
                    {
                        "parallel_config.tensor_parallel_size": 8,
                        "scheduler_config.max_num_seqs": 256,
                        # no max_num_batched_tokens
                    }
                )
            )
            with self.assertRaises(aic_fpm_gap.ParityError):
                aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=8)

    # ---- HOLE #2: TP mismatch (the exact #31 scenario) ---------------------------
    def test_tp_mismatch_fails_loud(self):
        """Effective TP=8 but caller passes requested_tp=4 -> fail loud (the #31 mismatch)."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_effective_vllm_config(subdir, tp=8, mnbt=2048, mns=128)
            with self.assertRaises(aic_fpm_gap.ParityError):
                aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=4)

    def test_tp_is_read_from_config_not_cli_default(self):
        """TP must be READ from parallel_config.tensor_parallel_size, not the CLI default."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_effective_vllm_config(subdir, tp=4, mnbt=40960, mns=256)
            resolved = aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=4)
        self.assertEqual(resolved["tp"], 4)  # read from config, not the requested 8-style default


class ParityGuardGoodPathTests(unittest.TestCase):
    """The happy path: a populated, matching config passes and stamps provenance."""

    def test_good_synthetic_config_passes_and_stamps(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_effective_vllm_config(subdir, tp=4, mnbt=40960, mns=256)
            resolved = aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=4)
        self.assertEqual(resolved["vllm_max_num_batched_tokens"], 40960)
        self.assertEqual(resolved["vllm_max_num_seqs"], 256)
        self.assertEqual(resolved["tp"], 4)
        self.assertEqual(resolved["config_source"], "effective")
        self.assertIn("effective_vllm_config.json", resolved["config_path"])

    def test_golden_aws_dfw_c1_config_passes(self):
        """The REAL aws-dfw c1 effective config (tp=4, mnbt=40960, mns=256) passes as a GOOD fixture."""
        if not GOLDEN_GOOD.exists():
            self.skipTest(f"golden fixture not present: {GOLDEN_GOOD}")
        resolved = aic_fpm_gap.resolve_and_verify_runtime_config(GOLDEN_GOOD.parent, requested_tp=4)
        self.assertEqual(resolved["tp"], 4)
        self.assertEqual(resolved["vllm_max_num_batched_tokens"], 40960)
        self.assertEqual(resolved["vllm_max_num_seqs"], 256)
        self.assertEqual(resolved["config_source"], "effective")

    def test_vllm_metadata_effective_config_is_read_when_no_flat_file(self):
        """When only vllm_metadata.json exists (populated effective_config), it is read."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_vllm_metadata(
                subdir,
                effective_config={
                    "parallel_config.tensor_parallel_size": 4,
                    "scheduler_config.max_num_batched_tokens": 40960,
                    "scheduler_config.max_num_seqs": 256,
                },
            )
            resolved = aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=4)
        self.assertEqual(resolved["tp"], 4)
        self.assertEqual(resolved["vllm_max_num_batched_tokens"], 40960)
        self.assertEqual(resolved["config_source"], "metadata")


class ParityGuardEscapeHatchTests(unittest.TestCase):
    """The scoped escape hatch downgrades a hard-fail to a loud warning + stamp."""

    def test_allow_mismatch_flag_downgrades_tp_mismatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_effective_vllm_config(subdir, tp=8, mnbt=2048, mns=128)
            with self.assertLogs(level=logging.WARNING) as captured:
                resolved = aic_fpm_gap.resolve_and_verify_runtime_config(
                    subdir, requested_tp=4, allow_mismatch=True
                )
        # Downgraded: no raise, loud warning, provenance stamped, config values still surfaced.
        self.assertTrue(captured.output, "expected a WARNING when downgrading a config mismatch")
        self.assertEqual(resolved["tp"], 8)  # the config's real TP, not the requested 4
        self.assertTrue(resolved.get("config_mismatch"))
        self.assertEqual(resolved["config_source"], "effective")

    def test_allow_mismatch_env_var_downgrades(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            _write_effective_vllm_config(subdir, tp=8, mnbt=2048, mns=128)
            os.environ["FPM_ALLOW_CONFIG_MISMATCH"] = "1"
            try:
                with self.assertLogs(level=logging.WARNING):
                    resolved = aic_fpm_gap.resolve_and_verify_runtime_config(subdir, requested_tp=4)
            finally:
                os.environ.pop("FPM_ALLOW_CONFIG_MISMATCH", None)
        self.assertTrue(resolved.get("config_mismatch"))

    def test_allow_mismatch_does_not_rescue_a_missing_config(self):
        """The escape hatch is for a MISMATCH between real configs, not for a MISSING one.

        A missing/null config has no ground-truth config to trust, so the escape hatch must
        NOT silently resurrect the 2048/128 landmine — it still fails loud.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as td, self.assertRaises(aic_fpm_gap.ParityError):
            aic_fpm_gap.resolve_and_verify_runtime_config(
                Path(td), requested_tp=8, allow_mismatch=True
            )


class TwoLaneParityTests(unittest.TestCase):
    """attribute.py drives AIC from the PROFILED lane's config but computes the wall from the
    CLEAN --fpm-run lane; the two lanes must be proven to share one config (codex refutation).
    resolve_and_verify_runtime_config supports this via requested_mnbt/requested_mns cross-checks.
    """

    def test_clean_lane_mnbt_mismatch_vs_profiled_fails_loud(self):
        """Clean lane mnbt=2048 but the profiled lane drove AIC at mnbt=40960 -> fail loud."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            clean = Path(td)
            _write_effective_vllm_config(clean, tp=4, mnbt=2048, mns=256)
            with self.assertRaises(aic_fpm_gap.ParityError):
                # requested_mnbt/mns come from the profiled lane's run_rc.
                aic_fpm_gap.resolve_and_verify_runtime_config(
                    clean, requested_tp=4, requested_mnbt=40960, requested_mns=256
                )

    def test_clean_lane_matching_profiled_passes(self):
        """Clean lane == profiled lane (tp=4, mnbt=40960, mns=256) -> passes, no mismatch."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            clean = Path(td)
            _write_effective_vllm_config(clean, tp=4, mnbt=40960, mns=256)
            resolved = aic_fpm_gap.resolve_and_verify_runtime_config(
                clean, requested_tp=4, requested_mnbt=40960, requested_mns=256
            )
        self.assertFalse(resolved["config_mismatch"])
        self.assertEqual(resolved["vllm_max_num_batched_tokens"], 40960)

    def test_clean_lane_mismatch_downgraded_by_escape_hatch(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            clean = Path(td)
            _write_effective_vllm_config(clean, tp=4, mnbt=2048, mns=256)
            with self.assertLogs(level=logging.WARNING):
                resolved = aic_fpm_gap.resolve_and_verify_runtime_config(
                    clean, requested_tp=4, requested_mnbt=40960, requested_mns=256,
                    allow_mismatch=True,
                )
        self.assertTrue(resolved["config_mismatch"])


class ResolveFpmSourceTupleTests(unittest.TestCase):
    """_resolve_fpm_source returns a 3-tuple (fpm_csv, subdir, peer_subdirs); both callers unpack it."""

    def test_returns_three_tuple_for_flat_tp_layout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fpm_run = Path(td)
            result = aic_fpm_gap._resolve_fpm_source(fpm_run, tp=8, out_dir=fpm_run)
        self.assertEqual(len(result), 3)
        fpm_csv, subdir, peers = result
        self.assertEqual(subdir, fpm_run / "tp8_ep1_past4096")
        self.assertEqual(peers, [subdir])

    def test_returns_all_cdirs_as_peers_for_concurrency_layout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            fpm_run = Path(td)
            model_dir = fpm_run / "fpm" / "Qwen-Qwen3-32B"
            for c in ("c1", "c16"):
                (model_dir / c).mkdir(parents=True)
                (model_dir / c / "fpm_metrics_phase.csv").write_text("step,latency\n")
            fpm_csv, subdir, peers = aic_fpm_gap._resolve_fpm_source(
                fpm_run, tp=8, out_dir=fpm_run
            )
        peer_names = sorted(p.name for p in peers)
        self.assertEqual(peer_names, ["c1", "c16"])


class ResolveConfigLaneTests(unittest.TestCase):
    """attribute._resolve_config_lane layout-resolves a lane so a concurrency root whose config
    lives under fpm/<model>/c1/ is NOT a false-positive hard-fail (codex round-2 finding).
    """

    def _lane(self):
        from collector.layerwise.diagnostics import aic_fpm_attribute as A
        return A._resolve_config_lane, aic_fpm_gap

    def test_flat_run_with_config_at_root_uses_run_dir(self):
        import tempfile

        resolve_lane, gap_mod = self._lane()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_effective_vllm_config(run_dir, tp=4, mnbt=40960, mns=256)
            subdir, peers = resolve_lane(gap_mod, run_dir, 4, run_dir)
        self.assertEqual(subdir, run_dir)
        self.assertEqual(peers, [run_dir])

    def test_concurrency_root_resolves_to_cdir_not_false_positive(self):
        """A concurrency root (config under fpm/<model>/c1/) resolves to the c-dir, then PASSES."""
        import tempfile

        resolve_lane, gap_mod = self._lane()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            model_dir = run_dir / "fpm" / "Qwen-Qwen3-32B"
            for c in ("c1", "c16"):
                (model_dir / c).mkdir(parents=True)
                (model_dir / c / "fpm_metrics_phase.csv").write_text("step,latency\n")
                _write_effective_vllm_config(model_dir / c, tp=4, mnbt=40960, mns=256)
            subdir, peers = resolve_lane(gap_mod, run_dir, 4, run_dir)
            # The resolved subdir carries a real config -> the guard PASSES (no false-positive fail).
            resolved = gap_mod.resolve_and_verify_runtime_config(
                subdir, requested_tp=4, peer_subdirs=peers
            )
        self.assertEqual(resolved["tp"], 4)
        self.assertEqual(resolved["vllm_max_num_batched_tokens"], 40960)


class ReadRuntimeConfigBackCompatTests(unittest.TestCase):
    """`_read_runtime_config` stays as a thin shim but is hardened to fail loud on the holes."""

    def test_flattened_dotted_key_is_read(self):
        """A real-shaped config with the flattened dotted key must be honored (unchanged behavior)."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            (subdir / "effective_vllm_config.json").write_text(
                json.dumps(
                    {
                        "parallel_config.tensor_parallel_size": 8,
                        "scheduler_config.max_num_batched_tokens": 8192,
                        "scheduler_config.max_num_seqs": 16,
                    }
                )
            )
            result = aic_fpm_gap._read_runtime_config(subdir)
        self.assertEqual(result["vllm_max_num_batched_tokens"], 8192)
        self.assertEqual(result["vllm_max_num_seqs"], 16)

    def test_missing_config_fails_loud_not_silent_fallback(self):
        """The old silent 2048/128 fallback is GONE: a missing config now fails loud."""
        import tempfile

        with tempfile.TemporaryDirectory() as td, self.assertRaises(aic_fpm_gap.ParityError):
            aic_fpm_gap._read_runtime_config(Path(td))


class CliTpSelectionTests(unittest.TestCase):
    def test_main_uses_explicit_tp_prediction_lane(self):
        import tempfile
        from unittest import mock

        previous_model = aic_fpm_gap.MODEL_NAME
        previous_tp_values = aic_fpm_gap.TP_VALUES
        observed_tp_values = []

        def fake_run(*_args, **_kwargs):
            observed_tp_values.append(aic_fpm_gap.TP_VALUES)
            return {"summary": []}

        try:
            with (
                tempfile.TemporaryDirectory() as td,
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "aic_fpm_gap",
                        "--tp",
                        "1",
                        "--repo-root",
                        td,
                        "--fpm-run",
                        td,
                        "--out-dir",
                        td,
                        "--no-html",
                    ],
                ),
                mock.patch.object(aic_fpm_gap, "run", side_effect=fake_run),
            ):
                aic_fpm_gap.main()
        finally:
            aic_fpm_gap.MODEL_NAME = previous_model
            aic_fpm_gap.TP_VALUES = previous_tp_values

        self.assertEqual(observed_tp_values, [(1,)])


if __name__ == "__main__":
    unittest.main()
