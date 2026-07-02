"""Tests for aic_fpm_gap._read_runtime_config.

The real effective_vllm_config.json uses FLATTENED dotted keys, e.g.
{"scheduler_config.max_num_batched_tokens": 8192, "scheduler_config.max_num_seqs": 16}.
The helper reads that flattened key.

Config-parity hardening (AIC-1205 / task #31): the OLD silent 2048/128 fallback on a
missing/unreadable config was a proven landmine — it silently produced AIC predictions at a
config that differed from the FPM ground truth (a ~5x error swing). The helper now FAILS LOUD
(ParityError) instead of fabricating constants; see tests/test_config_parity_guard.py for the
full guard contract.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collector.layerwise.diagnostics import aic_fpm_gap


class ReadRuntimeConfigTests(unittest.TestCase):
    def test_flattened_dotted_key_is_read(self):
        """A real-shaped config with the flattened dotted key must be honored."""
        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)
            (subdir / "effective_vllm_config.json").write_text(
                json.dumps(
                    {
                        "scheduler_config.max_num_batched_tokens": 8192,
                        "scheduler_config.max_num_seqs": 16,
                    }
                )
            )
            result = aic_fpm_gap._read_runtime_config(subdir)
        # Current code misses the flattened key and falls back to 2048 -> RED.
        self.assertEqual(result["vllm_max_num_batched_tokens"], 8192)
        self.assertEqual(result["vllm_max_num_seqs"], 16)

    def test_missing_config_fails_loud(self):
        """A missing config file now FAILS LOUD instead of the silent 2048/128 fallback.

        The old silent fallback (2048/128) was a config-parity landmine: it produced AIC
        predictions at a config that did NOT match the FPM ground truth. The helper now
        raises ParityError (AIC-1205 / task #31).
        """
        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)  # no effective_vllm_config.json written
            with self.assertRaises(aic_fpm_gap.ParityError):
                aic_fpm_gap._read_runtime_config(subdir)


if __name__ == "__main__":
    unittest.main()
