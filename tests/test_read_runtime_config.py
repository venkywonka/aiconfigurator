"""F3 RED-step tests for aic_fpm_gap._read_runtime_config.

The real effective_vllm_config.json uses FLATTENED dotted keys, e.g.
{"scheduler_config.max_num_batched_tokens": 8192, "scheduler_config.max_num_seqs": 16}.
The current helper only checks a top-level key or a nested 'scheduler_config' object,
so it misses the flattened key and silently falls back to 2048/128.
"""
import json
import logging
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

    def test_missing_config_returns_fallback_and_warns(self):
        """A missing config file returns the 2048/128 fallback and warns loudly."""
        with tempfile.TemporaryDirectory() as td:
            subdir = Path(td)  # no effective_vllm_config.json written
            with self.assertLogs(level=logging.WARNING) as captured:
                result = aic_fpm_gap._read_runtime_config(subdir)
        self.assertEqual(result["vllm_max_num_batched_tokens"], 2048)
        self.assertEqual(result["vllm_max_num_seqs"], 128)
        self.assertTrue(
            captured.output,
            "expected a WARNING-level log when falling back to constants",
        )


if __name__ == "__main__":
    unittest.main()
