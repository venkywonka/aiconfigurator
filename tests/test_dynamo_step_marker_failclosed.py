# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F1 (RED): the dynamo step marker must FAIL CLOSED when explicitly required.

When ``LAYERWISE_DYNAMO_STEP_MARKER=1`` and the monkeypatch target is
unavailable (torch/vllm absent -> the ``from vllm.v1.worker...`` import inside
``_install`` fails), the marker MUST abort the worker rather than silently
continuing unmarked. The desired behavior (does NOT exist yet):

  (a) ``dynamo_step_marker._install()`` raises ``SystemExit`` -- a
      ``BaseException`` that escapes a bare ``except Exception:`` (so the
      surrounding ``_try_import`` fail-open layer cannot re-swallow it).
  (b) With the env var unset or ``!= '1'``, ``_install()`` is a no-op and does
      NOT raise.
  (c) ``sitecustomize._try_import('dynamo_step_marker', required=True)``
      re-raises (propagates ``SystemExit``) on install failure, while
      ``_try_import(<any failing module>, required=False)`` swallows and
      returns normally.

These assertions FAIL against the current code (the marker swallows + returns
None; ``_try_import`` has no ``required`` parameter) -- that is the intended RED.

IMPORTANT: importing ``dynamo_step_marker`` runs ``_install()`` at import time.
We import it with the env var UNSET first (so the import-time install is a
no-op) and stub ``torch`` so the module's transitive ``import torch`` succeeds.
The vllm patch target stays genuinely absent, so calling ``_install()`` later
with the env var set exercises the unavailable-target path.
"""

import importlib
import os
import sys
import unittest
from unittest import mock

# The marker transitively imports torch at module load (via vllm_step_marker)
# and runs _install() as an import-time side effect. Stub torch so the import
# succeeds, and keep the env var UNSET so the import-time _install() is a no-op.
# vllm is deliberately NOT stubbed -> the in-_install vllm import will fail,
# which is exactly the "monkeypatch target unavailable" condition we test.
os.environ.pop("LAYERWISE_DYNAMO_STEP_MARKER", None)
# vllm_step_marker (imported transitively) ALSO runs _install() at module load
# and defaults LAYERWISE_STEP_MARKER to "1"; disable it so that import-time
# install is a no-op too (mirrors tests/test_vllm_step_marker.py).
os.environ["LAYERWISE_STEP_MARKER"] = "0"
sys.modules.setdefault("torch", mock.Mock())
sys.modules.setdefault("torch.cuda", mock.Mock())
sys.modules.setdefault("torch.cuda.nvtx", mock.Mock())

dsm = importlib.import_module("collector.layerwise.vllm.dynamo_step_marker")
sitecustomize = importlib.import_module("collector.layerwise.vllm.sitecustomize")


class DynamoStepMarkerFailClosedTests(unittest.TestCase):
    def setUp(self):
        # Ensure no leaked state forces the env on/off between tests.
        self._saved = os.environ.get("LAYERWISE_DYNAMO_STEP_MARKER")
        # vllm must remain unimportable so the patch target is unavailable.
        self.assertNotIn(
            "vllm",
            sys.modules,
            "test precondition: vllm must be absent so the patch target fails",
        )

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("LAYERWISE_DYNAMO_STEP_MARKER", None)
        else:
            os.environ["LAYERWISE_DYNAMO_STEP_MARKER"] = self._saved

    # (a) -----------------------------------------------------------------
    def test_install_raises_systemexit_when_required_and_target_unavailable(self):
        """env=1 + unavailable target -> _install() raises SystemExit.

        SystemExit is a BaseException, so it escapes a bare ``except Exception:``
        in the surrounding _try_import layer. Current code swallows the failure
        and returns None, so this assertion FAILS (correct RED).
        """
        os.environ["LAYERWISE_DYNAMO_STEP_MARKER"] = "1"

        with self.assertRaises(SystemExit):
            dsm._install()

    def test_install_failure_escapes_bare_except_exception(self):
        """The raised exception must NOT be catchable by ``except Exception``.

        Simulates the _try_import fail-open layer: a bare ``except Exception``
        wrapping the install must NOT be able to swallow the failure. Current
        code never raises at all, so ``raised`` stays False -> FAILS (RED).
        """
        os.environ["LAYERWISE_DYNAMO_STEP_MARKER"] = "1"

        raised = False
        try:
            try:
                dsm._install()
            except Exception:  # noqa: BLE001 - intentionally simulate fail-open layer
                self.fail("install failure was swallowed by a bare 'except Exception'")
        except BaseException:  # noqa: BLE001 - SystemExit should land here
            raised = True

        self.assertTrue(
            raised,
            "expected a BaseException (SystemExit) to escape 'except Exception'",
        )

    # (b) -----------------------------------------------------------------
    def test_install_is_noop_when_env_unset(self):
        os.environ.pop("LAYERWISE_DYNAMO_STEP_MARKER", None)
        # Must not raise (no SystemExit, no other exception).
        self.assertIsNone(dsm._install())

    def test_install_is_noop_when_env_not_one(self):
        os.environ["LAYERWISE_DYNAMO_STEP_MARKER"] = "0"
        self.assertIsNone(dsm._install())

    # (c) -----------------------------------------------------------------
    def test_try_import_reraises_when_required(self):
        """_try_import('dynamo_step_marker', required=True) propagates SystemExit.

        The dynamo marker re-imports already-loaded -> import is a no-op, so the
        failure must come from the env-gated install. With required=True the
        SystemExit must propagate. Current _try_import has no ``required`` param,
        so this raises TypeError -> acceptable RED (feature missing).
        """
        os.environ["LAYERWISE_DYNAMO_STEP_MARKER"] = "1"

        with self.assertRaises(SystemExit):
            sitecustomize._try_import(
                "collector.layerwise.vllm.dynamo_step_marker", required=True
            )

    def test_try_import_swallows_when_not_required(self):
        """_try_import(<failing module>, required=False) swallows and returns None.

        A genuinely-missing module raises ModuleNotFoundError on import; with
        required=False the fail-open behavior must be preserved (no raise).
        Current _try_import has no ``required`` param -> TypeError -> RED.
        """
        result = sitecustomize._try_import(
            "definitely_not_a_real_module_xyz_layerwise", required=False
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
