"""Install layerwise vLLM hooks inside spawned vLLM subprocesses.

vLLM physical tensor-parallel runs use spawned EngineCore/worker processes, so
patches imported by the parent layerwise worker do not automatically apply
where scheduling/model execution actually happens.  Python imports
``sitecustomize`` from ``PYTHONPATH`` during interpreter startup; the layerwise
worker places this directory on ``PYTHONPATH`` before creating the vLLM engine.
"""

from __future__ import annotations

import os
import sys


def _try_import(module: str, required: bool = False) -> None:
    """Import an instrumentation module.

    When ``required`` is False (default) the import is best-effort: failures are
    logged and swallowed so worker startup is never broken. When ``required`` is
    True the failure is re-raised -- the marker was explicitly requested, so a
    failed install must FAIL CLOSED (abort the worker) rather than produce an
    unattributable trace.
    """

    try:
        __import__(module)
    except Exception as exc:  # pragma: no cover - exercised in vLLM subprocesses
        print(f"[layerwise-sitecustomize] failed to import {module}: {exc!r}", file=sys.stderr)
        if required:
            raise
        return

    if required:
        # An already-loaded module's ``__import__`` is a no-op, so its
        # import-time install (which is env-gated) does not re-run. Re-invoke
        # the install explicitly so an explicitly-required marker FAILS CLOSED:
        # any SystemExit it raises propagates (aborting the worker) instead of
        # leaving the process running unmarked.
        mod = sys.modules.get(module)
        installer = getattr(mod, "_install", None)
        if callable(installer):
            installer()


if os.environ.get("LAYERWISE_SCHEDULER_TIMING", "0") == "1":
    _try_import("vllm_scheduler_timing_patch")

if os.environ.get("LAYERWISE_STEP_MARKER") == "1":
    _try_import("vllm_step_marker")

if os.environ.get("LAYERWISE_DYNAMO_STEP_MARKER") == "1":
    # Fail CLOSED: an explicitly-required marker that fails to install must
    # abort the worker (SystemExit from _install escapes anyway, but mark it
    # required so an ordinary import failure also propagates).
    _try_import("dynamo_step_marker", required=True)
