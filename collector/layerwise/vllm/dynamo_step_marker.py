# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DYNAMO-side per-step NVTX marker for the FPM / attribute path.

Why this exists (Phase-0 finding)
----------------------------------
The FPM worker is launched as ``python3 -m dynamo.vllm`` under nsys. In that
launch context ``vllm_step_marker.py`` is NEVER imported (no repo mount /
sitecustomize on PYTHONPATH), and even if it were, its counter-mode label
assumes ``isl=1`` single-stream (``past_kv = n - 1``) which is WRONG for a real
multi-request FPM workload.

So per-step NVTX for the FPM/attribute path must wrap the GPU forward
(``GPUModelRunner.execute_model``) -- reading the REAL per-step batch state from
the ``scheduler_output`` arg -- and emit a label in the EXACT format the existing
nsys parser (``collector/layerwise/common/parse_nsys_step_sweep.py`` /
``collector/layerwise/diagnostics/analyze_nsys_comm_overlap.py``) keys on.

NOTE (Phase-0 smoke #5): an earlier version hooked
``InstrumentedScheduler.update_from_output``, but in vLLM v1 the per-step order is
``schedule() -> execute_model() [GPU kernels] -> update_from_output()``. NVTX around
``update_from_output`` brackets only post-step CPU bookkeeping (~0 kernels inside the
window), so kernels were attributed to no step. We MUST wrap ``execute_model`` (the
forward) -- the same method ``vllm_step_marker`` wraps -- but with REAL-batch labels
instead of its single-stream counter-mode labels.

Label format (must match ``vllm_step_marker.py:_run_marked_step``)::

    bench_step::N<step:07d>::bs<decode_batch>::past<mean_kv:06d>

Injection is from the aiconfigurator side via monkeypatch (no edits to the
dynamo source tree). Enable with ``LAYERWISE_DYNAMO_STEP_MARKER=1``.
"""

from __future__ import annotations

import logging
import os
import sys

# Reuse the windowed cudaProfilerStart/Stop gating helpers from the existing
# step marker. Do NOT duplicate them: a single implementation keeps the window
# semantics identical across the two markers.
from collector.layerwise.vllm.vllm_step_marker import (
    _PROFILER_STATE,
    _advance_profiler_window,
    _parse_profiler_window,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-logic helpers (unit-testable without torch / dynamo)
# ---------------------------------------------------------------------------


def _decode_batch_and_kv(scheduler_output) -> tuple[int, int]:
    """Extract ``(decode_batch, mean_kv)`` from a vLLM ``SchedulerOutput``.

    A request is a DECODE request iff it is NOT in the context (prefill) phase.

    ``decode_batch`` is the count of scheduled cached requests in decode phase;
    ``mean_kv`` is the rounded mean of their ``num_computed_tokens`` (the
    per-request KV context length), or ``0`` when there are no decode requests.

    Tolerates missing attributes (uses ``getattr``) so it degrades safely to
    ``(0, 0)`` rather than crashing the worker hot path.
    """
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is None:
        return 0, 0

    req_ids = getattr(cached, "req_ids", None)
    num_computed_tokens = getattr(cached, "num_computed_tokens", None)
    is_context_phase = getattr(cached, "is_context_phase", None)
    if req_ids is None or num_computed_tokens is None or is_context_phase is None:
        return 0, 0

    decode_kv: list[int] = []
    for i, req_id in enumerate(req_ids):
        try:
            if is_context_phase(req_id):
                continue
        except Exception:
            # If phase classification fails for a request, treat it as
            # non-decode so we never over-count the decode batch.
            continue
        if i < len(num_computed_tokens):
            decode_kv.append(int(num_computed_tokens[i]))

    decode_batch = len(decode_kv)
    if decode_batch == 0:
        return 0, 0
    mean_kv = round(sum(decode_kv) / decode_batch)
    return decode_batch, int(mean_kv)


def _bench_step_label(step: int, decode_batch: int, mean_kv: int) -> str:
    """Return the EXACT ``bench_step::...`` NVTX label the parser keys on.

    Format mirrors ``vllm_step_marker.py:_run_marked_step``: 7-digit zero-padded
    step, unpadded batch size, 6-digit zero-padded past_kv.
    """
    return f"bench_step::N{step:07d}::bs{decode_batch}::past{mean_kv:06d}"


# ---------------------------------------------------------------------------
# Monkeypatch installer (validated on hardware; not unit-tested here)
# ---------------------------------------------------------------------------


def _install() -> None:
    """Monkeypatch ``GPUModelRunner.execute_model`` to emit per-step NVTX.

    Wraps the GPU FORWARD (where the step's kernels run), reading the REAL
    per-step batch state from the ``scheduler_output`` arg. NOT
    ``update_from_output`` -- that runs after the forward and would bracket no
    kernels (see module docstring / Phase-0 smoke #5). No-op unless
    ``LAYERWISE_DYNAMO_STEP_MARKER=1``; wrapped in try/except so a missing
    torch/vllm import (or any patch failure) never crashes worker startup.
    """
    if os.environ.get("LAYERWISE_DYNAMO_STEP_MARKER") != "1":
        return

    try:
        import torch.cuda.nvtx as nvtx

        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        # Idempotent: in production the marker is installed twice (once at module
        # import, once when ``sitecustomize._try_import(required=True)``
        # re-invokes _install for the fail-closed cached-module path). Re-wrapping
        # an already-wrapped ``execute_model`` would emit two nested
        # ``bench_step::`` NVTX ranges per step and corrupt attribution, so bail
        # out if our wrapper is already installed.
        if getattr(GPUModelRunner.execute_model, "_layerwise_dynamo_marked", False):
            return

        orig = GPUModelRunner.execute_model
        state = {"n": 0}

        def patched(self, scheduler_output, *args, **kwargs):
            # Read the REAL per-step batch state BEFORE running the forward.
            decode_batch, mean_kv = _decode_batch_and_kv(scheduler_output)

            state["n"] += 1
            n = state["n"]

            label = _bench_step_label(n, decode_batch, mean_kv)
            spans = _parse_profiler_window(
                os.environ.get("LAYERWISE_CUDA_PROFILER_WINDOW")
            )

            nvtx.range_push(label)
            if spans:
                from collector.layerwise.vllm.worker import (
                    _cuda_profiler_call as _prof_call,
                )

                _advance_profiler_window(n, spans, _PROFILER_STATE, _prof_call)
            try:
                return orig(self, scheduler_output, *args, **kwargs)
            finally:
                if spans:
                    from collector.layerwise.vllm.worker import (
                        _cuda_profiler_call as _prof_call,
                    )

                    _advance_profiler_window(n, spans, _PROFILER_STATE, _prof_call)
                nvtx.range_pop()

        patched._layerwise_dynamo_marked = True
        GPUModelRunner.execute_model = patched
        logger.warning(
            "[dynamo-step-marker] installed GPUModelRunner.execute_model wrapper (real-batch labels)"
        )
    except Exception as exc:
        # Fail CLOSED: the marker was explicitly required (env == '1'), so a
        # patch/import failure must abort the worker rather than produce an
        # unattributable (unmarked) trace. Raise SystemExit -- a BaseException
        # that escapes the surrounding ``except Exception`` fail-open layer in
        # ``sitecustomize._try_import`` so the hole cannot be silently reopened.
        print(
            f"[dynamo-step-marker] install failed (LAYERWISE_DYNAMO_STEP_MARKER=1, aborting worker): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


_install()
