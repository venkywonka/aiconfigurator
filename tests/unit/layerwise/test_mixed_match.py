# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Step-marker tests for the mixed (one un-chunked prefill + B decodes) phase.

The matcher operates purely on the passed ``scheduler_output`` (torch-free), so
these tests fake ``scheduler_output``/``runner`` and never need a GPU or model.

Torch is imported by the marker at module load (and ``_install`` runs as a side
effect). We disable the install and stub torch so the module imports locally;
``setdefault`` is a no-op in-container where real torch is present.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = pytest.mark.unit

os.environ.setdefault("LAYERWISE_STEP_MARKER", "0")
sys.modules.setdefault("torch", mock.Mock())
sys.modules.setdefault("torch.cuda", mock.Mock())
sys.modules.setdefault("torch.cuda.nvtx", mock.Mock())

from collector.layerwise.vllm import vllm_step_marker as marker  # noqa: E402


def _new_req(req_id, prompt_len, num_computed=0):
    """A freshly scheduled (prefill) request."""
    return SimpleNamespace(
        req_id=req_id,
        num_prompt_tokens=prompt_len,
        num_computed_tokens=num_computed,
    )


def _mixed_scheduler_output(
    *,
    prefill_id,
    prefill_tokens,
    prefill_prompt_len,
    prefill_computed,
    decode_ids,
    decode_past,
):
    """Fake a scheduler_output where one new req takes ``prefill_tokens`` and
    each cached (decode) req takes exactly one token at ``decode_past`` KV."""
    num_scheduled = {prefill_id: prefill_tokens}
    for dec_id in decode_ids:
        num_scheduled[dec_id] = 1
    return SimpleNamespace(
        scheduled_new_reqs=[
            _new_req(prefill_id, prefill_prompt_len, prefill_computed)
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(decode_ids),
            num_computed_tokens=[decode_past for _ in decode_ids],
        ),
        num_scheduled_tokens=num_scheduled,
    )


def _runner_with_decodes(decode_ids, prompt_len):
    return SimpleNamespace(
        requests={
            dec_id: SimpleNamespace(num_prompt_tokens=prompt_len)
            for dec_id in decode_ids
        }
    )


# --------------------------------------------------------------------------- #
# Task 5: _mixed_match                                                          #
# --------------------------------------------------------------------------- #


def test_mixed_match_accepts_one_unchunked_prefill_plus_b_decodes():
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    so = _mixed_scheduler_output(
        prefill_id="prefill0",
        prefill_tokens=P,
        prefill_prompt_len=P,
        prefill_computed=0,
        decode_ids=decode_ids,
        decode_past=K,
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, prefill_tokens, decode_bs, past = marker._mixed_match(runner, so, control)

    assert matched is True
    assert prefill_tokens == P
    assert decode_bs == B
    assert past == K
    assert marker._LAST_MIXED_MATCH_META == {
        "prefill_tokens": P,
        "decode_bs": B,
        "past": K,
    }


def test_mixed_match_rejects_chunked_prefill():
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    # Prefill split: only 1024 of the 2048-token prompt scheduled this step.
    so = _mixed_scheduler_output(
        prefill_id="prefill0",
        prefill_tokens=1024,
        prefill_prompt_len=P,
        prefill_computed=0,
        decode_ids=decode_ids,
        decode_past=K,
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


def test_mixed_match_rejects_wrong_decode_count():
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B - 1)]  # one short
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    so = _mixed_scheduler_output(
        prefill_id="prefill0",
        prefill_tokens=P,
        prefill_prompt_len=P,
        prefill_computed=0,
        decode_ids=decode_ids,
        decode_past=K,
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


def test_mixed_match_rejects_wrong_past():
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    so = _mixed_scheduler_output(
        prefill_id="prefill0",
        prefill_tokens=P,
        prefill_prompt_len=P,
        prefill_computed=0,
        decode_ids=decode_ids,
        decode_past=K - 1,  # decodes not yet at the target KV
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


def test_mixed_match_rejects_pure_ctx_step():
    # No cached decode reqs at all: a pure prefill step.
    P, B, K = 2048, 4, 4096
    runner = _runner_with_decodes([], prompt_len=K)
    so = SimpleNamespace(
        scheduled_new_reqs=[_new_req("prefill0", P, 0)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={"prefill0": P},
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


def test_mixed_match_rejects_pure_gen_step():
    # No new prefill req: a pure decode step.
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    so = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(decode_ids),
            num_computed_tokens=[K for _ in decode_ids],
        ),
        num_scheduled_tokens={dec_id: 1 for dec_id in decode_ids},
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


def test_mixed_match_rejects_two_new_prefills():
    P, B, K = 2048, 4, 4096
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    num_scheduled = {"prefill0": P, "prefill1": P}
    for dec_id in decode_ids:
        num_scheduled[dec_id] = 1
    so = SimpleNamespace(
        scheduled_new_reqs=[_new_req("prefill0", P, 0), _new_req("prefill1", P, 0)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(decode_ids),
            num_computed_tokens=[K for _ in decode_ids],
        ),
        num_scheduled_tokens=num_scheduled,
    )
    control = {"prefill_tokens": P, "decode_bs": B, "past": K}

    matched, *_ = marker._mixed_match(runner, so, control)
    assert matched is False


# --------------------------------------------------------------------------- #
# Task 6: marker dispatch + progress emission + GPU-event timing               #
# --------------------------------------------------------------------------- #


def _read_progress_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            import json as _json

            rows.append(_json.loads(line))
    return rows


def _arm_real_progress(monkeypatch, tmp_path, work_unit_id="wu1"):
    """Point the real _write_progress at a temp JSONL so datapoint_id (built
    inside _write_progress) is exercised end-to-end."""
    progress_file = tmp_path / "progress.jsonl"
    monkeypatch.setenv("LAYERWISE_PROGRESS_FILE", str(progress_file))
    monkeypatch.setenv("LAYERWISE_WORK_UNIT_ID", work_unit_id)
    monkeypatch.delenv("LAYERWISE_ATTEMPT_ID", raising=False)
    monkeypatch.delenv("LAYERWISE_MEASURE_RUN", raising=False)
    monkeypatch.delenv("LAYERWISE_CUDA_PROFILER_WINDOW", raising=False)
    return progress_file


class _FakeEvent:
    def __init__(self, elapsed=12.5):
        self._elapsed = elapsed

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return self._elapsed


def test_progress_datapoint_id_mixed_branch():
    P, B, K = 2048, 64, 4096
    dpid = marker._progress_datapoint_id(
        "wu1", "mixed", 0, None, 0, prefill_tokens=P, decode_requests=B, decode_past_kv=K
    )
    assert dpid == "wu1:mixed:P2048:B64:K4096"


def test_run_marked_step_mixed_emits_gpu_time(monkeypatch, tmp_path):
    P, B, K = 2048, 4, 4096
    progress_file = _arm_real_progress(monkeypatch, tmp_path)

    def fake_orig(runner, scheduler_output, intermediate_tensors):
        return "ok"

    monkeypatch.setattr(marker.torch.cuda, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(
        marker.torch.cuda, "Event", lambda **k: _FakeEvent(12.5), raising=False
    )

    ret = marker._run_marked_step(
        fake_orig,
        SimpleNamespace(),
        SimpleNamespace(),
        None,
        step=K + 1,
        batch_size=B,
        past_kv=K,
        control={
            "trigger": "mixed",
            "phase": "mixed",
            "prefill_tokens": P,
            "decode_bs": B,
            "past": K,
            "measure_execute_model_gpu_time": True,
        },
    )

    assert ret == "ok"
    rows = _read_progress_rows(progress_file)
    completed = [r for r in rows if r["event"] == "completed_execution"]
    assert completed, "expected a completed_execution progress event"
    row = completed[0]
    assert row["execute_model_gpu_time_ms"] == 12.5
    assert row["phase"] == "mixed"
    assert row["datapoint_id"] == f"wu1:mixed:P{P}:B{B}:K{K}"


def test_dispatch_trigger_routes_mixed(monkeypatch, tmp_path):
    """Integration-style: the trigger dispatcher (called first by patched())
    routes a matching mixed scheduler_output through the timed marked step."""
    P, B, K = 2048, 4, 4096
    progress_file = _arm_real_progress(monkeypatch, tmp_path)
    decode_ids = [f"dec{i}" for i in range(B)]
    runner = _runner_with_decodes(decode_ids, prompt_len=K)
    so = _mixed_scheduler_output(
        prefill_id="prefill0",
        prefill_tokens=P,
        prefill_prompt_len=P,
        prefill_computed=0,
        decode_ids=decode_ids,
        decode_past=K,
    )

    control = {
        "trigger": "mixed",
        "phase": "mixed",
        "prefill_tokens": P,
        "decode_bs": B,
        "past": K,
        "measure_execute_model_gpu_time": True,
    }

    monkeypatch.setattr(marker.torch.cuda, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(
        marker.torch.cuda, "Event", lambda **k: _FakeEvent(7.25), raising=False
    )

    captured = {}

    def fake_orig(self_runner, scheduler_output, intermediate_tensors=None):
        captured["called"] = True
        return "done"

    handled, ret = marker._dispatch_trigger(fake_orig, runner, so, None, control)

    assert handled is True
    assert ret == "done"
    assert captured.get("called") is True
    rows = _read_progress_rows(progress_file)
    completed = [r for r in rows if r["event"] == "completed_execution"]
    assert completed, "mixed trigger did not route through a timed marked step"
    assert completed[0]["execute_model_gpu_time_ms"] == 7.25
    assert completed[0]["datapoint_id"] == f"wu1:mixed:P{P}:B{B}:K{K}"


def test_dispatch_trigger_no_trigger_falls_through():
    """With no trigger key, the dispatcher reports unhandled so patched()
    continues to its env-driven counting path."""
    handled, ret = marker._dispatch_trigger(
        lambda *a, **k: "orig", SimpleNamespace(), SimpleNamespace(), None, {}
    )
    assert handled is False
    assert ret is None
