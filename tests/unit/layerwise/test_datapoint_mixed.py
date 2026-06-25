# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collector.layerwise.vllm.data import DataPoint


def test_pure_ctx_still_constructs_from_old_raw():
    raw = {"phase": "ctx", "batch_size": 1, "new_tokens": 128, "past_kv": 0}
    dp = DataPoint(**raw)
    assert dp.shape_key == "ctx:bs1:new128:past0"


def test_mixed_datapoint_shape_key():
    dp = DataPoint("mixed", 0, 0, 0, prefill_tokens=2048, decode_requests=64, decode_past_kv=4096)
    assert dp.shape_key == "mixed:P2048:B64:K4096"
    assert dp.datapoint_id("wu1") == "wu1:mixed:P2048:B64:K4096"
