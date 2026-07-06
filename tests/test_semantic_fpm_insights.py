# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from collector.layerwise.diagnostics.semantic_fpm_insights import (
    SCHEMA_VERSION,
    SemanticShape,
    ShapeValidationError,
    build_semantic_query,
    canonical_semantic_key,
    derive_phase,
    round_half_up_ratio,
    stable_bin_id,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("total", "count", "expected"),
    [
        (0, 1, 0),
        (1, 3, 0),
        (1, 2, 1),
        (2, 4, 1),
        (3, 2, 2),
        (4, 3, 1),
        (5, 3, 2),
    ],
)
def test_round_half_up_ratio_is_integer_only(total: int, count: int, expected: int):
    assert round_half_up_ratio(total, count) == expected


@pytest.mark.parametrize(("total", "count"), [(-1, 1), (1, 0), (1, -1)])
def test_round_half_up_ratio_rejects_invalid_inputs(total: int, count: int):
    with pytest.raises(ValueError):
        round_half_up_ratio(total, count)


@pytest.mark.parametrize(
    ("ctx_requests", "decode_requests", "expected"),
    [(1, 0, "context"), (0, 1, "decode"), (1, 1, "mixed"), (0, 0, "idle")],
)
def test_derive_phase_uses_request_counts(ctx_requests: int, decode_requests: int, expected: str):
    assert derive_phase(ctx_requests, decode_requests) == expected


def test_semantic_key_uses_exact_counts_half_up_means_and_absent_axes():
    context = SemanticShape(
        ctx_requests=2,
        decode_requests=0,
        ctx_new_tokens=5,
        ctx_kv_tokens=1,
        decode_kv_tokens=0,
    )
    decode = SemanticShape(
        ctx_requests=0,
        decode_requests=3,
        ctx_new_tokens=0,
        ctx_kv_tokens=0,
        decode_kv_tokens=10,
    )
    mixed = SemanticShape(
        ctx_requests=2,
        decode_requests=3,
        ctx_new_tokens=5,
        ctx_kv_tokens=7,
        decode_kv_tokens=11,
    )

    assert context.semantic_key == (2, 0, 3, 1, None)
    assert decode.semantic_key == (0, 3, None, None, 3)
    assert mixed.semantic_key == (2, 3, 3, 4, 4)
    assert context.serialized_key == "[2,0,3,1,null]"
    assert decode.serialized_key == "[0,3,null,null,3]"


def test_shape_validation_rejects_structural_errors_with_stable_reason():
    cases = [
        (SemanticShape(-1, 0, 1, 0, 0), "negative_value"),
        (SemanticShape(0, 1, 1, 0, 1), "zero_count_nonzero_total"),
        (SemanticShape(1, 0, 1, 0, 1), "zero_count_nonzero_total"),
        (SemanticShape(1, 0, 0, 0, 0), "active_context_without_new_tokens"),
        (SemanticShape(0, 1, 0, 0, 0), "active_decode_without_kv_tokens"),
    ]

    for shape, reason in cases:
        with pytest.raises(ShapeValidationError) as exc_info:
            shape.validate()
        assert exc_info.value.process_code == "invalid_shape"
        assert exc_info.value.reason == reason


def test_shape_validation_rejects_source_phase_disagreement():
    shape = SemanticShape(1, 0, 8, 0, 0)
    with pytest.raises(ShapeValidationError) as exc_info:
        shape.validate(source_phase="decode")
    assert exc_info.value.process_code == "phase_mismatch"
    assert exc_info.value.reason == "source_phase_disagrees_with_counts"


def test_idle_shape_is_valid_but_has_no_active_token_axes():
    shape = SemanticShape(0, 0, 0, 0, 0)
    assert shape.validate(source_phase="idle") == "idle"
    assert shape.semantic_key == (0, 0, None, None, None)


def test_canonical_semantic_key_is_whitespace_free_json():
    key = (2, 3, 4, 5, 6)
    serialized = canonical_semantic_key(key)
    assert serialized == "[2,3,4,5,6]"
    assert json.loads(serialized) == [2, 3, 4, 5, 6]


def test_stable_bin_id_hashes_versioned_canonical_identity():
    shape = SemanticShape(2, 3, 5, 7, 11)
    first = stable_bin_id(
        configuration_fingerprint="cfg-sha",
        concurrency=16,
        phase="mixed",
        semantic_key=shape.semantic_key,
    )
    second = stable_bin_id(
        configuration_fingerprint="cfg-sha",
        concurrency=16,
        phase="mixed",
        semantic_key=shape.semantic_key,
    )

    assert first == second == "a53af006bfb14dbb99870c642639cf582528ed93184908d8684a82ebc2737a76"
    assert first != stable_bin_id(
        configuration_fingerprint="cfg-sha",
        concurrency=64,
        phase="mixed",
        semantic_key=shape.semantic_key,
    )


def test_build_semantic_query_reconstructs_canonical_integer_shape():
    shape = SemanticShape(2, 3, 5, 7, 11)
    query = build_semantic_query(
        configuration_fingerprint="cfg-sha",
        concurrency=16,
        shape=shape,
        source_phase="mixed",
    )

    assert query.schema_version == SCHEMA_VERSION
    assert query.configuration_fingerprint == "cfg-sha"
    assert query.concurrency == 16
    assert query.phase == "mixed"
    assert query.semantic_key == "[2,3,3,4,4]"
    assert query.ctx_requests == 2
    assert query.decode_requests == 3
    assert query.ctx_new_per_request == 3
    assert query.ctx_kv_per_request == 4
    assert query.decode_kv_per_request == 4
    assert query.query_ctx_new_total == 6
    assert query.query_ctx_kv_total == 8
    assert query.query_decode_kv == 4


def test_build_semantic_query_rejects_nonpositive_concurrency():
    with pytest.raises(ValueError, match="concurrency"):
        build_semantic_query(
            configuration_fingerprint="cfg-sha",
            concurrency=0,
            shape=SemanticShape(0, 1, 0, 0, 1),
            source_phase="decode",
        )


def test_build_semantic_query_rejects_structurally_ineligible_idle_shape():
    with pytest.raises(ValueError, match="idle"):
        build_semantic_query(
            configuration_fingerprint="cfg-sha",
            concurrency=1,
            shape=SemanticShape(0, 0, 0, 0, 0),
            source_phase="idle",
        )
