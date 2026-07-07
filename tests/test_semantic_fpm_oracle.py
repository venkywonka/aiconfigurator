# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ORACLE = Path(__file__).parent / "fixtures/semantic_fpm_insights/job_354281415_expected.json"


def _load_oracle():
    return json.loads(ORACLE.read_text())


def test_job_354281415_oracle_pins_artifact_identity_and_policies():
    oracle = _load_oracle()

    assert oracle["oracle_schema"] == "job-354281415-semantic-fpm/v1"
    assert oracle["source"]["job_id"] == 354281415
    assert oracle["source"]["archive_bytes"] == 6554156865
    assert oracle["source"]["archive_sha256"] == ("de4bd64b5f327fa2e73e131d5e7ec3346367814ece0181fe305ec05a795b22be")
    assert oracle["policies"] == {
        "alignment_version": "profiled-monotonic-v1",
        "mapping_hash_serialization": (
            "sha256(canonical-json[{counter_id,fpm_sequence_index,marker_canonical_index,marker_step,measure_run}])"
        ),
        "marker_encoder_version": "python-round-half-even-v1",
        "schema_version": "fpm-semantic-insights/v1",
        "semantic_key_version": "one-token-half-up-v1",
    }


def test_job_354281415_alignment_oracle_is_complete_and_self_consistent():
    oracle = _load_oracle()
    expected = {
        "c1": (4612, 5662, 5652, 10, 6, "c799214e308844abcc81b3c38dcb20cf84ba2e4694adaabccde9edac579171f7"),
        "c16": (4099, 5157, 5129, 0, 1, "4a0b14e5f9d5e86c1714df866dedce7ceb5c3d915b31a0621f3c370ba518dcab"),
        "c64": (4200, 5282, 5229, 0, 1, "9ee7980935f2c0e92e30f86e67ae23a8902987b7cbb4a1f439886b5b11a2e7a6"),
        "c128": (4242, 5358, 5272, 0, 1, "ed596b1dfdb35462ea5cec883bdf136b7f54e720d659de6339e2a30d7b4e5748"),
    }

    for cohort, (mapped, raw, canonical, skipped, segments, mapping_hash) in expected.items():
        actual = oracle["alignment"][cohort]
        assert actual["mapped_rows"] == actual["profiled_fpm_rows"] == mapped
        assert actual["raw_markers_per_rank"] == raw
        assert actual["canonical_markers_per_rank"] == canonical
        assert len(actual["nonmonotonic"]) == raw - canonical
        assert len(actual["skipped_markers"]) == skipped
        assert len(actual["segments"]) == segments
        assert sum(segment["mapped_rows"] for segment in actual["segments"]) == mapped
        assert actual["mapping_hash"] == mapping_hash
        assert len(actual["mapping_hash"]) == 64
        first = actual["first_mapping"]["marker_canonical_index"]
        last = actual["last_mapping"]["marker_canonical_index"]
        assert actual["prefix_markers"] == first
        assert actual["suffix_markers"] == canonical - last - 1
        assert last - first + 1 == mapped + skipped


def test_job_354281415_population_oracle_pins_all_rows_and_clean_only_mass():
    population = _load_oracle()["population"]

    assert population["source_rows"] == 34311
    assert population["clean_rows"] == 17158
    assert population["profiled_rows"] == 17153
    assert population["clean_bins"] == 17009
    assert population["profiled_bins"] == 17027
    assert population["shared_bins"] == 16530
    assert population["shared_clean_rows"] == 16679
    assert population["shared_profiled_rows"] == 16646
    assert population["clean_only_bins"] == population["clean_only_rows"] == 479
    assert population["clean_support"] == {"singleton": 16869, "sparse": 140, "repeated": 0}
    assert population["max_clean_count"] == 4
    assert population["clean_only_by_cohort_phase"] == {
        "c1": {"context": 0, "decode": 0, "mixed": 0},
        "c16": {"context": 1, "decode": 16, "mixed": 10},
        "c64": {"context": 1, "decode": 107, "mixed": 39},
        "c128": {"context": 1, "decode": 228, "mixed": 76},
    }
    assert sum(item["clean_rows"] for item in population["by_cohort"].values()) == population["clean_rows"]
    assert sum(item["profiled_rows"] for item in population["by_cohort"].values()) == population["profiled_rows"]
    assert sum(item["clean_bins"] for item in population["by_cohort"].values()) == population["clean_bins"]
    assert sum(item["shared_bins"] for item in population["by_cohort"].values()) == population["shared_bins"]


def test_job_354281415_oracle_freezes_full_clean_population_predictor_proxy():
    predictor = _load_oracle()["predictor_oracle"]
    assert predictor["status"] == "frozen_artifact_proxy_v1"
    assert predictor["policy"] == "artifact-proxy-v1"
    assert predictor["call_count"] == 17009
    assert predictor["status_reason"] == {
        "ok/predicted": 11276,
        "unavailable/missing_surface": 5606,
        "unavailable/unsupported_phase": 127,
    }
    assert predictor["clean_only"] == {
        "attempted": 479,
        "eligible": 65,
        "reasons": {"missing_surface": 289, "predicted": 65, "unsupported_phase": 125},
    }
    assert predictor["by_cohort_phase"]["c1/decode"]["eligible"] == 4507
    assert predictor["by_cohort_phase"]["c16/decode"]["eligible"] == 3546
    assert predictor["by_cohort_phase"]["c64/decode"]["eligible"] == 2037
    assert predictor["by_cohort_phase"]["c128/decode"]["eligible"] == 1186
    assert predictor["production_exact_surface_eligible"] == 0
    assert predictor["production_exact_surface_reason"] == ("cached_layerwise_decode_max_num_seqs_64_vs_runtime_256")
    assert predictor["layerwise_sha256"] == ("17781fac0d806c3683642c85e0f5e76f6046ab8739ad154fc1e2db01646c18f9")
    assert sum(predictor["evaluated_shapes"].values()) == 11276
    assert predictor["prediction_vector_sha256"] == ("27565b556f21dc49b3f598892c7bf1214b25643afa27444a5e974adc47e00402")
    assert len(predictor["top_absolute_relative_errors"]) == 20
    assert predictor["top_absolute_relative_errors"][0] == {
        "bin_id": "b3814eae03a39e200db0897382f29c0d9cbed982836821d2e86bbeee0111f1ec",
        "concurrency": 1,
        "phase": "decode",
        "relative_error": 0.7835103103671003,
        "semantic_key": "[0,1,null,null,9725]",
    }
