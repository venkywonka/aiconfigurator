# Job 354281415 semantic-FPM oracle

`job_354281415_expected.json` is the artifact-backed acceptance oracle for the
semantic FPM/Nsight reducer and cached-artifact predictor proxy. The generator
verifies the archive, fresh layerwise CSV, all eight FPM CSVs, and all four
Nsight SQLite inputs by SHA-256 before writing.

From the Aiconfigurator repository root:

```bash
python3 tests/fixtures/semantic_fpm_insights/generate_job_354281415_expected.py \
  --archive ~/scratch/gitlab_ci/fpm_job_354281415/result.tar.gz \
  --artifact-root ~/scratch/agent-slop/aiconfigurator/fpm-nsys-semantic-step-key-design/existing-artifact/fpm/Qwen-Qwen3-32B \
  --sqlite-root ~/scratch_big/aic-auto-collector/fpm_job_354281415-semantic-key/fpm/Qwen-Qwen3-32B \
  --layerwise-csv ~/scratch/agent-slop/aiconfigurator/fpm-nsys-semantic-step-key-design/existing-artifact/layerwise/Qwen-Qwen3-32B/layerwise.csv \
  --output tests/fixtures/semantic_fpm_insights/job_354281415_expected.json
```

The full command completed on 2026-07-06. Its current output SHA-256 is
`72c687fba597ebb73acc124a5320febf01ef7465be17393c1177a8069b3453c5`.

The `predictor_oracle` block freezes 17,009 adapter calls and 11,276 eligible
predictions under explicit `artifact-proxy-v1`. It also freezes zero production
`conservative-v1` eligibility for this cached artifact because its decode
layerwise surface was collected at `max_num_seqs=64`, while the FPM runtime used
256. A new production run must collect and use the exact 256 surface.
