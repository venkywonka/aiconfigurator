# Job 354281415 semantic-FPM oracle

`job_354281415_expected.json` is the artifact-backed, pre-predictor acceptance
oracle for the semantic FPM/Nsight reducer. The generator verifies the archive,
all eight FPM CSVs, and all four Nsight SQLite inputs by SHA-256 before writing.

From the Aiconfigurator repository root:

```bash
python3 tests/fixtures/semantic_fpm_insights/generate_job_354281415_expected.py \
  --archive ~/scratch/gitlab_ci/fpm_job_354281415/result.tar.gz \
  --artifact-root ~/scratch/agent-slop/aiconfigurator/fpm-nsys-semantic-step-key-design/existing-artifact/fpm/Qwen-Qwen3-32B \
  --sqlite-root ~/scratch_big/aic-auto-collector/fpm_job_354281415-semantic-key/fpm/Qwen-Qwen3-32B \
  --output tests/fixtures/semantic_fpm_insights/job_354281415_expected.json
```

The full command completed on 2026-07-06. The regenerated and committed files
were byte-identical (`cmp` exit 0), both with SHA-256
`f422520b341b8ecefdea38b4ddc5d4cbd45a9eaed4a0399838330de628b3d44c`.

The `predictor_oracle` block intentionally remains `pending_adapter_freeze` in
this checkpoint. It must be replaced by the independently reviewed
`conservative-v1` adapter results before production pipeline integration.
