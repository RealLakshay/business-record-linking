# Code Red Business Entity Resolution

This package implements phases 0–6: audit-ready TSV ingestion, normalization,
disk-backed blocking, pair features, supervised classifier training, and
macro-$F_{0.5}$ threshold tuning. It uses only the supplied records and labels;
there are no external lookups.

## Run

From the `student_resource` directory:

```bash
python -m pip install -r code_red_submission/code/business_entity_resolution/requirements.txt
python code_red_submission/code/business_entity_resolution/src/pipeline.py \
  --data-root . \
  --output-dir code_red_submission/output \
  --work-dir code_red_submission/work
```

The command builds a SQLite index for S2/S3, samples training entities by a
stable hash, trains LightGBM, tunes the threshold on a disjoint entity split,
then writes both required output files. SQLite files can be large and should be
placed on a disk with sufficient free space. Use `--skip-inference` to run only
training and threshold tuning during development.

Validate generated files with:

```bash
python utils/validate_submission.py \
  --matching code_red_submission/output/matching_results.tsv \
  --candidate code_red_submission/output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The candidate file is the exact final candidate set scored during test
inference. The default candidate cap is 30 per S1 record and can be changed
with `--candidate-cap` after checking blocking recall on training data.