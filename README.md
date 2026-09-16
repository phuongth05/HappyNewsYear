# Knowledge-rich image captioning

This repository currently contains the first data-layer milestone for the
research project: a unified sample schema, an inference-safe sample view, a
GoodNews adapter, and dataset auditing.

## GoodNews layout

Place the official release artifacts under a configurable root, for example:

```text
data/goodnews/
├── article+caption.json
├── img_splits.json
└── images/
    └── <article-id>_<image-index>.jpg
```

Copy `configs/dataset/goodnews.example.yaml` if local field aliases or paths
need to be changed. Relative data paths are resolved from the config file.
The official `val` split is exposed to project code as `dev`; it is never
regenerated.

## Audit

```powershell
python scripts/audit_dataset.py `
  --dataset goodnews `
  --config configs/dataset/goodnews.example.yaml `
  --output artifacts/dataset_audits/goodnews.json
```

Add `--fail-on-issues` in CI when missing, empty, duplicate, long-article, or
split-integrity findings should produce a non-zero exit code.

## Tests

```powershell
python -m pytest
```

Model and evidence code must consume `InferenceSample`, obtained through
`DatasetAdapter.iter_inference_samples`. It contains neither the reference
caption nor legacy entities of uncertain provenance.

## Saved-prediction evaluation

Evaluation accepts JSONL or a JSON array with `sample_id`, `prediction`,
`reference`, and `metadata`. It never invokes caption generation.

```powershell
python scripts/evaluate.py `
  --predictions runs/b0/predictions.jsonl `
  --output runs/b0/metrics.json `
  --metrics cider,spice,entity
```

CLIPScore is enabled explicitly because it requires `metadata.image_path` and
pretrained CLIP weights. Its image and text embeddings are cached in SQLite.

```powershell
python scripts/evaluate.py `
  --predictions runs/b0/predictions.jsonl `
  --output runs/b0/metrics.json `
  --metrics cider,spice,entity,clipscore `
  --embedding-cache artifacts/cache/clip_embeddings.sqlite3
```

Compare aligned per-sample outputs with a paired bootstrap:

```powershell
python scripts/compare_metrics.py `
  --baseline runs/b0/metrics_per_sample.csv `
  --candidate runs/b1/metrics_per_sample.csv `
  --metric EntityF1 `
  --output runs/comparisons/b1_vs_b0_entity_f1.json
```

CIDEr and SPICE use `pycocoevalcap==1.2`. Both use the bundled Stanford PTB
tokenizer. SPICE additionally requires its Stanford CoreNLP 3.6 artifacts and
a compatible Java runtime; use `--spice-java` to select an explicit Java
executable. Metric failures are recorded in the JSON and cause a non-zero exit
unless `--allow-metric-errors` is intentionally supplied.

## B0: image-only captioning

The B0 runner uses pinned unconditional BLIP generation. Its model-facing
`ImageOnlyInput` contains only `sample_id` and `image_path`; the adapter also
rejects processor outputs containing text-token keys. The default config is a
deterministic ten-sample development smoke run:

```powershell
python scripts/run_captioning.py `
  --config configs/experiments/b0_image_only.yaml
```

Install generation dependencies with `pip install -e ".[captioning]"`. The
runner writes predictions in the unified evaluator format, invokes that same
saved-file evaluator, and records the resolved config, model revision,
generation/token settings, runtime, Git state, metrics, and ten qualitative
examples under `runs/b0_image_only/dev_smoke/`.
