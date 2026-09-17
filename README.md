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

See `docs/goodnews_setup.md` for the exact consumed JSON layout, official
acquisition routes, current SharePoint-access limitation, and preflight audit.

## Audit

```powershell
python scripts/audit_dataset.py `
  --dataset goodnews `
  --root "D:/Datasets/GoodNews" `
  --output artifacts/dataset_audits/goodnews_validation.json `
  --check-only `
  --fail-on-issues
```

Remove `--check-only` after validation to compute tokenizer-exact statistics
for the real release. This loads only the pinned InstructBLIP processor and
tokenizer, not model weights. See `docs/goodnews_setup.md` for the full list of
checks and external-root configuration.

For a faster experiment preflight, verify all metadata and paths but decode a
fixed sample of images, then validate the exact 50 experiment samples:

```powershell
python scripts/audit_dataset.py --dataset goodnews --root "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews" --quick --verify-images 10000 --output artifacts/dataset_audits/goodnews_quick.json --fail-on-issues
python scripts/audit_experiment_subset.py --dataset-root "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews" --split dev --max-samples 50 --selection first_by_sample_id
```

## Tests

```powershell
python -m pytest
```

## Cloud GPU: controlled GoodNews validation

The same CUDA workflow runs the fixed 50-sample B0/B1/B1-random validation on
Kaggle, molab/marimo, or another CUDA host. It accepts `--dataset-root`,
`--output-root`, and `--bundle`; no provider path is embedded in core code.
See `docs/kaggle_goodnews_validation.md` and
`docs/molab_goodnews_validation.md`.

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
deterministic 50-sample development validation run:

```powershell
python scripts/run_captioning.py `
  --config configs/experiments/b0_image_only.yaml
```

Install generation dependencies with `pip install -e ".[captioning]"`. The
runner writes predictions in the unified evaluator format, invokes that same
saved-file evaluator, and records the resolved config, model revision,
generation/token settings, runtime, Git state, metrics, and ten qualitative
examples under `runs/b0_image_only/dev_smoke/`.

## Controlled InstructBLIP B0/B1

The original BLIP decoder-prefix B1 remains reproducible but is now explicitly
exploratory/legacy. The definitive controlled implementation uses pinned
InstructBLIP Flan-T5 XL; see
[`docs/b1_architecture_selection.md`](docs/b1_architecture_selection.md).

Run the controlled development conditions in this order:

```powershell
python scripts/run_captioning.py --config configs/experiments/b0_instructblip_image_only.yaml
python scripts/run_captioning.py --config configs/experiments/b1_instructblip_article_context.yaml
python scripts/run_captioning.py --config configs/experiments/b1_instructblip_random_article.yaml
```

All three configs share the checkpoint, prompt template, context policy,
decoding, seed, subset, and evaluator. The runner rejects B1 when any controlled
field differs from its declared B0. The random condition is only a sanity
control.

## B1: image plus full article (exploratory/legacy)

B1 uses the same pinned BLIP checkpoint, development subset, seed, decoding,
and evaluator as B0. The only model-input change is a raw article decoder
prefix. It retains at most the first 256 article tokens and records original
tokens, used tokens, and `(original-used)/original` truncation ratio for every
prediction plus an aggregate `context_statistics.json`.

Install the fixed NER evaluator and run B0 before B1 so paired outputs exist:

```powershell
python -m pip install -e ".[captioning,coco-eval,ner]"
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python scripts/run_captioning.py --config configs/experiments/b0_image_only.yaml
python scripts/run_captioning.py --config configs/experiments/b1_full_article.yaml
python scripts/run_captioning.py --config configs/experiments/b1_random_article.yaml
python scripts/analyze_context_sanity.py `
  --b0-dir runs/goodnews_validation_50/b0 `
  --b1-dir runs/goodnews_validation_50/b1 `
  --b1-random-dir runs/goodnews_validation_50/b1_random `
  --output runs/goodnews_validation_50/context_sanity.json
```

The B1 runner validates all controlled B0/B1 configuration fields and then
writes 95% paired-bootstrap comparisons for CIDEr and entity precision,
recall, and F1 under `runs/b1_full_article/dev_smoke/comparisons/`.
