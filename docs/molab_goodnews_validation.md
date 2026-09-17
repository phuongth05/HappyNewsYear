# molab/marimo GoodNews validation (50 dev samples)

This guide runs the same controlled B0, B1-correct, and B1-random validation
used elsewhere. It does not change the model, prompts, decoding, evaluator,
seed, or approved sample IDs, and it does not implement B2.

## Assumed molab environment

- a CUDA-capable GPU with at least 14 GiB total and 12 GiB currently free VRAM;
- at least 32 GB system RAM;
- Python 3.10 or newer;
- Git access;
- Internet access for the pinned Hugging Face checkpoint and Python packages.

The workflow does not check for a particular GPU model. CUDA availability and
capacity are validated from the reported total/free memory before model load.

## Prepare the repository

From a molab terminal:

```bash
git clone https://github.com/phuongth05/HappyNewsYear.git MERGE
cd MERGE
python -m pip install -e ".[captioning,coco-eval,ner]"
python -m spacy download en_core_web_sm
python -m pip install marimo
```

The repository revision used for the run should contain the workflow changes
and should be recorded by Git. The runner records the commit and dirty state in
each condition manifest.

## Compact approved dataset

Build this package locally from the validated full release; do not reconstruct
it from captions, URLs, or another mirror:

```powershell
python scripts/package_goodnews_subset.py `
  --source-root "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews" `
  --ids "configs/dataset/goodnews_validation_50_ids.json" `
  --output "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews_validation_50"
```

The package contains only `article+caption.json`, `img_splits.json`, the exact
50 images, and `subset_manifest.json`. It preserves the official `val` identity
(project split `dev`), raw article/caption values, and image bytes. The manifest
states explicitly that this is only for the approved 50-sample sanity
experiment and is not a replacement for full GoodNews.

For the currently validated release, the measured unpacked package size is
1,008,532 bytes (about 0.96 MiB). A future source release with byte-identical
semantics but different image encodings could have a different size; rely on
the generated manifest checksums rather than size alone.

For upload through a browser, create an archive without adding the full source
dataset:

```powershell
Compress-Archive `
  -Path "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews_validation_50/*" `
  -DestinationPath "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews_validation_50.zip"
```

The measured ZIP for the current package is 832,528 bytes (about 0.79 MiB).
Upload it to molab, extract it, and pass the extracted directory—not the ZIP
path—to `--dataset-root`.

Upload the complete `GoodNews_validation_50/` directory to persistent molab
storage. After upload, its root must look like:

```text
<molab-data>/GoodNews_validation_50/
  article+caption.json
  img_splits.json
  subset_manifest.json
  images/                 # exactly 50 images
```

## Command-line validation

Choose paths appropriate for the current molab workspace:

```bash
python scripts/audit_experiment_subset.py \
  --dataset-root "/path/to/GoodNews_validation_50" \
  --split dev \
  --max-samples 50 \
  --selection first_by_sample_id
```

Then run the authoritative cloud-agnostic workflow:

```bash
python scripts/run_goodnews_validation.py \
  --dataset-root "/path/to/GoodNews_validation_50" \
  --output-root "/path/to/outputs/goodnews_validation_50" \
  --bundle "/path/to/outputs/goodnews_validation_50.zip"
```

These are the only three runtime paths consumed by the authoritative runner.
It refuses to overwrite existing predictions or bundle files.

## marimo notebook

Start the thin orchestration notebook:

```bash
marimo edit notebooks/molab_goodnews_validation.py
```

Set the repository, dataset, output, and bundle paths in its UI. The notebook
only invokes the same audit, GPU preflight, and validation CLIs, then displays
the saved `context_sanity.json`. It contains no generation, dataset-selection,
evaluation, or packaging logic of its own.

## Outputs and capacity notes

The output layout and scientific controls are identical to the Kaggle workflow.
The unquantized FP16 model is expected to peak around 10–13 GiB, but actual peak
allocated/reserved memory is recorded for every condition. If CUDA is missing,
free VRAM is below 12 GiB, total VRAM is below 14 GiB, or a run OOMs, execution
stops without changing decoding or introducing quantization.
