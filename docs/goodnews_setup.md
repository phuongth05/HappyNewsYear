# GoodNews official release setup

This project does not redistribute GoodNews and never downloads it from an
unofficial mirror. The authoritative sources are the
[official GoodNews repository](https://github.com/furkanbiten/GoodNews) and the
[CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Biten_Good_News_Everyone_Context_Driven_Entity-Aware_Captioning_for_News_Images_CVPR_2019_paper.pdf).

## Required directory structure

The repository accepts the official release or a manually downloaded copy in
any directory, including outside this Git repository:

```text
D:/Datasets/GoodNews/
├── article+caption.json
├── img_splits.json
└── images/
    ├── <article-id>_<image-index>.jpg
    └── ...
```

The official downloader writes images as `<article-id>_<integer-index>.jpg`.
For example, article ID `abc` and image index `0` resolve to `abc_0.jpg`.
JSON object keys are strings after parsing, even when the upstream Python code
created integer image indexes.

## `article+caption.json`

The upstream collection scripts create an article-keyed JSON object. The
fields directly established by those scripts are:

```json
{
  "<article-id>": {
    "article_url": "https://...",
    "article": "full cleaned article text",
    "images": {
      "0": "caption for image zero",
      "1": "caption for image one"
    }
  }
}
```

`article_url` is metadata and is not required by this project. `article` and
`images` are required. There is one caption per image entry. Compatible copies
may use an image object containing `caption`, `raw`, or `text` plus an explicit
filename/ID; those aliases are configurable, but the scientific pipeline is
defined against the official article/image relationship above.

## `img_splits.json`

The official README states that this artifact records each image and its
`train`, `val`, or `test` split. It does not include the downloadable file in
Git or publish an example of its exact top-level JSON serialization. Therefore
we do not claim an unverified single serialization format.

The verifier accepts the representations already supported by the loader:

- image ID/filename mapped to `train`, `val`, or `test`;
- `train`/`val`/`test` mapped to lists of image IDs;
- rows containing an image ID field and `split`.

In every representation IDs are normalized to filename stems and must resolve
to exactly one `<article-id>_<image-index>` entry. Only the official labels
`train`, `val`, and `test` pass release validation. Project code maps official
`val` to internal `dev`; it does not regenerate a split.

## Release scale

The CVPR paper reports approximately 466,000 image-caption samples: about
424,000 train, 18,000 validation, and 23,000 test images. These rounded values
do not sum exactly to the rounded total and are not used as validation
thresholds. The official README does not state an exact release count or exact
unique-article count. The audit report always derives counts from the supplied
files.

## Acquisition and preprocessing

The official README links all of the following from the authors' SharePoint:

- cleaned `article+caption.json`;
- official `img_splits.json`;
- `img_urls.json`;
- a complete image archive.

Preferred route: manually download the cleaned annotations, official split
file, and complete image archive. Some author-hosted links may require a
Microsoft login.

No resizing, HDF5 conversion, caption anonymization, sentence embedding, or
new split generation is required for this project. InstructBLIP performs image
preprocessing at inference time. Do not run upstream `clean_captions.py` to
make a new random split.

If only `img_urls.json` is available, the official Python-2-era downloader can
be ported and run manually. The authors note that some URLs are broken. The
checked-in upstream script also reads `img_urls_all.json`, despite the README
calling the download `img_urls.json`; resolve that filename manually rather
than changing this project's dataset format.

No unofficial mirror has been selected or integrated. If one is used for
transport, its contents must still pass the official-layout verifier without
mirror-specific conversion in model code.

## External path configuration

The dataset config supports an absolute external root and defaults to the
three official artifact names:

```yaml
dataset:
  name: goodnews
  root: D:/Datasets/GoodNews
```

The longer form can override filenames for a compatible local copy:

```yaml
dataset:
  name: goodnews
  root: D:/Datasets/GoodNews
  annotations_path: article+caption.json
  splits_path: img_splits.json
  images_root: images
```

## Verification

Quick validation parses the complete annotation and split artifacts, but treats
the IDs referenced by the official train/val/test splits as the authoritative
experimental population. Missing annotations/images, empty articles/captions,
duplicates, and overlaps in that population are fatal. Annotation entries that
are not referenced by an official split are retained and reported separately as
warnings; they do not invalidate the experiment release. The audit then uses
`random.Random(2026)` to Pillow-verify only the requested number of official
split images:

```powershell
python scripts/audit_dataset.py `
  --dataset goodnews `
  --root "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews" `
  --quick `
  --verify-images 10000 `
  --output artifacts/dataset_audits/goodnews_quick.json `
  --fail-on-issues
```

Quick reports explicitly contain `"verification_scope": "sampled"` and
`"full_image_verification_completed": false`. Passing quick mode means the
sampled images decoded successfully; it never claims that all images are free
of corruption.

To validate the exact same first 50 sorted development IDs used by B0, B1, and
B1-random:

```powershell
python scripts/audit_experiment_subset.py `
  --dataset-root "D:/KIEMCOM/HK1-N4/KLTN/Datasets/GoodNews" `
  --split dev `
  --max-samples 50 `
  --selection first_by_sample_id
```

This command uses the same shared selector as the experiment runner, prints all
50 IDs, and fully verifies their images, articles, and captions.

Quick mode remains O(number of annotations + number of split paths) because
the full structural and existence checks are intentional. Its expensive image
decode component is O(N) instead of O(all images). At 10,000 versus roughly
647,000 images, the decode portion is about 65 times smaller; actual wall-clock
speedup is lower because JSON parsing and path checks are shared fixed costs.

Release validation without loading a Hugging Face tokenizer:

```powershell
python scripts/audit_dataset.py `
  --dataset goodnews `
  --root "D:/Datasets/GoodNews" `
  --output artifacts/dataset_audits/goodnews_validation.json `
  --check-only `
  --fail-on-issues
```

This still opens every present image referenced by the official splits with
Pillow and calls `verify()`, so on the full release it is I/O intensive. It
checks files, JSON, IDs, splits, captions, articles, duplicates, missing images,
and corrupted images. It never modifies the dataset or regenerates splits.

After check-only passes, compute statistics with the exact pinned B1
InstructBLIP language tokenizer:

```powershell
python scripts/audit_dataset.py `
  --dataset goodnews `
  --root "D:/Datasets/GoodNews" `
  --output artifacts/dataset_audits/goodnews_validation.json `
  --fail-on-issues
```

This downloads only the processor/tokenizer files when they are not cached,
not the model weights. Statistics are computed over the supplied real release
and include split counts, unique articles, article/caption token distributions,
images per article, and truncation at the B1 limit of 384 tokens. Both
unique-article and per-image-context truncation rates are reported.

## Running the same 50 development samples

Set `root` in `configs/dataset/goodnews.example.yaml` (or point all three
experiment configs at an equivalent local dataset config), then run in order:

```powershell
python scripts/run_captioning.py --config configs/experiments/b0_instructblip_image_only.yaml
python scripts/run_captioning.py --config configs/experiments/b1_instructblip_article_context.yaml
python scripts/run_captioning.py --config configs/experiments/b1_instructblip_random_article.yaml
```

All three configs use `split: dev`, `max_samples: 50`, and
`first_by_sample_id`, so they select the same official validation sample IDs.
B1-random changes only the seeded article donor assignment and is a sanity
control, not a main baseline.
