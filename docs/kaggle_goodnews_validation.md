# Kaggle GoodNews validation (50 dev samples)

This workflow runs only the controlled B0, B1-correct, and B1-random
InstructBLIP validation. It does not implement or run B2.

## Kaggle setup

Enable a GPU accelerator and attach the validated GoodNews dataset. Its mount
must directly contain:

```text
/kaggle/input/<goodnews-dataset-folder>/
  article+caption.json
  img_splits.json
  images/
```

The folder name is supplied at runtime and is never hard-coded. In Kaggle
notebook cells, run:

```bash
!git clone https://github.com/phuongth05/HappyNewsYear.git /kaggle/working/MERGE
%cd /kaggle/working/MERGE
!python -m pip install -e ".[captioning,coco-eval,ner]"
!python -m spacy download en_core_web_sm
```

The checkpoint is about 16.1 GB on disk at the pinned Hugging Face revision,
so either enable Kaggle Internet for the first download or make the exact
revision available in the Hugging Face cache. Do not substitute a different
checkpoint silently.

## One-command run

Replace only the input folder placeholder:

```bash
!python scripts/run_goodnews_validation.py \
  --dataset-root "/kaggle/input/<goodnews-dataset-folder>" \
  --output-root "/kaggle/working/MERGE/runs/goodnews_validation_50" \
  --bundle "/kaggle/working/goodnews_validation_50.zip"
```

The workflow refuses existing predictions and an existing ZIP. Use a new
output location or explicitly remove an obsolete run before starting again.

Before model loading, it repeats full validation of the exact 50 selected
images/articles/captions, compares the selected ordered IDs with
`configs/dataset/goodnews_validation_50_ids.json`, validates all controlled
configuration fields, and runs CUDA preflight in a short-lived process.

The three model conditions then run in separate processes in this order:

1. `b0`
2. `b1`
3. `b1_random`
4. saved-output context sanity analysis

Each model process calls `gc.collect()` and `torch.cuda.empty_cache()` before
exit. Process termination then releases its CUDA context before the next run.
An OOM is fatal: the workflow does not enable quantization, alter decoding, or
retry with different settings.

## Pinned scientific controls

- model: `Salesforce/instructblip-flan-t5-xl`
- revision: `bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d`
- dtype: float16
- batch size: 1
- seed: 2026
- split: GoodNews dev (official `val`)
- samples: first 50 ordered by `sample_id`
- instruction: `Describe the image in one factual news-style sentence.`
- article cap: 384 tokens with deterministic head truncation
- decoding: beam search with 3 beams and at most 30 new tokens

B0 and B1 use the same checkpoint, processor, instruction, decoding,
evaluation, seed, and samples. Only article-context access changes. B1-random
uses a seed-derived cyclic derangement and can never receive its own article.

## Outputs

The run root is:

```text
/kaggle/working/MERGE/runs/goodnews_validation_50/
```

Each of `b0/`, `b1/`, and `b1_random/` contains `predictions.jsonl`,
`metrics.json`, `metrics_per_sample.csv`, `resolved_config.json`,
`runtime.json`, and `gpu_memory.json`, plus manifests and supporting runtime
artifacts. B1 prediction metadata includes original/used article token counts,
truncation ratio, maximum context tokens, generation tokens, and selected
article sample ID.

The run root additionally contains:

- `gpu_preflight.json`
- `subset_preflight.json`
- `scientific_controls.json`
- `context_sanity.json`
- `qualitative_examples.json` (all 50 aligned samples)
- `qualitative_examples.md` (10 deterministic evenly spaced samples)
- `workflow_runtime.json`
- generated runtime configs recording the external dataset path

The final archive is:

```text
/kaggle/working/goodnews_validation_50.zip
```

It is built only from the experiment output root; dataset images and model
weights are not included.

## T4 memory expectation

The checkpoint files total about 16.1 GB in their stored representation. FP16
model weights are expected to occupy roughly 8 GB of device memory. With the
vision encoder, Q-Former, 384-token input, three-beam generation, CUDA kernels,
and allocator reserve, a practical peak estimate is roughly 10–13 GiB on a
16-GB T4. This is an estimate, not a guarantee; the workflow records actual
peak allocated and peak reserved bytes for every condition.
