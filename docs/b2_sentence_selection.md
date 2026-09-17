# B2: top-k relevant article sentences

B2 is the next controlled GoodNews validation stage after the saved 50-sample
InstructBLIP B1 run. It does not implement atomic evidence, claim matching, or
provenance.

## Scientific controls

Every B2 variant uses the same approved ordered 50 development IDs and the
same pinned `Salesforce/instructblip-flan-t5-xl` revision, image processing,
instruction, 384-token context ceiling, generation settings, seed, and saved
prediction evaluator as B1. The only changed component is article-context
selection.

The generator receives only the selected article sentences. Reference
captions, reference entities, generated captions, and test annotations are not
available to either retriever. Selected sentences are restored to their
original article order before prompt construction.

Six variants are explicit configs:

- BM25 with k = 1, 3, and 5. The query is the input article headline, with the
  lead sentence as a documented fallback when the headline is absent.
- CLIP image-sentence cosine similarity with k = 1, 3, and 5. The checkpoint is
  `openai/clip-vit-base-patch32` at revision
  `b97b0100e55e367c057773c2a614676470b0d575`.

Ranking is precomputed in a separate process. The semantic embedding cache is
reused by k = 1/3/5, and the CLIP process exits before InstructBLIP is loaded.
This avoids simultaneous residency of both models.

## Context-budget behavior

Candidates are considered in relevance-rank order. A candidate is selected
only when the complete sentence still fits the exact InstructBLIP prompt under
the 384-token context policy. Over-budget candidates are logged and skipped.
The selected set is then reordered by original sentence index. If no complete
sentence fits, the highest-ranked sentence is used with the existing head
truncation policy and the fallback is explicitly recorded.

Each prediction logs:

- retrieval method and k;
- candidate and selected sentence counts;
- selected sentence IDs, relevance ranks, scores, and text;
- context token count and original article token count;
- fraction of article tokens and sentences represented;
- budget-skipped sentence IDs and any truncation fallback.

The same information is also written to `selected_evidence.jsonl`, with
aggregate statistics in `context_statistics.json`.

## Install and run on a CUDA host

Install the existing captioning/evaluation dependencies:

```bash
python -m pip install -e ".[captioning,coco-eval,ner]"
python -m spacy download en_core_web_sm
```

The B1 directory must be the completed saved run for the approved 50 samples
and must contain `predictions.jsonl`, `metrics.json`,
`metrics_per_sample.csv`, and `resolved_config.json`.

Run all six variants and the paired analysis with one cloud-agnostic command:

```bash
python scripts/run_b2_validation.py \
  --dataset-root "/path/to/GoodNews_validation_50" \
  --b1-dir "/path/to/goodnews_validation_50/b1" \
  --output-root "/path/to/outputs/goodnews_b2_validation_50" \
  --bundle "/path/to/outputs/goodnews_b2_validation_50.zip"
```

The runner refuses a non-empty output directory or an existing bundle. It
first verifies the exact 50 IDs and the saved B1 controls, then performs GPU
preflight, both rankings, all six runs, 10,000-resample paired bootstrap
comparisons, and packaging.

To debug retrieval without loading InstructBLIP, run either ranker directly:

```bash
python scripts/rank_goodnews_sentences.py \
  --dataset-root "/path/to/GoodNews_validation_50" \
  --method bm25 \
  --output artifacts/b2/bm25.jsonl

python scripts/rank_goodnews_sentences.py \
  --dataset-root "/path/to/GoodNews_validation_50" \
  --method semantic \
  --output artifacts/b2/semantic.jsonl \
  --cache artifacts/b2/clip_sentence_embeddings.sqlite3
```

## Analysis outputs and selection rule

`b2_analysis.json` reports CIDEr and Entity Precision/Recall/F1 for every
variant, 10,000-resample paired confidence intervals against B1, context-token
efficiency, retrieval-score summaries, and BM25-versus-semantic sentence-set
difference rates. `qualitative_examples.json` contains every aligned sample,
caption, and selected sentence. `qualitative_review.md` is an intentionally
unlabeled worksheet for coding these requested behaviors:

- A: irrelevant article information is removed;
- B: selection loses an important fact;
- C: BM25 and semantic retrieval choose meaningfully different evidence;
- D: direct article copying is reduced;
- E: a sentence is visually relevant but not reference-relevant.

Select and report ten examples only after human review, ideally at least two
per category when the data actually contains them. Do not infer those labels
from CIDEr alone. Choose a B2 configuration using development metrics,
context efficiency, and the qualitative coding together; never use the test
split for this choice.

