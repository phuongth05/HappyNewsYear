# M3 atomic evidence extraction audit

This stage asks whether changing evidence granularity alone is promising. It
does not generate captions, retrieve additional sentences, train an extractor,
or implement evidence-claim matching.

## Frozen input boundary

The only accepted input is the completed B2 semantic-k3
`selected_evidence.jsonl`. The loader requires the approved ordered 50 IDs,
`retrieval_method=semantic`, `retrieval_k=3`, aligned sentence IDs/texts/ranks/
scores, and original article order. It has no dataset-root or article-loading
argument. Reference captions, generated captions, full articles, gold entities,
and claim annotations therefore cannot enter the extractor.

Frozen retrieval:

- `openai/clip-vit-base-patch32`
- revision `b97b0100e55e367c057773c2a614676470b0d575`
- k = 3

## Extraction method

The initial audit uses `en_core_web_sm` dependency parsing and NER with
deterministic rules (`spacy_dependency_atomic_v1`). It is an existing
information-extraction model, not a newly trained component. There is no LLM
prompt or sampling; the recorded prompt version is
`not_applicable_rule_based_v1` and temperature is null.

Named entities become complete propositions such as `Denver is a location.`
rather than isolated keywords. Dependency clauses retain a subject, predicate,
arguments, and relevant prepositional phrases. If neither NER nor a supported
clause is found, the source sentence is retained as an `external_fact` and is
expected to be reviewed for excessive coarseness.

The exact output schema is:

```json
{
  "evidence_id": "ae_<stable-sha256-prefix>",
  "text": "Complete factual proposition.",
  "type": "entity|object|attribute|relation|event|location|time|external_fact",
  "source_sentence_id": 12,
  "source_span": {"start": 0, "end": 24, "text": "supporting source text"},
  "source_rank": 1,
  "metadata": {
    "ranking_score": 0.31,
    "extractor_rule": "dependency_clause"
  }
}
```

`source_span` is null rather than fabricated if alignment cannot be established.
Schema-invalid structured output is retried once and then fails explicitly.
Validated results are cached in SQLite using the source, provenance, model, and
rule versions.

## Run

Place or extract the completed B2 artifact at the path declared by the config,
then run:

```bash
python scripts/extract_atomic_evidence.py \
  --config configs/experiments/m3_atomic_evidence_audit.yaml
```

This loads spaCy and the pinned InstructBLIP tokenizer only. It does not load
InstructBLIP weights and does not run generation.

## Context and token controls

The untyped representation is deterministic:

```text
Evidence:
- First proposition.
- Second proposition.
```

Units stay in original source-sentence order and then source-span order. The
normal atomic context is capped at 384 InstructBLIP tokens without padding. A
second context is greedily capped at `min(384, B2 selected-sentence tokens)` for
token-matched analysis. Both contexts retain only whole evidence units.

## Outputs

Under `artifacts/m3_atomic_evidence_50/`:

- `atomic_evidence.jsonl`: all validated units and frozen B2 provenance;
- `atomic_contexts.jsonl`: 384-token and token-matched untyped contexts;
- `atomic_evidence_audit.json`: aggregate quality/token statistics;
- `manual_review_100.jsonl`: 100 source sentences with blank human labels;
- `manual_review_100.csv`: editable review worksheet;
- `atomic_extraction_cache.sqlite3`: deterministic extraction cache.
- `config.resolved.json` and `manifest.json`: resolved paths, source/config
  hashes, extractor version, Git state, and an explicit no-generation record.

Human reviewers fill only these values:

- atomicity: `good`, `too_coarse`, or `over_split`;
- faithfulness: `supported`, `unsupported`, or `uncertain`;
- type_correct: `yes` or `no`;
- notes: free text.

Heuristic flags are triage signals, not ground-truth quality labels. Caption
generation must not begin until this worksheet has been reviewed.
