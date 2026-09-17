# Controlled B0/B1 architecture decision

## Candidates

| Candidate | Pinned checkpoint | Parameters | Native image + text conditioning | Context | T4 feasibility |
|---|---|---:|---|---|---|
| BLIP-2 Flan-T5 XL | `Salesforce/blip2-flan-t5-xl@0eb0d3b46c14c1f8c7680bca2693baafdb90bb28` | 3,942,446,592 | Yes. Text conditions the language encoder; the Q-Former itself is image-only. It is not instruction-tuned. | T5 declares 512 positions; 32 image query tokens | Expected feasible at FP16, batch 1 (about 10–12 GB peak; estimate, not measured here) |
| InstructBLIP Flan-T5 XL | `Salesforce/instructblip-flan-t5-xl@bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d` | 4,022,969,088 | Yes. Instruction/article text conditions both the Q-Former and the encoder-decoder language model. | T5 declares 512 positions; 32 image query tokens | Expected feasible but close at FP16, batch 1 (about 10–13 GB peak; estimate, not measured here) |
| InstructBLIP Vicuna 7B | `Salesforce/instructblip-vicuna-7b@19103d0c5b5263c8a7891012e08573439fb6607f` | 7,913,726,720 | Yes, but its decoder-only generation makes prompt/continuation accounting less clean. | 2,048 tokens; 32 image query tokens | Not practical unquantized on a 16 GB T4; quantization would introduce another experimental variable |
| Existing repository BLIP | `Salesforce/blip-image-captioning-base@82a37760796d32b1411fe092ab5d4e227313294b` | about 247M | No native long multimodal instruction interface. The legacy B1 uses article tokens as a decoder prefix. | 512 decoder positions | Easy, but scientifically unsuitable as definitive B1 |

No multimodal instruction model adapter existed in the repository before this change. The only supported generator was the BLIP adapter, which remains available as an exploratory/legacy baseline.

Parameter counts for BLIP-2 and InstructBLIP are the exact Hugging Face repository tensor counts. Memory figures are engineering estimates from FP16 parameter storage plus activations and runtime overhead; they must be measured on the actual Colab runtime before a full run.

## Decision

Use InstructBLIP Flan-T5 XL. It provides the cleanest controlled intervention:

- B0: image plus the fixed caption instruction.
- B1: image plus article context plus the same fixed caption instruction.
- Both conditions use the same checkpoint, image processor, tokenizers, generation parameters, seed, split, and evaluator.
- The language model is encoder-decoder. Article text is encoder conditioning, not a decoder prefix that generation continues.

The implementation pins the revision, rejects a decoder-only checkpoint, and makes the processor's 32 image-query-token expansion explicit.

## Exact input formats

B0 text input:

```text
Describe the image in one factual news-style sentence.
```

B1 text input:

```text
Article context:
<article prefix>

Describe the image in one factual news-style sentence.
```

The prompt configuration is structurally identical in B0 and B1. Only B1 inserts the article context block. B1-random uses the same format with a seeded, non-self donor article.

## Context policy

- `max_context_tokens`: 384 language-tokenizer tokens from the head of the input article.
- The prompt is also checked against the Q-Former tokenizer.
- The complete language input, including 32 image query tokens, must fit the checkpoint's declared 512-position encoder window.
- If prompt overhead makes 384 article tokens impossible, the implementation reduces the article prefix and reports the actual count.
- `generation_tokens`: 30. Because Flan-T5 is encoder-decoder, these decoder tokens do not consume encoder positions, but they are logged separately.
- Per sample: `original_article_tokens`, `used_article_tokens`, `truncation_ratio`, `max_context_tokens`, and `generation_tokens`.

This is therefore named `B1_instructblip_article_context`, not “full article.”

## Generation API

The adapter uses `InstructBlipProcessor(images=..., text=...)` and
`InstructBlipForConditionalGeneration.generate(...)`. The processor produces image, language-model, and Q-Former inputs. Reference captions, reference entities, claims, and annotations cannot enter the narrow model-facing input dataclasses.

## Sources

- [InstructBLIP Flan-T5 XL model card](https://huggingface.co/Salesforce/instructblip-flan-t5-xl)
- [InstructBLIP Flan-T5 XL configuration](https://huggingface.co/Salesforce/instructblip-flan-t5-xl/blob/main/config.json)
- [BLIP-2 Flan-T5 XL model card](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- [InstructBLIP paper](https://arxiv.org/abs/2305.06500)
- [Transformers InstructBLIP documentation](https://huggingface.co/docs/transformers/model_doc/instructblip)
