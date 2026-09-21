"""Pinned premise-evidence, hypothesis-claim NLI matcher."""

from __future__ import annotations

from typing import Sequence

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"


class NliMatcher:
    method = "nli_entailment_v1"

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        revision: str = NLI_REVISION,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device_request = device
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = "unloaded"
        self.entailment_index = 1

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = (
            "cuda" if self.device_request == "auto" and torch.cuda.is_available()
            else "cpu" if self.device_request == "auto"
            else self.device_request
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision
        ).to(self.device)
        self.model.eval()
        labels = {
            int(key): str(value).casefold()
            for key, value in self.model.config.id2label.items()
        }
        matches = [index for index, label in labels.items() if label == "entailment"]
        if matches != [1]:
            raise ValueError(f"unexpected NLI entailment label mapping: {labels}")
        self.entailment_index = matches[0]

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Pairs are always (evidence premise, claim hypothesis)."""

        if self.model is None or self.tokenizer is None:
            self.load()
        if not pairs:
            return []
        import torch

        results: list[float] = []
        for offset in range(0, len(pairs), self.batch_size):
            batch = pairs[offset : offset + self.batch_size]
            premises = [pair[0] for pair in batch]
            hypotheses = [pair[1] for pair in batch]
            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[:, self.entailment_index]
            results.extend(float(value) for value in probabilities.detach().cpu().tolist())
        return results

    def model_info(self) -> dict[str, object]:
        return {
            "method": self.method,
            "model": self.model_name,
            "revision": self.revision,
            "premise": "atomic_evidence_text",
            "hypothesis": "caption_claim_text",
            "score": "softmax_entailment_probability",
            "entailment_label_index": self.entailment_index,
        }