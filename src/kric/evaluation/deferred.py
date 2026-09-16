"""Declared interfaces for claim/evidence metrics not implemented in this stage."""

from __future__ import annotations

from collections.abc import Sequence

from .base import EvaluationMetric
from .records import EvaluationRecord, MetricResult


class DeferredMetric(EvaluationMetric):
    def __init__(self, name: str, required_inputs: Sequence[str], reason: str):
        self.name = name
        self.required_inputs = tuple(required_inputs)
        self.reason = reason

    def evaluate(self, records: Sequence[EvaluationRecord]) -> MetricResult:
        return MetricResult(
            summary={},
            details={
                "status": "deferred",
                "required_inputs": list(self.required_inputs),
                "reason": self.reason,
            },
        )


def deferred_metrics() -> dict[str, DeferredMetric]:
    return {
        "unsupported_claim_rate": DeferredMetric(
            "unsupported_claim_rate",
            ("atomic_claims", "evidence", "claim_support_labels"),
            "Requires the later atomic-claim and evidence-support stages.",
        ),
        "claim_support_f1": DeferredMetric(
            "claim_support_f1",
            ("atomic_claims", "evidence", "claim_support_labels"),
            "Requires gold or validated claim-support decisions.",
        ),
        "attribution": DeferredMetric(
            "attribution",
            ("predicted_claim_evidence_links", "gold_claim_evidence_links"),
            "Requires claim-level provenance annotations.",
        ),
    }

