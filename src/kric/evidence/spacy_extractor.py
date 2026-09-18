"""Deterministic atomic propositions from a fixed spaCy parser/NER pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable

from .schema import AtomicEvidence, FrozenSentence, SourceSpan, stable_evidence_id


RULE_SET_VERSION = "spacy_dependency_atomic_v1"
PROMPT_VERSION = "not_applicable_rule_based_v1"

_ENTITY_TYPE = {
    "PERSON": ("entity", "person"),
    "ORG": ("entity", "organization"),
    "NORP": ("entity", "group"),
    "GPE": ("location", "location"),
    "LOC": ("location", "location"),
    "FAC": ("location", "place"),
    "DATE": ("time", "time"),
    "TIME": ("time", "time"),
    "EVENT": ("event", "event"),
    "WORK_OF_ART": ("object", "work"),
    "PRODUCT": ("object", "product"),
}


def _normalize_sentence(text: str) -> str:
    return " ".join(text.split())


def _finish(text: str) -> str:
    clean = re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split())).strip()
    if clean and clean[-1] not in ".!?":
        clean += "."
    return clean[:1].upper() + clean[1:] if clean else clean


class SpacyAtomicExtractor:
    """Dependency-based IE baseline; consumes one already-selected sentence."""

    def __init__(self, model_name: str = "en_core_web_sm", nlp: Any | None = None) -> None:
        if nlp is None:
            import spacy

            nlp = spacy.load(model_name)
        self.nlp = nlp
        self.model_name = model_name
        meta = getattr(nlp, "meta", {})
        self.model_version = str(meta.get("version", "unknown"))

    def info(self) -> dict[str, Any]:
        try:
            spacy_version = version("spacy")
        except PackageNotFoundError:
            spacy_version = "unknown"
        return {
            "method": "deterministic_spacy_dependency_and_ner",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "spacy_version": spacy_version,
            "rule_set_version": RULE_SET_VERSION,
            "prompt_version": PROMPT_VERSION,
            "temperature": None,
            "structured_llm": False,
        }

    def cache_key(self, sentence: FrozenSentence) -> str:
        payload = {
            "extractor": self.info(),
            "sample_id": sentence.sample_id,
            "sentence_id": sentence.sentence_id,
            "text": sentence.text,
            "source_rank": sentence.source_rank,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _span(source: str, start: int, end: int) -> SourceSpan | None:
        if start < 0 or end <= start or end > len(source):
            return None
        return SourceSpan(start=start, end=end, text=source[start:end])

    def _unit(
        self,
        sentence: FrozenSentence,
        text: str,
        evidence_type: str,
        span: SourceSpan | None,
        **metadata: Any,
    ) -> AtomicEvidence:
        finished = _finish(text)
        return AtomicEvidence(
            evidence_id=stable_evidence_id(
                sentence.sample_id,
                sentence.sentence_id,
                evidence_type,
                finished,
                span,
            ),
            text=finished,
            type=evidence_type,
            source_sentence_id=sentence.sentence_id,
            source_span=span,
            source_rank=sentence.source_rank,
            metadata={
                "ranking_score": sentence.ranking_score,
                "extractor_rule": metadata.pop("extractor_rule"),
                **metadata,
            },
        )

    @staticmethod
    def _subject(root: Any) -> Any | None:
        for child in root.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                return child
        if root.dep_ == "conj":
            for ancestor in root.ancestors:
                for child in ancestor.children:
                    if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
                        return child
        return None

    @staticmethod
    def _clause_tokens(root: Any, subject: Any) -> list[Any]:
        allowed = {
            "aux",
            "auxpass",
            "neg",
            "prt",
            "dobj",
            "obj",
            "iobj",
            "dative",
            "attr",
            "acomp",
            "oprd",
            "prep",
            "agent",
            "xcomp",
        }
        tokens = set(subject.subtree)
        tokens.add(root)
        for child in root.children:
            if child.dep_ in allowed:
                tokens.update(child.subtree)
        return sorted(
            (token for token in tokens if not token.is_space and not token.is_punct),
            key=lambda token: token.i,
        )

    def extract(self, sentence: FrozenSentence) -> list[AtomicEvidence]:
        source = sentence.text
        doc = self.nlp(source)
        units: list[AtomicEvidence] = []

        for entity in doc.ents:
            mapped = _ENTITY_TYPE.get(entity.label_)
            if mapped is None:
                continue
            evidence_type, noun = mapped
            if evidence_type == "time":
                proposition = f"The time is {entity.text}"
            elif evidence_type == "location":
                proposition = f"{entity.text} is a {noun}"
            else:
                article = "an" if noun[:1].lower() in "aeiou" else "a"
                proposition = f"{entity.text} is {article} {noun}"
            units.append(
                self._unit(
                    sentence,
                    proposition,
                    evidence_type,
                    self._span(source, entity.start_char, entity.end_char),
                    extractor_rule="named_entity_proposition",
                    entity_label=entity.label_,
                )
            )

        roots: Iterable[Any] = (
            token
            for token in doc
            if token.pos_ in {"VERB", "AUX"}
            and token.dep_ in {"ROOT", "conj", "ccomp", "advcl"}
        )
        for root in roots:
            subject = self._subject(root)
            if subject is None:
                continue
            tokens = self._clause_tokens(root, subject)
            if len(tokens) < 3:
                continue
            start = tokens[0].idx
            end = tokens[-1].idx + len(tokens[-1].text)
            proposition = " ".join(token.text for token in tokens)
            has_attribute = any(
                child.dep_ in {"attr", "acomp", "oprd"} for child in root.children
            )
            evidence_type = (
                "attribute"
                if has_attribute
                else "relation"
                if root.lemma_.casefold() in {"be", "have", "include", "contain"}
                else "event"
            )
            units.append(
                self._unit(
                    sentence,
                    proposition,
                    evidence_type,
                    self._span(source, start, end),
                    extractor_rule="dependency_clause",
                    root_lemma=root.lemma_,
                )
            )

        if not units:
            units.append(
                self._unit(
                    sentence,
                    source,
                    "external_fact",
                    self._span(source, 0, len(source)),
                    extractor_rule="sentence_fallback",
                )
            )

        deduplicated: list[AtomicEvidence] = []
        seen = set()
        for unit in units:
            key = (unit.type, unit.text.casefold())
            if key not in seen:
                seen.add(key)
                deduplicated.append(unit)
        return deduplicated
