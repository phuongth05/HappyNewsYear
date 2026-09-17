"""Deterministic, dependency-light sentence segmentation for news articles."""

from __future__ import annotations

import re


_DOT = "<KRIPERIOD>"
_ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
    "Sen.", "Rep.", "Gov.", "Gen.", "Lt.", "Col.", "Capt.",
    "Inc.", "Ltd.", "Co.", "Corp.", "U.S.", "U.K.", "No.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
    "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "a.m.", "p.m.",
    "e.g.", "i.e.", "vs.",
)


def _protect_periods(text: str) -> str:
    text = re.sub(r"(?<=\d)\.(?=\d)", _DOT, text)
    text = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda match: match.group(0)[:-1].replace(".", _DOT) + ".",
        text,
    )
    for abbreviation in _ABBREVIATIONS:
        # Titles must stay attached to a following name. Other abbreviations
        # retain their final period so they can still terminate a sentence.
        protect_final = abbreviation.lower() in {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
            "sen.", "rep.", "gov.", "gen.", "lt.", "col.", "capt.",
            "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
            "sep.", "sept.", "oct.", "nov.", "dec.",
        }
        protected = abbreviation.replace(".", _DOT)
        if not protect_final:
            protected = protected[: -len(_DOT)] + "."
        text = re.sub(
            re.escape(abbreviation),
            protected,
            text,
            flags=re.IGNORECASE,
        )
    return text


def segment_sentences(text: str) -> list[str]:
    """Split an article deterministically while handling common news abbreviations.

    Paragraph boundaries always create a boundary. Within paragraphs, a
    boundary requires terminal punctuation followed by a likely sentence
    starter. This intentionally avoids downloading a mutable NLP model.
    """

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    sentences: list[str] = []
    paragraphs = re.split(r"\n\s*\n|\n", text)
    for paragraph in paragraphs:
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if not paragraph:
            continue
        protected = _protect_periods(paragraph)
        pieces = re.split(
            r"(?<=[.!?])\s+(?=[\"'“‘(\[]*[A-Z0-9])",
            protected,
        )
        for piece in pieces:
            restored = piece.replace(_DOT, ".").strip()
            if restored:
                sentences.append(restored)
    return sentences
