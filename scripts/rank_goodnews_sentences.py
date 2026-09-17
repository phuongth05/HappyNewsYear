"""Precompute reference-free GoodNews sentence rankings for B2."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402
from kric.retrieval.sentences import (  # noqa: E402
    ClipSentenceRanker,
    SentenceCandidate,
    bm25_rank,
)


DEFAULT_IDS = PROJECT_ROOT / "configs" / "dataset" / "goodnews_validation_50_ids.json"
CLIP_NAME = "openai/clip-vit-base-patch32"
CLIP_REVISION = "b97b0100e55e367c057773c2a614676470b0d575"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        for key in ("main", "print_headline", "name", "text"):
            if key in value and _text(value[key]):
                return _text(value[key])
        return " ".join(_text(value[key]) for key in sorted(value) if _text(value[key]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(part for item in value if (part := _text(item)))
    return " ".join(str(value).split())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def rank_dataset(
    dataset_root: Path,
    ids_path: Path,
    *,
    method: str,
    output: Path,
    cache_path: Path | None = None,
    semantic_device: str = "auto",
) -> dict[str, Any]:
    expected_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if len(expected_ids) != 50 or len(set(expected_ids)) != 50:
        raise ValueError("B2 requires the approved manifest of exactly 50 unique IDs")
    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=dataset_root / "article+caption.json",
            splits_path=dataset_root / "img_splits.json",
            images_root=dataset_root / "images",
        )
    )
    dataset.assert_split_integrity()
    samples = select_samples(
        dataset, split="dev", max_samples=50, strategy="first_by_sample_id"
    )
    if [sample.sample_id for sample in samples] != expected_ids:
        raise RuntimeError("B2 ranking IDs differ from the approved 50-sample manifest")
    if method not in {"bm25", "semantic"}:
        raise ValueError("method must be bm25 or semantic")
    semantic_ranker = None
    if method == "semantic":
        semantic_ranker = ClipSentenceRanker(
            cache_path or output.with_name("clip_sentence_embeddings.sqlite3"),
            model_name=CLIP_NAME,
            model_revision=CLIP_REVISION,
            device=semantic_device,
        )

    rows = []
    for sample in samples:
        candidates = [
            SentenceCandidate(index, sentence)
            for index, sentence in enumerate(sample.article_sentences)
        ]
        if not candidates:
            raise RuntimeError(f"article has no segmented sentences: {sample.sample_id}")
        if method == "bm25":
            query = _text(sample.metadata.get("headline"))
            query_source = "article_headline"
            if not query:
                query = candidates[0].text
                query_source = "lead_sentence_fallback"
            ranked = bm25_rank(candidates, query)
            retriever = {
                "method": "bm25",
                "query_source": query_source,
                "query": query,
                "k1": 1.2,
                "b": 0.75,
            }
        else:
            ranked = semantic_ranker.rank(sample.image_path, candidates)
            retriever = {"method": "semantic", **semantic_ranker.model_info()}
        rows.append(
            {
                "sample_id": sample.sample_id,
                "candidate_sentence_count": len(candidates),
                "retriever": retriever,
                "ranked_sentences": [
                    {
                        "sentence_id": sentence.sentence_id,
                        "text": sentence.text,
                        "score": sentence.score,
                        "rank": sentence.rank,
                    }
                    for sentence in ranked
                ],
            }
        )
    _write_jsonl(output, rows)
    return {
        "method": method,
        "samples": len(rows),
        "output": str(output.resolve()),
        "reference_caption_used": False,
        "generated_caption_used": False,
        "semantic_model": (
            {"name": CLIP_NAME, "revision": CLIP_REVISION}
            if method == "semantic"
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--method", required=True, choices=("bm25", "semantic"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--semantic-device", default="auto")
    args = parser.parse_args()
    result = rank_dataset(
        args.dataset_root.expanduser().resolve(),
        args.ids.expanduser().resolve(),
        method=args.method,
        output=args.output.expanduser().resolve(),
        cache_path=args.cache.expanduser().resolve() if args.cache else None,
        semantic_device=args.semantic_device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
