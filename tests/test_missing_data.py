import json
from pathlib import Path

from kric.data.audit import audit_dataset
from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset


def test_audit_reports_missing_and_empty_data(tmp_path: Path) -> None:
    annotations = [
        {
            "article_id": "duplicate",
            "article": "one two three four",
            "images": {"0": {"caption": "", "entities": ["Alice"]}},
        },
        {
            "article_id": "duplicate",
            "article": "",
            "images": {"0": "A caption."},
        },
    ]
    splits = {"duplicate_0.jpg": "train"}
    annotations_path = tmp_path / "annotations.json"
    splits_path = tmp_path / "splits.json"
    images_root = tmp_path / "images"
    images_root.mkdir()
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=annotations_path,
            splits_path=splits_path,
            images_root=images_root,
            long_article_words=3,
        )
    )
    report = audit_dataset(dataset, long_article_words=3)

    assert report["number_of_samples"] == 2
    assert report["issues"]["missing_images"]["count"] == 2
    assert report["issues"]["empty_articles"]["count"] == 1
    assert report["issues"]["empty_captions"]["count"] == 1
    assert report["issues"]["duplicate_sample_ids"]["count"] == 1
    assert report["issues"]["unusually_long_articles"]["count"] == 1
    assert report["issues"]["split_integrity"]["count"] == 1
    assert report["statistics"]["entity_density"]["available"] is True

