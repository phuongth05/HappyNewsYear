import json
import random
from pathlib import Path

from PIL import Image

from kric.data.goodnews import GoodNewsConfig
from kric.data.goodnews_verify import verify_goodnews_release


def _image(path: Path) -> None:
    Image.new("RGB", (2, 2), color="white").save(path, format="JPEG")


def _config(root: Path) -> GoodNewsConfig:
    return GoodNewsConfig(
        annotations_path=root / "article+caption.json",
        splits_path=root / "img_splits.json",
        images_root=root / "images",
    )


def test_release_verification_and_statistics(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    annotations = {
        "article-a": {
            "article": "one two three four five",
            "images": {"0": "A useful caption.", "1": "Another caption."},
        },
        "article-b": {
            "article": "short article",
            "images": {"0": "Test caption."},
        },
    }
    splits = {
        "article-a_0.jpg": "train",
        "article-a_1.jpg": "val",
        "article-b_0.jpg": "test",
    }
    (tmp_path / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (tmp_path / "img_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    for sample_id in ("article-a_0", "article-a_1", "article-b_0"):
        _image(images / f"{sample_id}.jpg")

    report = verify_goodnews_release(
        _config(tmp_path),
        count_tokens=lambda text: len(text.split()),
        tokenizer_name="synthetic-whitespace-test-tokenizer",
        context_limit=3,
    )
    assert report["valid"] is True
    assert report["release_counts"] == {
        "total_annotation_samples": 3,
        "official_split_samples": 3,
        "annotations_outside_official_splits": 0,
        "unique_articles": 2,
        "samples_by_official_split": {"train": 1, "val": 1, "test": 1},
    }
    assert report["statistics"]["article_token_length_unique_articles"]["median"] == 3.5
    assert report["statistics"]["images_per_article"]["mean"] == 1.5
    assert report["statistics"]["percentage_sample_contexts_truncated_at_context_limit"] == 66.666667
    assert report["verification_scope"] == "full"
    assert report["full_image_verification_completed"] is True


def test_quick_mode_checks_all_paths_but_decodes_seeded_sample(
    tmp_path: Path, monkeypatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    annotations = {
        f"article-{index}": {
            "article": f"article text {index}",
            "images": {"0": f"caption {index}"},
        }
        for index in range(5)
    }
    splits = {f"article-{index}_0.jpg": "val" for index in range(5)}
    (tmp_path / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (tmp_path / "img_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    for sample_id in sorted(Path(name).stem for name in splits):
        _image(images / f"{sample_id}.jpg")

    from PIL import Image as PilImage

    opened: list[str] = []
    original_open = PilImage.open

    def tracking_open(path, *args, **kwargs):
        opened.append(Path(path).stem)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(PilImage, "open", tracking_open)
    report = verify_goodnews_release(
        _config(tmp_path), verify_images=2, verification_seed=2026
    )
    expected = random.Random(2026).sample(sorted(Path(name).stem for name in splits), 2)
    assert opened == expected
    assert report["valid"] is True
    assert report["verification_scope"] == "sampled"
    assert report["full_image_verification_completed"] is False
    assert report["image_verification"] == {
        "verification_scope": "sampled",
        "seed": 2026,
        "requested_sample_size": 2,
        "total_image_files_found": 5,
        "total_split_referenced_images": 5,
        "resolved_split_referenced_images": 5,
        "sampled_images_selected": 2,
        "sampled_images_verified": 2,
        "successfully_verified": 2,
        "corrupted_images_found_in_sample": 0,
        "missing_referenced_images": 0,
        "sampled_split_distribution": {"train": 0, "val": 2, "test": 0},
    }


def test_release_verifier_reports_split_content_and_image_issues(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    # Preserve a duplicate JSON key to verify that json.load() cannot silently
    # overwrite duplicate article IDs.
    (tmp_path / "article+caption.json").write_text(
        '{"article-a":{"article":"","images":{"0":""}},'
        '"article-a":{"article":"text","images":{"0":"caption"}},'
        '"article-b":{"article":"text","images":{"0":"caption"}}}',
        encoding="utf-8",
    )
    (tmp_path / "img_splits.json").write_text(
        json.dumps(
            {
                "train": ["article-a_0.jpg", "unknown_0.jpg"],
                "val": ["article-a_0.jpg"],
                "test": [],
            }
        ),
        encoding="utf-8",
    )
    (images / "article-a_0.jpg").write_bytes(b"not an image")
    # article-b_0.jpg is deliberately missing.

    report = verify_goodnews_release(_config(tmp_path))
    assert report["valid"] is False
    assert report["issues"]["duplicate_json_keys"]["count"] == 1
    assert report["issues"]["samples_in_multiple_splits"]["count"] == 1
    assert report["issues"]["split_ids_missing_annotations"]["count"] == 1
    assert report["warnings"]["annotation_ids_missing_splits"]["count"] == 1
    assert report["issues"]["missing_images"]["count"] == 0
    assert report["issues"]["corrupted_images"]["count"] == 1
    assert report["statistics"]["status"] == "not_computed_check_only"


def test_missing_required_artifacts_return_structured_failure(tmp_path: Path) -> None:
    report = verify_goodnews_release(_config(tmp_path))
    assert report["valid"] is False
    assert report["issues"]["missing_required_artifacts"]["count"] == 3


def test_url_mapping_mislabeled_as_split_file_is_reported_clearly(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    annotations = {"article": {"article": "text", "images": {"0": "caption"}}}
    urls = {"article": {"0": "https://example.test/image.jpg"}}
    (tmp_path / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (tmp_path / "img_splits.json").write_text(json.dumps(urls), encoding="utf-8")
    _image(images / "article_0.jpg")
    report = verify_goodnews_release(_config(tmp_path), verify_images=1)
    assert report["valid"] is False
    assert report["issues"]["schema_errors"]["count"] == 1
    assert "img_urls.json" in report["issues"]["schema_errors"]["examples"][0]
    assert report["issues"]["non_official_split_labels"]["count"] == 0


def test_out_of_split_empty_annotation_is_informational_not_fatal(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    annotations = {
        "official": {"article": "article", "images": {"0": "caption"}},
        "extra": {"article": "", "images": {"0": ""}},
    }
    splits = {"official_0.jpg": "val"}
    (tmp_path / "article+caption.json").write_text(json.dumps(annotations), encoding="utf-8")
    (tmp_path / "img_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    _image(images / "official_0.jpg")

    report = verify_goodnews_release(_config(tmp_path), verify_images=1)
    assert report["valid"] is True
    assert report["release_counts"]["annotations_outside_official_splits"] == 1
    assert report["warnings"]["annotation_ids_missing_splits"]["count"] == 1
    assert report["content_validation"] == {
        "empty_captions_all_annotations": 1,
        "empty_captions_official_splits": {
            "train": 0,
            "val": 0,
            "test": 0,
            "total": 0,
        },
        "empty_captions_outside_official_splits": 1,
        "empty_caption_partition_consistent": True,
        "empty_articles_all_annotations": 1,
        "empty_articles_official_splits": {
            "train": 0,
            "val": 0,
            "test": 0,
            "total": 0,
        },
        "empty_articles_outside_official_splits": 1,
        "empty_article_partition_consistent": True,
    }
