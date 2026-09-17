import json
from pathlib import Path

from PIL import Image

from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset
from kric.data.selection import select_samples
from kric.evaluation.io import file_sha256
from scripts.package_goodnews_subset import package_subset


def _source_release(root: Path) -> list[str]:
    images = root / "images"
    images.mkdir(parents=True)
    annotations = {}
    splits = {}
    expected = []
    for index in range(50):
        article_id = f"article_{index:03d}"
        sample_id = f"{article_id}_0"
        expected.append(sample_id)
        annotations[article_id] = {
            "article": f"Original article text {index} — unchanged.",
            "headline": f"Headline {index}",
            "images": {"0": f"Original caption {index}."},
        }
        splits[f"{sample_id}.jpg"] = "val"
        Image.new("RGB", (3, 3), (index, 20, 30)).save(
            images / f"{sample_id}.jpg", "JPEG"
        )
    (root / "article+caption.json").write_text(
        json.dumps(annotations, ensure_ascii=False), encoding="utf-8"
    )
    (root / "img_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return expected


def _dataset(root: Path) -> GoodNewsDataset:
    return GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=root / "article+caption.json",
            splits_path=root / "img_splits.json",
            images_root=root / "images",
        )
    )


def test_packaged_subset_is_value_and_image_equivalent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _source_release(source)
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(expected), encoding="utf-8")
    output = tmp_path / "GoodNews_validation_50"

    result = package_subset(source, ids, output)
    assert result["verification"] == {
        "source_and_packaged_ids_identical": True,
        "article_values_identical": True,
        "caption_values_identical": True,
        "image_checksums_identical": True,
        "all_copied_images_pillow_valid": True,
        "copied_image_count": 50,
    }

    source_samples = {
        sample.sample_id: sample
        for sample in select_samples(
            _dataset(source), split="dev", max_samples=50, strategy="first_by_sample_id"
        )
    }
    packaged = select_samples(
        _dataset(output), split="dev", max_samples=50, strategy="first_by_sample_id"
    )
    assert [sample.sample_id for sample in packaged] == expected
    for sample in packaged:
        original = source_samples[sample.sample_id]
        assert sample.article_text == original.article_text
        assert sample.reference_caption == original.reference_caption
        assert file_sha256(sample.image_path) == file_sha256(original.image_path)

    copied = [path for path in (output / "images").iterdir() if path.is_file()]
    assert len(copied) == 50
    manifest = json.loads((output / "subset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ordered_sample_ids"] == expected
    assert manifest["not_a_full_dataset_replacement"] is True
