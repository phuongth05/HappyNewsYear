import json
from pathlib import Path

from kric.data import DatasetSplit, GoodNewsConfig, GoodNewsDataset


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_official_split_is_preserved_and_val_maps_to_dev(tmp_path: Path) -> None:
    annotations = {
        "train-story": {"article": "Training article.", "images": {"0": "Train caption."}},
        "dev-story": {"article": "Development article.", "images": {"0": "Dev caption."}},
        "test-story": {"article": "Test article.", "images": {"0": "Test caption."}},
    }
    splits = {
        "train-story_0.jpg": "train",
        "dev-story_0.jpg": "val",
        "test-story_0.jpg": "test",
    }
    annotations_path = tmp_path / "article+caption.json"
    splits_path = tmp_path / "img_splits.json"
    images_root = tmp_path / "images"
    images_root.mkdir()
    _write_json(annotations_path, annotations)
    _write_json(splits_path, splits)
    for sample_id in ("train-story_0", "dev-story_0", "test-story_0"):
        (images_root / f"{sample_id}.jpg").write_bytes(b"image")

    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=annotations_path,
            splits_path=splits_path,
            images_root=images_root,
        )
    )

    assert dataset.split_integrity_errors() == ()
    assert [sample.sample_id for sample in dataset.iter_samples(DatasetSplit.TRAIN)] == ["train-story_0"]
    dev = list(dataset.iter_samples(DatasetSplit.DEV))
    assert [sample.sample_id for sample in dev] == ["dev-story_0"]
    assert dev[0].metadata["official_split"] == "val"
    assert [sample.sample_id for sample in dataset.iter_samples(DatasetSplit.TEST)] == ["test-story_0"]

    inference = list(dataset.iter_inference_samples("dev"))[0]
    assert "reference_caption" not in inference.to_dict()


def test_config_paths_resolve_from_root_dir(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    config_path = tmp_path / "goodnews.yaml"
    config_path.write_text(
        "\n".join(
            (
                "dataset: goodnews",
                "root_dir: data",
                "annotations_path: annotations.json",
                "splits_path: splits.json",
                "images_root: images",
            )
        ),
        encoding="utf-8",
    )

    config = GoodNewsConfig.from_file(config_path)
    assert config.annotations_path == data_root / "annotations.json"
    assert config.splits_path == data_root / "splits.json"
    assert config.images_root == data_root / "images"
    assert config.article_fields == ("article", "article_text", "text", "body")
    assert config.image_path_fields == ("image_path", "filename", "filepath")
