"""Package the approved 50-sample GoodNews validation subset without rewriting content."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402
from kric.evaluation.io import file_sha256  # noqa: E402


EXPECTED_SAMPLES = 50


def _load_ids(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError("ID manifest must contain a JSON list of strings")
    if len(values) != EXPECTED_SAMPLES or len(set(values)) != EXPECTED_SAMPLES:
        raise ValueError("ID manifest must contain exactly 50 unique sample IDs")
    return values


def _images_field(record: Mapping[str, Any], aliases: tuple[str, ...]) -> str:
    for name in aliases:
        if name in record:
            return name
    raise ValueError("selected article has no configured images field")


def _filtered_images(images: Any, selected_indices: set[str]) -> Any:
    if isinstance(images, Mapping):
        missing = selected_indices - {str(key) for key in images}
        if missing:
            raise KeyError(f"selected image indices missing from article record: {sorted(missing)}")
        return {
            key: copy.deepcopy(value)
            for key, value in images.items()
            if str(key) in selected_indices
        }
    if isinstance(images, list):
        selected = []
        found: set[str] = set()
        for position, value in enumerate(images):
            if isinstance(value, Mapping):
                explicit = next(
                    (value[name] for name in ("image_index", "index", "id") if name in value),
                    None,
                )
                index = str(explicit) if explicit is not None else str(position)
            else:
                index = str(position)
            if index in selected_indices:
                selected.append(copy.deepcopy(value))
                found.add(index)
        if found != selected_indices:
            raise KeyError(
                f"selected image indices missing from article list: {sorted(selected_indices - found)}"
            )
        return selected
    raise TypeError("selected article images field must be an object or list")


def package_subset(source_root: Path, ids_path: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    ids_path = ids_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    expected_ids = _load_ids(ids_path)
    source_config = GoodNewsConfig(
        annotations_path=source_root / "article+caption.json",
        splits_path=source_root / "img_splits.json",
        images_root=source_root / "images",
    )
    source_dataset = GoodNewsDataset(source_config)
    source_dataset.assert_split_integrity()
    source_samples = select_samples(
        source_dataset,
        split="dev",
        max_samples=EXPECTED_SAMPLES,
        strategy="first_by_sample_id",
    )
    source_ids = [sample.sample_id for sample in source_samples]
    if source_ids != expected_ids:
        raise RuntimeError("full dataset selection does not match the approved 50 IDs")

    raw_annotations = json.loads(
        source_config.annotations_path.read_text(encoding="utf-8")
    )
    if not isinstance(raw_annotations, Mapping):
        raise TypeError("official subset packager requires article-keyed annotations")

    selected_by_article: dict[str, set[str]] = defaultdict(set)
    for sample in source_samples:
        selected_by_article[str(sample.metadata["article_id"])].add(
            str(sample.metadata["image_index"])
        )

    packaged_annotations: dict[str, Any] = {}
    for article_id, selected_indices in selected_by_article.items():
        if article_id not in raw_annotations:
            raise KeyError(f"selected article missing from raw annotations: {article_id}")
        raw_record = raw_annotations[article_id]
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"article {article_id!r} is not an object")
        record = copy.deepcopy(dict(raw_record))
        field = _images_field(raw_record, source_config.images_fields)
        record[field] = _filtered_images(raw_record[field], selected_indices)
        packaged_annotations[article_id] = record

    images_root = temporary / "images"
    images_root.mkdir(parents=True)
    image_manifest = []
    destination_names: set[str] = set()
    for sample in source_samples:
        source_image = Path(sample.image_path)
        destination_name = f"{sample.sample_id}{source_config.image_extension}"
        if destination_name in destination_names:
            raise RuntimeError(f"duplicate packaged image name: {destination_name}")
        destination_names.add(destination_name)
        destination_image = images_root / destination_name
        shutil.copy2(source_image, destination_image)
        source_sha256 = file_sha256(source_image)
        destination_sha256 = file_sha256(destination_image)
        if source_sha256 != destination_sha256:
            raise RuntimeError(f"image checksum mismatch after copy: {sample.sample_id}")
        try:
            with Image.open(destination_image) as image:
                image.verify()
        except Exception as error:
            raise RuntimeError(
                f"copied image failed Pillow verification: {sample.sample_id}: {error}"
            ) from error
        image_manifest.append(
            {
                "sample_id": sample.sample_id,
                "article_id": sample.metadata["article_id"],
                "image_index": sample.metadata["image_index"],
                "official_split": "val",
                "filename": destination_name,
                "sha256": destination_sha256,
                "bytes": destination_image.stat().st_size,
            }
        )

    annotations_path = temporary / "article+caption.json"
    splits_path = temporary / "img_splits.json"
    annotations_path.write_text(
        json.dumps(packaged_annotations, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    splits_path.write_text(
        json.dumps(
            {f"{sample_id}{source_config.image_extension}": "val" for sample_id in expected_ids},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    packaged_config = GoodNewsConfig(
        annotations_path=annotations_path,
        splits_path=splits_path,
        images_root=images_root,
    )
    packaged_dataset = GoodNewsDataset(packaged_config)
    packaged_dataset.assert_split_integrity()
    packaged_samples = select_samples(
        packaged_dataset,
        split="dev",
        max_samples=EXPECTED_SAMPLES,
        strategy="first_by_sample_id",
    )
    packaged_ids = [sample.sample_id for sample in packaged_samples]
    if packaged_ids != expected_ids:
        raise RuntimeError("packaged dataset does not resolve the approved ordered IDs")
    packaged_by_id = {sample.sample_id: sample for sample in packaged_samples}
    for source_sample in source_samples:
        packaged_sample = packaged_by_id[source_sample.sample_id]
        if packaged_sample.article_text != source_sample.article_text:
            raise RuntimeError(f"article changed during packaging: {source_sample.sample_id}")
        if packaged_sample.reference_caption != source_sample.reference_caption:
            raise RuntimeError(f"caption changed during packaging: {source_sample.sample_id}")

    copied_images = [path for path in images_root.rglob("*") if path.is_file()]
    if len(copied_images) != EXPECTED_SAMPLES:
        raise RuntimeError(f"expected exactly 50 copied images, found {len(copied_images)}")

    payload_bytes = sum(path.stat().st_size for path in copied_images)
    payload_bytes += annotations_path.stat().st_size + splits_path.stat().st_size
    manifest = {
        "schema_version": 1,
        "purpose": "approved GoodNews 50-sample B0/B1/B1-random sanity experiment only",
        "not_a_full_dataset_replacement": True,
        "source_root": str(source_root),
        "source_annotations_sha256": file_sha256(source_config.annotations_path),
        "source_splits_sha256": file_sha256(source_config.splits_path),
        "approved_ids_manifest": str(ids_path),
        "approved_ids_manifest_sha256": file_sha256(ids_path),
        "split": "val",
        "project_split": "dev",
        "selection": "first_by_sample_id",
        "samples": EXPECTED_SAMPLES,
        "ordered_sample_ids": expected_ids,
        "unique_articles": len(packaged_annotations),
        "payload_bytes_excluding_manifest": payload_bytes,
        "annotations_sha256": file_sha256(annotations_path),
        "splits_sha256": file_sha256(splits_path),
        "images": image_manifest,
        "verification": {
            "source_and_packaged_ids_identical": True,
            "article_values_identical": True,
            "caption_values_identical": True,
            "image_checksums_identical": True,
            "all_copied_images_pillow_valid": True,
            "copied_image_count": len(copied_images),
        },
    }
    manifest_path = temporary / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output_root)
    manifest["output_root"] = str(output_root)
    manifest["package_bytes"] = sum(
        path.stat().st_size for path in output_root.rglob("*") if path.is_file()
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = package_subset(args.source_root, args.ids, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
