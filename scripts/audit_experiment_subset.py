"""Validate the exact deterministic GoodNews subset consumed by experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.data.goodnews import GoodNewsConfig, GoodNewsDataset  # noqa: E402
from kric.data.selection import select_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument(
        "--selection", default="first_by_sample_id", choices=("first_by_sample_id",)
    )
    return parser.parse_args()


def audit_subset(
    root: Path,
    *,
    split: str,
    max_samples: int,
    selection: str,
) -> dict:
    root = root.expanduser().resolve()
    dataset = GoodNewsDataset(
        GoodNewsConfig(
            annotations_path=root / "article+caption.json",
            splits_path=root / "img_splits.json",
            images_root=root / "images",
        )
    )
    dataset.assert_split_integrity()
    samples = select_samples(
        dataset,
        split=split,
        max_samples=max_samples,
        strategy=selection,
    )
    if len(samples) != max_samples:
        raise ValueError(
            f"requested {max_samples} samples from {split!r}, found {len(samples)}"
        )

    from PIL import Image

    issues: list[str] = []
    for sample in samples:
        if not sample.article_text.strip():
            issues.append(f"{sample.sample_id}: empty article")
        if not sample.reference_caption.strip():
            issues.append(f"{sample.sample_id}: empty caption")
        image_path = Path(sample.image_path)
        if not image_path.is_file():
            issues.append(f"{sample.sample_id}: missing image: {image_path}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as error:
            issues.append(
                f"{sample.sample_id}: corrupt image: {type(error).__name__}: {error}"
            )
    if issues:
        raise RuntimeError("subset validation failed:\n" + "\n".join(issues))
    return {
        "dataset": "goodnews",
        "split": split,
        "max_samples": max_samples,
        "selection": selection,
        "valid": True,
        "sample_ids": [sample.sample_id for sample in samples],
    }


def main() -> int:
    args = parse_args()
    report = audit_subset(
        args.dataset_root,
        split=args.split,
        max_samples=args.max_samples,
        selection=args.selection,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
