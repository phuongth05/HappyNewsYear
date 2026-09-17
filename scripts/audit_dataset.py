"""Validate a GoodNews release and optionally compute tokenizer-exact statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.data.goodnews import GoodNewsConfig  # noqa: E402
from kric.data.goodnews_verify import verify_goodnews_release  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "dataset_audits" / "goodnews_validation.json"
DEFAULT_TOKENIZER = "Salesforce/instructblip-flan-t5-xl"
DEFAULT_TOKENIZER_REVISION = "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("goodnews",))
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--config", type=Path, help="GoodNews dataset YAML")
    location.add_argument(
        "--root",
        type=Path,
        help="Directory containing article+caption.json, img_splits.json, and images/",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate artifacts and images without loading the B1 tokenizer or computing statistics.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run all structural/path checks but Pillow-verify only a deterministic image sample.",
    )
    parser.add_argument(
        "--verify-images",
        type=int,
        help="Number of split-referenced images to Pillow-verify in --quick mode (default: 10000).",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit 1 when any validation issue is detected.",
    )
    parser.add_argument("--context-limit", type=int, default=384)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--max-issue-examples", type=int, default=20)
    return parser.parse_args()


def _config(args: argparse.Namespace) -> GoodNewsConfig:
    if args.config is not None:
        config = GoodNewsConfig.from_file(args.config)
        return GoodNewsConfig(
            annotations_path=config.annotations_path,
            splits_path=config.splits_path,
            images_root=config.images_root,
            image_extension=config.image_extension,
            long_article_words=config.long_article_words,
            max_issue_examples=args.max_issue_examples,
            article_fields=config.article_fields,
            images_fields=config.images_fields,
            caption_fields=config.caption_fields,
            image_path_fields=config.image_path_fields,
            entity_fields=config.entity_fields,
            metadata_fields=config.metadata_fields,
            entities_source=config.entities_source,
        )
    root = args.root.expanduser().resolve()
    return GoodNewsConfig(
        annotations_path=root / "article+caption.json",
        splits_path=root / "img_splits.json",
        images_root=root / "images",
        max_issue_examples=args.max_issue_examples,
    )


def _token_counter(name: str, revision: str):
    try:
        from transformers import InstructBlipProcessor
    except ImportError as error:
        raise RuntimeError(
            "statistics require the captioning dependencies: "
            "python -m pip install -e \".[captioning]\""
        ) from error
    processor = InstructBlipProcessor.from_pretrained(name, revision=revision)
    tokenizer = processor.tokenizer
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def print_summary(report: dict) -> None:
    print(f"Dataset: {report['dataset']}")
    print(f"Mode: {report['mode']}")
    print(f"Valid: {report['valid']}")
    counts = report.get("release_counts")
    if counts:
        print(f"Total annotation samples: {counts['total_annotation_samples']}")
        print(f"Official split samples: {counts['official_split_samples']}")
        print(
            "Annotations outside official splits: "
            f"{counts['annotations_outside_official_splits']}"
        )
        print(f"Unique articles: {counts['unique_articles']}")
        print("Official splits:", json.dumps(counts["samples_by_official_split"], sort_keys=True))
    print("Fatal issues:")
    for name, details in report["issues"].items():
        print(f"  {name}: {details['count']}")
    print("Warnings/informational:")
    for name, details in report.get("warnings", {}).items():
        print(f"  {name}: {details['count']}")
    statistics = report.get("statistics", {})
    print(f"Statistics: {statistics.get('status', 'not_available')}")


def main() -> int:
    args = parse_args()
    if args.context_limit <= 0:
        raise ValueError("--context-limit must be positive")
    if args.verify_images is not None and not args.quick:
        raise ValueError("--verify-images requires --quick")
    if args.verify_images is not None and args.verify_images <= 0:
        raise ValueError("--verify-images must be positive")
    config = _config(args)
    counter = None
    tokenizer_description = None
    if not args.check_only and not args.quick:
        counter = _token_counter(args.tokenizer, args.tokenizer_revision)
        tokenizer_description = f"{args.tokenizer}@{args.tokenizer_revision}"
    report = verify_goodnews_release(
        config,
        count_tokens=counter,
        tokenizer_name=tokenizer_description,
        context_limit=args.context_limit,
        verify_images=(args.verify_images or 10_000) if args.quick else None,
        verification_seed=2026,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print(f"Report: {output}")
    return 1 if args.fail_on_issues and not report["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
