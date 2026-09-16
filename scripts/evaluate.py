"""Evaluate saved captions without running model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.clipscore import ClipScoreMetric  # noqa: E402
from kric.evaluation.coco import CocoCaptionMetrics  # noqa: E402
from kric.evaluation.deferred import deferred_metrics  # noqa: E402
from kric.evaluation.entities import (  # noqa: E402
    EntityMetric,
    MetadataEntityExtractor,
    SpacyEntityExtractor,
)
from kric.evaluation.io import load_predictions  # noqa: E402
from kric.evaluation.runner import EvaluationRunner  # noqa: E402


METRIC_CHOICES = {
    "cider",
    "spice",
    "clipscore",
    "entity",
    "unsupported_claim_rate",
    "claim_support_f1",
    "attribution",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON summary path")
    parser.add_argument(
        "--per-sample-output",
        type=Path,
        help="CSV path; defaults to <output-stem>_per_sample.csv",
    )
    parser.add_argument(
        "--metrics",
        default="cider,spice,entity",
        help="Comma-separated metrics. Add clipscore when metadata.image_path is available.",
    )
    parser.add_argument("--entity-extractor", choices=("metadata", "spacy"), default="metadata")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--entity-match-labels", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-revision", help="Optional immutable Hugging Face revision")
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-text-prefix", default="A photo depicts ")
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--coco-timeout-seconds", type=int, default=600)
    parser.add_argument("--spice-java", default="java", help="Java executable used only for SPICE")
    parser.add_argument("--spice-memory", default="8G")
    parser.add_argument(
        "--allow-metric-errors",
        action="store_true",
        help="Exit zero after recording unavailable/failed metrics in the JSON summary",
    )
    return parser.parse_args()


def build_metrics(args: argparse.Namespace) -> list:
    names = [name.strip().lower() for name in args.metrics.split(",") if name.strip()]
    if not names:
        raise ValueError("at least one metric must be requested")
    unknown = set(names) - METRIC_CHOICES
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate metric names: {names}")

    plugins = []
    deferred = deferred_metrics()
    for name in names:
        if name == "cider":
            plugins.append(CocoCaptionMetrics(("CIDEr",), timeout_seconds=args.coco_timeout_seconds))
        elif name == "spice":
            plugins.append(
                CocoCaptionMetrics(
                    ("SPICE",),
                    timeout_seconds=args.coco_timeout_seconds,
                    spice_java=args.spice_java,
                    spice_memory=args.spice_memory,
                )
            )
        elif name == "entity":
            extractor = (
                MetadataEntityExtractor()
                if args.entity_extractor == "metadata"
                else SpacyEntityExtractor(args.spacy_model)
            )
            plugins.append(EntityMetric(extractor, match_labels=args.entity_match_labels))
        elif name == "clipscore":
            cache_path = args.embedding_cache or args.output.with_name("clip_embeddings.sqlite3")
            plugins.append(
                ClipScoreMetric(
                    cache_path,
                    model_name=args.clip_model,
                    model_revision=args.clip_revision,
                    device=args.clip_device,
                    batch_size=args.clip_batch_size,
                    text_prefix=args.clip_text_prefix,
                )
            )
        else:
            plugins.append(deferred[name])
    return plugins


def main() -> int:
    args = parse_args()
    records = load_predictions(args.predictions)
    per_sample_path = args.per_sample_output or args.output.with_name(
        f"{args.output.stem}_per_sample.csv"
    )
    runner = EvaluationRunner(build_metrics(args))
    summary, had_errors = runner.run(
        records,
        predictions_path=args.predictions,
        summary_path=args.output,
        per_sample_path=per_sample_path,
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    for name, status in summary["metric_runs"].items():
        print(f"{name}: {status['status']}")
        if status["status"] == "error":
            print(f"  {status['error_type']}: {status['message']}")
    print(f"Summary: {args.output.resolve()}")
    print(f"Per-sample: {per_sample_path.resolve()}")
    return 1 if had_errors and not args.allow_metric_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
