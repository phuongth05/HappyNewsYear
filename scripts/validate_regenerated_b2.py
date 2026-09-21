"""Validate a regenerated B2 semantic-k3 run against exact frozen inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.b2_regenerated import (  # noqa: E402
    REGENERATED_MANIFEST_NAME,
    create_regenerated_baseline_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-b2-dir", required=True, type=Path)
    parser.add_argument("--regenerated-b2-dir", required=True, type=Path)
    parser.add_argument(
        "--ids",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset/goodnews_validation_50_ids.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Defaults to <regenerated-b2-dir>/{REGENERATED_MANIFEST_NAME}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    historical_dir = args.historical_b2_dir.expanduser().resolve()
    regenerated_dir = args.regenerated_b2_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else regenerated_dir / REGENERATED_MANIFEST_NAME
    )
    result = create_regenerated_baseline_manifest(
        historical_dir=historical_dir,
        regenerated_dir=regenerated_dir,
        ids_path=args.ids.expanduser().resolve(),
        output_path=output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Regenerated baseline manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())