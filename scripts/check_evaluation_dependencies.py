"""Validate evaluator dependencies before any caption-model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import write_json  # noqa: E402
from kric.evaluation.preflight import check_evaluation_dependencies  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--entity-extractor", default="spacy")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check_evaluation_dependencies(
        tuple(item.strip() for item in args.metrics.split(",") if item.strip()),
        entity_extractor=args.entity_extractor,
        spacy_model=args.spacy_model,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

