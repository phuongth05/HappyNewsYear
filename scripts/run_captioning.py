"""Run a configuration-driven captioning experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.config import B0ExperimentConfig  # noqa: E402
from kric.captioning.runner import run_b0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing predictions file in this run directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = B0ExperimentConfig.from_file(args.config)
    result = run_b0(config, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
