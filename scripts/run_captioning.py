"""Run a configuration-driven captioning experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.config import (  # noqa: E402
    B0ExperimentConfig,
    B2ExperimentConfig,
    load_experiment_config,
)
from kric.captioning.runner import run_b0, run_b1, run_b2  # noqa: E402


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
    config = load_experiment_config(args.config)
    try:
        if isinstance(config, B0ExperimentConfig):
            result = run_b0(config, overwrite=args.overwrite)
        elif isinstance(config, B2ExperimentConfig):
            result = run_b2(config, overwrite=args.overwrite)
        else:
            result = run_b1(config, overwrite=args.overwrite)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        # The cloud workflow runs each condition in a separate process. Free
        # this process's model tensors and CUDA cache before it exits so the
        # next controlled condition starts from released device memory.
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
