"""Print and save CUDA preflight information without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def collect_gpu_preflight(torch_module: Any | None = None) -> dict[str, Any]:
    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": available,
    }
    if not available:
        return report
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    report.update(
        {
            "gpu_model": torch.cuda.get_device_name(0),
            "total_vram_bytes": int(total_bytes),
            "free_vram_bytes": int(free_bytes),
            "reported_device_total_memory_bytes": int(properties.total_memory),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect_gpu_preflight()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["cuda_available"]:
        raise SystemExit("CUDA is required for this validation workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
