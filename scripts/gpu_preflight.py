"""Print and save CUDA preflight information without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
MINIMUM_TOTAL_VRAM_BYTES = 14 * GIB
MINIMUM_FREE_VRAM_BYTES = 12 * GIB


def collect_gpu_preflight(
    torch_module: Any | None = None,
    *,
    minimum_total_vram_bytes: int = MINIMUM_TOTAL_VRAM_BYTES,
    minimum_free_vram_bytes: int = MINIMUM_FREE_VRAM_BYTES,
) -> dict[str, Any]:
    torch = torch_module
    if torch is None:
        import torch as imported_torch

        torch = imported_torch
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": available,
        "minimum_total_vram_bytes": minimum_total_vram_bytes,
        "minimum_free_vram_bytes": minimum_free_vram_bytes,
    }
    if not available:
        report["sufficient_vram"] = False
        return report
    properties = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    report.update(
        {
            "gpu_model": torch.cuda.get_device_name(0),
            "total_vram_bytes": int(total_bytes),
            "free_vram_bytes": int(free_bytes),
            "total_vram_gib": round(int(total_bytes) / GIB, 3),
            "free_vram_gib": round(int(free_bytes) / GIB, 3),
            "reported_device_total_memory_bytes": int(properties.total_memory),
            "sufficient_vram": (
                int(total_bytes) >= minimum_total_vram_bytes
                and int(free_bytes) >= minimum_free_vram_bytes
            ),
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
    if not report["sufficient_vram"]:
        raise SystemExit(
            "Insufficient free/total VRAM for the unquantized FP16 "
            "InstructBLIP Flan-T5 XL validation"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
