"""Generate the deterministic M4 claim-support human-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.ranking import read_jsonl  # noqa: E402
from kric.matching.review import sample_review_claims  # noqa: E402

FIELDS = [
    "sample_id", "caption_variant", "caption", "claim_id", "claim_text", "claim_type",
    "cosine_top1_id", "cosine_top1_text", "cosine_top1_score",
    "nli_top1_id", "nli_top1_text", "nli_top1_score", "union_top3_evidence_json",
    "support_label", "best_evidence_ids", "matcher_preference", "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-dir", required=True, type=Path)
    parser.add_argument("--matcher-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    claims_dir = args.claims_dir.expanduser().resolve()
    matcher_dir = args.matcher_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claims_manifest = json.loads((claims_dir / "manifest.json").read_text(encoding="utf-8"))
    matcher_manifest = json.loads((matcher_dir / "manifest.json").read_text(encoding="utf-8"))
    claims_path = claims_dir / "all_claims.jsonl"
    cosine_path = matcher_dir / "cosine_rankings.jsonl"
    nli_path = matcher_dir / "nli_rankings.jsonl"
    if claims_manifest.get("outputs", {}).get("all_claims.jsonl") != file_sha256(claims_path):
        raise ValueError("claims hash mismatch")
    for name, path in (("cosine_rankings.jsonl", cosine_path), ("nli_rankings.jsonl", nli_path)):
        if matcher_manifest.get("outputs", {}).get(name) != file_sha256(path):
            raise ValueError(f"matcher ranking hash mismatch: {name}")
    rows, sampling = sample_review_claims(
        read_jsonl(claims_path),
        read_jsonl(cosine_path),
        read_jsonl(nli_path),
        target=args.target,
        seed=args.seed,
    )
    csv_path = output_dir / "human_review_claim_support.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_claim_support_human_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **sampling,
        "claims_manifest_sha256": file_sha256(claims_dir / "manifest.json"),
        "matcher_manifest_sha256": file_sha256(matcher_dir / "manifest.json"),
        "review_csv_sha256": file_sha256(csv_path),
        "allowed_support_labels": ["supported", "unsupported", "uncertain"],
        "allowed_matcher_preferences": ["cosine", "nli", "tie", "neither"],
    }
    write_json(output_dir / "sampling_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())