"""Run frozen cosine and NLI evidence-claim matcher baselines."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.evaluation.io import file_sha256, write_json  # noqa: E402
from kric.matching.cosine import COSINE_MODEL, COSINE_REVISION, CosineMatcher  # noqa: E402
from kric.matching.nli import NLI_MODEL, NLI_REVISION, NliMatcher  # noqa: E402
from kric.matching.ranking import (  # noqa: E402
    rank_claims_against_same_sample_evidence,
    ranking_audit,
    read_jsonl,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--claims-dir", required=True, type=Path)
    parser.add_argument("--atomic-dir", required=True, type=Path)
    parser.add_argument(
        "--primary-ids",
        type=Path,
        default=PROJECT_ROOT / "configs/dataset/m3_complete_49_ids.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--methods", default="cosine,nli")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if raw.get("experiment") != "M4_matcher_baselines_49":
        raise ValueError("unexpected M4 matcher config")
    if raw["cosine"]["model"] != COSINE_MODEL or raw["cosine"]["revision"] != COSINE_REVISION:
        raise ValueError("cosine model pin changed")
    if raw["nli"]["model"] != NLI_MODEL or raw["nli"]["revision"] != NLI_REVISION:
        raise ValueError("NLI model pin changed")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or set(methods) - {"cosine", "nli"} or len(methods) != len(set(methods)):
        raise ValueError("methods must be unique values from cosine,nli")
    claims_dir = args.claims_dir.expanduser().resolve()
    atomic_dir = args.atomic_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claims_manifest = json.loads((claims_dir / "manifest.json").read_text(encoding="utf-8"))
    atomic_manifest = json.loads((atomic_dir / "manifest.json").read_text(encoding="utf-8"))
    claims_path = claims_dir / "all_claims.jsonl"
    evidence_path = atomic_dir / "atomic_evidence.jsonl"
    if claims_manifest.get("outputs", {}).get("all_claims.jsonl") != file_sha256(claims_path):
        raise ValueError("claim artifact hash mismatch")
    if atomic_manifest.get("outputs", {}).get("atomic_evidence.jsonl", {}).get("sha256") != file_sha256(evidence_path):
        raise ValueError("atomic evidence hash mismatch")
    primary_ids = json.loads(args.primary_ids.read_text(encoding="utf-8"))
    if primary_ids != claims_manifest.get("ordered_primary_ids"):
        raise ValueError("claims do not use the exact primary 49 IDs")
    primary_set = set(primary_ids)
    claims = read_jsonl(claims_path)
    if any(str(row.get("sample_id")) not in primary_set for row in claims):
        raise ValueError("claim artifact contains a non-primary sample")
    evidence = [
        row for row in read_jsonl(evidence_path) if str(row.get("sample_id")) in primary_set
    ]
    if [str(row["sample_id"]) for row in evidence] != primary_ids:
        raise ValueError("atomic evidence does not resolve the ordered primary 49")
    matchers = {}
    if "cosine" in methods:
        matchers["cosine"] = CosineMatcher(
            raw["cosine"]["model"],
            raw["cosine"]["revision"],
            raw["cosine"].get("device"),
            int(raw["cosine"]["batch_size"]),
        )
    if "nli" in methods:
        matchers["nli"] = NliMatcher(
            raw["nli"]["model"],
            raw["nli"]["revision"],
            str(raw["nli"].get("device", "auto")),
            int(raw["nli"]["batch_size"]),
        )
    audits = {}
    models = {}
    output_hashes = {}
    for name, matcher in matchers.items():
        rankings = rank_claims_against_same_sample_evidence(claims, evidence, matcher)
        path = output_dir / f"{name}_rankings.jsonl"
        write_jsonl(path, rankings)
        audits[name] = ranking_audit(rankings)
        models[name] = matcher.model_info()
        output_hashes[path.name] = file_sha256(path)
    audit = {
        "primary_samples": 49,
        "claims": len(claims),
        "methods": methods,
        "matchers": audits,
        "no_cross_sample_matching": True,
    }
    write_json(output_dir / "audit.json", audit)
    manifest = {
        "schema_version": 1,
        "experiment": "M4_matcher_baselines_49",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ordered_primary_ids": primary_ids,
        "claims_manifest_sha256": file_sha256(claims_dir / "manifest.json"),
        "claims_sha256": file_sha256(claims_path),
        "atomic_manifest_sha256": file_sha256(atomic_dir / "manifest.json"),
        "atomic_evidence_sha256": file_sha256(evidence_path),
        "models": models,
        "outputs": output_hashes,
        "reference_passed_to_matcher": False,
        "article_passed_to_matcher": False,
        "cross_sample_matching": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())