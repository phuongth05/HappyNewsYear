"""Extract and audit atomic evidence from frozen B2 semantic-k3 sentences."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kric.captioning.instructblip import _InstructBlipCaptioner  # noqa: E402
from kric.evidence.pipeline import run_atomic_extraction  # noqa: E402
from kric.evidence.spacy_extractor import SpacyAtomicExtractor  # noqa: E402


DEFAULT_IDS = PROJECT_ROOT / "configs" / "dataset" / "goodnews_validation_50_ids.json"
TOKENIZER_NAME = "Salesforce/instructblip-flan-t5-xl"
TOKENIZER_REVISION = "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d"


def _token_counter(name: str, revision: str):
    from transformers import InstructBlipProcessor

    processor = InstructBlipProcessor.from_pretrained(name, revision=revision)
    tokenizer = processor.tokenizer
    return lambda text: len(_InstructBlipCaptioner._token_ids(tokenizer, text))


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"available": True, "commit": commit, "dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"available": False, "commit": None, "dirty": None, "detail": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--selected-evidence", type=Path)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--context-budget", type=int, default=384)
    parser.add_argument("--manual-review-count", type=int, default=100)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    args = parser.parse_args()
    if args.config:
        config_path = args.config.expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        base = config_path.parent
        selected_evidence = (base / raw["source"]["selected_evidence"]).resolve()
        ids_path = (base / raw["source"]["ids"]).resolve()
        output_dir = (base / raw["output_dir"]).resolve()
        cache = (base / raw["cache"]).resolve()
        spacy_model = raw["extractor"]["model"]
        context_budget = int(raw["audit"]["context_budget"])
        manual_review_count = int(raw["audit"]["manual_review_count"])
        tokenizer_name = raw["tokenizer"]["name"]
        tokenizer_revision = raw["tokenizer"]["revision"]
        if (
            raw.get("experiment") != "M3_atomic_evidence_audit"
            or raw["source"].get("retrieval_method") != "semantic"
            or int(raw["source"].get("retrieval_k", 0)) != 3
            or raw["source"].get("retrieval_model") != "openai/clip-vit-base-patch32"
            or raw["source"].get("retrieval_revision")
            != "b97b0100e55e367c057773c2a614676470b0d575"
            or raw["extractor"].get("method")
            != "deterministic_spacy_dependency_and_ner"
            or raw["extractor"].get("rule_set_version")
            != "spacy_dependency_atomic_v1"
            or raw["extractor"].get("prompt_version")
            != "not_applicable_rule_based_v1"
        ):
            raise ValueError("M3 config changed a frozen retrieval/extraction setting")
    else:
        if args.selected_evidence is None or args.output_dir is None:
            parser.error("provide --config or both --selected-evidence and --output-dir")
        selected_evidence = args.selected_evidence.expanduser().resolve()
        ids_path = (args.ids or DEFAULT_IDS).expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        cache = (
            args.cache.expanduser().resolve()
            if args.cache
            else output_dir / "atomic_extraction_cache.sqlite3"
        )
        spacy_model = args.spacy_model
        context_budget = args.context_budget
        manual_review_count = args.manual_review_count
        tokenizer_name = args.tokenizer
        tokenizer_revision = args.tokenizer_revision
        raw = {
            "experiment": "M3_atomic_evidence_audit",
            "source": {"selected_evidence": str(selected_evidence), "ids": str(ids_path)},
            "extractor": {"model": spacy_model},
            "tokenizer": {"name": tokenizer_name, "revision": tokenizer_revision},
            "audit": {
                "context_budget": context_budget,
                "manual_review_count": manual_review_count,
            },
            "output_dir": str(output_dir),
            "cache": str(cache),
        }
        config_path = None
    report = run_atomic_extraction(
        selected_evidence,
        ids_path,
        output_dir,
        extractor=SpacyAtomicExtractor(spacy_model),
        token_counter=_token_counter(tokenizer_name, tokenizer_revision),
        cache_path=cache,
        context_budget=context_budget,
        manual_review_count=manual_review_count,
    )
    resolved = {
        **raw,
        "source": {**raw["source"], "selected_evidence": str(selected_evidence), "ids": str(ids_path)},
        "output_dir": str(output_dir),
        "cache": str(cache),
    }
    (output_dir / "config.resolved.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_names = (
        "atomic_evidence.jsonl",
        "atomic_contexts.jsonl",
        "atomic_evidence_audit.json",
        "manual_review_100.jsonl",
        "manual_review_100.csv",
        "config.resolved.json",
    )
    manifest = {
        "schema_version": 1,
        "experiment": "M3_atomic_evidence_audit",
        "config": {
            "path": str(config_path) if config_path else None,
            "sha256": (
                hashlib.sha256(config_path.read_bytes()).hexdigest()
                if config_path
                else None
            ),
        },
        "git": _git_state(),
        "extractor": report["extractor"],
        "input": report["input"],
        "caption_generation_run": False,
        "learned_matcher_used": False,
        "outputs": {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
            }
            for name in output_names
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
