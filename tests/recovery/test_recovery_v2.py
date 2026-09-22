from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kric.recovery import m3
from kric.recovery.common import require_recovery_output
from kric.recovery.compatibility import map_reviewed_claims
from kric.recovery.threshold_transfer import transfer_diagnostic


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path):
    unit = {"evidence_id": "e1", "text": "Alice arrived.", "type": "event", "provenance": {"sample_id": "s1"}}
    atomic = tmp_path / "atomic.jsonl"; full = tmp_path / "full.jsonl"; selected = tmp_path / "selected.jsonl"
    _write_jsonl(atomic, [{"sample_id": "s1", "atomic_units": [unit], "failed_source_sentences": 0}])
    _write_jsonl(full, [{"sample_id": "s1", "context": "Evidence:\n- Alice arrived."}])
    _write_jsonl(selected, [{"sample_id": "s1", "context_token_count": 10}])
    return atomic, full, selected


def test_recovery_namespace_and_no_overwrite(tmp_path: Path):
    with pytest.raises(ValueError): require_recovery_output(tmp_path / "original")
    target = tmp_path / "x_regenerated_v2"; target.mkdir(); (target / "keep").write_text("x")
    with pytest.raises(FileExistsError): require_recovery_output(target)


def test_hash_mismatch_is_rejected(tmp_path: Path):
    atomic, full, selected = _inputs(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        m3.reconstruct_token_matched(atomic_evidence=atomic, atomic_contexts_full=full, selected_evidence=selected, primary_ids=["s1"], output_dir=tmp_path/"bad_regenerated_v2", token_counter=lambda text: len(text.split()))


def test_deterministic_reconstruction_and_provenance(tmp_path: Path, monkeypatch):
    atomic, full, selected = _inputs(tmp_path)
    expected = hashlib.sha256(selected.read_bytes()).hexdigest(); monkeypatch.setattr(m3, "FROZEN_SELECTED_SHA256", expected)
    outputs=[]
    for suffix in ("a_regenerated_v2", "b_regenerated_v2"):
        out=tmp_path/suffix
        out.mkdir()  # Cloud mounts commonly pre-create an empty output directory.
        manifest=m3.reconstruct_token_matched(atomic_evidence=atomic,atomic_contexts_full=full,selected_evidence=selected,primary_ids=["s1"],output_dir=out,token_counter=lambda text:len(text.split()))
        assert manifest["recovery_status"] == "deterministically_reconstructed_from_partial_original_artifacts"
        assert manifest["every_reconstructed_sample_deterministic"] is True
        outputs.append((out/"atomic_contexts_token_matched.jsonl").read_bytes())
        assert (out/"atomic_evidence.jsonl").read_bytes() == atomic.read_bytes()
        assert not (out/"failures.jsonl").exists()
    assert outputs[0] == outputs[1]


def test_claim_mapping_is_one_to_one_same_sample_and_variant_aware():
    originals=[{"sample_id":"s1","claim_id":"o1","claim_text":"Alice arrived","caption_variant":"b2","claim_type":"event"},{"sample_id":"s1","claim_id":"o2","claim_text":"Bob left","caption_variant":"atomic_token_matched","claim_type":"event"},{"sample_id":"s2","claim_id":"o3","claim_text":"Never cross sample","caption_variant":"b2","claim_type":"event"}]
    regenerated=[{"sample_id":"s1","claim_id":"n1","claim_text":"Alice arrived","caption_variant":"b2"},{"sample_id":"s1","claim_id":"n2","claim_text":"Bob departed","caption_variant":"atomic_token_matched"},{"sample_id":"s3","claim_id":"n3","claim_text":"Never cross sample","caption_variant":"b2"}]
    def scorer(pairs): return [1.0 if a.casefold()==b.casefold() else .9 for a,b in pairs]
    rows=map_reviewed_claims(originals,regenerated,scorer)
    assigned=[row["regenerated_claim_id"] for row in rows if row["regenerated_claim_id"]]
    assert len(assigned)==len(set(assigned)); assert rows[0]["regenerated_claim_id"]=="n1"; assert rows[1]["regenerated_claim_id"]=="n2"; assert rows[2]["regenerated_claim_id"]==""
    assert rows[1]["variant_match"] is True


def test_threshold_transfer_metrics_and_conservative_recommendation():
    weak=[{"high_confidence_mapping":True,"old_score":.9,"new_score":.1,"support_label":"supported","caption_variant":"b2"} for _ in range(7)]
    report=transfer_diagnostic(weak,.5,10)
    assert report["high_confidence_mapping_coverage"]==.7
    assert report["transfer_recommendation"]=="recalibration_required"
    strong=[]
    for i in range(20):
        score=.2+i*.03; strong.append({"high_confidence_mapping":True,"old_score":score,"new_score":score+.001,"support_label":"supported" if score>=.5 else "unsupported","caption_variant":"atomic_token_matched"})
    report=transfer_diagnostic(strong,.5,20)
    assert report["pearson"] > .99 and report["spearman"] > .99
    assert report["threshold_decision_agreement"] == 1.0
    assert report["transfer_recommendation"] == "safe_to_reuse"


def test_explicit_non_deterministic_claim_provenance_label_present():
    script=Path("scripts/run_m4_claim_recovery_v2.py").read_text(encoding="utf-8")
    assert '"claim_universe_status"' in script
    assert '"regenerated_non_deterministic"' in script
