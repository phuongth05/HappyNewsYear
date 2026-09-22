"""Safely restore the historical semantic-k3 B2 directory from its backup ZIP."""
from __future__ import annotations
import argparse, json, sys, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.recovery.common import FROZEN_SELECTED_SHA256,HISTORICAL_B2_PREDICTIONS_SHA256,read_jsonl,sha256,write_json  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--bundle",required=True,type=Path); p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/semantic_k3_original_recovered"); p.add_argument("--regenerated-dir",type=Path); p.add_argument("--spacy-predecessor-dir",type=Path); a=p.parse_args()
    out=a.output_dir.expanduser().resolve()
    if "_original_recovered" not in out.name: raise ValueError("historical restore output must use _original_recovered namespace")
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite {out}")
    prefix="semantic_k3/"; extracted=[]; out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.bundle) as archive:
        matches=[n for n in archive.namelist() if n.replace('\\','/').split('goodnews_b2_validation_50/',1)[-1].startswith(prefix)]
        if not matches: raise ValueError("bundle has no semantic_k3 directory")
        for name in matches:
            relative=name.replace('\\','/').split(prefix,1)[1]
            parts=PurePosixPath(relative).parts
            if not relative or any(part in {'..',''} for part in parts): continue
            destination=out.joinpath(*parts); destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(archive.read(name)); extracted.append(relative)
    selected_hash=sha256(out/"selected_evidence.jsonl"); prediction_hash=sha256(out/"predictions.jsonl")
    if selected_hash!=FROZEN_SELECTED_SHA256 or prediction_hash!=HISTORICAL_B2_PREDICTIONS_SHA256: raise ValueError("restored historical B2 hash mismatch")
    report={"provenance_status":"historical_b2_original_recovered_from_verified_zip","created_utc":datetime.now(timezone.utc).isoformat(),"bundle_sha256":sha256(a.bundle),"selected_evidence_sha256":selected_hash,"predictions_sha256":prediction_hash,"ordered_sample_ids":[str(r["sample_id"]) for r in read_jsonl(out/"selected_evidence.jsonl")],"extracted_files":sorted(extracted),"primary_b2_artifact":True}
    if a.regenerated_dir:
        rs=sha256(a.regenerated_dir/"selected_evidence.jsonl"); rp=sha256(a.regenerated_dir/"predictions.jsonl"); report["regenerated_comparison"]={"selected_evidence_sha256":rs,"selected_evidence_identical":rs==selected_hash,"predictions_sha256":rp,"predictions_identical":rp==prediction_hash,"note":"predictions differ even when selected evidence is byte-identical"}
    if a.spacy_predecessor_dir: report["excluded_spacy_predecessor"]={"path":str(a.spacy_predecessor_dir.resolve()),"classification":"historical_spacy_predecessor_only","allowed_for_llm_atomic_reconstruction":False,"method":"deterministic_spacy_dependency_and_ner","rule_set_version":"spacy_dependency_atomic_v1","known_total_atomic_units":715}
    write_json(out/"recovery_manifest.json",report); print(json.dumps(report,indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
