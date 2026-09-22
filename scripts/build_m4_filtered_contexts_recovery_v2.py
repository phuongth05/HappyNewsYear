"""Build M4 filtered contexts only after an explicit regenerated-v2 threshold policy."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.captioning.instructblip import _InstructBlipCaptioner  # noqa:E402
from kric.captioning.m3 import MODEL_NAME,MODEL_REVISION  # noqa:E402
from kric.matching.cosine import COSINE_MODEL,COSINE_REVISION  # noqa:E402
from kric.matching.filtering import build_support_filtered_contexts  # noqa:E402
from kric.recovery.common import read_jsonl,require_recovery_output,sha256,write_json,write_jsonl  # noqa:E402
ALLOWED={"transferred_from_original_m4_human_review","recalibrated_on_regenerated_claim_universe"}
def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--threshold-policy",required=True,type=Path);p.add_argument("--claims-dir",required=True,type=Path);p.add_argument("--matcher-dir",required=True,type=Path);p.add_argument("--atomic-dir",required=True,type=Path);p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m4_filtered_contexts_49_regenerated_v2");p.add_argument("--primary-ids",type=Path,default=PROJECT_ROOT/"configs/dataset/m3_complete_49_ids.json");a=p.parse_args();out=require_recovery_output(a.output_dir);policy=json.loads(a.threshold_policy.read_text(encoding="utf-8"));origin=policy.get("threshold_origin")
    if origin not in ALLOWED or policy.get("selected_threshold") is None:raise ValueError("threshold policy must explicitly decide transfer or recalibration")
    claim_manifest=json.loads((a.claims_dir/"manifest.json").read_text(encoding="utf-8"));matcher_manifest=json.loads((a.matcher_dir/"manifest.json").read_text(encoding="utf-8"));atomic_manifest=json.loads((a.atomic_dir/"manifest.json").read_text(encoding="utf-8"))
    if claim_manifest.get("claim_universe_status")!="regenerated_non_deterministic" or matcher_manifest.get("claim_universe_status")!="regenerated_non_deterministic":raise ValueError("claims/rankings are not regenerated-v2 artifacts")
    if atomic_manifest.get("recovery_status")!="deterministically_reconstructed_from_partial_original_artifacts":raise ValueError("atomic source is not deterministic recovery")
    model=matcher_manifest.get("model",{})
    if model.get("model")!=COSINE_MODEL or model.get("revision")!=COSINE_REVISION:raise ValueError("cosine pin changed")
    ids=json.loads(a.primary_ids.read_text(encoding="utf-8"));claims=a.claims_dir/"atomic_token_matched_claims.jsonl";rankings=a.matcher_dir/"cosine_rankings.jsonl";atomic=a.atomic_dir/"atomic_evidence.jsonl";matched=a.atomic_dir/"atomic_contexts_token_matched.jsonl"
    from transformers import InstructBlipProcessor
    tokenizer=InstructBlipProcessor.from_pretrained(MODEL_NAME,revision=MODEL_REVISION).tokenizer
    counter=lambda text:len(_InstructBlipCaptioner._token_ids(tokenizer,text))
    contexts,audit=build_support_filtered_contexts(primary_ids=ids,claim_rows=read_jsonl(claims),ranking_rows=read_jsonl(rankings),atomic_rows=read_jsonl(atomic),token_matched_rows=read_jsonl(matched),threshold=float(policy["selected_threshold"]),token_counter=counter);out.mkdir(parents=True, exist_ok=True);write_jsonl(out/"filtered_contexts.jsonl",contexts);audit.update({"threshold_origin":origin,"threshold_policy_sha256":sha256(a.threshold_policy)});write_json(out/"audit.json",audit);manifest={"schema_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),"experiment":"M4_support_filtered_contexts_49_regenerated_v2","threshold_origin":origin,"selected_threshold":float(policy["selected_threshold"]),"claim_universe_status":"regenerated_non_deterministic","ordered_primary_ids":ids,"tokenizer":{"model":MODEL_NAME,"revision":MODEL_REVISION},"provenance":{"threshold_policy":sha256(a.threshold_policy),"claims":sha256(claims),"rankings":sha256(rankings),"atomic_evidence":sha256(atomic),"token_matched_contexts":sha256(matched)},"outputs":{"filtered_contexts.jsonl":sha256(out/"filtered_contexts.jsonl"),"audit.json":sha256(out/"audit.json")}};write_json(out/"manifest.json",manifest);print(json.dumps(manifest,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
