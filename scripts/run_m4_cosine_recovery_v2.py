"""Regenerate pinned cosine rankings for an M4 regenerated-v2 claim universe."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.matching.cosine import COSINE_MODEL,COSINE_REVISION,CosineMatcher  # noqa:E402
from kric.matching.ranking import rank_claims_against_same_sample_evidence,ranking_audit  # noqa:E402
from kric.recovery.common import read_jsonl,require_recovery_output,sha256,write_json,write_jsonl  # noqa:E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--claims",required=True,type=Path);p.add_argument("--atomic-evidence",required=True,type=Path);p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m4_matcher_baselines_49_regenerated_v2");p.add_argument("--device");a=p.parse_args();out=require_recovery_output(a.output_dir);claims=read_jsonl(a.claims);evidence=read_jsonl(a.atomic_evidence);matcher=CosineMatcher(COSINE_MODEL,COSINE_REVISION,a.device);rows=rank_claims_against_same_sample_evidence(claims,evidence,matcher);out.mkdir(parents=True);write_jsonl(out/"cosine_rankings.jsonl",rows);write_json(out/"audit.json",ranking_audit(rows));manifest={"schema_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),"experiment":"M4_matcher_baselines_49_regenerated_v2","claim_universe_status":"regenerated_non_deterministic","model":matcher.model_info(),"same_sample_only":True,"deterministic_tie_breaking":"score_desc_then_evidence_id","claims_sha256":sha256(a.claims),"atomic_evidence_sha256":sha256(a.atomic_evidence),"outputs":{"cosine_rankings.jsonl":sha256(out/"cosine_rankings.jsonl"),"audit.json":sha256(out/"audit.json")},"nli_status":"not_run_optional"};write_json(out/"manifest.json",manifest);print(json.dumps(manifest,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
