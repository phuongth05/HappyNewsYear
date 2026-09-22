"""Regenerate an explicitly new, non-deterministic M4 claim universe."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.claims.llm_claim_extractor import StructuredClaimExtractor  # noqa:E402
from kric.evidence.llm_structured import LlmEndpointConfig  # noqa:E402
from kric.recovery.common import read_jsonl,require_recovery_output,sha256,write_json,write_jsonl  # noqa:E402
def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path);p.add_argument("--source",action="append",required=True,help="variant=predictions.jsonl");p.add_argument("--primary-ids",type=Path,default=PROJECT_ROOT/"configs/dataset/m3_complete_49_ids.json");p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m4_claims_49_regenerated_v2");a=p.parse_args();out=require_recovery_output(a.output_dir);cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"));ids=json.loads(a.primary_ids.read_text(encoding="utf-8"));allowed={"b2","atomic_full","atomic_token_matched"};sources={}
    for spec in a.source:
        variant,path=spec.split("=",1)
        if variant not in allowed or variant in sources:raise ValueError("sources must have unique known variants")
        sources[variant]=Path(path)
    endpoint=LlmEndpointConfig(api_url=cfg["api_url"],model=cfg["model"],api_key_env=cfg["api_key_env"],model_revision=cfg.get("model_revision"),structured_output_mode=cfg["structured_output_mode"],temperature=cfg.get("temperature"),timeout_seconds=float(cfg["timeout_seconds"]),max_attempts=int(cfg["max_attempts"]));out.mkdir(parents=True, exist_ok=True);extractor=StructuredClaimExtractor(endpoint,out/"claim_response_cache.sqlite3");all_rows=[];failures=[];hashes={}
    for variant,path in sources.items():
        rows=read_jsonl(path);by={str(r["sample_id"]):r for r in rows}
        if any(i not in by for i in ids):raise ValueError(f"{variant} missing primary IDs")
        claims=[]
        for sample_id in ids:
            prediction=str(by[sample_id]["prediction"])
            try:
                result=extractor.extract(prediction,sample_id=sample_id,caption_variant=variant)
                claims.extend({"sample_id":sample_id,"caption_variant":variant,"caption":prediction,"claim_id":c["claim_id"],"claim_text":c["text"],"claim_type":c["type"]} for c in result["claims"])
            except Exception as exc:failures.append({"sample_id":sample_id,"caption_variant":variant,"error_type":type(exc).__name__,"error":str(exc)})
        name=f"{variant}_claims.jsonl";write_jsonl(out/name,claims);all_rows.extend(claims);hashes[variant]=sha256(path)
    write_jsonl(out/"all_claims.jsonl",all_rows);write_jsonl(out/"failures.jsonl",failures);write_json(out/"audit.json",{"requested_captions":len(ids)*len(sources),"claims":len(all_rows),"failures":len(failures),"variants":list(sources),"api":extractor.stats()})
    names=["all_claims.jsonl","failures.jsonl","audit.json",*[f"{v}_claims.jsonl" for v in sources]];manifest={"schema_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),"experiment":"M4_caption_claim_decomposition_49_regenerated_v2","claim_universe_status":"regenerated_non_deterministic","ordered_primary_ids":ids,"variants":list(sources),"scope_limitation":None if set(sources)==allowed else "only explicitly supplied prediction variants were regenerated; no substitution occurred","source_prediction_sha256":hashes,"llm_configuration":{"api_url":cfg["api_url"],"requested_model":cfg["model"],"model_revision":cfg.get("model_revision"),"requested_temperature":cfg.get("temperature"),"effective_temperature":"provider_default" if cfg.get("temperature") is None else cfg.get("temperature")},"api_key_stored_or_logged":False,"reference_passed":False,"article_passed":False,"image_passed":False,"evidence_passed":False,"outputs":{n:sha256(out/n) for n in names}}
    write_json(out/"manifest.json",manifest);print(json.dumps(manifest,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
