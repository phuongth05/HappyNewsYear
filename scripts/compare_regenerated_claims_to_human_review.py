"""Map old reviewed claims one-to-one to the regenerated claim universe."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.matching.cosine import COSINE_MODEL,COSINE_REVISION,CosineMatcher  # noqa:E402
from kric.recovery.common import read_csv,read_jsonl,require_recovery_output,write_json  # noqa:E402
from kric.recovery.compatibility import compatibility_report,map_reviewed_claims  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--human-review",required=True,type=Path); p.add_argument("--regenerated-claims",required=True,type=Path); p.add_argument("--regenerated-predictions",required=True,type=Path); p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m4_recovery_claim_compatibility_v2")
    a=p.parse_args(); out=require_recovery_output(a.output_dir)
    originals=read_csv(a.human_review); regenerated=read_jsonl(a.regenerated_claims); predictions=read_jsonl(a.regenerated_predictions)
    prediction_ids={str(row["sample_id"]) for row in predictions}
    if any(str(row.get("sample_id")) not in prediction_ids for row in regenerated): raise ValueError("regenerated claim has no regenerated prediction")
    matcher=CosineMatcher(COSINE_MODEL,COSINE_REVISION)
    rows=map_reviewed_claims(originals,regenerated,matcher.score_pairs); out.mkdir(parents=True, exist_ok=True)
    fields=list(rows[0]) if rows else []
    with (out/"claim_mapping.csv").open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    report=compatibility_report(rows); report.update({"cosine_model":COSINE_MODEL,"cosine_revision":COSINE_REVISION,"one_to_one":True,"same_sample_only":True,"prefer_same_variant":True})
    write_json(out/"compatibility_report.json",report); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
