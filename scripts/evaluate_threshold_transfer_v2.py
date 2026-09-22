"""Diagnose transfer of the original M4 threshold to regenerated claims."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.recovery.common import read_csv,read_jsonl,write_json  # noqa:E402
from kric.recovery.threshold_transfer import transfer_diagnostic  # noqa:E402

def _threshold(value:dict)->float:
    for path in (("calibration","selected_threshold"),("selected_threshold",),("threshold",),("selection","threshold"),("selected","threshold")):
        cur=value
        try:
            for key in path: cur=cur[key]
            return float(cur)
        except (KeyError,TypeError,ValueError): pass
    raise ValueError("cannot find frozen threshold in manifest")

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--human-review",required=True,type=Path); p.add_argument("--claim-mapping",required=True,type=Path); p.add_argument("--regenerated-rankings",required=True,type=Path); p.add_argument("--threshold-manifest",required=True,type=Path); p.add_argument("--output",required=True,type=Path); a=p.parse_args()
    review={(r["sample_id"],r["claim_id"]):r for r in read_csv(a.human_review)}; mappings=read_csv(a.claim_mapping); rankings={str(r["claim_id"]):r for r in read_jsonl(a.regenerated_rankings)}
    rows=[]
    for m in mappings:
        old=review[(m["sample_id"],m["claim_id"])]; new_id=m.get("regenerated_claim_id",""); ranked=rankings.get(new_id,{})
        rows.append({"caption_variant":m.get("caption_variant",""),"high_confidence_mapping":str(m.get("high_confidence_mapping","")).casefold() in {"true","1"},"old_score":old.get("cosine_top1_score"),"new_score":ranked.get("best_score"),"support_label":old.get("support_label","")})
    report=transfer_diagnostic(rows,_threshold(json.loads(a.threshold_manifest.read_text(encoding="utf-8"))),len(review)); write_json(a.output,report); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
