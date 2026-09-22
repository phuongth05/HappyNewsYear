"""Build a deterministic blank-label review pack for regenerated M4 claims."""
from __future__ import annotations
import argparse,csv,json,random,sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.recovery.common import read_jsonl,require_recovery_output,write_json  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--claims",required=True,type=Path); p.add_argument("--rankings",required=True,type=Path); p.add_argument("--old-threshold",required=True,type=float); p.add_argument("--size",type=int,default=80); p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m4_recovery_review_v2"); a=p.parse_args()
    if not 60<=a.size<=100: raise ValueError("review size must be 60..100")
    out=require_recovery_output(a.output_dir); claims={str(r["claim_id"]):r for r in read_jsonl(a.claims)}; rankings=read_jsonl(a.rankings); groups=defaultdict(list)
    for r in rankings:
        claim=claims[str(r["claim_id"])]; item={**claim,"cosine_top1_score":r.get("best_score"),"cosine_top1_evidence_id":r.get("top1_evidence_id"),"support_label":"","annotator_notes":""}; groups[(str(claim.get("caption_variant","")),str(claim.get("claim_type","")))].append(item)
    rng=random.Random(2026)
    for values in groups.values(): values.sort(key=lambda r:(abs(float(r["cosine_top1_score"])-a.old_threshold),str(r["claim_id"]))); near=values[:max(1,len(values)//2)]; far=values[max(1,len(values)//2):]; rng.shuffle(far); values[:]=near+far
    selected=[]; keys=sorted(groups)
    while len(selected)<a.size and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected)<a.size: selected.append(groups[key].pop(0))
    out.mkdir(parents=True, exist_ok=True); fields=list(selected[0])
    with (out/"human_review_recovery.csv").open("w",encoding="utf-8-sig",newline="") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(selected)
    write_json(out/"sampling_manifest.json",{"claim_universe_status":"regenerated_non_deterministic","seed":2026,"target_size":a.size,"actual_size":len(selected),"old_threshold":a.old_threshold,"labels_prepopulated":False,"sampling":"variant/type stratified round-robin; near-threshold first"}); return 0
if __name__=="__main__": raise SystemExit(main())
