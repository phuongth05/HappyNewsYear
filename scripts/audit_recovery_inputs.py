"""Audit partial original M3/M4 recovery inputs without modifying them."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.recovery.common import read_csv, write_json  # noqa:E402
from kric.recovery.m3 import audit_inputs  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("atomic-evidence","atomic-contexts-full","selected-evidence","human-review-csv","frozen-annotations","primary-ids"): p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--output",type=Path,default=PROJECT_ROOT/"artifacts/recovery_v2/input_audit.json")
    a=p.parse_args(); ids=json.loads(a.primary_ids.read_text(encoding="utf-8"))
    report=audit_inputs(atomic_evidence=a.atomic_evidence,atomic_contexts_full=a.atomic_contexts_full,selected_evidence=a.selected_evidence,human_review_csv=read_csv(a.human_review_csv),frozen_annotations=read_csv(a.frozen_annotations),primary_ids=ids)
    report["created_utc"]=datetime.now(timezone.utc).isoformat(); write_json(a.output,report); print(json.dumps(report,indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
