"""Deterministically reconstruct missing M3 token-matched contexts from backups."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.captioning.instructblip import _InstructBlipCaptioner  # noqa:E402
from kric.captioning.m3 import MODEL_NAME, MODEL_REVISION  # noqa:E402
from kric.recovery.m3 import reconstruct_token_matched  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--atomic-evidence",required=True,type=Path); p.add_argument("--atomic-contexts-full",required=True,type=Path); p.add_argument("--selected-evidence",required=True,type=Path); p.add_argument("--primary-ids",type=Path,default=PROJECT_ROOT/"configs/dataset/m3_complete_49_ids.json"); p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"artifacts/m3_atomic_regenerated_v2")
    a=p.parse_args()
    from transformers import InstructBlipProcessor
    processor=InstructBlipProcessor.from_pretrained(MODEL_NAME,revision=MODEL_REVISION)
    tokenizer=getattr(processor,"tokenizer",None)
    if tokenizer is None: raise RuntimeError("InstructBLIP processor has no tokenizer")
    def count(text:str)->int: return len(_InstructBlipCaptioner._token_ids(tokenizer,text))
    manifest=reconstruct_token_matched(atomic_evidence=a.atomic_evidence,atomic_contexts_full=a.atomic_contexts_full,selected_evidence=a.selected_evidence,primary_ids=json.loads(a.primary_ids.read_text(encoding="utf-8")),output_dir=a.output_dir,token_counter=count)
    print(json.dumps(manifest,indent=2,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
