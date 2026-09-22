"""Regenerate only M3 Atomic Token-Matched captions in a recovery-v2 namespace."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import yaml
PROJECT_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT_ROOT/"src"))
from kric.captioning.instructblip import InstructBlipArticleCaptioner  # noqa:E402
from kric.captioning.m3 import MODEL_NAME,MODEL_REVISION,EXPECTED_GENERATION,EXPECTED_PROMPT,build_atomic_model_inputs,controlled_captioner_settings,validate_generator_control  # noqa:E402
from kric.data.goodnews import GoodNewsConfig,GoodNewsDataset  # noqa:E402
from kric.data.selection import select_samples  # noqa:E402
from kric.recovery.common import FROZEN_SELECTED_SHA256,HISTORICAL_B2_PREDICTIONS_SHA256,ORIGINAL_M3_TOKEN_MATCHED_PREDICTIONS_SHA256,read_jsonl,require_recovery_output,sha256,write_json,write_jsonl  # noqa:E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True,type=Path);p.add_argument("--dataset-root",required=True,type=Path);p.add_argument("--b2-dir",required=True,type=Path);p.add_argument("--atomic-dir",required=True,type=Path);p.add_argument("--output-root",type=Path,default=Path("/marimo/outputs/m3_generation_regenerated_v2"));a=p.parse_args();out=require_recovery_output(a.output_root);cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"));validate_generator_control(cfg)
    if sha256(a.b2_dir/"selected_evidence.jsonl")!=FROZEN_SELECTED_SHA256 or sha256(a.b2_dir/"predictions.jsonl")!=HISTORICAL_B2_PREDICTIONS_SHA256:raise ValueError("B2 input is not the recovered historical semantic-k3 run")
    manifest=json.loads((a.atomic_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("recovery_status")!="deterministically_reconstructed_from_partial_original_artifacts":raise ValueError("atomic inputs are not validated regenerated-v2 recovery artifacts")
    primary=json.loads((PROJECT_ROOT/"configs/dataset/m3_complete_49_ids.json").read_text(encoding="utf-8"));contexts_by={str(r["sample_id"]):r for r in read_jsonl(a.atomic_dir/"atomic_contexts_token_matched.jsonl")};contexts=[contexts_by[i] for i in primary]
    root=a.dataset_root.resolve();dataset=GoodNewsDataset(GoodNewsConfig(annotations_path=root/"article+caption.json",splits_path=root/"img_splits.json",images_root=root/"images"));dataset.assert_split_integrity();samples50=select_samples(dataset,split="dev",max_samples=50,strategy="first_by_sample_id");sample_by={s.sample_id:s for s in samples50};samples=[sample_by[i] for i in primary];build_atomic_model_inputs(samples,contexts,primary)
    out.mkdir(parents=True);subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts/check_evaluation_dependencies.py"),"--metrics","cider,entity","--entity-extractor","spacy","--spacy-model","en_core_web_sm","--output",str(out/"evaluation_preflight.json")],check=True);subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts/gpu_preflight.py"),"--output",str(out/"gpu_preflight.json")],check=True)
    model,generation,prompt,context,seed=controlled_captioner_settings(cfg);captioner=InstructBlipArticleCaptioner(model,generation,prompt,context,seed);captioner.load();inputs=build_atomic_model_inputs(samples,contexts,primary);started=time.perf_counter();generated=captioner.generate(inputs);by_id={r.sample_id:r for r in generated};rows=[]
    for sample,row in zip(samples,contexts,strict=True):
        item=by_id[sample.sample_id];used=int(item.context_stats["used_article_tokens"])
        if used!=int(row["context_tokens"]) or used>int(row["b2_context_token_budget"]):raise RuntimeError(f"token accounting drift: {sample.sample_id}")
        rows.append({"sample_id":sample.sample_id,"prediction":item.text,"reference":sample.reference_caption,"metadata":{"dataset":"goodnews","official_split":sample.metadata.get("official_split"),"image_path":sample.image_path,"experiment":"m3_atomic_token_matched_regenerated_v2","model_name":MODEL_NAME,"model_revision":MODEL_REVISION,"context_tokens":used,"context_kind":row["context_kind"],"included_evidence_ids":row["included_evidence_ids"],"dropped_evidence_ids":row["dropped_evidence_ids"],"all_source_sentences_completed":row["all_source_sentences_completed"]}})
    pred=out/"predictions.jsonl";write_jsonl(pred,rows);subprocess.run([sys.executable,str(PROJECT_ROOT/"scripts/evaluate.py"),"--predictions",str(pred),"--output",str(out/"metrics.json"),"--per-sample-output",str(out/"metrics_per_sample.csv"),"--metrics","cider,entity","--entity-extractor","spacy","--spacy-model","en_core_web_sm"],check=True);actual=sha256(pred);status="byte_identical_to_original" if actual==ORIGINAL_M3_TOKEN_MATCHED_PREDICTIONS_SHA256 else "regenerated_not_byte_identical"
    report={"schema_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),"experiment":"M3_atomic_token_matched_generation_regenerated_v2","generation_recovery_status":status,"original_predictions_sha256":ORIGINAL_M3_TOKEN_MATCHED_PREDICTIONS_SHA256,"regenerated_predictions_sha256":actual,"exact_match":actual==ORIGINAL_M3_TOKEN_MATCHED_PREDICTIONS_SHA256,"prediction_level_exact_string_match_rate":None,"prediction_level_comparison_unavailable_reason":"original M3 prediction rows are unavailable","generator_control":{"model":MODEL_NAME,"revision":MODEL_REVISION,"prompt":EXPECTED_PROMPT,"generation":EXPECTED_GENERATION,"seed":2026,"max_context_tokens":384},"ordered_primary_ids":primary,"runtime_seconds":time.perf_counter()-started,"outputs":{"predictions.jsonl":actual,"metrics.json":sha256(out/"metrics.json"),"metrics_per_sample.csv":sha256(out/"metrics_per_sample.csv")},"original_aggregate_metrics_available":False,"caption_variants_run":["atomic_token_matched"]}
    write_json(out/"manifest.json",report);print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
