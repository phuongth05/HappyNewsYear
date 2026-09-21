"""Deterministic, self-contained M4 claim-support annotation interface."""

from __future__ import annotations

import csv
import io
import json
from collections import OrderedDict
from typing import Any, Mapping, Sequence


ANNOTATION_FIELDS = ("support_label", "best_evidence_ids", "matcher_preference", "notes")


def parse_best_evidence_ids(value: str) -> list[str]:
    normalized = value.replace(",", ";").replace("|", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def toggle_best_evidence_id(value: str, evidence_id: str, candidate_order: Sequence[str]) -> str:
    selected = set(parse_best_evidence_ids(value))
    if evidence_id in selected:
        selected.remove(evidence_id)
    else:
        selected.add(evidence_id)
    known = [item for item in candidate_order if item in selected]
    extra = sorted(selected - set(candidate_order))
    return ";".join(known + extra)


def group_candidate_evidence(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for item in items:
        key = (str(item["source_sentence_id"]), str(item["source_sentence_text"]))
        group = groups.setdefault(
            key,
            {
                "source_sentence_id": key[0],
                "source_sentence_text": key[1],
                "evidence": [],
            },
        )
        group["evidence"].append(dict(item))
    return list(groups.values())


def apply_annotations(
    rows: Sequence[Mapping[str, str]], annotations: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for source in rows:
        row = dict(source)
        update = annotations.get(str(row.get("claim_id", "")), {})
        for field in ANNOTATION_FIELDS:
            if field in update:
                row[field] = str(update[field])
        output.append(row)
    return output


def export_csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fieldnames), extrasaction="raise", lineterminator="\r\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + stream.getvalue()


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_annotation_html(
    fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]], *, source_sha256: str
) -> str:
    prepared: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        context = json.loads(row["union_top3_evidence_context_json"])
        if not isinstance(context, list):
            raise ValueError(f"candidate evidence context is not a list for {row.get('claim_id')}")
        row["_candidate_context"] = context
        row["_candidate_groups"] = group_candidate_evidence(context)
        prepared.append(row)
    payload = _safe_json(
        {
            "fieldnames": list(fieldnames),
            "rows": prepared,
            "source_sha256": source_sha256,
            "storage_key": f"m4_claim_support_v2_{source_sha256[:16]}",
        }
    )
    return _HTML_TEMPLATE.replace("__M4_DATA__", payload)


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>M4 Claim Support Annotation</title>
<style>
:root{font-family:Arial,Helvetica,sans-serif;color:#18212f;background:#f4f6f8;line-height:1.45}
*{box-sizing:border-box}body{margin:0}.shell{max-width:1280px;margin:auto;padding:20px}.top{position:sticky;top:0;z-index:4;background:#f4f6f8;padding:8px 0 14px;border-bottom:1px solid #d7dde5}.title{font-size:22px;margin:0 0 4px}.muted,.score{color:#667085;font-size:13px}.progress-track{height:8px;background:#dfe4ea;border-radius:5px;overflow:hidden;margin:10px 0}.progress-fill{height:100%;background:#2d6a4f;width:0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.controls input,.controls select,button,textarea{font:inherit}.controls input,.controls select{padding:7px;border:1px solid #b8c1cc;border-radius:5px;background:#fff}button{border:1px solid #aeb8c4;background:#fff;padding:7px 11px;border-radius:5px;cursor:pointer}button:hover{background:#edf2f7}button.active{background:#244c66;color:#fff;border-color:#244c66}.panel,.card{background:#fff;border:1px solid #d7dde5;border-radius:8px;padding:16px;margin-top:14px}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#667085}.claim{font-size:20px;font-weight:700;margin:7px 0 16px}.caption{font-size:16px}.matcher-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.matcher-card h3{margin:0 0 10px;font-size:15px}.evidence-text{font-size:16px;font-weight:600;margin:5px 0}.parent-source{background:#f6f8fa;border-left:3px solid #9aa7b4;padding:9px;margin-top:9px}.strict .parent-source{display:none}.candidate-group{border-top:1px solid #e1e6eb;padding-top:12px;margin-top:12px}.candidate-group:first-child{border-top:0}.evidence-item{padding:9px;margin-top:7px;border:1px solid #dfe4ea;border-radius:6px}.evidence-id{font-family:Consolas,monospace;color:#24527a;background:#eef4f8}.evidence-id.selected{background:#244c66;color:#fff}.annotation-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.choice-row{display:flex;gap:7px;flex-wrap:wrap;margin:7px 0 14px}.choice-row button.selected{background:#2d6a4f;color:#fff;border-color:#2d6a4f}textarea{width:100%;min-height:80px;padding:8px;border:1px solid #b8c1cc;border-radius:5px}.notice{background:#fff8db;border:1px solid #e4cc72;padding:10px;border-radius:6px;margin-top:10px}.instructions summary{cursor:pointer;font-weight:700}.hidden{display:none!important}@media(max-width:800px){.matcher-grid,.annotation-grid{grid-template-columns:1fr}.shell{padding:10px}}
</style>
</head>
<body><div class="shell">
<header class="top">
  <h1 class="title">M4 Claim Support Annotation</h1>
  <div id="progressText" class="muted"></div><div class="progress-track"><div id="progressFill" class="progress-fill"></div></div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search claim, caption, sample ID">
    <select id="variantFilter"><option value="">All caption variants</option></select>
    <select id="typeFilter"><option value="">All claim types</option></select>
    <select id="statusFilter"><option value="">All rows</option><option value="unannotated">Unannotated</option><option value="annotated">Annotated</option></select>
    <button id="prev">Previous</button><button id="next">Next</button><button id="nextUnannotated">Next unannotated</button>
  </div>
  <div class="controls" style="margin-top:8px">
    <button id="contextMode" class="active">Context-Assisted View</button><button id="strictMode">Strict Evidence View</button>
    <button id="save">Save</button><button id="load">Load</button><input id="loadFile" type="file" accept="application/json" class="hidden"><button id="exportCsv">Export CSV</button>
  </div>
  <div class="notice">Parent source sentences are provided only to resolve ambiguity and provenance. Judge support based on the candidate atomic evidence. Do not use unrelated facts from the parent sentence as additional evidence.</div>
</header>
<details class="panel instructions"><summary>Annotation guidance</summary>
  <p><strong>SUPPORTED:</strong> At least one candidate atomic evidence unit directly or sufficiently supports the claim.</p>
  <p><strong>UNSUPPORTED:</strong> No candidate atomic evidence unit supports the claim.</p>
  <p><strong>UNCERTAIN:</strong> Evidence is related but insufficient, ambiguous, malformed, or requires unsupported inference.</p>
  <p><strong>BEST EVIDENCE IDS:</strong> Select only evidence units that actually support the claim. Lexical similarity alone is insufficient.</p>
  <p><strong>MATCHER PREFERENCE:</strong> cosine or nli when that ranking places useful support better; tie for comparable quality; neither when neither retrieves useful support.</p>
  <p>Judge support within the presented candidate evidence pool. Do not judge whether the claim is true in the real world.</p>
</details>
<main id="review"></main>
</div>
<script id="m4-data" type="application/json">__M4_DATA__</script>
<script>
"use strict";
const DATA=JSON.parse(document.getElementById("m4-data").textContent);
const rows=DATA.rows, fields=DATA.fieldnames, storageKey=DATA.storage_key;
let visible=[], position=0, mode="context";
const $=id=>document.getElementById(id);
function candidateIds(row){return row._candidate_context.map(x=>x.evidence_id)}
function parseIds(value){return String(value||"").replace(/[|,]/g,";").split(";").map(x=>x.trim()).filter(Boolean)}
function setBest(row,id){const selected=new Set(parseIds(row.best_evidence_ids));selected.has(id)?selected.delete(id):selected.add(id);const order=candidateIds(row);row.best_evidence_ids=[...order.filter(x=>selected.has(x)),...[...selected].filter(x=>!order.includes(x)).sort()].join(";");persist();render()}
function persist(){const a={};for(const r of rows)a[r.claim_id]={support_label:r.support_label||"",best_evidence_ids:r.best_evidence_ids||"",matcher_preference:r.matcher_preference||"",notes:r.notes||""};localStorage.setItem(storageKey,JSON.stringify(a))}
function applyState(a){for(const r of rows){const x=a[r.claim_id];if(!x)continue;for(const f of ["support_label","best_evidence_ids","matcher_preference","notes"])if(Object.prototype.hasOwnProperty.call(x,f))r[f]=String(x[f]??"")}persist();refresh()}
function restore(){try{const raw=localStorage.getItem(storageKey);if(raw)applyState(JSON.parse(raw))}catch(e){console.warn("Could not restore local annotations",e)}}
function isAnnotated(r){return Boolean(r.support_label)}
function refresh(){const q=$("search").value.toLowerCase(),v=$("variantFilter").value,t=$("typeFilter").value,s=$("statusFilter").value;visible=rows.map((r,i)=>i).filter(i=>{const r=rows[i],hay=[r.sample_id,r.claim_id,r.claim_text,r.caption].join(" ").toLowerCase();return(!q||hay.includes(q))&&(!v||r.caption_variant===v)&&(!t||r.claim_type===t)&&(!s||(s==="annotated"?isAnnotated(r):!isAnnotated(r)))});position=Math.min(position,Math.max(0,visible.length-1));render()}
function node(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text??"");if(className)e.className=className;return e}
function details(parent,label,value,className=""){const d=node("div",undefined,className),l=node("div",label,"label");d.append(l,node("div",value));parent.append(d);return d}
function evidenceButton(item,row){const b=node("button",item.evidence_id,"evidence-id"+(parseIds(row.best_evidence_ids).includes(item.evidence_id)?" selected":""));b.type="button";b.title="Toggle as best evidence";b.onclick=()=>setBest(row,item.evidence_id);return b}
function matcherCard(row,matcher){const id=row[matcher+"_top1_id"],item=row._candidate_context.find(x=>x.evidence_id===id)||{};const card=node("section",undefined,"card matcher-card");card.append(node("h3",matcher.toUpperCase()+" TOP-1"));card.append(evidenceButton({evidence_id:id},row));details(card,"Evidence text",row[matcher+"_top1_text"],"evidence-text");details(card,"Evidence type",item.evidence_type||"");details(card,"Source sentence ID",row[matcher+"_top1_source_sentence_id"]||"");details(card,"Full parent source sentence",row[matcher+"_top1_source_sentence_text"]||"","parent-source");details(card,"Raw score",row[matcher+"_top1_score"],"score");return card}
function evidenceSelectionText(x){const parts=[];for(const m of ["cosine","nli"]){if(x[m+"_rank"]!=null)parts.push(m+" rank "+x[m+"_rank"])}return parts.join(", ")}
function render(){const host=$("review");host.replaceChildren();const done=rows.filter(isAnnotated).length;$("progressText").textContent=`${done} / ${rows.length} annotated · ${visible.length} rows in current filter`;$("progressFill").style.width=(100*done/rows.length)+"%";if(!visible.length){host.append(node("div","No rows match the filters.","panel"));return}const row=rows[visible[position]];const main=node("article",undefined,"panel "+(mode==="strict"?"strict":""));const meta=node("div",undefined,"meta");details(meta,"Sample ID",row.sample_id);details(meta,"Caption variant",row.caption_variant);details(meta,"Claim type",row.claim_type);details(meta,"Position",`${position+1} / ${visible.length}`);main.append(meta,node("div",row.claim_text,"claim"));details(main,"Generated caption",row.caption,"caption");const matchers=node("div",undefined,"matcher-grid");matchers.append(matcherCard(row,"cosine"),matcherCard(row,"nli"));main.append(matchers);const context=node("section",undefined,"card");context.append(node("h2","Candidate Evidence Context"));context.append(node("p","Scores are not calibrated probabilities and should not be used as annotation labels.","muted"));for(const group of row._candidate_groups){const g=node("div",undefined,"candidate-group");g.append(node("h3","Source sentence #"+group.source_sentence_id));const parent=node("div",undefined,"parent-source");parent.append(node("div","Original source sentence:","label"),node("div",group.source_sentence_text));g.append(parent);for(const item of group.evidence){const e=node("div",undefined,"evidence-item");e.append(evidenceButton(item,row),node("div",item.evidence_text,"evidence-text"));details(e,"Type",item.evidence_type);details(e,"Selected by",evidenceSelectionText(item));g.append(e)}context.append(g)}main.append(context);const annotation=node("section",undefined,"card annotation-grid");annotation.append(choiceBlock("Support label",["supported","unsupported","uncertain"],row,"support_label"),choiceBlock("Matcher preference",["cosine","nli","tie","neither"],row,"matcher_preference"));const notes=node("div");notes.append(node("div","Notes","label"));const textarea=node("textarea");textarea.value=row.notes||"";textarea.oninput=()=>{row.notes=textarea.value;persist()};notes.append(textarea);annotation.append(notes);details(annotation,"Best evidence IDs",row.best_evidence_ids||"None selected");main.append(annotation);host.append(main)}
function choiceBlock(label,values,row,field){const box=node("div");box.append(node("div",label,"label"));const choices=node("div",undefined,"choice-row");for(const value of values){const b=node("button",value,String(row[field]||"")===value?"selected":"");b.type="button";b.onclick=()=>{row[field]=value;persist();render()};choices.append(b)}box.append(choices);return box}
function move(delta){if(!visible.length)return;position=(position+delta+visible.length)%visible.length;render();window.scrollTo({top:0,behavior:"smooth"})}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function csvCell(v){const s=String(v??"");return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
function exportCsv(){const lines=[fields.map(csvCell).join(",")];for(const r of rows)lines.push(fields.map(f=>csvCell(r[f])).join(","));download("human_review_claim_support_context_annotated.csv","\ufeff"+lines.join("\r\n")+"\r\n","text/csv;charset=utf-8")}
function saveJson(){const annotations={};for(const r of rows)annotations[r.claim_id]={support_label:r.support_label||"",best_evidence_ids:r.best_evidence_ids||"",matcher_preference:r.matcher_preference||"",notes:r.notes||""};download("m4_claim_support_annotations.json",JSON.stringify({source_sha256:DATA.source_sha256,annotations},null,2),"application/json")}
function populate(id,values){for(const v of [...new Set(values)].sort()){const o=node("option",v);o.value=v;$(id).append(o)}}
populate("variantFilter",rows.map(r=>r.caption_variant));populate("typeFilter",rows.map(r=>r.claim_type));
for(const id of ["search","variantFilter","typeFilter","statusFilter"])$(id).addEventListener(id==="search"?"input":"change",()=>{position=0;refresh()});
$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("nextUnannotated").onclick=()=>{if(!visible.length)return;for(let n=1;n<=visible.length;n++){const p=(position+n)%visible.length;if(!isAnnotated(rows[visible[p]])){position=p;render();return}}};
$("contextMode").onclick=()=>{mode="context";$("contextMode").classList.add("active");$("strictMode").classList.remove("active");render()};$("strictMode").onclick=()=>{mode="strict";$("strictMode").classList.add("active");$("contextMode").classList.remove("active");render()};
$("save").onclick=saveJson;$("load").onclick=()=>$("loadFile").click();$("loadFile").onchange=async e=>{const f=e.target.files[0];if(!f)return;const value=JSON.parse(await f.text());if(value.source_sha256&&value.source_sha256!==DATA.source_sha256){alert("Annotation file belongs to a different review CSV.");return}applyState(value.annotations||value)};$("exportCsv").onclick=exportCsv;
document.addEventListener("keydown",e=>{if(["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return;if(e.key==="ArrowLeft")move(-1);else if(e.key==="ArrowRight")move(1);else if(["1","2","3"].includes(e.key)){const r=rows[visible[position]];r.support_label={"1":"supported","2":"unsupported","3":"uncertain"}[e.key];persist();render()}else if(["c","n","t","x","C","N","T","X"].includes(e.key)){const r=rows[visible[position]];r.matcher_preference={c:"cosine",n:"nli",t:"tie",x:"neither"}[e.key.toLowerCase()];persist();render()}});
restore();refresh();
</script></body></html>'''


__all__ = [
    "ANNOTATION_FIELDS",
    "apply_annotations",
    "build_annotation_html",
    "export_csv_text",
    "group_candidate_evidence",
    "parse_best_evidence_ids",
    "toggle_best_evidence_id",
]
