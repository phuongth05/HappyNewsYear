from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 2: return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    numerator = sum((x-ma)*(y-mb) for x,y in zip(a,b,strict=True)); da=sum((x-ma)**2 for x in a); db=sum((y-mb)**2 for y in b)
    return numerator / math.sqrt(da*db) if da and db else None


def _ranks(values: Sequence[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        average_rank = ((position + 1) + end) / 2
        for index in ordered[position:end]:
            result[index] = average_rank
        position = end
    return result

def transfer_diagnostic(rows: Sequence[Mapping[str, Any]], threshold: float, total_reviewed: int) -> dict[str, Any]:
    high=[row for row in rows if bool(row.get("high_confidence_mapping")) and row.get("old_score") not in (None,"") and row.get("new_score") not in (None,"") and str(row.get("support_label","")).casefold() in {"supported","unsupported"}]
    old=[float(row["old_score"]) for row in high]; new=[float(row["new_score"]) for row in high]
    labels=[1 if str(row["support_label"]).casefold()=="supported" else 0 for row in high]
    old_d=[x>=threshold for x in old]; new_d=[x>=threshold for x in new]
    agreement=sum(x==y for x,y in zip(old_d,new_d,strict=True))/len(high) if high else 0.0
    recall=sum(p and y for p,y in zip(new_d,labels,strict=True))/sum(labels) if sum(labels) else None
    negatives=sum(y==0 for y in labels); specificity=sum((not p) and y==0 for p,y in zip(new_d,labels,strict=True))/negatives if negatives else None
    pearson=_pearson(old,new); spearman=_pearson(_ranks(old),_ranks(new)) if high else None
    mae=statistics.fmean(abs(x-y) for x,y in zip(old,new,strict=True)) if high else None
    coverage=len(high)/total_reviewed if total_reviewed else 0.0
    if coverage < .80 or pearson is None or pearson < .90 or spearman is None or spearman < .90 or agreement < .90: recommendation="recalibration_required"
    elif mae is not None and mae <= .03 and agreement >= .95 and min(pearson,spearman) >= .95: recommendation="safe_to_reuse"
    elif mae is not None and mae > .05: recommendation="recalibrate_recommended"
    else: recommendation="reuse_with_caution"
    variants={}
    for variant in sorted({str(row.get("caption_variant","")) for row in high}):
        subset=[row for row in high if str(row.get("caption_variant",""))==variant]
        shifts=[float(row["new_score"])-float(row["old_score"]) for row in subset]
        variants[variant]={"count":len(subset),"mean_score_shift":statistics.fmean(shifts)}
    return {"high_confidence_mapped_claims":len(high),"high_confidence_mapping_coverage":coverage,"pearson":pearson,"spearman":spearman,"mean_absolute_score_shift":mae,"threshold_decision_agreement":agreement,"supported_recall":recall,"unsupported_specificity":specificity,"per_variant_score_shifts":variants,"old_threshold":threshold,"transfer_recommendation":recommendation,"threshold_automatically_accepted":False}
