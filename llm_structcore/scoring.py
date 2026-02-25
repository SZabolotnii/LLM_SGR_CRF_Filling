from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """
    Multiclass macro-F1, compatible with sklearn.metrics.f1_score(..., average="macro")
    for string labels, with zero_division=0 behavior.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    labels = sorted(set(y_true) | set(y_pred))
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


@dataclass(frozen=True)
class TPFPFN:
    tp: int
    fp: int
    fn: int


def compute_tp_fp_fn(y_true: Iterable[str], y_pred: Iterable[str], *, not_available: str = "unknown") -> TPFPFN:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t != not_available and p != not_available and p == t:
            tp += 1
        elif t == not_available and p != not_available:
            fp += 1
        elif t != not_available and p == not_available:
            fn += 1
        elif t != not_available and p != not_available and p != t:
            fp += 1
    return TPFPFN(tp=tp, fp=fp, fn=fn)


@dataclass(frozen=True)
class SubmissionScore:
    macro_f1: float
    tp: int
    fp: int
    fn: int


def score_records(reference: list[dict], submission: list[dict], *, not_available: str = "unknown") -> SubmissionScore:
    if len(reference) != len(submission):
        raise ValueError(f"Length mismatch: reference={len(reference)} submission={len(submission)}")

    patient_f1s: list[float] = []
    totals = Counter(tp=0, fp=0, fn=0)

    for ref, sub in zip(reference, submission):
        ref_id = str(ref["document_id"])
        sub_doc_id = str(sub["document_id"])
        sub_id, _lang = sub_doc_id.split("_", 1)
        if ref_id != sub_id:
            raise ValueError(f"Document order/id mismatch: ref={ref_id} sub={sub_doc_id}")

        y_true = [a["ground_truth"] for a in ref["annotations"]]
        y_pred = [a["prediction"] for a in sub["predictions"]]
        if len(y_true) != len(y_pred):
            raise ValueError(f"Item count mismatch for doc_id={sub_doc_id}: {len(y_true)} vs {len(y_pred)}")

        patient_f1s.append(macro_f1(y_true, y_pred))
        tpfpfn = compute_tp_fp_fn(y_true, y_pred, not_available=not_available)
        totals["tp"] += tpfpfn.tp
        totals["fp"] += tpfpfn.fp
        totals["fn"] += tpfpfn.fn

    return SubmissionScore(
        macro_f1=sum(patient_f1s) / len(patient_f1s) if patient_f1s else 0.0,
        tp=int(totals["tp"]),
        fp=int(totals["fp"]),
        fn=int(totals["fn"]),
    )

