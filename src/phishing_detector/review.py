"""Score a small, human-reviewed URL set and report false positives/negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .features import normalize_url
from .model import MODEL_PATH, ROOT, load_bundle

DEFAULT_INPUT = ROOT / "data" / "curated" / "review_urls.csv"


def _label_to_int(value: object) -> int:
    label = str(value).strip().lower()
    mapping = {"good": 0, "legitimate": 0, "benign": 0, "0": 0,
               "bad": 1, "phishing": 1, "1": 1}
    if label not in mapping:
        raise ValueError(f"Unsupported label {value!r}; use good/bad or legitimate/phishing.")
    return mapping[label]


def evaluate_review_frame(frame: pd.DataFrame, bundle: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Evaluate rows without fitting or changing the saved model."""
    if "URL" not in frame.columns or "Label" not in frame.columns:
        raise ValueError("Review CSV must contain URL and Label columns.")
    rows = frame.copy()
    rows["URL"] = rows["URL"].fillna("").astype(str).str.strip()
    rows = rows[rows["URL"].ne("")].copy()
    if rows.empty:
        raise ValueError("Review set has no non-empty URLs.")
    rows["Actual"] = rows["Label"].map(_label_to_int).astype("int8")
    conflicts = rows.groupby("URL")["Actual"].nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    if conflict_urls:
        raise ValueError(f"Review set has conflicting labels for {len(conflict_urls)} URL(s). Resolve them first.")
    rows = rows.drop_duplicates(subset="URL").reset_index(drop=True)

    bundle = bundle or load_bundle()
    threshold = float(bundle.get("threshold", 0.5))
    scores = bundle["pipeline"].predict_proba(rows["URL"].map(normalize_url).tolist())[:, 1]
    predictions = (scores >= threshold).astype("int8")
    rows["ModelScore"] = scores
    rows["DecisionThreshold"] = threshold
    rows["Predicted"] = pd.Series(predictions).map({0: "legitimate", 1: "phishing"})
    rows["Outcome"] = "correct"
    rows.loc[(rows["Actual"] == 0) & (predictions == 1), "Outcome"] = "false_positive"
    rows.loc[(rows["Actual"] == 1) & (predictions == 0), "Outcome"] = "false_negative"
    rows.loc[(rows["Actual"] == 1) & (predictions == 1), "Outcome"] = "true_positive"
    rows.loc[(rows["Actual"] == 0) & (predictions == 0), "Outcome"] = "true_negative"

    tn, fp, fn, tp = confusion_matrix(rows["Actual"], predictions, labels=[0, 1]).ravel()
    has_both_classes = rows["Actual"].nunique() == 2
    summary = {
        "model_path": str(MODEL_PATH),
        "threshold": threshold,
        "review_urls": int(len(rows)),
        "legitimate_urls": int((rows["Actual"] == 0).sum()),
        "phishing_urls": int((rows["Actual"] == 1).sum()),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "metrics_meaningful_for_both_classes": has_both_classes,
        "accuracy": float(accuracy_score(rows["Actual"], predictions)) if has_both_classes else None,
        "phishing_precision": float(precision_score(rows["Actual"], predictions, zero_division=0)) if has_both_classes else None,
        "phishing_recall": float(recall_score(rows["Actual"], predictions, zero_division=0)) if has_both_classes else None,
        "phishing_f1": float(f1_score(rows["Actual"], predictions, zero_division=0)) if has_both_classes else None,
        "note": "A curated review set is a diagnostic, not a representative performance estimate. Keep it separate from training.",
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-out", type=Path, default=ROOT / "reports" / "review_results.csv")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports" / "review_summary.json")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Review file not found: {args.input}")
    source = pd.read_csv(args.input, dtype={"URL": "string", "Label": "string"})
    summary, results = evaluate_review_frame(source, load_bundle())
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_out, index=False)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Per-URL results: {args.results_out}\nSummary: {args.summary_out}")


if __name__ == "__main__":
    main()
