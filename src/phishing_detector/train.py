"""Train and evaluate the first URL text-classification model."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from .features import normalize_url
from .model import MODEL_PATH, ROOT


def domain_group(value: str) -> str:
    """Approximate grouping key for URLs with or without a scheme."""
    candidate = str(value).strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", candidate, flags=re.I):
        candidate = "http://" + candidate
    try:
        host = (urlsplit(candidate).hostname or "").lower()
    except ValueError:
        host = ""
    return host or normalize_url(value).split("/", 1)[0] or "__empty__"


def choose_threshold(y_true, scores) -> float:
    """Pick validation threshold maximizing F2 (recall-weighted)."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in [i / 100 for i in range(10, 91, 2)]:
        score = fbeta_score(y_true, scores >= threshold, beta=2, zero_division=0)
        if score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/phishing_site_urls.csv")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-out", type=Path, default=ROOT / "reports/evaluation.json")
    parser.add_argument("--time-column", help="Timestamp column for chronological train/validation/test evaluation")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset not found: {args.data}\nCopy phishing_site_urls.csv into data/raw/ or pass --data.")

    print(f"Loading {args.data} ...", flush=True)
    usecols = ["URL", "Label"] + ([args.time_column] if args.time_column else [])
    frame = pd.read_csv(args.data, usecols=usecols, dtype={"URL": "string", "Label": "string"})
    frame = frame.dropna(subset=["URL", "Label"]).copy()
    frame["URL"] = frame["URL"].astype(str).str.strip()
    frame["Label"] = frame["Label"].astype(str).str.strip().str.lower()
    frame = frame[frame["Label"].isin(["good", "bad"]) & frame["URL"].ne("")]
    if args.time_column:
        frame["observed_at"] = pd.to_datetime(frame[args.time_column], errors="coerce", utc=True)
        missing_time_rows = int(frame["observed_at"].isna().sum())
        frame = frame.dropna(subset=["observed_at"])
    else:
        missing_time_rows = 0

    conflicts = frame.groupby("URL")["Label"].nunique()
    conflict_urls = set(conflicts[conflicts > 1].index)
    before_dedup = len(frame)
    frame = frame[~frame["URL"].isin(conflict_urls)]
    if args.time_column:
        frame = frame.sort_values("observed_at")
    frame = frame.drop_duplicates(subset="URL", keep="first").reset_index(drop=True)
    frame["target"] = frame["Label"].map({"good": 0, "bad": 1}).astype("int8")
    frame["text"] = frame["URL"].map(normalize_url)
    groups = frame["URL"].map(domain_group)

    if args.time_column:
        # Chronological 70/15/15 partition. Validation and test contain later observations.
        first_cut = frame["observed_at"].quantile(0.70)
        second_cut = frame["observed_at"].quantile(0.85)
        train_idx = frame.index[frame["observed_at"] <= first_cut].to_numpy()
        val_idx = frame.index[(frame["observed_at"] > first_cut) & (frame["observed_at"] <= second_cut)].to_numpy()
        test_idx = frame.index[frame["observed_at"] > second_cut].to_numpy()
        if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
            raise SystemExit("Temporal split created an empty partition. Check timestamp coverage and --time-column.")
        split_description = "Chronological split: oldest 70% train, next 15% validation, newest 15% test"
        split_dates = {
            "train_end": frame["observed_at"].iloc[train_idx[-1]].isoformat(),
            "validation_start": frame["observed_at"].iloc[val_idx[0]].isoformat(),
            "validation_end": frame["observed_at"].iloc[val_idx[-1]].isoformat(),
            "test_start": frame["observed_at"].iloc[test_idx[0]].isoformat(),
            "test_end": frame["observed_at"].iloc[test_idx[-1]].isoformat(),
        }
        train_hosts = set(groups.iloc[train_idx])
        test_hosts = set(groups.iloc[test_idx])
        split_dates["hostname_overlap_train_test"] = len(train_hosts & test_hosts)
    else:
        # First reserve a domain-disjoint test split, then carve validation groups from train.
        outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        trainval_idx, test_idx = next(outer.split(frame, frame["target"], groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=43)
        train_rel, val_rel = next(inner.split(frame.iloc[trainval_idx], frame["target"].iloc[trainval_idx], groups.iloc[trainval_idx]))
        train_idx, val_idx = trainval_idx[train_rel], trainval_idx[val_rel]
        split_description = "GroupShuffleSplit by approximate hostname; 20% test, validation from remaining groups"
        split_dates = None

    for partition_name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if frame["target"].iloc[indices].nunique() != 2:
            raise SystemExit(f"The {partition_name} partition must contain both labels; adjust timestamps or provide more data.")

    x_train, y_train = frame["text"].iloc[train_idx], frame["target"].iloc[train_idx]
    x_val, y_val = frame["text"].iloc[val_idx], frame["target"].iloc[val_idx]
    x_test, y_test = frame["text"].iloc[test_idx], frame["target"].iloc[test_idx]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                 max_features=150_000, sublinear_tf=True, dtype=np.float32)

    print(f"Rows after cleanup: {len(frame):,}; vectorizing {len(train_idx):,} training URLs ...", flush=True)
    fit_start = time.perf_counter()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    vectorize_seconds = time.perf_counter() - fit_start

    candidates = {
        "Logistic Regression": LogisticRegression(C=4.0, solver="liblinear", max_iter=250,
                                                    class_weight="balanced", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "SGD Logistic Classifier": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30,
                                                  tol=1e-3, class_weight="balanced", random_state=42),
    }
    comparisons = []
    estimators = {}
    for name, estimator in candidates.items():
        print(f"Fitting {name} ...", flush=True)
        candidate_start = time.perf_counter()
        estimator.fit(x_train_vec, y_train)
        candidate_fit_seconds = time.perf_counter() - candidate_start
        val_scores = estimator.predict_proba(x_val_vec)[:, 1]
        threshold = choose_threshold(y_val.to_numpy(), val_scores)
        val_pred = val_scores >= threshold
        comparisons.append({
            "model": name,
            "validation_threshold": threshold,
            "validation_f2": float(fbeta_score(y_val, val_pred, beta=2, zero_division=0)),
            "validation_recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "validation_precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "validation_average_precision": float(average_precision_score(y_val, val_scores)),
            "fit_seconds": round(candidate_fit_seconds, 2),
        })
        estimators[name] = estimator

    # Select using validation data only; the held-out test set is used once for final reporting.
    winner = max(comparisons, key=lambda row: (row["validation_f2"], row["validation_recall"],
                                               row["validation_average_precision"]))
    model_name = winner["model"]
    threshold = float(winner["validation_threshold"])
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", estimators[model_name])])
    start = time.perf_counter()
    # Measure the full deployed inference path, including text vectorization.
    test_scores = pipeline.predict_proba(x_test)[:, 1]
    predict_seconds = time.perf_counter() - start
    test_pred = test_scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "phishing_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "phishing_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "phishing_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "phishing_f2": float(fbeta_score(y_test, test_pred, beta=2, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }
    report = {
        "model": f"Character TF-IDF (3-5 grams) + {model_name}",
        "selection_criterion": "Highest validation F2 after per-model validation threshold selection; ties broken by validation recall, then average precision",
        "selected_model": model_name,
        "model_comparison": comparisons,
        "positive_class": "bad / phishing",
        "split": split_description,
        "rows": {"raw": int(before_dedup), "after_cleanup": int(len(frame)), "conflicting_urls_excluded": int(len(conflict_urls)),
                 "rows_missing_timestamp": missing_time_rows, "train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))},
        "time_split": split_dates,
        "class_counts_after_cleanup": {"legitimate": int((frame.target == 0).sum()), "phishing": int((frame.target == 1).sum())},
        "vectorization_seconds": round(vectorize_seconds, 2),
        "fit_seconds": round(sum(row["fit_seconds"] for row in comparisons), 2),
        "test_predict_seconds": round(predict_seconds, 3),
        "test_urls_per_second": round(len(test_idx) / max(predict_seconds, 1e-9), 1),
        "metrics": metrics,
        "notes": ["Evaluation numbers describe this dataset and split; they do not guarantee performance on future phishing campaigns.",
                  "Hostname grouping is approximate because the source often omits URL schemes and public-suffix metadata."] if not args.time_column else
                 ["Chronological metrics require trustworthy observation timestamps; a crawl/import date is not a substitute for the URL's first-seen date.",
                  "Hostname overlap is reported, not purged, in a chronological split."],
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "model_name": report["model"]}, args.model_out, compress=3)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved model: {args.model_out}\nSaved report: {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
