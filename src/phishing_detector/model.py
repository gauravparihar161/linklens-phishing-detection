"""Model loading and single URL inference."""

from __future__ import annotations

from pathlib import Path

import joblib

from .features import normalize_url

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "url_classifier.joblib"


def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first with: "
            "python -m phishing_detector.train --data data/raw/phishing_site_urls.csv"
        )
    return joblib.load(path)


def predict_one(url: str, bundle: dict | None = None) -> dict[str, float | str]:
    bundle = bundle or load_bundle()
    score = float(bundle["pipeline"].predict_proba([normalize_url(url)])[0, 1])
    threshold = float(bundle.get("threshold", 0.5))
    return {
        "score": score,
        "threshold": threshold,
        "label": "phishing" if score >= threshold else "likely legitimate",
    }
