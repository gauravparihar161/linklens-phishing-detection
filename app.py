from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phishing_detector.features import explain_url
from phishing_detector.model import MODEL_PATH, load_bundle, predict_one
from phishing_detector.review import evaluate_review_frame
from phishing_detector.threat_feed import load_feed, lookup_url, refresh_feed

REPORT_PATH = ROOT / "reports" / "evaluation.json"
REVIEW_PATH = ROOT / "data" / "curated" / "review_urls.csv"
# Keep the optional research feed off unless explicitly enabled. This is safer
# for public cloud deployments where a shared cache could redistribute feed data.
COMMUNITY_FEED_ENABLED = os.getenv("LINKLENS_ENABLE_COMMUNITY_FEED", "false").strip().lower() in {"1", "true", "yes"}


def current_feed() -> dict:
    if COMMUNITY_FEED_ENABLED:
        return load_feed()
    return {"available": False, "urls": set(), "updated_at": None, "count": 0}

st.set_page_config(page_title="LinkLens | URL Risk Check", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --ink:#14221f; --muted:#687773; --paper:#f5f7f3; --line:#dfe7e1; --green:#176b55; --mint:#e4f3eb; }
.stApp { background:var(--paper); color:var(--ink); font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1120px; padding-top:1.5rem; padding-bottom:4rem; }
.brand { display:flex; align-items:center; gap:11px; font-weight:800; letter-spacing:-.04em; font-size:1.22rem; }
.brand-mark { display:grid; place-items:center; height:34px; width:34px; border-radius:11px; color:white; background:var(--green); font-size:19px; }
.eyebrow { color:var(--green); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:800; }
.hero-title { font-size:clamp(2.4rem,5vw,4.4rem); line-height:1.02; letter-spacing:-.065em; font-weight:800; margin:.8rem 0 1rem; color:var(--ink); }
.hero-copy { max-width:660px; color:var(--muted); font-size:1.08rem; line-height:1.7; }
.surface { border:1px solid var(--line); border-radius:22px; background:#fff; padding:26px; box-shadow:0 12px 42px rgba(20,34,31,.045); }
.section-title { letter-spacing:-.035em; font-size:1.35rem; font-weight:800; margin:0 0 .3rem; }
.small-copy { color:var(--muted); font-size:.93rem; line-height:1.6; }
.result { border-radius:18px; padding:22px; background:#f5f7f3; border:1px solid var(--line); }
.result-danger { background:#fff0ed; border-color:#f7cdc4; }
.result-safe { background:#eaf6ee; border-color:#cde7d4; }
.result-review { background:#fff7e7; border-color:#f2e0ae; }
.risk-title { font-size:1.6rem; font-weight:800; letter-spacing:-.04em; }
.mono { font-family:ui-monospace,'Cascadia Code',Consolas,monospace; }
div.stButton > button[kind="primary"] { background:var(--green); border:0; border-radius:12px; min-height:48px; font-weight:700; }
div.stButton > button[kind="primary"]:hover { background:#10513f; }
div[data-testid="stTextInput"] input { border-radius:12px; min-height:50px; font-family:ui-monospace,'Cascadia Code',Consolas,monospace; }
.footer-note { color:#7e8b86; font-size:.82rem; border-top:1px solid var(--line); padding-top:18px; }
</style>
""", unsafe_allow_html=True)

left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="brand"><span class="brand-mark">↗</span>LinkLens <span style="color:#8b9993;font-weight:500;font-size:.82rem;letter-spacing:0">URL Risk Check</span></div>', unsafe_allow_html=True)
with right:
    st.markdown('<div style="text-align:right;padding-top:7px;color:#687773;font-size:.83rem">LOCAL ANALYSIS · NO SITE VISITS</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="eyebrow">A smarter first look at a link</div><div class="hero-title">Pause before<br>you click.</div><div class="hero-copy">Paste a link to inspect its URL patterns. LinkLens uses a machine-learning model to estimate phishing risk without opening the website.</div>', unsafe_allow_html=True)
st.write("")

model_available = MODEL_PATH.exists()
if not model_available:
    st.warning("The model is not trained yet. Follow the setup steps in README.md, then restart the app to enable live predictions.")

left_col, right_col = st.columns([1.55, 1], gap="large")
with left_col:
    st.markdown('<div class="surface"><div class="section-title">Analyze a URL</div><div class="small-copy">We inspect the text you provide. The application does not connect to the destination.</div>', unsafe_allow_html=True)
    with st.form("url_analysis", clear_on_submit=False):
        url = st.text_input("URL to check", placeholder="https://example.com/account", label_visibility="collapsed", max_chars=4096)
        submitted = st.form_submit_button("Analyze URL  →", type="primary", use_container_width=True, disabled=not model_available)
    if submitted:
        cleaned = url.strip()
        if not cleaned:
            st.session_state["analysis_error"] = "Enter a URL to analyze."
            st.session_state.pop("last_url", None)
        elif any(ord(char) < 32 for char in cleaned):
            st.session_state["analysis_error"] = "The URL contains control characters. Remove them and try again."
            st.session_state.pop("last_url", None)
        elif not re.search(r"[A-Za-z0-9-]", cleaned):
            st.session_state["analysis_error"] = "Enter a URL containing a hostname or URL text."
            st.session_state.pop("last_url", None)
        else:
            st.session_state["last_url"] = cleaned
            st.session_state.pop("analysis_error", None)

    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])

    last_url = st.session_state.get("last_url")
    if last_url and model_available:
        result = predict_one(last_url)
        details = explain_url(last_url)
        score = float(result["score"])
        scheme = details.get("scheme_status")
        if not scheme:  # Also handles Streamlit runs with a previously imported helper module.
            scheme = "https" if last_url.lower().startswith("https://") else "http" if last_url.lower().startswith("http://") else "unknown"
        feed = current_feed()
        feed_match = lookup_url(last_url, feed)
        if feed_match is True:
            label, css, note = "Known phishing · threat-feed match", "result-danger", "This exact URL appears in the locally cached OpenPhish community feed. Do not visit it or enter information."
        elif score >= min(1.0, float(result["threshold"]) + 0.20):
            label, css, note = "High risk · likely phishing", "result-danger", "The model found patterns commonly associated with phishing URLs. Avoid entering credentials or payment details."
        elif score >= float(result["threshold"]):
            label, css, note = "Suspicious · review carefully", "result-review", "The model score is above its decision threshold. Verify the sender and destination independently."
        else:
            label, css, note = "Lower model risk · not confirmed safe", "result-safe", "The model found no strong warning patterns. A lower score does not prove the site is safe."
        st.markdown(f'<div class="result {css}"><div class="eyebrow">COMBINED ASSESSMENT</div><div class="risk-title">{label}</div><div class="small-copy">{note}</div></div>', unsafe_allow_html=True)
        st.write("")
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Model phishing score", f"{score:.1%}")
        metric_b.metric("Model decision cutoff", f"{float(result['threshold']):.0%}")
        st.progress(min(max(score, 0.0), 1.0), text="Model score for the phishing class")
        if feed_match is True:
            st.error("Threat feed: exact URL match")
        elif not feed.get("available"):
            st.info("Threat feed: unavailable · load a feed to check for known URLs")
        else:
            try:
                updated_at = datetime.fromisoformat(feed["updated_at"])
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds() / 3600
                freshness = "current snapshot" if age_hours <= 24 else "stale snapshot"
                feed_message = f"Threat feed: no exact match in {freshness} · {feed['count']:,} URLs · updated {updated_at.strftime('%Y-%m-%d %H:%M UTC')}"
                if age_hours > 24:
                    st.warning(feed_message + " · refresh before relying on this check")
                else:
                    st.success(feed_message)
            except (TypeError, ValueError):
                st.warning("Threat feed: cached list loaded, but its update time is unavailable")
        st.markdown("**URL details**")
        sig_a, sig_b, sig_c, sig_d = st.columns(4)
        sig_a.metric("URL length", f"{details['url_length']} chars")
        sig_b.metric("Domain length", f"{details['domain_length']} chars")
        sig_c.metric("Subdomains", str(details["subdomain_count"]))
        sig_d.metric("Scheme", str(scheme).upper())
        findings = []
        if details["ip_address_host"]: findings.append("The hostname is an IP address")
        if details["suspicious_terms"]: findings.append("Terms found: " + ", ".join(details["suspicious_terms"]))
        if details["special_character_count"] >= 4: findings.append("Several URL special characters appear")
        if details["url_length"] >= 100: findings.append("The URL is unusually long")
        if not findings: findings.append("No simple lexical warning signals were triggered")
        for finding in findings: st.markdown(f"- {finding}")
        st.caption("The classifier score and threat-feed match are separate signals. URL details are not a full model explanation.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="surface"><div class="section-title">How it works</div><div class="small-copy">Three quick steps, with the URL kept as text.</div>', unsafe_allow_html=True)
    for number, title, desc in [
        ("01", "Paste the link", "The app checks the input and normalizes basic formatting."),
        ("02", "Inspect URL patterns", "Character patterns are passed to the trained classifier."),
        ("03", "Review the score", "See a risk estimate, threshold, and readable URL signals."),
    ]:
        st.markdown(f'<div style="display:flex;gap:14px;padding:17px 0;border-bottom:1px solid #edf1ed"><span class="mono" style="color:#176b55;font-size:.83rem;padding-top:3px">{number}</span><div><b>{title}</b><div class="small-copy">{desc}</div></div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="eyebrow">Privacy by design</div><div class="small-copy">No browser automation, DNS lookup, or remote page fetch. The model sees only the URL text submitted here.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Known-threat feed**")
    feed = current_feed()
    if not COMMUNITY_FEED_ENABLED:
        st.caption("Community feed is disabled for this deployment. Model-only analysis remains available.")
    elif feed.get("available"):
        st.caption(f"Local cache: {feed['count']:,} feed URLs. Submitted links are matched on this machine.")
    else:
        st.caption("No feed is cached yet. The URL you analyze is never uploaded to the provider.")
    if COMMUNITY_FEED_ENABLED:
        st.markdown("Feed: [OpenPhish Community](https://openphish.com/phishing_feeds.html) · [Terms of Use](https://openphish.com/terms.html)")
        terms_reviewed = st.checkbox("I have read and agree to the provider terms for personal/research use", key="feed_terms_reviewed")
        if st.button("Refresh local phishing feed", disabled=not terms_reviewed, use_container_width=True):
            try:
                with st.spinner("Downloading feed and updating local cache…"):
                    info = refresh_feed()
                st.success(f"Updated local feed with {info['entry_count']:,} URLs.")
                st.rerun()
            except Exception as exc:
                st.error(f"Feed refresh failed; any existing cache is unchanged. {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
tab_metrics, tab_review, tab_method = st.tabs(["Model snapshot", "Review URLs", "Method & limitations"])
with tab_metrics:
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Phishing recall", f"{metrics['phishing_recall']:.1%}")
        m2.metric("Phishing precision", f"{metrics['phishing_precision']:.1%}")
        m3.metric("F1 score", f"{metrics['phishing_f1']:.1%}")
        m4.metric("Average precision", f"{metrics['average_precision']:.1%}")
        st.caption(f"{report['split']} · {report['rows']['test']:,} test URLs · threshold {metrics['threshold']:.2f}")
        if report.get("model_comparison"):
            st.markdown(f"**Selected for predictions:** {report['selected_model']} · selection is based on validation F2, with recall and average precision as tie-breakers.")
            comparison = pd.DataFrame(report["model_comparison"]).rename(columns={
                "model": "Model", "validation_threshold": "Validation cutoff",
                "validation_f2": "Validation F2", "validation_recall": "Validation recall",
                "validation_precision": "Validation precision",
                "validation_average_precision": "Validation average precision",
                "fit_seconds": "Fit seconds",
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.caption("Models are ranked on validation data only. The held-out test metrics above describe the selected model and are not used to choose it.")
    else:
        st.info("Evaluation metrics will appear here after you train the model. Metrics shown by the finished project come from the held-out test split, not example values.")
with tab_review:
    st.markdown("### Curated false-positive review")
    st.markdown("Score reviewed URLs without changing the model. Keep this challenge set separate from training so it remains an independent diagnostic.")
    if REVIEW_PATH.exists():
        review_source = pd.read_csv(REVIEW_PATH, dtype={"URL": "string", "Label": "string"})
        st.caption(f"Review file: `data/curated/review_urls.csv` · {len(review_source):,} starter case(s). Add manually verified rows with URL, Label, Source, and VerifiedAt columns.")
        if st.button("Evaluate curated URLs", type="primary", disabled=not model_available):
            try:
                summary, review_results = evaluate_review_frame(review_source)
                st.session_state["review_summary"] = summary
                st.session_state["review_results"] = review_results
            except Exception as exc:
                st.error(f"Review evaluation failed: {exc}")
        summary = st.session_state.get("review_summary")
        review_results = st.session_state.get("review_results")
        if summary and review_results is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Reviewed URLs", summary["review_urls"])
            c2.metric("False positives", summary["false_positives"])
            c3.metric("Missed phishing URLs", summary["false_negatives"])
            if summary["metrics_meaningful_for_both_classes"]:
                c4, c5, c6 = st.columns(3)
                c4.metric("Review-set precision", f"{summary['phishing_precision']:.1%}")
                c5.metric("Review-set recall", f"{summary['phishing_recall']:.1%}")
                c6.metric("Review-set F1", f"{summary['phishing_f1']:.1%}")
            else:
                st.info("Add both legitimate and phishing examples before interpreting review-set precision, recall, or F1.")
            st.dataframe(review_results.drop(columns=["Actual"], errors="ignore"), use_container_width=True, hide_index=True)
            st.caption(summary["note"])
            st.download_button("Download scored review CSV", review_results.drop(columns=["Actual"], errors="ignore").to_csv(index=False),
                               file_name="linklens_review_results.csv", mime="text/csv")
    else:
        st.warning("Review file not found. Create `data/curated/review_urls.csv` with URL and Label columns.")
with tab_method:
    st.markdown("**Model selection:** character-level TF-IDF (3–5 character n-grams) with Logistic Regression, Multinomial Naive Bayes, and SGD Logistic Classifier. The validation-set F2 score selects the winning model and its threshold; held-out test results are reserved for final evaluation.")
    st.markdown("**Evaluation:** exact duplicates are removed and conflicting URL labels are excluded. Approximate hostnames are grouped between train, validation, and test splits to reduce memorization leakage.")
    st.markdown("**Decision logic:** a local exact feed match is reported separately from the model score. A feed miss does not mean safe; stale or unavailable feed status is shown. Missing schemes are labeled unknown; HTTPS is context, not proof of safety.")
    st.markdown("**Temporal evaluation:** use a dataset with trustworthy first-seen timestamps via `--time-column`. The provided CSV has no timestamp, so it cannot produce a real time-based score. New domains and campaigns can still behave differently.")

st.write("")
st.markdown('<div class="footer-note">LinkLens is a portfolio demonstration, not a browser protection product. Treat every result as a risk signal and verify links through a trusted channel.</div>', unsafe_allow_html=True)
