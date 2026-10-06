# LinkLens — Phishing URL Risk Checker

A local portfolio project that estimates whether a URL resembles phishing based on its text. It includes a Streamlit interface, a reproducible scikit-learn training pipeline, domain-grouped evaluation, and readable URL signals.

> This is a learning/demo tool. It does not open URLs or establish whether a site is safe. Never use its score as the only security control.

## What the user sees

1. A user pastes a link into the URL field and selects **Analyze URL**.
2. LinkLens validates the text and does not connect to the destination.
3. The same normalization used at training time is applied; character n-grams go to the trained classifier.
4. The result shows a risk label, model score, decision cutoff, explicit scheme state (`HTTP`, `HTTPS`, or `unknown`), and basic URL signals.
5. If the user has reviewed and accepted the provider terms, they can manually refresh a local phishing-feed cache. Submitted URLs are checked against that cache locally and are never uploaded to the feed provider.
6. The Review URLs tab scores a curated CSV without fitting or changing the model and highlights false positives and false negatives.
7. The Model snapshot tab shows held-out phishing precision, recall, F1, and average precision after training.

Before a model has been trained, the app displays a setup notice and disables live predictions. It does not show mock scores.

## Architecture

```text
CSV (URL, Label)
   └─ audit, drop exact duplicates, exclude conflicting labels
       └─ approximate-host grouped train/validation/test split
   └─ normalize URL text + show protocol state separately
       └─ character TF-IDF (3–5 grams)
           ├─ Logistic Regression / Multinomial Naive Bayes / SGD Logistic Classifier
           ├─ validation F2 selects winner and its threshold
           ├─ held-out metrics → reports/evaluation.json
           └─ saved winning pipeline → models/url_classifier.joblib

Browser → Streamlit validation → local threat-feed exact match + saved pipeline
        → separate feed verdict and model score + simple URL signals
```

## Project layout

```text
app.py
src/phishing_detector/features.py   # Shared normalization and readable URL signals
src/phishing_detector/model.py      # Model loading and inference
src/phishing_detector/train.py      # Cleaning, grouped split, fit, evaluation, save
src/phishing_detector/review.py     # Curated review-set scoring and error reporting
data/raw/                            # Local source dataset (git-ignored)
data/curated/review_urls.csv         # Hand-reviewed diagnostic cases
models/                              # Trained model artifact
reports/                             # Evaluation report
Dockerfile + compose.yaml            # Reproducible container build and run
```

## Run locally on Windows

Use Python 3.12 or newer. In PowerShell, from the project directory:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Place the provided `phishing_site_urls.csv` at `data/raw/phishing_site_urls.csv`, or pass its current path directly to the training command. Train and save the model and evaluation report:

```powershell
python -m phishing_detector.train --data "C:\Users\Garry\OneDrive\Desktop\phishing_site_urls.csv"
```

The supplied CSV has no trustworthy first-seen timestamp, so the command above uses the existing approximate-host grouped split. To run a **chronological** evaluation with a newer dataset that has a trustworthy timestamp column, use:

```powershell
python -m phishing_detector.train --data data\new_labeled_urls.csv --time-column first_seen_at --model-out models\temporal_model.joblib --report-out reports\temporal_evaluation.json
```

The time-aware path assigns the oldest 70% of observations to training, the next 15% to validation, and newest 15% to test. It reports date ranges and hostname overlap. Repeated URLs are deduplicated to their earliest timestamp. Do not substitute the date you downloaded the old CSV for the URLs' first-seen dates; that would create a misleading temporal test.

Launch the interactive app:

```powershell
streamlit run app.py
```

Streamlit prints a local address (typically `http://localhost:8501`) to open in your browser. The app does not fetch the submitted URL. When hosted remotely, URL text is sent to your application server for analysis, so avoid submitting private links or URLs containing access tokens.

## Review false positives and false negatives

`data/curated/review_urls.csv` starts with the official Gemini web app URL as a manually verified legitimate challenge case. It is deliberately separate from the training dataset. Add reviewed examples with these columns:

```csv
URL,Label,Source,VerifiedAt
https://example.com/,good,Official site reviewed,2026-10-07
```

Use `good`/`legitimate` for benign URLs and `bad`/`phishing` for phishing URLs. Record how and when ground truth was verified; don't label examples using LinkLens's own prediction. Run:

```powershell
python -m phishing_detector.review
```

The command writes per-URL scores and outcomes to `reports/review_results.csv` and a summary to `reports/review_summary.json`. The **Review URLs** tab runs the same evaluation and allows downloading scored results. A tiny or one-class challenge set can identify a particular error but cannot estimate general precision/recall; the UI withholds those metrics until both classes are present. Keep this review set separate from training when comparing models.

## Build and run the deployable container

The image includes the trained model and evaluation report but excludes the raw training dataset. Train first if `models/url_classifier.joblib` or `reports/evaluation.json` is missing. With Docker Desktop installed:

```powershell
docker compose up --build -d
docker compose logs -f linklens
```

Open `http://localhost:8501`; stop the container with `docker compose down`. The container runs as a non-root user and has a health check. Compose disables the OpenPhish Community Feed by default. Its terms limit use to personal/research purposes and prohibit redistributing the feed, so don't enable it for public hosting unless your use is permitted. For a public deployment, use a threat feed whose license explicitly permits serving your application, or keep feed checks disabled.

## Model comparison and evaluation choices

- **Candidates:** character TF-IDF (3–5 character n-grams) with Logistic Regression, Multinomial Naive Bayes, and SGD Logistic Classifier. URL spelling, delimiters, and obfuscated strings are meaningful, making character features a useful shared representation.
- **Winner selection:** each candidate gets a phishing-recall-weighted F2 threshold selected on the validation partition. The candidate with highest validation F2 wins; ties use validation recall, then average precision. The winner is saved and used for all app predictions. The held-out test set is not used for model selection.
- **Positive class:** `bad` = phishing (1); `good` = legitimate (0).
- **Cleaning:** exact URL duplicates are removed; URLs carrying both labels are excluded from the experiment.
- **Split:** approximate hostnames are grouped so a host is kept in one partition. The host extraction is necessarily approximate because this source often omits schemes and public-suffix data.
- **Threshold:** selected on validation data to maximize F2, weighting phishing recall more heavily. This is a portfolio choice, not a universal operational policy.
- **Protocol:** the training normalizer strips HTTP(S) scheme to match the original training representation. The UI reports the explicit input scheme or `unknown`; the scheme is not used as a learned signal until enough labeled protocol-aware data is collected and evaluated.
- **Threat feed:** manually refreshes the OpenPhish Community Feed to an ignored local cache. URL checks are exact, scheme-insensitive full-URL comparisons run locally. A match is shown separately from the ML score. The user must read and accept the [OpenPhish Terms of Use](https://openphish.com/terms.html) before using that feed; the feed is limited to personal/research use and must not be redistributed. If you do not accept those terms, use a feed whose license permits your use or leave the feature unavailable.
- **Curated review set:** scores manually verified examples and reports per-URL false-positive/false-negative outcomes without modifying the fitted model. The starter Gemini example is a diagnostic case, not a representative performance estimate.
- **Feed freshness:** the UI shows the cache timestamp and warns when it is older than 24 hours. The community feed's published cadence is limited; feed absence is never presented as proof of safety.
- **Test report:** includes each candidate's validation metrics, the selected winner, and the winner's held-out phishing precision/recall/F1/F2, average precision, ROC-AUC, balanced accuracy, accuracy, confusion matrix, training time, and test inference throughput.

Accuracy alone is not a good summary. Precision says how many flagged URLs were phishing; recall says how many phishing URLs were caught. Average precision summarizes the precision-recall tradeoff across thresholds. The confusion matrix makes false alarms and missed phishing cases visible.

## Expected output

The app presents an input field and **Analyze URL** button. A cached feed match is shown as **Known phishing**; otherwise the model returns **High risk**, **Suspicious**, or **Lower model risk**. The model score and cutoff remain visible as separate values. Feed misses and lower model scores are not described as “safe.” The explicit URL scheme is shown as `HTTP`, `HTTPS`, or `UNKNOWN`; this remains context rather than a learned safety signal.

The training command prints and writes a report similar in structure to this (the values below are placeholders, not results):

```json
{
  "model": "Character TF-IDF (3-5 grams) + Logistic Regression",
  "split": "GroupShuffleSplit by approximate hostname",
  "metrics": {
    "phishing_precision": "measured value",
    "phishing_recall": "measured value",
    "phishing_f1": "measured value",
    "average_precision": "measured value"
  }
}
```

Do not quote metric values on a resume until you have run the project and can explain the split and evaluation.

## Dataset notes and limitations

The supplied file has columns `URL` and `Label`; in the initial inspection it contained 549,346 rows, labels `good`/`bad`, exact duplicates, one conflicting URL, and very few explicit HTTP(S) schemes. Training cleans these issues and writes actual counts to the report. HTTPS is shown only when explicit, and omitted schemes are `unknown`. The classifier uses URL text only; it does not query DNS, inspect page content, or identify newly registered domains. A feed match adds known-threat evidence, but no-match does not imply safety. The old CSV has no observation timestamp, so temporal performance cannot be measured from it.

`joblib` model files should only be loaded from a trusted source. Retrain using the supplied data rather than loading an unknown model artifact.
