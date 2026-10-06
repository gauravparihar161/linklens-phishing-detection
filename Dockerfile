FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --shell /usr/sbin/nologin linklens \
    && mkdir -p /app/data/cache /app/models /app/reports \
    && chown -R linklens:linklens /app

COPY --chown=linklens:linklens app.py ./app.py
COPY --chown=linklens:linklens src/ ./src/
COPY --chown=linklens:linklens models/url_classifier.joblib ./models/url_classifier.joblib
COPY --chown=linklens:linklens reports/evaluation.json ./reports/evaluation.json
COPY --chown=linklens:linklens data/curated/review_urls.csv ./data/curated/review_urls.csv

USER linklens
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
