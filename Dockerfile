# Spotify Track Performance Intelligence — production image
FROM python:3.12-slim

WORKDIR /app

# System deps needed by lightgbm/xgboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Directories the pipeline writes to must exist and be writable
RUN mkdir -p data/interim data/processed models reports/figures reports/logs mlruns

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Default: serve the dashboard. Override the command to run training/pipeline
# steps instead, e.g.:
#   docker run <image> python main.py
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
