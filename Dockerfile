# SweepScreen prediction service (CPU).
#   docker build -t sweepscreen .
#   docker run -p 8000:8000 -v $PWD/models:/app/models -v $PWD/logs:/app/logs sweepscreen
# models/ must contain registry.json, the registered bundle, and pretrained/dinov2-small/.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 \
    SWEEPSCREEN_MODELS=/app/models SWEEPSCREEN_ENCODER=/app/models/pretrained/dinov2-small \
    SWEEPSCREEN_LOG=/app/logs/inference.jsonl
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch wheel first, then the remaining pins
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^torch==' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

COPY sweepscreen ./sweepscreen
COPY service ./service

RUN useradd --create-home app && mkdir -p /app/logs && chown app /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
