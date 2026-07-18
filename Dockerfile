FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r backend/requirements.txt

RUN ollama serve > /tmp/ollama.log 2>&1 & \
    for i in $(seq 1 60); do \
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi; \
    sleep 2; \
    done; \
    ollama pull qwen2.5:1.5b \
    && python ingest.py

EXPOSE 7860

CMD ["sh", "-lc", "ollama serve > /tmp/ollama.log 2>&1 & while ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do sleep 2; done; exec uvicorn backend.main:app --host 0.0.0.0 --port 7860"]
