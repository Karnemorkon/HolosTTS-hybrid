FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends         ffmpeg git build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV HF_HOME=/app/models PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
