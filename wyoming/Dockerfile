FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1 HOLOS_URL=http://holos-hybrid:8002 WYOMING_PORT=10200

EXPOSE 10200

CMD ["python3.11", "app.py"]
