FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends         ffmpeg git build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ORT_VARIANT: cpu (default) | openvino | cuda - підміна onnxruntime-білду.
ARG ORT_VARIANT=cpu
RUN pip install --no-cache-dir -r requirements.txt && if [ "$ORT_VARIANT" = "openvino" ]; then pip uninstall -y onnxruntime && pip install --no-cache-dir onnxruntime-openvino==1.24.1; fi && if [ "$ORT_VARIANT" = "cuda" ]; then pip uninstall -y onnxruntime && pip install --no-cache-dir onnxruntime-gpu==1.31.0; fi

COPY app/ ./app/

ENV HF_HOME=/app/models PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
