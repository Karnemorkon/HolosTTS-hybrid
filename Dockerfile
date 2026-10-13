ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# python3.11: в офіційних python-образах уже є; на NVIDIA/cuda-базах (ubuntu) - ставимо з deadsnakes.
RUN if ! command -v python3.11 >/dev/null 2>&1; then DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.11 python3.11-venv && python3.11 -m ensurepip --upgrade; fi

RUN apt-get update && apt-get install -y --no-install-recommends         ffmpeg git build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ORT_VARIANT: cpu (default) | openvino | cuda - підміна onnxruntime-білду.
ARG ORT_VARIANT=cpu
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt && if [ "$ORT_VARIANT" = "openvino" ]; then python3.11 -m pip uninstall -y onnxruntime && python3.11 -m pip install --no-cache-dir onnxruntime-openvino==1.24.1; fi && if [ "$ORT_VARIANT" = "cuda" ]; then python3.11 -m pip uninstall -y onnxruntime && python3.11 -m pip install --no-cache-dir onnxruntime-gpu==1.31.0; fi

COPY app/ ./app/

ENV HF_HOME=/app/models PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["python3.11", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
