FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
    libgl1 libxcb1 libx11-6 libxext6 libxrender1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt 

COPY train.py /app/train.py

# Defaults (override via docker run -e ...)
ENV MLFLOW_TRACKING_URI="http://192.168.18.109:5000"
ENV MLFLOW_EXPERIMENT_NAME=""
ENV MLFLOW_RUN_NAME=""

# MinIO/S3-compatible artifacts (needed if your MLflow server stores artifacts on MinIO)
ENV MLFLOW_S3_ENDPOINT_URL="http://192.168.18.109:9000"
ENV AWS_ACCESS_KEY_ID="minio_user"
ENV AWS_SECRET_ACCESS_KEY="minio_password"
ENV AWS_DEFAULT_REGION="us-east-1"

# Training inputs
ENV DATA_YAML=""
ENV MODEL=""
ENV TASK=""
ENV EPOCHS=""
ENV IMGSZ="1024"
ENV BATCH="auto"
ENV DEVICE="0"
ENV WORKERS=""
ENV PROJECT_DIR="/outputs/ultralytics"
ENV EXPORT_DIR="/outputs/exported"
ENV SEED="0"

# Make sure these exist (volumes will mount over them)
RUN mkdir -p /outputs/ultralytics /outputs/exported

ENTRYPOINT ["python", "/app/train.py"]