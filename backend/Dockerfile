# Build context: project root (cd /Skin-Doc && docker build -f backend/Dockerfile .)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LESION_MODEL_WEIGHTS=/app/backend/ml/weights/best_model.pth \
    LESION_IMG_SIZE=384

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir huggingface-hub>=0.22.0

# App code
COPY . /app

# Create weight directory
RUN mkdir -p /app/backend/ml/weights

# Non-root user
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup: auto-download model if missing, then serve
CMD ["sh", "-c", \
    "if [ ! -f /app/backend/ml/weights/best_model.pth ]; then \
        echo 'Downloading model weights from HuggingFace...' && \
        python3 -c \"from huggingface_hub import hf_hub_download; \
            hf_hub_download('arjg/skin-doc-model', 'finetuned_isic2019.pth', \
            local_dir='/app/backend/ml/weights', local_dir_use_symlinks=False); \
            import shutil; shutil.copy('/app/backend/ml/weights/finetuned_isic2019.pth', \
            '/app/backend/ml/weights/best_model.pth'); \
            print('Model ready.')\"; \
    fi && \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000"]