# Multi-Cloud Dockerfile for DAVID-Net Inference Microservice
# Compatible with Render, Koyeb, Hugging Face Spaces, and Docker Compose
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first for modern wheel and dependency resolution
RUN pip install --no-cache-dir --upgrade pip

# 1. Install CPU-only PyTorch using --extra-index-url so PyPI remains available for build tools
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining lightweight API requirements
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# 3. Copy application codebase (filtered by .dockerignore)
COPY . /app

# Pre-bake model weights and config during Docker build so container starts in <1 second with zero runtime download delay
RUN python -c "from huggingface_hub import hf_hub_download; import shutil, os; os.makedirs('/app/checkpoints', exist_ok=True); \
p = hf_hub_download('MIHMahmudEli/davidnet-q1-experiments', 'experiments/EXP_011_davidnet_s42/best_model/model.safetensors'); shutil.copy(p, '/app/checkpoints/david_net_best.safetensors'); \
c = hf_hub_download('MIHMahmudEli/davidnet-q1-experiments', 'experiments/EXP_011_davidnet_s42/best_model/config.json'); shutil.copy(c, '/app/checkpoints/config.json'); \
print('Checkpoint pre-baked successfully!')"

ENV PYTHONPATH=/app
ENV DAVID_CONFIG=/app/configs/david_net.yaml
# Hugging Face Hub source for automatic checkpoint pull on startup
ENV DAVID_HF_REPO=MIHMahmudEli/davidnet-q1-experiments
ENV DAVID_HF_SUBFOLDER=experiments/EXP_011_davidnet_s42/best_model

EXPOSE 7860

# Dynamically listen on cloud-provided $PORT (e.g. Render/Koyeb) or default to 7860
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
