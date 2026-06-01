FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e .

# Download Triad weights at build time (optional — can also mount at runtime)
# RUN mkdir -p /app/weights

ENV PYTHONUNBUFFERED=1
ENV TRIAGEMRI_ROOT=/app

EXPOSE 7860 8000

CMD ["python", "-c", "from triagemri.config import load_config; print('Triage-MRI ready')"]
