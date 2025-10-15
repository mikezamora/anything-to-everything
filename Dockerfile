# Multi-stage build for EPUB to Audiobook Converter
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    git \
    wget \
    curl \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer) to a standard location
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY *.py ./
COPY checkpoints/ ./checkpoints/

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Create necessary directories
RUN mkdir -p \
    work/segments \
    work/ollama \
    work/ollama_characters \
    work/ollama_segmentation \
    work/character_segmentation \
    inputs \
    outputs \
    uploads/epub \
    uploads/voice \
    uploads/emotion \
    jobs/pending \
    jobs/running \
    jobs/completed \
    jobs/failed

# Expose ports
EXPOSE 6969

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6969/ || exit 1

# Default command (can be overridden)
CMD ["uv", "run", "-m", "webui", "--port", "6969", "--host", "0.0.0.0"]
