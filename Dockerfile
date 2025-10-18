FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    LLVM_CONFIG=/usr/bin/llvm-config-14

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsndfile1 \
    build-essential \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    llvm-14 \
    llvm-14-dev \
    libllvm14 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv (fast Python package installer) to a standard location
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./

# clone lib files  
RUN git clone https://github.com/index-tts/index-tts.git lib/index-tts

# install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --extra deepspeed

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# RUN uv tool install "huggingface-hub[cli,hf_xet]"
# RUN hf download IndexTeam/IndexTTS-2 --local-dir=./checkpoints

# Expose ports
EXPOSE 6969

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6969/ || exit 1

# Default command (can be overridden)
CMD ["uv", "run", "src/webui.py", "--port", "6969", "--host", "0.0.0.0"]
