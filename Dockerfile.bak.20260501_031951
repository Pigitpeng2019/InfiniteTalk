# =============================================================================
# Dockerfile for InfiniteTalk — Audio-driven Video Generation
# Base: nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# GPU requirement: ≥16GB VRAM (480p), ≥24GB VRAM (720p)
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

LABEL maintainer="InfiniteTalk Team"
LABEL description="InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    wget \
    curl \
    git \
    ninja-build \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

# ── Install PyTorch 2.4.1 + CUDA 12.1 ────────────────────────────────────────
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Install xformers 0.0.28 ──────────────────────────────────────────────────
RUN pip install -U xformers==0.0.28 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Install flash-attn (pre-compiled) ─────────────────────────────────────────
RUN pip install misaki[en] ninja psutil packaging wheel
# flash-attn may take a long time to build. Try prebuilt wheel first.
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation || \
    (echo "Prebuilt flash-attn not found, building from source..." && \
     MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation)

# ── Install xfuser ───────────────────────────────────────────────────────────
RUN pip install xfuser>=0.4.1

# ── Install optimum-quanto ───────────────────────────────────────────────────
RUN pip install optimum-quanto==0.2.6

# Install librosa via conda-forge alternative (pip version suffices)
RUN pip install librosa

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

LABEL maintainer="InfiniteTalk Team"
LABEL description="InfiniteTalk Runtime - Audio-driven Video Generation"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:/usr/local/bin:/usr/bin
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=8418
ENV GRADIO_DEBUG=false

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

# ── Copy Python packages from builder ────────────────────────────────────────
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# ── Pre-install remaining requirements ───────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install project dependencies (skip torch/torchvision/xformers/flash-attn since copied above)
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ─────────────────────────────────────────────
RUN mkdir -p /app/weights /app/outputs /app/.cache

# ── Environment for HuggingFace ──────────────────────────────────────────────
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TORCH_HOME=/app/.cache/torch
ENV no_proxy=localhost,127.0.0.1,::1

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8418').status) if urllib.request.urlopen('http://localhost:8418').status == 200 else exit(1)" || exit 1

# ── Expose ports ─────────────────────────────────────────────────────────────
EXPOSE 8418 8419

# ── Default command: Gradio UI ───────────────────────────────────────────────
CMD ["python", "app.py", \
     "--ckpt_dir", "/app/weights/Wan2.1-I2V-14B-480P", \
     "--wav2vec_dir", "/app/weights/chinese-wav2vec2-base", \
     "--infinitetalk_dir", "/app/weights/InfiniteTalk/single/infinitetalk.safetensors", \
     "--num_persistent_param_in_dit", "0", \
     "--motion_frame", "9"]
