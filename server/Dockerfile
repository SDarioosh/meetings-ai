# Meetings.ai server for NVIDIA GPUs (CUDA 12 + cuDNN 9, as required by faster-whisper / CTranslate2 4.x).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml ./
COPY meetings_server ./meetings_server
COPY templates ./templates
# INSTALL_EXTRAS="[diarize]" adds pyannote (+ PyTorch, ~3 GB) for speaker detection.
ARG INSTALL_EXTRAS="[diarize]"
RUN pip install --upgrade pip && pip install ".${INSTALL_EXTRAS}"

# Templates are read from here so you can mount your own over it.
ENV MEETINGS_TEMPLATES_DIR=/app/templates
ENTRYPOINT ["meetings"]
CMD ["watch", "/data/inbox", "--output-dir", "/data/meetings"]
