FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-opencv \
    python3-scipy \
    python3-matplotlib \
    git \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --no-deps --timeout 30000 && \
    pip3 install --no-cache-dir --pre \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    triton

COPY requirements/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --timeout 3000 -r /tmp/requirements.txt

CMD ["bash"]