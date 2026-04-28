# syntax=docker/dockerfile:1.7-labs

ARG UV_VERSION=0.11.7
ARG PYTHON_VERSION=3.12
ARG CUDA_IMAGE_VERSION=12.9.1
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.9
ARG CUDA_IMAGE_VERSION=12.9.1
ARG TORCH_VERSION=2.10.0
ARG TENSORRT_VERSION=10.16.1
ARG ONNXRUNTIME_VERSION=1.21.0
ARG PPLCV_VERSION=0.8.0
ARG MMCV_VERSION=">=2.3.0"
ARG MMENGINE_VERSION=">=0.10.8"

FROM nvidia/cuda:${CUDA_IMAGE_VERSION}-cudnn-devel-ubuntu24.04 AS builder

# Redeclare ARGs after FROM so they are available in this stage
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.9
ARG CUDA_IMAGE_VERSION=12.9.0
ARG TORCH_VERSION=2.10.0
ARG TENSORRT_VERSION=10.16.1.11
ARG ONNXRUNTIME_VERSION=1.21.0
ARG PPLCV_VERSION=0.8.0
ARG MMCV_VERSION=">=2.3.0"
ARG MMENGINE_VERSION=">=0.10.8"

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA="1" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_SITE_PACKAGES=false \
    VIRTUAL_ENV=/venv \
    LANG=C.UTF-8

COPY --from=uv /uv /uvx /bin/

# C++ dev tools + Python (provided by apt on the CUDA base image)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update && \
    apt-get install -y \
        build-essential \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        cmake \
        cppcheck \
        clang \
        lldb \
        llvm \
        gdb \
        curl \
        ffmpeg \
        git \
        libc6 \
        libgl1 \
        libglib2.0-0 \
        libopencv-dev \
        libsm6 \
        libssl-dev \
        libxext6 \
        libxrender-dev \
        ninja-build \
        wget

# Create venv with the exact Python version
RUN uv venv ${VIRTUAL_ENV} --python python${PYTHON_VERSION}

# Set up CUDA network repository (required for TensorRT)
# Remove any pre-existing CUDA repo entries from the base image to avoid Signed-By conflicts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    find /etc/apt/sources.list.d/ -name '*cuda*' -o -name '*nvidia*' | xargs rm -f \
    && wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor > /usr/share/keyrings/cuda-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list

# Install TensorRT development packages (official Debian method)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        tensorrt-dev=${TENSORRT_VERSION}* \
        python3-libnvinfer=${TENSORRT_VERSION}* \
        python3-libnvinfer-dev=${TENSORRT_VERSION}*

# Set TensorRT environment variables
ENV TENSORRT_DIR=/usr

# PyTorch
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==${TORCH_VERSION} torchvision \
        --extra-index-url https://download.pytorch.org/whl/cu$(echo "$CUDA_VERSION" | sed 's/\.//g')

# OnnxRuntime C++ SDK (for cmake -DONNXRUNTIME_DIR)
WORKDIR /workspace/
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && rm  onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

# OnnxRuntime GPU Python wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} tensorrt==${TENSORRT_VERSION}

# onedl-mmcv + onedl-mmengine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install onedl-mim && \
    uv run --no-project mim install --no-cache \
        "onedl-mmcv${MMCV_VERSION}" \
        "onedl-mmengine${MMENGINE_VERSION}" \
        --only-binary=onedl-mmcv

# Build ppl.cv from source (CUDA)
WORKDIR /workspace/
RUN git clone -b v${PPLCV_VERSION} --single-branch \
        https://github.com/VBTI-development/ppl.cv.git && \
    cd ppl.cv && \
    ./build.sh cuda
ENV PPLCV_DIR=/workspace/ppl.cv

ENV TENSORRT_DIR=/usr
ENV LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:${TENSORRT_DIR}/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV UV_SYSTEM_PYTHON=false
