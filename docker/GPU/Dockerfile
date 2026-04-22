# syntax=docker/dockerfile:1.7-labs
# This docker installs:
#   - TensorRT (via official apt repository)
#   - ONNX Runtime
#   - onedl-mim, onedl-mmcv, onedl-mmengine
#   - ppl.cv (built from source, CUDA)
#   - onedl-mmdeploy [cuda;cpu, ort;trt, mmpretrain;mmseg;mmdet;mmrotate]

ARG UV_VERSION=0.11.7
ARG CUDA_IMAGE_VERSION=12.9.1

# ─── uv binary ───────────────────────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ─── Builder stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_IMAGE_VERSION}-cudnn-devel-ubuntu24.04 AS builder

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.9
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

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update && \
    apt-get install -y \
        build-essential \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        cmake \
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
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv ${VIRTUAL_ENV} --python python${PYTHON_VERSION}

# Set up CUDA network repository and install TensorRT development packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    find /etc/apt/sources.list.d/ \( -name '*cuda*' -o -name '*nvidia*' \) -delete \
    && wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub \
        | gpg --dearmor > /usr/share/keyrings/cuda-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" \
        > /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        tensorrt-dev=${TENSORRT_VERSION}* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV TENSORRT_DIR=/usr

# Install PyTorch into the venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==${TORCH_VERSION} torchvision \
        --extra-index-url https://download.pytorch.org/whl/cu$(echo "$CUDA_VERSION" | sed 's/\.//g')

# Download and extract the OnnxRuntime C++ SDK to a fixed path
WORKDIR /workspace/
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && rm  onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && mv onnxruntime-linux-x64-${ONNXRUNTIME_VERSION} onnxruntime

ENV ONNXRUNTIME_DIR=/workspace/onnxruntime

# Install OnnxRuntime GPU Python wheel and TensorRT Python bindings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION} tensorrt==${TENSORRT_VERSION}

# Install onedl-mmcv and onedl-mmengine via mim
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install onedl-mim && \
    uv run --no-project mim install --no-cache \
        "onedl-mmcv${MMCV_VERSION}" \
        "onedl-mmengine${MMENGINE_VERSION}" \
        --only-binary=onedl-mmcv

# Build ppl.cv from source (CUDA backend)
WORKDIR /workspace/
RUN git clone -b v${PPLCV_VERSION} --single-branch \
        https://github.com/VBTI-development/ppl.cv.git && \
    cd ppl.cv && \
    ./build.sh cuda

ENV LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:${TENSORRT_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Build and install onedl-mmdeploy (C++ SDK + Python bindings)
COPY . /workspace/onedl-mmdeploy
RUN cd /workspace/onedl-mmdeploy && \
    mkdir -p build && cd build && \
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=OFF \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_INSTALL_PREFIX=/workspace/onedl-mmdeploy/install \
        -Dpplcv_DIR=/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES="mmpretrain;mmseg;mmdet;mmrotate" && \
    make -j$(nproc) && make install && \
    echo "Built MMDeploy for GPU devices successfully!" && \
    cd /workspace/onedl-mmdeploy && \
    uv pip install .

# ─── Runtime stage ───────────────────────────────────────────────────────────
ARG CUDA_IMAGE_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_IMAGE_VERSION}-cudnn-runtime-ubuntu24.04 AS runtime

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/venv \
    LANG=C.UTF-8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libopencv-core-dev \
        libopencv-imgproc-dev \
        libopencv-imgcodecs-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy TensorRT shared libraries from builder (avoids re-adding apt repo at runtime)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnvinfer*.so*      /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnvonnxparsers*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libnvparsers*.so*    /usr/lib/x86_64-linux-gnu/

# Copy Python venv (all installed Python packages)
COPY --from=builder /venv /venv

# Copy OnnxRuntime C++ SDK runtime libraries
COPY --from=builder /workspace/onnxruntime/lib /workspace/onnxruntime/lib

# Copy mmdeploy SDK install artifacts
COPY --from=builder /workspace/onedl-mmdeploy/install /workspace/onedl-mmdeploy/install

# Copy ppl.cv runtime libraries
COPY --from=builder /workspace/ppl.cv/cuda-build/install/lib /workspace/ppl.cv/lib

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime
ENV TENSORRT_DIR=/usr
ENV LD_LIBRARY_PATH="${ONNXRUNTIME_DIR}/lib:/workspace/onedl-mmdeploy/install/lib:/workspace/ppl.cv/lib:${TENSORRT_DIR}/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

LABEL org.opencontainers.image.source=https://github.com/vbti-development/onedl-mmdeploy
LABEL org.opencontainers.image.description="OneDL MMDeploy Docker Image with GPU support"
LABEL org.opencontainers.image.licenses=Apache-2.0
LABEL org.opencontainers.image.vendor="VBTI Products BV"
