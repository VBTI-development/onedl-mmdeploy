# syntax=docker/dockerfile:1.7-labs
# use buildkit
FROM nvcr.io/nvidia/tensorrt:24.04-py3 AS runtime

# This docker installs:
    # - ONNX
    # - TensorRT
    # - onedl-mim
    # - onedl-mmcv and onedl-mmengine
    # - ppl.cv
    # - onedl-mmdeploy [cuda;cpu, ort;trt, mmpretrain;mmseg;mmdet;mmrotate]

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.4
ARG TORCH_VERSION=2.5.1
ARG ONNXRUNTIME_VERSION=1.19.2
ARG PPLCV_VERSION=0.8.0
ENV FORCE_CUDA="1"

ARG MMCV_VERSION=">=2.3.0"
ARG MMENGINE_VERSION=">=0.10.8"
ARG MMDEPLOY_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# set the timezone to Europe Amsterdam
# and install libs
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update && \
    apt-get install -yq tzdata && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y \
        curl \
        ffmpeg \
        git \
        libc6 \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libopencv-dev \
        libsm6 \
        libspdlog-dev \
        libssl-dev \
        libxext6 \
        libxrender-dev \
        ninja-build \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This directory is not part of official OpenCV releases and is likely leftover
# from an old install or a third-party package.
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/typing

# install necessary python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja onedl-mim

# install torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION} torchvision --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION/.//}

### get onnxruntime
WORKDIR /workspace/
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m mim install onnxruntime-gpu==${ONNXRUNTIME_VERSION}

### install onedl-mm dependencies
RUN python3 -m mim install --no-cache "onedl-mmcv${MMCV_VERSION}" "onedl-mmengine${MMENGINE_VERSION}"  --only-binary=onedl-mmcv

### install pplcv sdk
WORKDIR /workspace/
RUN git clone -b v${PPLCV_VERSION} --single-branch https://github.com/VBTI-development/ppl.cv.git &&\
    cd ppl.cv &&\
    ./build.sh cuda

# ### install mmdeploy
ENV TENSORRT_DIR=/workspace/tensorrt
WORKDIR /workspace/

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

RUN if [ -n "${MMDEPLOY_VERSION}" ] ; then \
        echo "Checking out MMDeploy version: ${MMDEPLOY_VERSION}" && \
        MMDEPLOY_VERSION_=${MMDEPLOY_VERSION} ; \
    else \
        echo "Using main branch (no version specified)" && \
        MMDEPLOY_VERSION_=main ; \
    fi && \
    git clone --single-branch -b ${MMDEPLOY_VERSION_} https://github.com/vbti-development/onedl-mmdeploy &&\
    cd onedl-mmdeploy &&\
    git submodule update --init --recursive &&\
    mkdir -p build && cd build &&\
    cmake .. \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -Dpplcv_DIR=/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
    -DMMDEPLOY_CODEBASES="mmpretrain;mmseg;mmdet;mmrotate" &&\
    make -j$(nproc) && make install &&\
    export SPDLOG_LEVEL=debug &&\
    if [ -z ${MMDEPLOY_VERSION} ] ; then echo "Built MMDeploy for GPU devices successfully!" ; else echo "Built MMDeploy version v${MMDEPLOY_VERSION} for GPU devices successfully!" ; fi &&\
    cd ../ &&\
    python3 -m mim install --config-settings editable_mode=compat -e .

ENV LD_LIBRARY_PATH="$ONNXRUNTIME_DIR/lib:$BACKUP_LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/workspace/onedl-mmdeploy/build/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/:$LD_LIBRARY_PATH"

LABEL org.opencontainers.image.source=https://github.com/vbti-development/onedl-mmdeploy
LABEL org.opencontainers.image.description="OneDL MMDeploy Docker Image with GPU support"
LABEL org.opencontainers.image.licenses=Apache-2.0
LABEL org.opencontainers.image.vendor="VBTI Products BV"
