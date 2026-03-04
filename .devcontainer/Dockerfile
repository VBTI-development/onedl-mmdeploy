# FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-22.04
FROM nvcr.io/nvidia/tensorrt:24.04-py3

# C++ dev stuff
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    # [Optional] Additional packages required for using vcpkg package manager in manifest mode.
    # [Optional] Please uncomment if the below line in needed.
    # && apt-get -y install autoconf automake libtool m4 autoconf-archive \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"
ARG ONNXRUNTIME_VERSION=1.19.2
ARG PPLCV_VERSION=0.8.0

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=12.4
ARG TORCH_VERSION=2.5.1
ARG ONNXRUNTIME_VERSION=1.19.2
ARG PPLCV_VERSION=0.8.0
ENV FORCE_CUDA="1"

ARG MMCV_VERSION=">=2.3.0"
ARG MMENGINE_VERSION=">=0.10.8"

# and install libs
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache \
    --mount=type=cache,target=/var/lib/apt,sharing=locked,id=apt-lib \
    apt-get update && \
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

### get onnxruntime
WORKDIR /workspace/
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ENV LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$BACKUP_LD_LIBRARY_PATH:$TENSORRT_DIR/lib:$LD_LIBRARY_PATH
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python3 -m mim install onnxruntime-gpu==${ONNXRUNTIME_VERSION}


### build PPL.cv
WORKDIR /workspace
RUN git clone https://github.com/VBTI-development/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

ENV PPLCV_DIR=/workspace/ppl.cv


# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==${TORCH_VERSION} torchvision --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION/./} && \
    pip install onedl-mim && \
    FORCE_CUDA=1 python -m mim install "onedl-mmcv${MMCV_VERSION}" "onedl-mmengine${MMENGINE_VERSION}" --only-binary=onedl-mmcv && \
    python -m mim install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
    # pip install onnxruntime==${ONNXRUNTIME_VERSION}

# setup vscode user
# ARG USERNAME=vscode
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# # Create the user
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     #
#     # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME
# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENV UV_SYSTEM_PYTHON=true
# add tensorRT to python path
# ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"
