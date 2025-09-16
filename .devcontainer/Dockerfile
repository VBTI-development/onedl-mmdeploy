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

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

ENV ONNXRUNTIME_DIR=/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ENV LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$LD_LIBRARY_PATH

### build PPL.cv
WORKDIR /workspace
RUN git clone https://github.com/VBTI-development/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

ENV PPLCV_DIR=/workspace/ppl.cv

## setup vscode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

ENV UV_SYSTEM_PYTHON=true
# add tensorRT to python path
ENV PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

RUN pip install "torch==2.5.1" torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    pip install onedl-mim && \
    FORCE_CUDA=1 python -m mim install onedl-mmcv && \
    pip install onnxruntime==${ONNXRUNTIME_VERSION}
