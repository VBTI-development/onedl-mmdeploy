# Use Docker Image

This document guides how to install mmdeploy with [Docker](https://docs.docker.com/get-docker/).

## Get prebuilt docker images

Not available right now.

## Build docker images (optional)

If the prebuilt docker images do not meet your requirements,
then you can build your own image by running the following script.
The docker file is `docker/Release/Dockerfile`and its building argument is `MMDEPLOY_VERSION`,
which can be a [tag](https://github.com/vbti-development/onedl-mmdeploy/tags) or a branch from [mmdeploy](https://github.com/vbti-development/onedl-mmdeploy).

```shell
export MMDEPLOY_VERSION=main
export TAG=mmdeploy-${MMDEPLOY_VERSION}
docker build docker/Release/ -t ${TAG} --build-arg MMDEPLOY_VERSION=${MMDEPLOY_VERSION}
```

## Run docker container

After pulling or building the docker image, you can use `docker run` to launch the docker service:

```shell
export TAG=onedl/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy
docker run --gpus=all -it --rm $TAG
```

## FAQs

1. CUDA error: the provided PTX was compiled with an unsupported toolchain:

   As described [here](https://forums.developer.nvidia.com/t/cuda-error-the-provided-ptx-was-compiled-with-an-unsupported-toolchain/185754), update the GPU driver to the latest one for your GPU.

2. docker: Error response from daemon: could not select device driver "" with capabilities: [gpu].

   ```shell
   # Add the package repositories
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
