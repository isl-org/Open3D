# Example build command:
# docker build \
#   --build-arg UBUNTU_VERSION=focal \
#   -t open3d-gpu-ci-focal:latest \
#   -f ./util/docker/open3d-gpu/Dockerfile \
#    .
# Example run command:
#    docker run --rm --gpus all \
#        --env NPROC=4 \
#        --env SHARED=OFF \
#        --env BUILD_CUDA_MODULE=ON \
#        --env BUILD_RPC_INTERFACE=ON \
#        --env BUILD_TENSORFLOW_OPS=ON \
#        --env BUILD_PYTORCH_OPS=ON \
#        --env OPEN3D_ML_ROOT=/root/Open3D/Open3D-ML \
#        open3d-gpu-ci-focal:latest

ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION} AS open3d-gpu
# Persist ARG for the rest of the build
ARG UBUNTU_VERSION
LABEL name="open3d-dev/open3d-gpu-ci-${UBUNTU_VERSION}"
LABEL vendor="open3d.org"
LABEL architecture="x86_64"
LABEL os="linux"
LABEL version="0.12.0"
LABEL maintainer="open3d-info@osvf.org"

ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles SUDO=command
# Install Python 3, cmake>=3.12 and nvidia drivers (only if not installed)
COPY ./util/docker/open3d-gpu/scripts/env-setup.sh /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh
RUN /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh

# Install dependencies with apt-get and pip
COPY ./util/install_deps_ubuntu.sh ./util/ci_utils.sh /root/Open3D/util/
RUN /root/Open3D/util/install_deps_ubuntu.sh assume-yes

COPY ./util/ci_utils.sh /root/Open3D/util/
RUN bash -o errexit -c "source /root/Open3D/util/ci_utils.sh && \
        install_cuda_toolkit with-cudnn purge-cache"

# Persist PATH for cuda, cudnn and set requirements for host and container
ENV PATH=/usr/local/cuda/bin:$PATH \
         LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
         NVIDIA_VISIBLE_DEVICES=all \
         NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics" \
         NVIDIA_REQUIRE_CUDA="cuda>=11.0"

COPY . /root/Open3D

WORKDIR /root/Open3D
ENTRYPOINT /bin/bash
