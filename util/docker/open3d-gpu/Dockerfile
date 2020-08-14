# These defaults can be changed with docker build --build-arg VAR=value
ARG CONTAINER_BASE_OS=ubuntu18.04
ARG CUDA_VERSION=10.1
ARG CUDNN=cudnn7-

# Base container image has NVIDIA drivers, CUDA and (optionally) cuDNN installed
FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN}devel-${CONTAINER_BASE_OS}
LABEL maintainer="sameer.sheorey@intel.com"

ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles SUDO=command
# Install Python 3, cmake>=3.12 and nvidia drivers (only if not installed)
COPY ./util/docker/open3d-gpu/scripts/env-setup.sh /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh
RUN /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh

# Install dependencies with apt-get
COPY ./util/install_deps_ubuntu.sh /root/Open3D/util/install_deps_ubuntu.sh
RUN /root/Open3D/util/install_deps_ubuntu.sh assume-yes

COPY . /root/Open3D
WORKDIR /root/Open3D

ENTRYPOINT util/run_ci.sh
