LABEL maintainer="sameer.sheorey@intel.com"

# These defaults can be changed with docker build --build-arg VAR=value
ARG CONTAINER_BASE_OS=ubuntu18.04
ARG CUDA_VERSION=10.2
ARG NVIDIA_DRIVER_VERSION=440.64.00

# Base container image has Nvidia drivers AND cuda isntalled
FROM nvidia/cuda:${CUDA_VERSION}-devel-${CONTAINER_BASE_OS}
# Only driver: 
# FROM nvidia/driver:${NVIDIA_DRIVER_VERSION}-${CONTAINER_BASE_OS}

ENV DEBIAN_FRONTEND noninteractive
# Install Python 3, cmake>=3.12 and nvidia drivers (only if not installed)
COPY ./util/docker/open3d-gpu/scripts/env-setup.sh /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh
RUN util/docker/open3d-gpu/scripts/env-setup.sh

# Install dependencies with apt-get
COPY ./util/scripts/install-deps-ubuntu.sh /root/Open3D/util/scripts/install-deps-ubuntu.sh
RUN util/scripts/install-deps-ubuntu.sh assume-yes

COPY . /root/Open3D
WORKDIR /root/Open3D

ENTRYPOINT ['/bin/bash', 'util/scripts/run-ci.sh']
