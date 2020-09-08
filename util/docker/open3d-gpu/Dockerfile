# These defaults can be changed with docker build --build-arg VAR=value
ARG UBUNTU_VERSION=bionic
FROM ubuntu:${UBUNTU_VERSION}
# Persist ARG for the rest of the build
ARG UBUNTU_VERSION
ARG NVIDIA_DRIVER_VERSION=440
LABEL name="open3d-dev/open3d-gpu-ci-${UBUNTU_VERSION}" \
            vendor="open3d.org" \
            maintainer="sameer.sheorey@intel.com"

ENV DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles SUDO=command
# Install Python 3, cmake>=3.12 and nvidia drivers (only if not installed)
COPY ./util/docker/open3d-gpu/scripts/env-setup.sh /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh
RUN /root/Open3D/util/docker/open3d-gpu/scripts/env-setup.sh

# Install dependencies with apt-get and pip
COPY ./util/install_deps_ubuntu.sh ./util/ci_utils.sh /root/Open3D/util/
RUN /root/Open3D/util/install_deps_ubuntu.sh assume-yes
COPY ./util/ci_utils.sh /root/Open3D/util/
RUN bash -c "source /root/Open3D/util/ci_utils.sh && \
        install_cuda_toolkit with-cudnn purge-cache && \
        install_python_dependencies with-unit-test with-cuda purge-cache && \
        echo export PATH=\$PATH >> /root/.bashrc && \
        echo export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH >> /root/.bashrc"
# Need to persist PATH for cuda, cudnn

COPY . /root/Open3D
WORKDIR /root/Open3D

ENTRYPOINT util/run_ci.sh
