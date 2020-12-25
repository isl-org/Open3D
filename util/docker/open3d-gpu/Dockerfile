# eg: docker build --build-arg UBUNTU_VERSION=focal
ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}
# Persist ARG for the rest of the build
ARG UBUNTU_VERSION
ARG NVIDIA_DRIVER_VERSION=440
LABEL name="open3d-dev/open3d-gpu-ci-${UBUNTU_VERSION}" \
            vendor="open3d.org" \
            architecture="x86_64" \
            os="linux" \
            maintainer="sameer.sheorey@intel.com"

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
         NVIDIA_REQUIRE_CUDA="cuda>=10.1"

COPY . /root/Open3D
WORKDIR /root/Open3D

ENTRYPOINT util/run_ci.sh
