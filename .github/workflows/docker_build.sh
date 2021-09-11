#!/usr/bin/env bash
#
# docker_build.sh is used to build Open3D docker images for all supported
# scenarios. This can be used in CI and on local machines. The objective is to
# allow developers to emulate CI environments for debugging or build release
# artifacts such as Python wheels locally.
#
# Guidelines:
# - Use a flat list of options.
#   We don't want to have a cartesian product of different combinations of
#   options. E.g., to support Ubuntu {18.04, 20.04} with Python {3.7, 3.8}, we
#   don't specify the OS and Python version separately, instead, we have a flat
#   list of combinations: [u1804_py37, u1804_py38, u2004_py37, u2004_py38].
# - No external environment variables.
#   This script should not make assumptions on external environment variables.
#   This make the Docker image reproducible across different machines.
set -euo pipefail

__usage="USAGE:
    $(basename $0) [OPTION]

OPTION:
    openblas_x86_64    : Build with OpenBLAS x86_64
    openblas_arm64     : Build with OpenBLAS ARM64
    cuda_wheel_py36_dev: Build CUDA Python 3.6 wheel, developer mode
    cuda_wheel_py37_dev: Build CUDA Python 3.7 wheel, developer mode
    cuda_wheel_py38_dev: Build CUDA Python 3.8 wheel, developer mode
    cuda_wheel_py39_dev: Build CUDA Python 3.9 wheel, developer mode
    cuda_wheel_py36    : Build CUDA Python 3.6 wheel, release mode
    cuda_wheel_py37    : Build CUDA Python 3.7 wheel, release mode
    cuda_wheel_py38    : Build CUDA Python 3.8 wheel, release mode
    cuda_wheel_py39    : Build CUDA Python 3.9 wheel, release mode
    2-bionic           : Build CUDA GCloud container, ubuntu 1804
    3-ML-SHARED-bionic : Build CUDA GCloud container, ubuntu 1804, with ML, shared
    4-ML-bionic        : Build CUDA GCloud container, ubuntu 1804, with ML
    5-ML-focal         : Build CUDA GCloud container, ubuntu 2004, with ML
"

OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"

# Derive short git hash for Docker tag.
pushd ${OPEN3D_ROOT}
GIT_HASH="$(git rev-parse --short HEAD)"
popd

# Shared variables
CCACHE_VERSION=4.3
CMAKE_VERSION=cmake-3.19.7-Linux-x86_64

print_usage_and_exit() {
    echo "$__usage"
    exit 1
}

openblas() {
    options="$(echo "$@" | tr ' ' '|')"
    echo "[openblas()] options: ${options}"
    if [[ "x86_64" =~ ^($options)$ ]]; then
        BASE_IMAGE=ubuntu:20.04
        CMAKE_VER=cmake-3.19.7-Linux-x86_64 # TODO: unify with the rest
        CCACHE_TAR_NAME=open3d-x86_64-ci-ccache
    elif [[ "arm64" =~ ^($options)$ ]]; then
        BASE_IMAGE=arm64v8/ubuntu:20.04
        CMAKE_VER=cmake-3.19.7-Linux-aarch64 # TODO: unify with the rest
        CCACHE_TAR_NAME=open3d-arm64-ci-ccache
        docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    else
        echo "Invalid architecture."
        print_usage_and_exit
    fi
    echo "[cuda_wheel()] BASE_IMAGE: ${BASE_IMAGE}"
    echo "[cuda_wheel()] CMAKE_VER: ${CMAKE_VER}"
    echo "[cuda_wheel()] CCACHE_TAR_NAME: ${CCACHE_TAR_NAME}"

    # Docker build
    pushd ${OPEN3D_ROOT}
    docker build --build-arg BASE_IMAGE=${BASE_IMAGE} \
                 --build-arg CMAKE_VER=${CMAKE_VER} \
                 --build-arg CCACHE_TAR_NAME=${CCACHE_TAR_NAME} \
                 -t open3d-openblas-ci:latest \
                 -f .github/workflows/Dockerfile.openblas .
    popd

    # Extract ccache
    docker run -v ${PWD}:/opt/mount --rm open3d-openblas-ci:latest \
        cp /${CCACHE_TAR_NAME}.tar.gz /opt/mount
    sudo chown $(id -u):$(id -g) ${CCACHE_TAR_NAME}.tar.gz
}

cuda_wheel() {
    BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
    CCACHE_TAR_NAME=open3d-ubuntu-1804-cuda-ci-ccache
    BUILD_WHEEL=ON
    BUILD_ALL=OFF
    SHARED=OFF
    BUILD_TENSORFLOW_OPS=ON
    BUILD_PYTORCH_OPS=ON

    options="$(echo "$@" | tr ' ' '|')"
    echo "[cuda_wheel()] options: ${options}"
    if [[ "py36" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.6
    elif [[ "py37" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.7
    elif [[ "py38" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.8
    elif [[ "py39" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.9
    else
        echo "Invalid python version."
        print_usage_and_exit
    fi
    if [[ "dev" =~ ^($options)$ ]]; then
        DEVELOPER_BUILD=ON
    else
        DEVELOPER_BUILD=OFF
    fi
    echo "[cuda_wheel()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[cuda_wheel()] DEVELOPER_BUILD: ${DEVELOPER_BUILD}"

    DOCKER_TAG=open3d-ubuntu-cuda-ci:wheel-${PYTHON_VERSION}-${DEVELOPER_BUILD}-${GIT_HASH}

    pushd ${OPEN3D_ROOT}
    docker build \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        --build-arg DEVELOPER_BUILD=${DEVELOPER_BUILD} \
        --build-arg CCACHE_TAR_NAME=${CCACHE_TAR_NAME} \
        --build-arg CMAKE_VERSION=${CMAKE_VERSION} \
        --build-arg CCACHE_VERSION=${CCACHE_VERSION} \
        --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
        --build-arg BUILD_WHEEL=${BUILD_WHEEL} \
        --build-arg BUILD_ALL=${BUILD_ALL} \
        --build-arg SHARED=${SHARED} \
        --build-arg BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS} \
        --build-arg BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS} \
        -t ${DOCKER_TAG} \
        -f .github/workflows/Dockerfile.ubuntu-cuda .
    popd

    docker run -v ${PWD}:/opt/mount --rm ${DOCKER_TAG} \
        bash -c "cp /${CCACHE_TAR_NAME}.tar.gz /opt/mount"
    sudo chown $(id -u):$(id -g) ${CCACHE_TAR_NAME}.tar.gz

    python_package_dir=/root/Open3D/build/lib/python_package
    docker run -v ${PWD}:/opt/mount --rm ${DOCKER_TAG} \
        bash -c "cp ${python_package_dir}/pip_package/open3d*.whl                /opt/mount \
              && cp ${python_package_dir}/conda_package/linux-64/open3d*.tar.bz2 /opt/mount"
    sudo chown $(id -u):$(id -g) open3d*.whl
    sudo chown $(id -u):$(id -g) open3d*.tar.bz2
}

cuda_gcloud_build() {
    echo "[cuda_gcloud_build()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[cuda_gcloud_build()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[cuda_gcloud_build()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[cuda_gcloud_build()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[cuda_gcloud_build()] CCACHE_VERSION=${CCACHE_VERSION}"
    echo "[cuda_gcloud_build()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[cuda_gcloud_build()] BUILD_WHEEL=${BUILD_WHEEL}"
    echo "[cuda_gcloud_build()] BUILD_ALL=${BUILD_ALL}"
    echo "[cuda_gcloud_build()] SHARED=${SHARED}"
    echo "[cuda_gcloud_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[cuda_gcloud_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[cuda_gcloud_build()] DOCKER_TAG=${DOCKER_TAG}"

    pushd ${OPEN3D_ROOT}
    docker build \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        --build-arg DEVELOPER_BUILD=${DEVELOPER_BUILD} \
        --build-arg CCACHE_TAR_NAME=${CCACHE_TAR_NAME} \
        --build-arg CMAKE_VERSION=${CMAKE_VERSION} \
        --build-arg CCACHE_VERSION=${CCACHE_VERSION} \
        --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
        --build-arg BUILD_WHEEL=${BUILD_WHEEL} \
        --build-arg BUILD_ALL=${BUILD_ALL} \
        --build-arg SHARED=${SHARED} \
        --build-arg BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS} \
        --build-arg BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS} \
        -t ${DOCKER_TAG} \
        -f .github/workflows/Dockerfile.ubuntu-cuda .
    popd

    docker run -v ${PWD}:/opt/mount --rm ${DOCKER_TAG} \
        bash -c "cp /${CCACHE_TAR_NAME}.tar.gz /opt/mount"
    sudo chown $(id -u):$(id -g) ${CCACHE_TAR_NAME}.tar.gz
}

2-bionic() {
    BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
    DEVELOPER_BUILD=ON
    CCACHE_TAR_NAME=open3d-ubuntu-1804-cuda-ci-ccache
    PYTHON_VERSION=3.6
    BUILD_WHEEL=OFF
    BUILD_ALL=ON
    SHARED=OFF
    BUILD_TENSORFLOW_OPS=OFF
    BUILD_PYTORCH_OPS=OFF

    DOCKER_TAG=open3d-ubuntu-cuda-ci:2-bionic-${GIT_HASH}
    cuda_gcloud_build
}

3-ML-SHARED-bionic() {
    BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
    DEVELOPER_BUILD=ON
    CCACHE_TAR_NAME=open3d-ubuntu-1804-cuda-ci-ccache
    PYTHON_VERSION=3.6
    BUILD_WHEEL=OFF
    BUILD_ALL=ON
    SHARED=ON
    BUILD_TENSORFLOW_OPS=ON
    BUILD_PYTORCH_OPS=ON

    DOCKER_TAG=open3d-ubuntu-cuda-ci:3-ML-SHARED-bionic-${GIT_HASH}
    cuda_gcloud_build
}

4-ML-bionic() {
    BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
    DEVELOPER_BUILD=ON
    CCACHE_TAR_NAME=open3d-ubuntu-1804-cuda-ci-ccache
    PYTHON_VERSION=3.6
    BUILD_WHEEL=OFF
    BUILD_ALL=ON
    SHARED=OFF
    BUILD_TENSORFLOW_OPS=ON
    BUILD_PYTORCH_OPS=ON

    DOCKER_TAG=open3d-ubuntu-cuda-ci:4-ML-bionic-${GIT_HASH}
    cuda_gcloud_build
}

5-ML-focal() {
    BASE_IMAGE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
    DEVELOPER_BUILD=ON
    CCACHE_TAR_NAME=open3d-ubuntu-2004-cuda-ci-ccache
    PYTHON_VERSION=3.6
    BUILD_WHEEL=OFF
    BUILD_ALL=ON
    SHARED=OFF
    BUILD_TENSORFLOW_OPS=ON
    BUILD_PYTORCH_OPS=ON

    DOCKER_TAG=open3d-ubuntu-cuda-ci:5-ML-focal-${GIT_HASH}
    cuda_gcloud_build
}


if [[ "$#" -ne 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit
fi
echo "[$(basename $0)] building $1"
case "$1" in
    openblas_x86_64)
        openblas x86_64
        ;;
    openblas_arm64)
        openblas arm64
        ;;
    cuda_wheel_py36_dev)
        cuda_wheel py36 dev
        ;;
    cuda_wheel_py37_dev)
        cuda_wheel py37 dev
        ;;
    cuda_wheel_py38_dev)
        cuda_wheel py38 dev
        ;;
    cuda_wheel_py39_dev)
        cuda_wheel py39 dev
        ;;
    cuda_wheel_py36)
        cuda_wheel py36
        ;;
    cuda_wheel_py37)
        cuda_wheel py37
        ;;
    cuda_wheel_py38)
        cuda_wheel py38
        ;;
    cuda_wheel_py39)
        cuda_wheel py39
        ;;
    2-bionic)
        2-bionic
        ;;
    3-ML-SHARED-bionic)
        3-ML-SHARED-bionic
        ;;
    4-ML-bionic)
        4-ML-bionic
        ;;
    5-ML-focal)
        5-ML-focal
        ;;
    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit
        ;;
esac
