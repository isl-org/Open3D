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
#   options. E.g., to support Ubuntu {20.04, 24.04} with Python {3.7, 3.8}, we
#   don't specify the OS and Python version separately, instead, we have a flat
#   list of combinations: [u2004_py39, u2004_py310, u2404_py39, u2404_py310].
# - No external environment variables.
#   This script should not make assumptions on external environment variables.
#   This make the Docker image reproducible across different machines.
set -euo pipefail

export BUILDKIT_PROGRESS=plain

__usage_docker_build="USAGE:
    $(basename $0) [OPTION]

OPTION:
    # OpenBLAS AMD64 (Dockerfile.openblas)
    openblas-amd64-py38-dev     : OpenBLAS AMD64 3.8 wheel, developer mode
    openblas-amd64-py39-dev     : OpenBLAS AMD64 3.9 wheel, developer mode
    openblas-amd64-py310-dev    : OpenBLAS AMD64 3.10 wheel, developer mode
    openblas-amd64-py311-dev    : OpenBLAS AMD64 3.11 wheel, developer mode
    openblas-amd64-py312-dev    : OpenBLAS AMD64 3.12 wheel, developer mode
    openblas-amd64-py38         : OpenBLAS AMD64 3.8 wheel, release mode
    openblas-amd64-py39         : OpenBLAS AMD64 3.9 wheel, release mode
    openblas-amd64-py310        : OpenBLAS AMD64 3.10 wheel, release mode
    openblas-amd64-py311        : OpenBLAS AMD64 3.11 wheel, release mode
    openblas-amd64-py312        : OpenBLAS AMD64 3.12 wheel, release mode

    # OpenBLAS ARM64 (Dockerfile.openblas)
    openblas-arm64-py38-dev     : OpenBLAS ARM64 3.8 wheel, developer mode
    openblas-arm64-py39-dev     : OpenBLAS ARM64 3.9 wheel, developer mode
    openblas-arm64-py310-dev    : OpenBLAS ARM64 3.10 wheel, developer mode
    openblas-arm64-py311-dev    : OpenBLAS ARM64 3.11 wheel, developer mode
    openblas-arm64-py312-dev    : OpenBLAS ARM64 3.12 wheel, developer mode
    openblas-arm64-py38         : OpenBLAS ARM64 3.8 wheel, release mode
    openblas-arm64-py39         : OpenBLAS ARM64 3.9 wheel, release mode
    openblas-arm64-py310        : OpenBLAS ARM64 3.10 wheel, release mode
    openblas-arm64-py311        : OpenBLAS ARM64 3.11 wheel, release mode
    openblas-arm64-py312        : OpenBLAS ARM64 3.12 wheel, release mode

    # Ubuntu CPU CI (Dockerfile.ci)
    cpu-static                  : Ubuntu CPU static
    cpu-shared                  : Ubuntu CPU shared (cxx11_abi)
    cpu-shared-release          : Ubuntu CPU shared (cxx11_abi), release mode
    cpu-shared-ml               : Ubuntu CPU shared with ML (pre_cxx11_abi)
    cpu-shared-ml-release       : Ubuntu CPU shared with ML (pre_cxx11_abi), release mode

    # Sycl CPU CI (Dockerfile.ci)
    sycl-shared                : SYCL (oneAPI) with shared lib
    sycl-static                : SYCL (oneAPI) with static lib

    # ML CIs (Dockerfile.ci)
    2-focal                   : CUDA CI, 2-bionic, developer mode
    3-ml-shared-focal-release : CUDA CI, 3-ml-shared-bionic (pre_cxx11_abi), release mode
    3-ml-shared-focal         : CUDA CI, 3-ml-shared-bionic (pre_cxx11_abi), developer mode
    4-shared-focal            : CUDA CI, 4-shared-bionic (cxx11_abi), developer mode
    4-shared-focal-release    : CUDA CI, 4-shared-bionic (cxx11_abi), release mode
    5-ml-jammy                 : CUDA CI, 5-ml-focal, developer mode

    # CUDA wheels (Dockerfile.wheel)
    cuda_wheel_py38_dev        : CUDA Python 3.8 wheel, developer mode
    cuda_wheel_py39_dev        : CUDA Python 3.9 wheel, developer mode
    cuda_wheel_py310_dev       : CUDA Python 3.10 wheel, developer mode
    cuda_wheel_py311_dev       : CUDA Python 3.11 wheel, developer mode
    cuda_wheel_py312_dev       : CUDA Python 3.12 wheel, developer mode
    cuda_wheel_py38            : CUDA Python 3.8 wheel, release mode
    cuda_wheel_py39            : CUDA Python 3.9 wheel, release mode
    cuda_wheel_py310           : CUDA Python 3.10 wheel, release mode
    cuda_wheel_py311           : CUDA Python 3.11 wheel, release mode
    cuda_wheel_py312           : CUDA Python 3.12 wheel, release mode
"

HOST_OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Shared variables
CCACHE_VERSION=4.3
CMAKE_VERSION=cmake-3.29.2-linux-x86_64
CMAKE_VERSION_AARCH64=cmake-3.24.4-linux-aarch64
CUDA_VERSION=12.1.0-cudnn8
CUDA_VERSION_LATEST=12.1.0-cudnn8

print_usage_and_exit_docker_build() {
    echo "$__usage_docker_build"
    exit 1
}

openblas_print_env() {
    echo "[openblas_print_env()] DOCKER_TAG: ${DOCKER_TAG}"
    echo "[openblas_print_env()] BASE_IMAGE: ${BASE_IMAGE}"
    echo "[openblas_print_env()] CONDA_SUFFIX: ${CONDA_SUFFIX}"
    echo "[openblas_print_env()] CMAKE_VERSION: ${CMAKE_VERSION}"
    echo "[openblas_print_env()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[openblas_print_env()] DEVELOPER_BUILD: ${DEVELOPER_BUILD}"
}

openblas_export_env() {
    options="$(echo "$@" | tr ' ' '|')"
    echo "[openblas_export_env()] options: ${options}"

    if [[ "amd64" =~ ^($options)$ ]]; then
        echo "[openblas_export_env()] platform AMD64"
        export DOCKER_TAG=open3d-ci:openblas-amd64
        export BASE_IMAGE=ubuntu:20.04
        export CONDA_SUFFIX=x86_64
        export CMAKE_VERSION=${CMAKE_VERSION}
    elif [[ "arm64" =~ ^($options)$ ]]; then
        echo "[openblas_export_env()] platform ARM64"
        export DOCKER_TAG=open3d-ci:openblas-arm64
        export BASE_IMAGE=arm64v8/ubuntu:20.04
        export CONDA_SUFFIX=aarch64
        export CMAKE_VERSION=${CMAKE_VERSION_AARCH64}
    else
        echo "Invalid platform."
        print_usage_and_exit_docker_build
    fi

    if [[ "py38" =~ ^($options)$ ]]; then
        export PYTHON_VERSION=3.8
        export DOCKER_TAG=${DOCKER_TAG}-py38
    elif [[ "py39" =~ ^($options)$ ]]; then
        export PYTHON_VERSION=3.9
        export DOCKER_TAG=${DOCKER_TAG}-py39
    elif [[ "py310" =~ ^($options)$ ]]; then
        export PYTHON_VERSION=3.10
        export DOCKER_TAG=${DOCKER_TAG}-py310
    elif [[ "py311" =~ ^($options)$ ]]; then
        export PYTHON_VERSION=3.11
        export DOCKER_TAG=${DOCKER_TAG}-py311
    elif [[ "py312" =~ ^($options)$ ]]; then
        export PYTHON_VERSION=3.12
        export DOCKER_TAG=${DOCKER_TAG}-py312
    else
        echo "Invalid python version."
        print_usage_and_exit_docker_build
    fi

    if [[ "dev" =~ ^($options)$ ]]; then
        export DEVELOPER_BUILD=ON
        export DOCKER_TAG=${DOCKER_TAG}-dev
    else
        export DEVELOPER_BUILD=OFF
        export DOCKER_TAG=${DOCKER_TAG}-release
    fi

    # For docker_test.sh
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_PYTORCH_OPS=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_SYCL_MODULE=OFF
}

openblas_build() {
    openblas_print_env

    pushd "${HOST_OPEN3D_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg CONDA_SUFFIX="${CONDA_SUFFIX}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        -t "${DOCKER_TAG}" \
        -f docker/Dockerfile.openblas .
    popd

    docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
        bash -c "cp /*.whl /opt/mount \
              && chown $(id -u):$(id -g) /opt/mount/*.whl"
}

cuda_wheel_build() {
    BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    CCACHE_TAR_NAME=open3d-ubuntu-2004-cuda-ci-ccache

    options="$(echo "$@" | tr ' ' '|')"
    echo "[cuda_wheel_build()] options: ${options}"
    if [[ "py38" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.8
    elif [[ "py39" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.9
    elif [[ "py310" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.10
    elif [[ "py311" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.11
    elif [[ "py312" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.12
    else
        echo "Invalid python version."
        print_usage_and_exit_docker_build
    fi
    if [[ "dev" =~ ^($options)$ ]]; then
        DEVELOPER_BUILD=ON
    else
        DEVELOPER_BUILD=OFF
    fi
    echo "[cuda_wheel_build()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[cuda_wheel_build()] DEVELOPER_BUILD: ${DEVELOPER_BUILD}"
    echo "[cuda_wheel_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:?'env var must be set.'}"
    echo "[cuda_wheel_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:?'env var must be set.'}"

    pushd "${HOST_OPEN3D_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CCACHE_TAR_NAME="${CCACHE_TAR_NAME}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg CCACHE_VERSION="${CCACHE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg CI="${CI:-}" \
        -t open3d-ci:wheel \
        -f docker/Dockerfile.wheel .
    popd

    python_package_dir=/root/Open3D/build/lib/python_package
    docker run -v "${PWD}:/opt/mount" --rm open3d-ci:wheel \
        bash -c "cp ${python_package_dir}/pip_package/open3d*.whl /opt/mount \
              && cp /${CCACHE_TAR_NAME}.tar.xz /opt/mount \
              && chown $(id -u):$(id -g) /opt/mount/open3d*.whl \
              && chown $(id -u):$(id -g) /opt/mount/${CCACHE_TAR_NAME}.tar.xz"
}

ci_build() {
    echo "[ci_build()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_build()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_build()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_build()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[ci_build()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_build()] CCACHE_VERSION=${CCACHE_VERSION}"
    echo "[ci_build()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_build()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_build()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[ci_build()] PACKAGE=${PACKAGE}"
    echo "[ci_build()] BUILD_SYCL_MODULE=${BUILD_SYCL_MODULE}"

    pushd "${HOST_OPEN3D_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CCACHE_TAR_NAME="${CCACHE_TAR_NAME}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg CCACHE_VERSION="${CCACHE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg PACKAGE="${PACKAGE}" \
        --build-arg BUILD_SYCL_MODULE="${BUILD_SYCL_MODULE}" \
        --build-arg CI="${CI:-}" \
        -t "${DOCKER_TAG}" \
        -f docker/Dockerfile.ci .
    popd

    docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
        bash -cx "cp /open3d* /opt/mount \
               && chown $(id -u):$(id -g) /opt/mount/open3d*"
}

2-focal_export_env() {
    export DOCKER_TAG=open3d-ci:2-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-2-focal
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=OFF
    export BUILD_SYCL_MODULE=OFF
}

3-ml-shared-focal_export_env() {
    export DOCKER_TAG=open3d-ci:3-ml-shared-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-3-ml-shared-focal
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=ON
    # TODO: re-enable tensorflow support, off due to due to cxx11_abi issue with PyTorch
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

3-ml-shared-focal-release_export_env() {
    export DOCKER_TAG=open3d-ci:3-ml-shared-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=open3d-ci-3-ml-shared-focal
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=ON
    # TODO: re-enable tensorflow support, off due to due to cxx11_abi issue with PyTorch
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

4-shared-focal_export_env() {
    export DOCKER_TAG=open3d-ci:4-shared-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-4-shared-focal
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=ON
    # TODO: tensorflow tests moved here till PyTorch supports cxx11_abi
    export BUILD_TENSORFLOW_OPS=ON
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

4-shared-focal-release_export_env() {
    export DOCKER_TAG=open3d-ci:4-shared-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=open3d-ci-4-shared-focal
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=ON
    # TODO: tensorflow tests moved here till PyTorch supports cxx11_abi
    export BUILD_TENSORFLOW_OPS=ON
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

5-ml-jammy_export_env() {
    export DOCKER_TAG=open3d-ci:5-ml-jammy

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu22.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-5-ml-jammy
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    # TODO: re-enable tensorflow support, off due to due to cxx11_abi issue with PyTorch
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=OFF
    export BUILD_SYCL_MODULE=OFF
}

cpu-static_export_env() {
    export DOCKER_TAG=open3d-ci:cpu-static

    export BASE_IMAGE=ubuntu:20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-cpu
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=VIEWER
    export BUILD_SYCL_MODULE=OFF
}

cpu-shared_export_env() {
    export DOCKER_TAG=open3d-ci:cpu-shared

    export BASE_IMAGE=ubuntu:20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-cpu
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=OFF
    # TODO: tensorflow tests moved here till PyTorch supports cxx11_abi
    export BUILD_TENSORFLOW_OPS=ON
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

cpu-shared-ml_export_env() {
    export DOCKER_TAG=open3d-ci:cpu-shared-ml

    export BASE_IMAGE=ubuntu:20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-cpu
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=OFF
    # TODO: re-enable tensorflow support, off due to due to cxx11_abi issue with PyTorch
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

cpu-shared-release_export_env() {
    export DOCKER_TAG=open3d-ci:cpu-shared

    export BASE_IMAGE=ubuntu:20.04
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=open3d-ci-cpu
    export PYTHON_VERSION=3.12 # no TF versions after 2.13.2 for Python 3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=OFF
    # TODO: tensorflow tests moved here till PyTorch supports cxx11_abi
    export BUILD_TENSORFLOW_OPS=ON
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

cpu-shared-ml-release_export_env() {
    export DOCKER_TAG=open3d-ci:cpu-shared-ml

    export BASE_IMAGE=ubuntu:20.04
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=open3d-ci-cpu
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=OFF
    # TODO: re-enable tensorflow support, off due to due to cxx11_abi issue with PyTorch
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=ON
    export BUILD_SYCL_MODULE=OFF
}

sycl-shared_export_env() {
    export DOCKER_TAG=open3d-ci:sycl-shared

    # https://hub.docker.com/r/intel/oneapi-basekit
    # https://github.com/intel/oneapi-containers/blob/master/images/docker/basekit/Dockerfile.ubuntu-20.04
    export BASE_IMAGE=intel/oneapi-basekit:2024.1.0-devel-ubuntu20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-sycl
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=ON
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=OFF
    export BUILD_SYCL_MODULE=ON
}

sycl-static_export_env() {
    export DOCKER_TAG=open3d-ci:sycl-static

    # https://hub.docker.com/r/intel/oneapi-basekit
    # https://github.com/intel/oneapi-containers/blob/master/images/docker/basekit/Dockerfile.ubuntu-20.04
    export BASE_IMAGE=intel/oneapi-basekit:2024.1.0-devel-ubuntu20.04
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=open3d-ci-sycl
    export PYTHON_VERSION=3.8
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=OFF
    export BUILD_SYCL_MODULE=ON
}

function main() {
    if [[ "$#" -ne 1 ]]; then
        echo "Error: invalid number of arguments: $#." >&2
        print_usage_and_exit_docker_build
    fi
    echo "[$(basename $0)] building $1"
    case "$1" in
    # OpenBLAS AMD64
    openblas-amd64-py38-dev)
        openblas_export_env amd64 py38 dev
        openblas_build
        ;;
    openblas-amd64-py39-dev)
        openblas_export_env amd64 py39 dev
        openblas_build
        ;;
    openblas-amd64-py310-dev)
        openblas_export_env amd64 py310 dev
        openblas_build
        ;;
    openblas-amd64-py311-dev)
        openblas_export_env amd64 py311 dev
        openblas_build
        ;;
    openblas-amd64-py312-dev)
        openblas_export_env amd64 py312 dev
        openblas_build
        ;;
    openblas-amd64-py38)
        openblas_export_env amd64 py38
        openblas_build
        ;;
    openblas-amd64-py39)
        openblas_export_env amd64 py39
        openblas_build
        ;;
    openblas-amd64-py310)
        openblas_export_env amd64 py310
        openblas_build
        ;;
    openblas-amd64-py311)
        openblas_export_env amd64 py311
        openblas_build
        ;;
    openblas-amd64-py312)
        openblas_export_env amd64 py312
        openblas_build
        ;;

    # OpenBLAS ARM64
    openblas-arm64-py38-dev)
        openblas_export_env arm64 py38 dev
        openblas_build
        ;;
    openblas-arm64-py39-dev)
        openblas_export_env arm64 py39 dev
        openblas_build
        ;;
    openblas-arm64-py310-dev)
        openblas_export_env arm64 py310 dev
        openblas_build
        ;;
    openblas-arm64-py311-dev)
        openblas_export_env arm64 py311 dev
        openblas_build
        ;;
    openblas-arm64-py312-dev)
        openblas_export_env arm64 py312 dev
        openblas_build
        ;;
    openblas-arm64-py38)
        openblas_export_env arm64 py38
        openblas_build
        ;;
    openblas-arm64-py39)
        openblas_export_env arm64 py39
        openblas_build
        ;;
    openblas-arm64-py310)
        openblas_export_env arm64 py310
        openblas_build
        ;;
    openblas-arm64-py311)
        openblas_export_env arm64 py311
        openblas_build
        ;;
    openblas-arm64-py312)
        openblas_export_env arm64 py312
        openblas_build
        ;;

    # CPU CI
    cpu-static)
        cpu-static_export_env
        ci_build
        ;;
    cpu-shared)
        cpu-shared_export_env
        ci_build
        ;;
    cpu-shared-release)
        cpu-shared-release_export_env
        ci_build
        ;;
    cpu-shared-ml)
        cpu-shared-ml_export_env
        ci_build
        ;;
    cpu-shared-ml-release)
        cpu-shared-ml-release_export_env
        ci_build
        ;;

    # SYCL CI
    sycl-shared)
        sycl-shared_export_env
        ci_build
        ;;
    sycl-static)
        sycl-static_export_env
        ci_build
        ;;

    # CUDA wheels
    cuda_wheel_py38_dev)
        cuda_wheel_build py38 dev
        ;;
    cuda_wheel_py39_dev)
        cuda_wheel_build py39 dev
        ;;
    cuda_wheel_py310_dev)
        cuda_wheel_build py310 dev
        ;;
    cuda_wheel_py311_dev)
        cuda_wheel_build py311 dev
        ;;
    cuda_wheel_py312_dev)
        cuda_wheel_build py312 dev
        ;;
    cuda_wheel_py38)
        cuda_wheel_build py38
        ;;
    cuda_wheel_py39)
        cuda_wheel_build py39
        ;;
    cuda_wheel_py310)
        cuda_wheel_build py310
        ;;
    cuda_wheel_py311)
        cuda_wheel_build py311
        ;;
    cuda_wheel_py312)
        cuda_wheel_build py312
        ;;

    # ML CIs
    2-focal)
        2-focal_export_env
        ci_build
        ;;
    3-ml-shared-focal-release)
        3-ml-shared-focal-release_export_env
        ci_build
        ;;
    3-ml-shared-focal)
        3-ml-shared-focal_export_env
        ci_build
        ;;
    4-shared-focal-release)
        4-shared-focal-release_export_env
        ci_build
        ;;
    4-shared-focal)
        4-shared-focal_export_env
        ci_build
        ;;
    5-ml-jammy)
        5-ml-jammy_export_env
        ci_build
        ;;
    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit_docker_build
        ;;
    esac
}

# main() will be executed when ./docker_build.sh is called directly.
# main() will not be executed when ./docker_build.sh is sourced.
if [ "$0" = "$BASH_SOURCE" ]; then
    main "$@"
fi
