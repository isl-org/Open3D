#!/usr/bin/env bash
#
# docker_test.sh is used to test Open3D docker images built by docker_build.sh
#
# Guidelines:
# - Use a flat list of options. No additional arguments.
#   The option names should match exactly the ones used in docker_build.sh.
# - No external environment variables.
#   - This script should not make assumptions on external environment variables.
#   - Environment variables are imported from docker_build.sh.

set -euo pipefail

__usage="USAGE:
    $(basename $0) [OPTION]

OPTION:
    openblas-x86_64    : OpenBLAS x86_64
    openblas-arm64     : OpenBLAS ARM64
    2-bionic           : CUDA CI, 2-bionic
    3-ml-shared-bionic : CUDA CI, 3-ml-shared-bionic
    4-ml-bionic        : CUDA CI, 4-ml-bionic
    5-ml-focal         : CUDA CI, 5-ml-focal
"

HOST_OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. >/dev/null 2>&1 && pwd)"

print_usage_and_exit() {
    echo "$__usage"
    exit 1
}

cuda_print_env() {
    echo "[cuda_print_env()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[cuda_print_env()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[cuda_print_env()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[cuda_print_env()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[cuda_print_env()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[cuda_print_env()] CCACHE_VERSION=${CCACHE_VERSION}"
    echo "[cuda_print_env()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[cuda_print_env()] SHARED=${SHARED}"
    echo "[cuda_print_env()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[cuda_print_env()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
}

openblas_print_env() {
    echo "[openblas_print_env()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[openblas_print_env()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[openblas_print_env()] CMAKE_VER=${CMAKE_VER}"
    echo "[openblas_print_env()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
}

restart_docker_daemon_if_on_gcloud() {
    # Sometimes `docker run` may fail on the second run on Google Cloud with the
    # following erorr:
    # ```
    # docker: Error response from daemon: OCI runtime create failed:
    # container_linux.go:349: starting container process caused
    # "process_linux.go:449: container init caused \"process_linux.go:432:
    # running prestart hook 0 caused \\\"error running hook: exit status 1,
    # stdout: , stderr: nvidia-container-cli: initialization error:
    # nvml error: driver/library version mismatch\\\\n\\\"\"": unknown.
    # ```
    if curl metadata.google.internal -i | grep Google; then
        # https://stackoverflow.com/a/30921162/1255535
        echo "[restart_docker_daemon_if_on_gcloud()] Restarting Docker daemon on Google Cloud."
        sudo systemctl daemon-reload
        sudo systemctl restart docker
    else
        echo "[restart_docker_daemon_if_on_gcloud()] Skipped."
    fi
}

cpp_python_linking_uninstall_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_CUDA_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    echo "[cpp_python_linking_uninstall_test()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"

    # Config-dependent argument: gpu_run_args
    if [ "${BUILD_CUDA_MODULE}" == "ON" ]; then
        docker_run="docker run --gpus all"
    else
        docker_run="docker run"
    fi

    # Config-dependent argument: pytest_args
    if [ "${BUILD_PYTORCH_OPS}" == "OFF" ] || [ "${BUILD_TENSORFLOW_OPS}" == "OFF" ]; then
        pytest_args="--ignore python/test/ml_ops/"
    else
        pytest_args=""
    fi

    restart_docker_daemon_if_on_gcloud

    # C++ test
    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm ${DOCKER_TAG} /bin/bash -c "\
        cd build \
     && ./bin/tests --gtest_shuffle \
    "
    restart_docker_daemon_if_on_gcloud

    # Python test
    echo "pytest is randomized, add --randomly-seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        pytest python/test ${pytest_args} \
    "
    restart_docker_daemon_if_on_gcloud

    # C++ linking
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        git clone https://github.com/isl-org/open3d-cmake-find-package.git \
     && cd open3d-cmake-find-package \
     && mkdir build \
     && cd build \
     && cmake -DCMAKE_INSTALL_PREFIX=~/open3d_install .. \
     && make -j$(nproc) VERBOSE=1 \
     && ./Draw --skip-for-unit-test \
    "
    restart_docker_daemon_if_on_gcloud

    # Uninstall
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cd build \
     && make uninstall \
    "
}

if [[ "$#" -ne 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit
fi
echo "[$(basename $0)] building $1"
source "${HOST_OPEN3D_ROOT}/.github/workflows/docker_build.sh"
case "$1" in
    2-bionic)
        2-bionic_export_env
        cuda_print_env
        export BUILD_CUDA_MODULE=ON
        cpp_python_linking_uninstall_test
        ;;
    3-ml-shared-bionic)
        3-ml-shared-bionic_export_env
        cuda_print_env
        export BUILD_CUDA_MODULE=ON
        cpp_python_linking_uninstall_test
        ;;
    4-ml-bionic)
        4-ml-bionic_export_env
        cuda_print_env
        export BUILD_CUDA_MODULE=ON
        cpp_python_linking_uninstall_test
        ;;
    5-ml-focal)
        5-ml-focal_export_env
        cuda_print_env
        export BUILD_CUDA_MODULE=ON
        cpp_python_linking_uninstall_test
        ;;
    openblas-x86_64)
        openblas-x86_64_export_env
        openblas_print_env
        export BUILD_CUDA_MODULE=OFF
        export BUILD_PYTORCH_OPS=OFF
        export BUILD_TENSORFLOW_OPS=OFF
        cpp_python_linking_uninstall_test
        ;;
    openblas-arm64)
        openblas-arm64_export_env
        openblas_print_env
        export BUILD_CUDA_MODULE=OFF
        export BUILD_PYTORCH_OPS=OFF
        export BUILD_TENSORFLOW_OPS=OFF
        cpp_python_linking_uninstall_test
        ;;
    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit
        ;;
esac
