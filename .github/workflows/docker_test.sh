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

print_env() {
    echo "[print_env()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[print_env()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[print_env()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[print_env()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[print_env()] CCACHE_VERSION=${CCACHE_VERSION}"
    echo "[print_env()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[print_env()] SHARED=${SHARED}"
    echo "[print_env()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[print_env()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[print_env()] DOCKER_TAG=${DOCKER_TAG}"
}

test_cpp_python_linking_uninstall() {
    # C++ test
    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    docker run -i --rm --gpus all ${DOCKER_TAG} /bin/bash -c "\
        cd build \
     && ./bin/tests --gtest_shuffle \
    "

    # Python test
    echo "pytest is randomized, add --rondomly-seed=SEED to repeat the test sequence."
    pytest_args=""
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] || [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        pytest_args="${pytest_args} --ignore python/test/ml_ops/"
    fi
    docker run -i --rm --gpus all "${DOCKER_TAG}" /bin/bash -c "\
        pytest python/test ${pytest_args} \
    "

    # C++ linking
    docker run -i --rm --gpus all "${DOCKER_TAG}" /bin/bash -c "\
        git clone https://github.com/isl-org/open3d-cmake-find-package.git \
     && cd open3d-cmake-find-package \
     && mkdir build \
     && cd build \
     && cmake -DCMAKE_INSTALL_PREFIX=~/open3d_install .. \
     && make -j$(nproc) VERBOSE=1 \
     && ./Draw --skip-for-unit-test \
    "

    # Uninstall
    docker run -i --rm --gpus all "${DOCKER_TAG}" /bin/bash -c "\
        cd build \
     && make uninstall \
    "
}

2-bionic() {
    export_env_2-bionic
    print_env
    test_cpp_python_linking_uninstall
}

3-ml-shared-bionic() {
    export_env_3-ml-shared-bionic
    print_env
    test_cpp_python_linking_uninstall
}

4-ml-bionic() {
    export_env_4-ml-bionic
    print_env
    test_cpp_python_linking_uninstall
}

5-ml-focal() {
    export_env_5-ml-focal
    print_env
    test_cpp_python_linking_uninstall
}

if [[ "$#" -ne 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit
fi
echo "[$(basename $0)] building $1"
source "${HOST_OPEN3D_ROOT}/.github/workflows/docker_build.sh"
case "$1" in
    2-bionic)
        2-bionic
        ;;
    3-ml-shared-bionic)
        3-ml-shared-bionic
        ;;
    4-ml-bionic)
        4-ml-bionic
        ;;
    5-ml-focal)
        5-ml-focal
        ;;
    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit
        ;;
esac
