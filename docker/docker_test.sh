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

__usage_docker_test="USAGE:
    $(basename $0) [OPTION]

OPTION:
    # OpenBLAS AMD64 (Dockerfile.openblas)
    openblas-amd64-py310-dev    : OpenBLAS AMD64 3.10 wheel, developer mode
    openblas-amd64-py311-dev    : OpenBLAS AMD64 3.11 wheel, developer mode
    openblas-amd64-py312-dev    : OpenBLAS AMD64 3.12 wheel, developer mode
    openblas-amd64-py313-dev    : OpenBLAS AMD64 3.13 wheel, developer mode
    openblas-amd64-py314-dev    : OpenBLAS AMD64 3.14 wheel, developer mode
    openblas-amd64-py310        : OpenBLAS AMD64 3.10 wheel, release mode
    openblas-amd64-py311        : OpenBLAS AMD64 3.11 wheel, release mode
    openblas-amd64-py312        : OpenBLAS AMD64 3.12 wheel, release mode
    openblas-amd64-py313        : OpenBLAS AMD64 3.13 wheel, release mode
    openblas-amd64-py314        : OpenBLAS AMD64 3.14 wheel, release mode

    # OpenBLAS ARM64 (Dockerfile.openblas)
    openblas-arm64-py310-dev    : OpenBLAS ARM64 3.10 wheel, developer mode
    openblas-arm64-py311-dev    : OpenBLAS ARM64 3.11 wheel, developer mode
    openblas-arm64-py312-dev    : OpenBLAS ARM64 3.12 wheel, developer mode
    openblas-arm64-py313-dev    : OpenBLAS ARM64 3.13 wheel, developer mode
    openblas-arm64-py314-dev    : OpenBLAS ARM64 3.14 wheel, developer mode
    openblas-arm64-py310        : OpenBLAS ARM64 3.10 wheel, release mode
    openblas-arm64-py311        : OpenBLAS ARM64 3.11 wheel, release mode
    openblas-arm64-py312        : OpenBLAS ARM64 3.12 wheel, release mode
    openblas-arm64-py313        : OpenBLAS ARM64 3.13 wheel, release mode
    openblas-arm64-py314        : OpenBLAS ARM64 3.14 wheel, release mode

    # Ubuntu CPU CI (Dockerfile.ci)
    cpu-static                  : Ubuntu CPU static
    cpu-static-release          : Ubuntu CPU static, release mode
    cpu-shared-ml               : Ubuntu CPU shared with ML
    cpu-shared-ml-release       : Ubuntu CPU shared with ML, release mode

    # Sycl CPU CI (Dockerfile.ci)
    sycl-shared [cpp|python]   : SYCL (oneAPI) with shared lib. Optional 2nd
                                 arg runs only the C++ or Python tests
                                 (default: full test suite).
    sycl-static [cpp|python]   : SYCL (oneAPI) with static lib. See above.

    # ML CIs (Dockerfile.ci)
    2-noble                   : CUDA CI, 2-noble, developer mode
    3-ml-shared-noble-release : CUDA CI, 3-ml-shared-noble (cxx11_abi), release mode
    3-ml-shared-noble         : CUDA CI, 3-ml-shared-noble (cxx11_abi), developer mode
"

HOST_OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

print_usage_and_exit_docker_test() {
    echo "$__usage_docker_test"
    exit 1
}

ci_print_env() {
    echo "[ci_print_env()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_print_env()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_print_env()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_print_env()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[ci_print_env()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_print_env()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_print_env()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_print_env()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_print_env()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_print_env()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[ci_print_env()] PACKAGE=${PACKAGE}"
}

restart_docker_daemon_if_on_gcloud() {
    # Sometimes `docker run` may fail on the second run on Google Cloud with the
    # following error:
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

docker_run_setup() {
    # Sets shared variables used by cpp_test(), python_test(),
    # linking_test() and uninstall_test():
    # - docker_run : "docker run" command prefix with config-dependent flags
    # - pytest_args: pytest arguments, e.g. to skip ML ops tests
    # Expects the following environment variables to be set:
    # - BUILD_CUDA_MODULE
    # - BUILD_SYCL_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[docker_run_setup()] NPROC=${NPROC:=$(nproc)}"

    docker_run="docker run --cpus ${NPROC}"
    if [ "${BUILD_CUDA_MODULE}" == "ON" ]; then
        docker_run="${docker_run} --gpus all"
    fi
    if [ "${BUILD_SYCL_MODULE}" == "ON" ]; then
        # Only request the DRI render node if present. GCE VMs used for
        # Intel GPU CI have one, but GitHub-hosted runners are CPU-only and
        # have no /dev/dri; the SYCL runtime falls back to the CPU device
        # automatically in that case (see PrintSYCLDevices() in
        # cpp/open3d/core/SYCLUtils.cpp).
        if [ -e /dev/dri ]; then
            docker_run="${docker_run} --device=/dev/dri"
        fi
        if [ -n "${CI:-}" ]; then
            docker_run="${docker_run} --env CI=${CI}"
        fi
    fi

    if [ "${BUILD_PYTORCH_OPS}" == "OFF" ] || [ "${BUILD_TENSORFLOW_OPS}" == "OFF" ]; then
        pytest_args="--ignore python/test/ml_ops/"
    else
        pytest_args=""
    fi
}

cpp_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_CUDA_MODULE
    # - BUILD_SYCL_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[cpp_test()] DOCKER_TAG=${DOCKER_TAG}"
    docker_run_setup
    restart_docker_daemon_if_on_gcloud

    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    if [ "${BUILD_SYCL_MODULE}" == "ON" ]; then
        # SYCL CPU tests can time out due to kernel compilation time. Keep the
        # test shard count independent from the host CPU count so every CI run
        # executes the same four GoogleTest shards.
        gtest_shards=4
        echo "[cpp_test()] Running sharded gtests with GNU parallel."
        # Each shard is a separate process but shares the host /tmp by default.
        # Many IO tests write fixed basenames (e.g. test.xyzrgb) under
        # GetTempDirectoryPath(); isolate shards via TMPDIR so they do not race.
        ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -euo pipefail -c " \
            cd build \
         && seq 0 $((${gtest_shards} - 1)) | parallel -k --jobs ${gtest_shards} --halt never \
            'd=/tmp/open3d-gtest-shard-{}; mkdir -p "$d" && TMPDIR="$d" GTEST_TOTAL_SHARDS='"${gtest_shards}"' GTEST_SHARD_INDEX={} ./bin/tests --gtest_shuffle' \
        "
    else
        ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c " \
            cd build \
         && ./bin/tests --gtest_shuffle \
        "
    fi
    restart_docker_daemon_if_on_gcloud
}

python_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_CUDA_MODULE
    # - BUILD_SYCL_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[python_test()] DOCKER_TAG=${DOCKER_TAG}"
    docker_run_setup
    restart_docker_daemon_if_on_gcloud

    echo "pytest is randomized, add --randomly-seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c " \
        python  -W default -m pytest python/test ${pytest_args} -s"
    restart_docker_daemon_if_on_gcloud
}

linking_test() {
    # Command-line tools test and C++ linking (cmake/pkg-config) test.
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_SHARED_LIBS
    # - BUILD_CUDA_MODULE
    # - BUILD_SYCL_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[linking_test()] DOCKER_TAG=${DOCKER_TAG}"
    docker_run_setup

    # Command-line tools test
    echo "testing Open3D command-line tools"
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        open3d \
     && open3d -h \
     && open3d --help \
     && open3d -V \
     && open3d --version \
     && open3d example -h \
     && open3d example --help \
     && open3d example -l \
     && open3d example --list \
     && open3d example -l io \
     && open3d example --list io \
     && open3d example -s io/image_io \
     && open3d example --show io/image_io \
    "

    # C++ linking with new project
    if [ "${BUILD_SYCL_MODULE}" == "ON" ]; then
        cmake_compiler_args="-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx"
    else
        cmake_compiler_args=""
    fi

    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cd examples/cmake/open3d-cmake-find-package \
     && mkdir build \
     && pushd build \
     && echo Testing build with cmake \
     && cmake ${cmake_compiler_args} -DCMAKE_INSTALL_PREFIX=~/open3d_install .. \
     && make -j$(nproc) VERBOSE=1 \
     && ./Draw --skip-for-unit-test \
    "

    if [ "${BUILD_SHARED_LIBS}" == "ON" ] && [ "${BUILD_SYCL_MODULE}" == "OFF" ]; then
        ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
            cd examples/cmake/open3d-cmake-find-package \
         && mkdir build \
         && pushd build \
         && echo Testing build with pkg-config \
         && export PKG_CONFIG_PATH=~/open3d_install/lib/pkgconfig \
         && echo Open3D build options: \$(pkg-config --cflags --libs Open3D) \
         && c++ ../Draw.cpp -o Draw \$(pkg-config --cflags --libs Open3D) \
         && ./Draw --skip-for-unit-test \
        "
    fi
    restart_docker_daemon_if_on_gcloud
}

uninstall_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_CUDA_MODULE
    # - BUILD_SYCL_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[uninstall_test()] DOCKER_TAG=${DOCKER_TAG}"
    docker_run_setup
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cd build \
     && make uninstall \
    "
}

cpp_python_linking_uninstall_test() {
    # Runs the full test suite: C++ unit tests, Python unit tests,
    # command-line tools + C++ linking tests, and the uninstall test.
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_SHARED_LIBS
    # - BUILD_CUDA_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - BUILD_SYCL_MODULE
    # - NPROC (optional)
    cpp_test
    python_test
    linking_test
    uninstall_test
}

if [[ "$#" -lt 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit_docker_test
fi
echo "[$(basename $0)] building $1"
source "${HOST_OPEN3D_ROOT}/docker/docker_build.sh"
case "$1" in
# OpenBLAS AMD64
openblas-amd64-py310-dev)
    openblas_export_env amd64 py310 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py311-dev)
    openblas_export_env amd64 py311 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py312-dev)
    openblas_export_env amd64 py312 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py313-dev)
    openblas_export_env amd64 py313 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py314-dev)
    openblas_export_env amd64 py314 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py310)
    openblas_export_env amd64 py310
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py311)
    openblas_export_env amd64 py311
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py312)
    openblas_export_env amd64 py312
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py313)
    openblas_export_env amd64 py313
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-amd64-py314)
    openblas_export_env amd64 py314
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;

# OpenBLAS ARM64
openblas-arm64-py310-dev)
    openblas_export_env arm64 py310 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py311-dev)
    openblas_export_env arm64 py311 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py312-dev)
    openblas_export_env arm64 py312 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py313-dev)
    openblas_export_env arm64 py313 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py314-dev)
    openblas_export_env arm64 py314 dev
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py310)
    openblas_export_env arm64 py310
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py311)
    openblas_export_env arm64 py311
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py312)
    openblas_export_env arm64 py312
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py313)
    openblas_export_env arm64 py313
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;
openblas-arm64-py314)
    openblas_export_env arm64 py314
    openblas_print_env
    cpp_python_linking_uninstall_test
    ;;

# CPU CI
cpu-static)
    cpu-static_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-static-release)
    cpu-static-release_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-shared-ml)
    cpu-shared-ml_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-shared-ml-release)
    cpu-shared-ml-release_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;

# SYCL CI
# Optional 2nd arg selects a single test phase (cpp|python); this is used
# to split C++ tests (build-lib job) from Python tests (build-wheel job,
# which builds/tests the actual per-Python-version wheel separately) in
# .github/workflows/ubuntu-sycl.yml, avoiding duplicate test runs.
sycl-shared)
    sycl-shared_export_env
    ci_print_env
    case "${2:-all}" in
        cpp) cpp_test ;;
        python) python_test ;;
        all) cpp_python_linking_uninstall_test ;;
        *)
            echo "Error: invalid test phase: ${2}." >&2
            print_usage_and_exit_docker_test
            ;;
    esac
    ;;
sycl-static)
    sycl-static_export_env
    ci_print_env
    case "${2:-all}" in
        cpp) cpp_test ;;
        python) python_test ;;
        all) cpp_python_linking_uninstall_test ;;
        *)
            echo "Error: invalid test phase: ${2}." >&2
            print_usage_and_exit_docker_test
            ;;
    esac
    ;;

    # ML CIs
2-noble)
    2-noble_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
3-ml-shared-noble-release)
    3-ml-shared-noble-release_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
3-ml-shared-noble)
    3-ml-shared-noble_export_env
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
*)
    echo "Error: invalid argument: ${1}." >&2
    print_usage_and_exit_docker_test
    ;;
esac
