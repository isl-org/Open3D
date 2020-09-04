#!/usr/bin/env bash
set -euo pipefail

# Get build scripts and control environment variables
# shellcheck source=build_scripts.sh
source "$(dirname "$0")"/build_scripts.sh

date
echo "$rj_startts StartJob ReportInit"
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

if [ "$BUILD_CUDA_MODULE" == "ON" ] && \
    ! nvcc --version | grep -q "release ${CUDA_VERSION[1]}" 2>/dev/null ; then
    reportRun install_cuda_toolkit with-cudnn purge-cache
    nvcc --version
fi

date
reportJobStart "Installing Python unit test dependencies"
if [ "$BUILD_CUDA_MODULE" == "ON" ] ; then
    install_python_dependencies with-unit-test with-cuda purge-cache
else
    install_python_dependencies with-unit-test purge-cache
fi

echo "using python: $(which python)"
python --version
echo -n "Using pip: "
python -m pip --version
echo -n "Using pytest:"
python -m pytest --version
echo "using cmake: $(which cmake)"
cmake --version

build_all

echo "Building examples iteratively..."
date
reportRun make VERBOSE=1 -j"$NPROC" build-examples-iteratively
echo

# skip unit tests if built with CUDA, unless system contains Nvidia GPUs
if [ "$BUILD_CUDA_MODULE" == "OFF" ] || nvidia-smi -L | grep -q GPU ; then
    date
    echo "try importing Open3D python package"
    test_wheel
    echo "running Open3D unit tests..."
    run_cpp_unit_tests
    echo
fi

reportJobStart "test build C++ example"
echo "test building a C++ example with installed Open3D..."
date
[ "$BUILD_CUDA_MODULE" == "OFF" ] && runExample=ON || runExample=OFF
test_cpp_example "$runExample"
echo

echo "test uninstalling Open3D..."
date
make uninstall

reportJobFinishSession
