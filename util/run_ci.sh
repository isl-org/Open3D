#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=build_scripts.sh
source "$(dirname "$0")"/build_scripts.sh

# disable incompatible pytorch configurations
if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
    # we need cudnn for building pytorch ops
    if ! find $(dirname $(which nvcc))/.. -name "libcudnn*"; then
        export BUILD_PYTORCH_OPS=OFF
    fi
    # pytorch 1.6 requires at least python 3.6
    if ! python -c "import sys; sys.exit(0) if sys.version_info.major==3 and sys.version_info.minor > 5 else sys.exit(1)"; then
        export BUILD_PYTORCH_OPS=OFF
    fi
fi
date

echo "$rj_startts StartJob ReportInit"
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

reportJobStart "Installing Python unit test dependencies"
date
if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    install_cuda_toolkit
fi
install_python_dependencies "${unittestDependencies:=ON}"

echo "using python: $(which python)"
python --version
echo -n "Using pip: "
python -m pip --version
if [ "$unittestDependencies" == ON ] ; then
    echo -n "Using pytest:"
    python -m pytest --version
fi
echo "using cmake: $(which cmake)"
cmake --version

build_all

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
