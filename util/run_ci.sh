#!/usr/bin/env bash
set -euo pipefail

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "$(dirname "$0")"/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

install_python_dependencies with-unit-test purge-cache

build_all

echo "Building examples iteratively..."
make VERBOSE=1 -j"$NPROC" build-examples-iteratively
echo

df -h

echo "Running Open3D C++ unit tests..."
run_cpp_unit_tests

# Run on GPU only. CPU versions run on Github already
if nvidia-smi >/dev/null 2>&1; then
    echo "Try importing Open3D Python package"
    test_wheel lib/python_package/pip_package/open3d*.whl
    df -h
    echo "Running Open3D Python tests..."
    run_python_tests
    echo
fi

echo "Test building a C++ example with installed Open3D..."
test_cpp_example "${runExample:=ON}"
echo

echo "Test uninstalling Open3D..."
make uninstall
