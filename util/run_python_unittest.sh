#!/usr/bin/env bash
set -euo pipefail

PWD="$(cd $(dirname $0); pwd)"
OPEN3D_ROOT="${PWD}/../"

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "${PWD}/ci_utils.sh"

# # Run on CPU versions
echo "Try importing Open3D CPU Python package"
cpu_wheel_path=$(ls ${OPEN3D_ROOT}/build/lib/python_package/pip_package/open3d_cpu*.whl)
test_wheel $cpu_wheel_path
echo "Running Open3D CPU Python tests..."
run_python_tests
echo

if nvidia-smi >/dev/null 2>&1; then
    # Run on GPU versions
    echo "Try importing Open3D GPU Python package"
    gpu_wheel_path=$(ls ${OPEN3D_ROOT}/build/lib/python_package/pip_package/open3d*.whl | grep -v $cpu_wheel_path)
    test_wheel $gpu_wheel_path
    echo "Running Open3D GPU Python tests..."
    run_python_tests
    echo
fi
