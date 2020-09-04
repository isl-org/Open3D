#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=build_scripts.sh
source "$(dirname "$0")"/build_scripts.sh

echo "$rj_startts StartJob ReportInit"
date
if ! nvcc --version | grep -q "release ${CUDA_VERSION[1]}" 2>/dev/null ; then
    reportRun install_cuda_toolkit with-cudnn purge-cache
    nvcc --version
fi
echo

reportJobStart "Installing Python unit test dependencies"
date
install_python_dependencies

echo "using python: $(which python)"
python --version
echo -n "Using pip: "
python -m pip --version
echo "using cmake: $(which cmake)"
cmake --version

reportJobStart "Building Open3D wheel"
build_wheel

reportJobFinishSession
