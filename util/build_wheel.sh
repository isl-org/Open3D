#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=build_scripts.sh
source "$(dirname "$0")"/build_scripts.sh

if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    # disable pytorch build if CUDA is enabled for now until the problem with caffe2 and cudnn is solved
    BUILD_PYTORCH_OPS="OFF"
fi

echo "$rj_startts StartJob ReportInit"
reportJobStart "Installing Python unit test dependencies"
date
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

reportJobStart "Building Open3D wheel"
build_wheel

reportJobFinishSession
