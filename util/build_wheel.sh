#!/usr/bin/env bash

set -euo pipefail

# shellcheck source=build_scripts.sh
source "$(dirname "$0")"/build_scripts.sh

echo "$rj_startts StartJob ReportInit"
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
