#!/usr/bin/env bash
set -e

echo
echo generating...

cd ../../
mkdir -p build
cd build

# you can specify a custom install location and the python version
cmake -DCMAKE_INSTALL_PREFIX=~/open3d_install .. #-DPYTHON_EXECUTABLE=/usr/bin/python3.5

echo
