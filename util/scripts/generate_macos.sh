#!/usr/bin/env bash
set -e

echo
echo generating...

cd ../../

rm -rf build-xcode
mkdir build-xcode
cd build-xcode

# you can specify a custom install location and the python version
cmake .. -G Xcode -DCMAKE_INSTALL_PREFIX=~/open3d_install/ #-DPYTHON_EXECUTABLE=/usr/bin/python3.5

open Open3D.xcodeproj

echo
