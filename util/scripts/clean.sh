#!/usr/bin/env bash
set -e

echo
echo cleaning open3d_install...

cd ../../
cd build
make uninstall
cd ..
rm -rf ~/open3d_install/

echo
echo cleaning...

rm -rf build

# remove the CMake cache
find . -name CMakeFiles -type d -exec rm -rf {} +
find . -name CMakeCache.txt -exec rm -rf {} +
find . -name cmake_install.cmake -exec rm -rf {} +

echo
