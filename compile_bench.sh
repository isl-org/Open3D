#!/usr/bin/env bash

ccache --clear
rm -rf build
mkdir build
cd build

echo "=========================="
echo "Configure"
echo "=========================="
time cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_ISPC_MODULE=OFF \
      -DBUILD_CUDA_MODULE=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DCMAKE_INSTALL_PREFIX=~/open3d_install \
      -DUSE_BLAS=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF ..

echo "=========================="
echo "First make"
echo "=========================="
time make -j

echo "=========================="
echo "Second make (with ccache)"
echo "=========================="
cd ..
rm -rf build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_ISPC_MODULE=OFF \
      -DBUILD_CUDA_MODULE=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DCMAKE_INSTALL_PREFIX=~/open3d_install \
      -DUSE_BLAS=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF ..
time make -j

echo "=========================="
echo "make install-pip-package"
echo "=========================="
time make install-pip-package -j
