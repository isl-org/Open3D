#!/usr/bin/env bash
set -euo pipefail

pushd /root/Open3D
mkdir -p build
pushd build

# Configure
cmake -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_RPC_INTERFACE=OFF \
      -DCMAKE_INSTALL_PREFIX=/root/open3d_install \
      -DBUILD_UNIT_TESTS=ON \
      -DBUILD_BENCHMARKS=ON \
      -DBUILD_EXAMPLES=ON \
      ..

# Build
make -j$(nproc)
make -j$(nproc) install-pip-package

CUDA_VISIBLE_DEVICES=0 ./bin/tests
CUDA_VISIBLE_DEVICES=0 pytest ../python/test

popd
popd
