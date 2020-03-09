#!/usr/bin/env bash
set -e

python --version
cmake --version
echo

OPEN3D_INSTALL_DIR=~/open3d_install

echo "cmake configure the Open3D project..."
date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    pip install --upgrade pip
    pip install tensorflow==2.0.0
fi
mkdir build
cd build
if [ "$BUILD_DEPENDENCY_FROM_SOURCE" == "OFF" ]; then
    cmake -DBUILD_SHARED_LIBS=$SHARED \
        -DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS \
        -DBUILD_UNIT_TESTS=ON \
        -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
        -DPYTHON_EXECUTABLE=$(which python) \
        ..
else
    cmake -DBUILD_SHARED_LIBS=$SHARED \
        -DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS \
        -DBUILD_UNIT_TESTS=ON \
        -DBUILD_EIGEN3=ON \
        -DBUILD_GLEW=ON \
        -DBUILD_GLFW=ON \
        -DBUILD_PNG=ON \
        -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
        -DPYTHON_EXECUTABLE=$(which python) \
        ..
fi
echo

echo "build & install Open3D..."
date
make install -j$NPROC
echo

echo "running the Open3D unit tests..."
date
./bin/unitTests
echo

echo "test find_package(Open3D)..."
date
test=$(cmake --find-package \
    -DNAME=Open3D \
    -DCOMPILER_ID=GNU \
    -DLANGUAGE=C \
    -DMODE=EXIST \
    -DCMAKE_PREFIX_PATH="${OPEN3D_INSTALL_DIR}/lib/cmake")
if [ "$test" == "Open3D found." ]; then
    echo "PASSED find_package(Open3D) in specified install path."
else
    echo "FAILED find_package(Open3D) in specified install path."
    exit 1
fi
echo

echo "test building a C++ example with installed Open3D..."
date
cd ../docs/_static/C++
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
make
./TestVisualizer
echo

echo "cleanup the C++ example..."
date
cd ../
rm -rf build

echo "uninstall Open3D..."
date
cd ../../../build
make uninstall

echo "cleanup Open3D..."
date
cd ../
rm -rf build
rm -rf ${OPEN3D_INSTALL_DIR}
echo
