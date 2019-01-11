#!/bin/bash

python --version
cmake --version
echo

INSTALL_DIR=~/open3d_install

# configure the project
# install to a specified CMAKE_INSTALL_PREFIX
mkdir build
cd build
if [ "$BUILD_DEPENDENCY_FROM_SOURCE" == "OFF" ]; then
    cmake -DBUILD_SHARED_LIBS=$SHARED \
          -DBUILD_UNIT_TESTS=ON
          -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
          ..
else
    cmake -DBUILD_SHARED_LIBS=$SHARED \
          -DBUILD_UNIT_TESTS=ON \
          -DBUILD_EIGEN3=ON \
          -DBUILD_GLEW=ON \
          -DBUILD_GLFW=ON \
          -DBUILD_JPEG=ON \
          -DBUILD_JSONCPP=ON \
          -DBUILD_PNG=ON \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
          ..
fi
echo

# build Open3D
make install -j$(nproc)
echo

# run Open3D unit tests
./bin/unitTests

test=`cmake --find-package \
            -DNAME=Open3D \
            -DCOMPILER_ID=GNU \
            -DLANGUAGE=C \
            -DMODE=EXIST \
            -DCMAKE_PREFIX_PATH="${INSTALL_DIR}/lib/cmake"`
if [ "$test" == "Open3D found." ]; then
    echo "PASSED find_package(Open3D) in specified install path.";
else
    echo "FAILED find_package(Open3D) in specified install path.";
    exit 1;
fi
echo

# Test building a C++ example with installed Open3D
cd ../docs/_static/C++
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} ..
make
./TestVisualizer
echo

# cleanup the C++ example
cd ../
rm -rf build

# uninstall Open3D
cd ../../../build
make uninstall

# cleanup Open3D
cd ../
rm -rf build
rm -rf ${INSTALL_DIR}
echo
