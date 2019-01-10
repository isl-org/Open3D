#!/bin/bash

python --version
cmake --version

git submodule update --init --recursive

# Build and install to CMake's default install path
mkdir build
cd build
if [ "$BUILD_DEPENDENCY_FROM_SOURCE" == "OFF" ]; then
    cmake -DBUILD_SHARED_LIBS=$SHARED -DBUILD_UNIT_TESTS=ON ..
else
    cmake -DBUILD_SHARED_LIBS=$SHARED -DBUILD_UNIT_TESTS=ON -DBUILD_EIGEN3=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JPEG=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON ..
fi

# make -j brings 'virtual memory exhausted: Cannot allocate memory' message
# this is presumably due to limited memory space of travis-ci
# also set the time limit to 30 mins, this is a fix to issue #557
sudo make install -j$(nproc)

./bin/unitTests

test=`cmake --find-package -DNAME=Open3D -DCOMPILER_ID=GNU -DLANGUAGE=C -DMODE=EXIST`
if [ "$test" == "Open3D found." ]; then
    echo "PASSED find_package(Open3D) in default install path.";
else
    echo "FAILED find_package(Open3D) in default install path.";
    exit 1;
fi

sudo make uninstall

# Reconfig and install in a specified CMAKE_INSTALL_PREFIX
if [ "$BUILD_DEPENDENCY_FROM_SOURCE" == "OFF" ]; then
    cmake -DBUILD_SHARED_LIBS=$SHARED -DBUILD_UNIT_TESTS=ON -DCMAKE_INSTALL_PREFIX=~/open3d_install ..;
else
    cmake -DBUILD_SHARED_LIBS=$SHARED -DBUILD_UNIT_TESTS=ON -DBUILD_EIGEN3=ON -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JPEG=ON -DBUILD_JSONCPP=ON -DBUILD_PNG=ON -DCMAKE_INSTALL_PREFIX=~/open3d_install ..;
fi

sudo make install -j$(nproc)

test=`cmake --find-package -DNAME=Open3D -DCOMPILER_ID=GNU -DLANGUAGE=C -DMODE=EXIST -DCMAKE_PREFIX_PATH="~/open3d_install/lib/cmake"`
if [ "$test" == "Open3D found." ]; then
    echo "PASSED find_package(Open3D) in specified install path.";
else
    echo "FAILED find_package(Open3D) in specified install path.";
    exit 1;
fi

# Test building example with installed Open3D
cd ../docs/_static/C++
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=~/open3d_install ..
make
./TestVisualizer
