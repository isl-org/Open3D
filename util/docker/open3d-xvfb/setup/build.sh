#!/bin/sh

echo
echo building...

mkdir -p build
cd build
cmake -DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF ../src
make -j

#make install

echo
