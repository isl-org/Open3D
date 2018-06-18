#!/bin/sh

echo
echo building...

mkdir -p build
cd build
cmake ../src
make -j

#make install

echo
