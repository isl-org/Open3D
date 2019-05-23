#!/usr/bin/env bash
set -e

echo
echo building...

./generate.sh

cd ../../
cd build
make -j

# this will copy the libs and headers to ~/open3d_install/lib & ~/open3d_install/include
# make install

echo
