#!/bin/sh

# the google test library must be built with
# the same flags as the C++ code under test.

tar -xzf googletest-release-1.8.0.tar.gz
cd googletest-release-1.8.0
mkdir build
cd build
cmake ..
make -j
cd googlemock/gtest
cp lib*.a /usr/local/lib
cd ../../../googletest
cp -r include/gtest /usr/local/include/gtest
rm -rf /root/googletest-release-1.8.0*
cd ../..
