#!/bin/sh

wget https://github.com/abseil/googletest/archive/release-1.8.0.tar.gz -O /tmp/release-1.8.0.tar.gz
cd /tmp/
tar -xzvf /tmp/release-1.8.0.tar.gz
cd /tmp/googletest-release-1.8.0
mkdir build
cd build
cmake ..
make -j
cd googlemock/gtest
cp lib*.a /usr/local/lib
cd ../../../googletest
cp -r include/gtest /usr/local/include/gtest
cd ../..
