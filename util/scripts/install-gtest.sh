#!/bin/sh

wget https://github.com/google/googletest/archive/release-1.8.0.tar.gz -O /tmp/release-1.8.0.tar.gz
cd /tmp/
tar -xzvf /tmp/release-1.8.0.tar.gz
cd /tmp/googletest-release-1.8.0
mkdir build
cd build
cmake ..
make -j
cd googlemock/gtest
sudo cp lib*.a /usr/local/lib
cd ../../../googletest
sudo cp -r include/gtest /usr/local/include/gtest
cd ../..
