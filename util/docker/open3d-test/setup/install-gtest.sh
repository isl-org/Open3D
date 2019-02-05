#!/bin/bash

# the google test library must be built with
# the same flags as the C++ code under test.

# decompress the gtest source
tar -xzf googletest-release-1.8.0.tar.gz

printf "building googletest 1.8.0 ... "
cd googletest-release-1.8.0
mkdir build
cd build
cmake .. >/dev/null 2>&1
make -j$(nproc) >/dev/null 2>&1
printf "done\n"

printf "installing ... "
cd googlemock/gtest
cp lib*.a /usr/local/lib
cd ../../../googletest
cp -r include/gtest /usr/local/include/gtest
printf "done\n"

printf "cleanup ... "
rm -rf /root/googletest-release-1.8.0*
cd ../..
printf "done\n\n"
