#!/bin/sh

cd /usr/local/lib
sudo rm -f lib*.a
cd /usr/local/include
sudo rm -rf gtest

rm -rf /tmp/googletest-release-1.8.0
rm -rf /tmp/release-1.8.0.tar.gz
