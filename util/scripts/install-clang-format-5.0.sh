#!/bin/sh

sudo apt update
sudo apt install wget software-properties-common -y
sudo wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-5.0 main"
sudo apt update
sudo apt install clang-format-5.0 -y
clang-format-5.0 --version
