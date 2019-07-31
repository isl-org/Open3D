#!/usr/bin/env bash
# This script is used by CI only

# Install the latest version of CMake
# https://apt.kitware.com/
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg software-properties-common wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -

if [[ $(lsb_release -rs) == "16.04" ]]; then
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
elif [[ $(lsb_release -rs) == "18.04" ]]; then
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
else
    echo "Unsupported ubuntu version, please install latest CMake manually"
    exit 1
fi
sudo apt-get update
sudo apt-get --yes install cmake

# CMake has a different CMake installation, we need to reset path priority
# https://travis-ci.community/t/install-cmake-using-apt-again/4062
export PATH=/usr/local/bin:$PATH

./util/scripts/install-deps-ubuntu.sh
