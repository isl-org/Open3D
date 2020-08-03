#!/usr/bin/env bash

# Run this script from the CMakeLists.txt (top-level) directory

# exit when any command fails
set -e

sudo apt-get update
echo "Installing Python3 and setting as default python"
sudo apt-get --yes install python3 python3-pip
if  python -V | grep -q ' 2.' ; then
    sudo ln -s /usr/bin/python3.6 /usr/local/bin/python
    sudo ln -s /usr/bin/python3.6-config /usr/local/bin/python-config
    sudo ln -s /usr/bin/pip3 /usr/local/bin/pip
fi

# cmake not installed or too old
if ! which cmake || cmake -P CMakeLists.txt 2>&1 | grep -q "or higher is required"  ; then
    echo 'Installing backported cmake from https://apt.kitware.com/'
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    2>/dev/null \
    | gpg --dearmor - \
    | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    sudo apt-get --yes install cmake 
fi

if [ -n "${NVIDIA_DRIVER_VERSION}" ] ; then
    echo "Installing NVIDIA drivers."
    sudo apt-get --yes install nvidia-driver-${NVIDIA_DRIVER_VERSION}
fi

# Cleanup apt cache (for docker)
sudo rm -rf /var/lib/apt/lists/*
