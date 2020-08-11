#!/usr/bin/env bash

# Run this script from the CMakeLists.txt (top-level) directory

# exit when any command fails
set -e

# Use SUDO=command to run in docker (user is root, sudo is not installed)
SUDO=${SUDO:=sudo}

$SUDO apt-get update
$SUDO apt-get -y install git software-properties-common
echo "Installing Python3 and setting as default python"
$SUDO apt-get --yes install python3 python3-pip
if  ! which python || python -V 2>/dev/null | grep -q ' 2.' ; then
    $SUDO ln -s /usr/bin/python3.6 /usr/local/bin/python
    $SUDO ln -s /usr/bin/python3.6-config /usr/local/bin/python-config
    $SUDO ln -s /usr/bin/pip3 /usr/local/bin/pip
fi

# cmake not installed or too old
if ! which cmake || cmake -P CMakeLists.txt 2>&1 | grep -q "or higher is required"  ; then
    echo 'Installing backported cmake from https://apt.kitware.com/'
    $SUDO apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
    $SUDO apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    $SUDO apt-get --yes install cmake
fi

if [ -n "${NVIDIA_DRIVER_VERSION}" ] ; then
    echo "Installing NVIDIA drivers."
    $SUDO apt-get --yes "install nvidia-driver-${NVIDIA_DRIVER_VERSION}"
fi

# Cleanup apt cache (for docker)
$SUDO rm -rf /var/lib/apt/lists/*
