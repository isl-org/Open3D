#!/usr/bin/env bash

# Run this script from the CMakeLists.txt (top-level) directory

# exit when any command fails
set -eo pipefail

# Use SUDO=command to run in docker (user is root, sudo is not installed)
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs)"}

$SUDO apt-get update
$SUDO apt-get --yes install git software-properties-common
echo "Installing Python3 and setting as default python"
$SUDO apt-get --yes --no-install-recommends install python3 python3-pip python3-setuptools
if ! which python || python -V 2>/dev/null | grep -q ' 2.'; then
    echo 'Making python3 the default python'
    $SUDO ln -s /usr/bin/python3 /usr/local/bin/python
    $SUDO ln -s /usr/bin/python3-config /usr/local/bin/python-config
    $SUDO ln -s /usr/bin/pip3 /usr/local/bin/pip
fi

# cmake not installed or too old
if ! which cmake || cmake -P CMakeLists.txt 2>&1 | grep -q "or higher is required"; then
    echo "Installing backported cmake from https://apt.kitware.com/ for Ubuntu $UBUNTU_VERSION"
    $SUDO apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
    $SUDO apt-add-repository --yes \
        "deb https://apt.kitware.com/ubuntu/ $UBUNTU_VERSION main"
    $SUDO apt-get --yes --no-install-recommends install cmake
fi

if [ -n "${NVIDIA_DRIVER_VERSION}" ]; then
    echo "Installing NVIDIA drivers."
    $SUDO apt-get --yes --no-install-recommends install "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
fi

# Cleanup apt cache (for docker)
$SUDO apt-get clean
$SUDO rm -rf /var/lib/apt/lists/*
