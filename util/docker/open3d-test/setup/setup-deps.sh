#!/bin/bash

# install Open3D dependencies
apt-get update -qq
apt-get install -qq -y --no-install-recommends \
    libeigen3-dev \
    libglew-dev \
    libjsoncpp-dev \
    libpng-dev >/dev/null 2>&1

# GLFW is not available as a package on Ubuntu 14.04
if [ "${UBUNTU_VERSION}" != "14.04" ]; then
    apt-get install -qq -y --no-install-recommends libglfw3-dev >/dev/null 2>&1
fi

# cleanup
rm -rf /var/lib/apt/lists/*
