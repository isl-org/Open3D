#!/bin/bash

# install the minimal Open3D dependencies
# - xorg-dev replaced with just 4 out of ~60 dependencies:
#   - libxrandr-dev
#   - libxinerama-dev
#   - libxcursor-dev
#   - libxi-dev &&
# - by replacing xorg-dev we've:
#   - reduced the Docker image size substantially
#   - made sure libpng is not installed
apt-get update -qq
apt-get install -qq -y --no-install-recommends \
    build-essential \
    git \
    libglu1-mesa-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    tzdata >/dev/null 2>&1

# the cmake 3.1+ package has a different name on Ubuntu 14.04
if [ "${UBUNTU_VERSION}" = "14.04" ]; then
    apt-get install -qq -y --no-install-recommends cmake3 >/dev/null 2>&1;
else
    apt-get install -qq -y --no-install-recommends cmake >/dev/null 2>&1;
fi

# install googletest, work around SSL CA cert issue...
/bin/bash /root/install-gtest.sh
git config --global http.sslVerify false

# cleanup
rm /root/install-gtest.sh
rm -rf /var/lib/apt/lists/*
