#!/usr/bin/env bash
# Use: install_deps_ubuntu.sh [ assume-yes ]

set -ev

SUDO=${SUDO:=sudo} # SUDO=command in docker (running as root, sudo not available)
if [ "$1" == "assume-yes" ]; then
    APT_CONFIRM="--assume-yes"
else
    APT_CONFIRM=""
fi

deps=(
    # Open3D
    xorg-dev
    libglu1-mesa-dev
    python3-dev
    # Filament build-from-source
    libsdl2-dev
    libc++-dev
    libc++abi-dev
    ninja-build
    libxi-dev
    # ML
    libtbb-dev
    # Headless rendering
    libosmesa6-dev
    # RealSense
    libudev-dev
    autoconf
    libtool
)

# Ubuntu ARM64
if [ "$(uname -m)" == "aarch64" ]; then
    # For LAPACK
    deps+=("gfortran")

    # For compiling Filament from source
    # Ubuntu 18.04 ARM64's libc++-dev and libc++abi-dev are version 6, but we need 7+.
    source /etc/lsb-release
    if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "18.04" ]; then
        deps=("${deps[@]/libc++-dev/libc++-7-dev}")
        deps=("${deps[@]/libc++abi-dev/libc++abi-7-dev}")
    fi
fi

echo "apt-get install ${deps[*]}"
$SUDO apt-get update
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}
