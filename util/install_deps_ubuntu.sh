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
    libxcb-shm0
    libglu1-mesa-dev
    python3-dev
    # Filament build-from-source
    clang
    libc++-dev
    libc++abi-dev
    libsdl2-dev
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

# Filament build-from-source
# Ubuntu 18.04's clang/libc++-dev/libc++abi-dev are version 6, but we need 7+.
source /etc/lsb-release
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "18.04" ]; then
    deps=("${deps[@]/clang/clang-7}")
    deps=("${deps[@]/libc++-dev/libc++-7-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-7-dev}")
fi

# Ubuntu ARM64 requires gfortran to compile OpenBLAS.
if [ "$(uname -m)" == "aarch64" ]; then
    # For LAPACK
    deps+=("gfortran")
fi

echo "apt-get install ${deps[*]}"
$SUDO apt-get update
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}
