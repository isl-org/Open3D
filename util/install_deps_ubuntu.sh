#!/usr/bin/env bash
# Install Open3D build dependencies from Ubuntu repositories
# CUDA (v10.1) and CUDNN (v7.6.5) are optional dependencies and are not
# installed here
# Use: install_deps_ubuntu.sh [ assume-yes ]

set -ev

SUDO=${SUDO:=sudo} # SUDO=command in docker (running as root, sudo not available)
if [ "$1" == "assume-yes" ]; then
    APT_CONFIRM="--assume-yes"
else
    APT_CONFIRM=""
fi

dependencies=(
    # Open3D deps
    xorg-dev
    libglu1-mesa-dev
    python3-dev
    # Filament build-from-source deps
    libsdl2-dev
    libc++-7-dev
    libc++abi-7-dev
    ninja-build
    libxi-dev
    # ML deps
    libtbb-dev
    # Headless rendering deps
    libosmesa6-dev
    # RealSense deps
    libudev-dev
    autoconf
    libtool
)

$SUDO apt-get update
for package in "${dependencies[@]}"; do
    $SUDO apt-get install "$APT_CONFIRM" "$package"
done
