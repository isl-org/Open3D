#!/usr/bin/env bash
# Use: install_deps_ubuntu.sh [ assume-yes ]

set -evo pipefail

SUDO=${SUDO:=sudo} # SUDO=command in docker (running as root, sudo not available)
options="$(echo "$@" | tr ' ' '|')"
APT_CONFIRM=""
if [[ "assume-yes" =~ ^($options)$ ]]; then
    APT_CONFIRM="--assume-yes"
fi
deps=(
    git
    # Open3D
    xorg-dev
    libxcb-shm0
    libglu1-mesa-dev
    python3-dev
    libssl-dev
    # Prebuilt Filament runtime
    libsdl2-dev
    libxi-dev
    # Compute shaders
    glslang-tools
    # ML
    libtbb-dev
    # Headless / offscreen GPU rendering (EGL)
    libegl1-mesa-dev
    mesa-vulkan-drivers
    # RealSense
    libudev-dev
    libusb-1.0-0-dev
    autoconf
    libtool
)

# Special case for ARM64
if [ "$(uname -m)" == "aarch64" ]; then
    # For compiling LAPACK in OpenBLAS
    deps+=("gfortran")
fi

echo "apt-get install ${deps[*]}"
$SUDO apt-get update
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}
