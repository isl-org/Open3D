#!/usr/bin/env bash
# Use: install_deps_ubuntu.sh [ assume-yes ] [ no-filament-deps ]

set -ev

SUDO=${SUDO:=sudo} # SUDO=command in docker (running as root, sudo not available)
options="$(echo "$@" | tr ' ' '|')"
APT_CONFIRM=""
if [[ "assume-yes" =~ ^($options)$ ]]; then
    APT_CONFIRM="--assume-yes"
fi
FILAMENT_DEPS="yes"
if [[ "no-filament-deps" =~ ^($options)$ ]]; then
    FILAMENT_DEPS=""
fi

deps=(
    git
    # Open3D
    xorg-dev
    libxcb-shm0
    libglu1-mesa-dev
    python3-dev
    libssl-dev
    # filament linking
    libc++-dev
    libc++abi-dev
    libsdl2-dev
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

if [[ "$FILAMENT_DEPS" ]]; then     # Filament build-from-source
    deps+=(clang
        ninja-build
    )
fi

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)
# To avoid dependence on libunwind, we don't want to use clang / libc++ versions later than 11.
# Ubuntu 20.04's has versions 8, 10 or 12 while Ubuntu 22.04 has versions 11 and later.
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "20.04" ]; then
    deps=("${deps[@]/clang/clang-10}")
    deps=("${deps[@]/libc++-dev/libc++-10-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-10-dev}")
fi
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "22.04" ]; then
    deps=("${deps[@]/clang/clang-11}")
    deps=("${deps[@]/libc++-dev/libc++-11-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-11-dev}")
fi
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "24.04" ]; then
    deps=("${deps[@]/clang/clang-14}")
    deps=("${deps[@]/libc++-dev/libc++-14-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-14-dev}")
fi

# Special case for ARM64
if [ "$(uname -m)" == "aarch64" ]; then
    # For compiling LAPACK in OpenBLAS
    deps+=("gfortran")
fi

echo "apt-get install ${deps[*]}"
$SUDO apt-get update
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}
