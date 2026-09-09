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

if [[ "$FILAMENT_DEPS" ]]; then
    # Filament v1.76 source builds require Clang 17 for C++20 ranges.
    deps+=(clang ninja-build)
fi

# Special case for ARM64
if [ "$(uname -m)" == "aarch64" ]; then
    # For compiling LAPACK in OpenBLAS
    deps+=("gfortran")
fi

source /etc/os-release
if [[ "$ID" == "ubuntu" && "$VERSION_ID" == "22.04" ]]; then
    # Ubuntu 22.04 does not provide the required LLVM 17 packages.
    $SUDO apt-get update
    $SUDO apt-get install ${APT_CONFIRM} ca-certificates gnupg wget
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key |
        gpg --dearmor |
        $SUDO tee /usr/share/keyrings/apt.llvm.org.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/apt.llvm.org.gpg] https://apt.llvm.org/jammy/ llvm-toolchain-jammy-17 main" |
        $SUDO tee /etc/apt/sources.list.d/llvm-17.list >/dev/null
    deps=("${deps[@]/clang/clang-17}")
    deps=("${deps[@]/libc++-dev/libc++-17-dev}")
    deps=("${deps[@]/libc++abi-dev/libc++abi-17-dev}")
fi

echo "apt-get install ${deps[*]}"
$SUDO apt-get update
$SUDO apt-get install ${APT_CONFIRM} ${deps[*]}
