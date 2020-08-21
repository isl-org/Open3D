#!/usr/bin/env bash
set -ev

SUDO=${SUDO:=sudo}      # SUDO=command in docker (running as root, sudo not available)
BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE:=OFF}     # Is CUDA needed?
NEED_CUDNN=${NEED_CUDNN:=$BUILD_CUDA_MODULE}    # Is cuDNN needed? Default is
                                                # same as CUDA
CUDA_VERSION=("10-1" "10.1")
CUDNN_MAJOR_VERSION=7
CUDNN_VERSION="7.6.5.32-1+cuda10.1"

if [ "$1" == "assume-yes" ]; then
    APT_ARG="--yes"
else
    APT_ARG=""
fi

packages=(
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
)

if [ $BUILD_CUDA_MODULE == ON ] ; then
    if ! which nvcc >/dev/null ; then
        $SUDO apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /"
        packages=("${packages[@]}" "cuda-toolkit-${CUDA_VERSION[0]}")
        echo "Add CUDA executables location to PATH in /etc/environment"
        echo "(system) or ~/.bashrc (user) and restart your shell:"
        echo "# CUDA installation path"
        echo "export PATH=/usr/local/cuda-${CUDA_VERSION[1]}/bin\${PATH:+:\${PATH}}"
    fi
    if [ $NEED_CUDNN == ON ] ; then
        $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"
        packages=("${packages[@]}" \
            "libcudnn${CUDNN_MAJOR_VERSION}=$CUDNN_VERSION" \
            "libcudnn${CUDNN_MAJOR_VERSION}-dev=$CUDNN_VERSION" \
        )
    fi
fi

$SUDO apt-get update
for package in "${packages[@]}" ; do
    $SUDO apt-get install "$APT_ARG" "$package"
done
