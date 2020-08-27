#!/usr/bin/env bash
# Install Open3D build dependencies from Ubuntu repositories
# Use: install_deps_ubuntu.sh [ assume-yes ] [ install-cuda ]

set -ev

SUDO=${SUDO:=sudo}      # SUDO=command in docker (running as root, sudo not available)
APT_CONFIRM=""
INSTALL_CUDA=""
for arg in "$@" ; do
    case $arg in
        "assume-yes") APT_CONFIRM="--assume-yes" ;;
        "install-cuda") INSTALL_CUDA=ON ;;
        *) echo "Use: install_deps_ubuntu.sh [ assume-yes ] [ install-cuda ]"
           echo "install-cuda: Install CUDA and cuDNN"
           exit 1
           ;;
    esac
done

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
)

$SUDO apt-get update
for package in "${dependencies[@]}" ; do
    $SUDO apt-get install "$APT_CONFIRM" "$package"
done

######################################################
# The following scripts are disabled by default.
# They are intended to be used by the Open3D CI system only.
######################################################

CUDA_VERSION=("10-1" "10.1")
CUDNN_MAJOR_VERSION=7
CUDNN_VERSION="7.6.5.32-1+cuda10.1"

if [ $INSTALL_CUDA == ON ] ; then
    if ! nvcc --version >/dev/null ; then
        echo "Installing CUDA ${CUDA_VERSION[1]} with apt ..."
        $SUDO apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /"
        $SUDO apt-get install "$APT_CONFIRM" "cuda-toolkit-${CUDA_VERSION[0]}"
        echo "Add CUDA executables location to PATH in /etc/environment"
        echo "(system) or ~/.bashrc (user) and restart your shell:"
        echo "export PATH=/usr/local/cuda-${CUDA_VERSION[1]}/bin\${PATH:+:\${PATH}}"
        echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION[1]}/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
    else
        echo "CUDA found:"
        nvcc --version
    fi
    echo "Installing cuDNN ${CUDNN_VERSION} with apt ..."
    $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"
    $SUDO apt-get install "$APT_CONFIRM" \
        "libcudnn${CUDNN_MAJOR_VERSION}=$CUDNN_VERSION" \
        "libcudnn${CUDNN_MAJOR_VERSION}-dev=$CUDNN_VERSION"
fi
