#!/usr/bin/env bash
set -euox pipefail

# This script builds WebRTC for Open3D for Ubuntu and macOS. For Windows, see
# .github/workflows/webrtc.yml
#
# Usage:
# $ bash # Start a new shell
# Specify custom configuration by exporting environment variables
# GLIBCXX_USE_CXX11_ABI, WEBRTC_COMMIT and DEPOT_TOOLS_COMMIT, if required.
# $ source 3rdparty/webrtc/webrtc_build.sh
# $ install_dependencies_ubuntu   # Ubuntu only
# $ download_webrtc_sources
# $ build_webrtc
# A webrtc_<commit>_platform.tar.gz file will be created that can be used to
# build Open3D with WebRTC support.
#
# Procedure:
#
# 1) Download depot_tools, webrtc to following directories:
#    ├── Oepn3D
#    ├── depot_tools
#    └── webrtc
#        ├── .gclient
#        └── src
#
# 2) depot_tools and webrtc have compatible versions, see:
#    https://chromium.googlesource.com/chromium/src/+/master/docs/building_old_revisions.md
#
# 3) Apply the following patch to enable GLIBCXX_USE_CXX11_ABI selection:
#    - 0001-build-enable-rtc_use_cxx11_abi-option.patch        # apply to webrtc/src
#    - 0001-src-enable-rtc_use_cxx11_abi-option.patch          # apply to webrtc/src/build
#    - 0001-third_party-enable-rtc_use_cxx11_abi-option.patch  # apply to webrtc/src/third_party
#    Note that these patches may or may not be compatible with your custom
#    WebRTC commits. You may have to patch them manually.

# Date: Wed Apr 7 19:12:13 2021 +0200
WEBRTC_COMMIT=${WEBRTC_COMMIT:-60e674842ebae283cc6b2627f4b6f2f8186f3317}
# Date: Wed Apr 7 21:35:29 2021 +0000
DEPOT_TOOLS_COMMIT=${DEPOT_TOOLS_COMMIT:-e1a98941d3ab10549be6d82d0686bb0fb91ec903}

GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI:-0}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)} # POSIX: MacOS + Linux
SUDO=${SUDO:-sudo}                           # Set to command if running inside docker
export PATH="$PWD/../depot_tools":${PATH}    # $(basename $PWD) == Open3D
export DEPOT_TOOLS_UPDATE=0

install_dependencies_ubuntu() {
    options="$(echo "$@" | tr ' ' '|')"
    # Dependencies
    # python*       : resolve ImportError: No module named pkg_resources
    # libglib2.0-dev: resolve pkg_config("glib")
    $SUDO apt-get update
    $SUDO apt-get install -y \
        apt-transport-https \
        build-essential \
        ca-certificates \
        git \
        gnupg \
        libglib2.0-dev \
        python \
        python-pip \
        python-setuptools \
        python-wheel \
        software-properties-common \
        tree \
        curl
    curl https://apt.kitware.com/keys/kitware-archive-latest.asc \
        2>/dev/null | gpg --dearmor - |
        $SUDO sed -n 'w /etc/apt/trusted.gpg.d/kitware.gpg' # Write to file, no stdout
    source <(grep VERSION_CODENAME /etc/os-release)
    $SUDO apt-add-repository --yes "deb https://apt.kitware.com/ubuntu/ $VERSION_CODENAME main"
    $SUDO apt-get update
    $SUDO apt-get --yes install cmake
    cmake --version >/dev/null
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        $SUDO apt-get clean
        $SUDO rm -rf /var/lib/apt/lists/*
    fi
}

download_webrtc_sources() {
    # PWD=Open3D
    pushd ..
    echo Get depot_tools
    git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
    git -C depot_tools checkout $DEPOT_TOOLS_COMMIT
    command -V fetch

    echo Get WebRTC
    mkdir webrtc
    cd webrtc
    fetch webrtc

    # Checkout to a specific version
    # Ref: https://chromium.googlesource.com/chromium/src/+/master/docs/building_old_revisions.md
    git -C src checkout $WEBRTC_COMMIT
    git -C src submodule update --init --recursive
    echo gclient sync
    gclient sync -D --force --reset
    cd ..
    echo random.org
    curl "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" -o skipcache
    popd
}

build_webrtc() {
    # PWD=Open3D
    OPEN3D_DIR="$PWD"
    echo Apply patches
    cp 3rdparty/webrtc/{CMakeLists.txt,webrtc_common.cmake} ../webrtc
    git -C ../webrtc/src apply \
        "$OPEN3D_DIR"/3rdparty/webrtc/0001-src-enable-rtc_use_cxx11_abi-option.patch
    git -C ../webrtc/src/build apply \
        "$OPEN3D_DIR"/3rdparty/webrtc/0001-build-enable-rtc_use_cxx11_abi-option.patch
    git -C ../webrtc/src/third_party apply \
        "$OPEN3D_DIR"/3rdparty/webrtc/0001-third_party-enable-rtc_use_cxx11_abi-option.patch
    WEBRTC_COMMIT_SHORT=$(git -C ../webrtc/src rev-parse --short=7 HEAD)

    echo Build WebRTC
    mkdir ../webrtc/build
    pushd ../webrtc/build
    cmake -DCMAKE_INSTALL_PREFIX=../../webrtc_release \
        -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \
        ..
    make -j$NPROC
    make install
    popd # PWD=Open3D
    pushd ..
    tree -L 2 webrtc_release || ls webrtc_release/*

    echo Package WebRTC
    if [[ $(uname -s) == 'Linux' ]]; then
        tar -czf \
            "$OPEN3D_DIR/webrtc_${WEBRTC_COMMIT_SHORT}_linux_cxx-abi-${GLIBCXX_USE_CXX11_ABI}.tar.gz" \
            webrtc_release
    elif [[ $(uname -s) == 'Darwin' ]]; then
        tar -czf \
            "$OPEN3D_DIR/webrtc_${WEBRTC_COMMIT_SHORT}_macos.tar.gz" \
            webrtc_release
    fi
    popd # PWD=Open3D
    webrtc_package=$(ls webrtc_*.tar.gz)
    cmake -E sha256sum "$webrtc_package" | tee "checksum_${webrtc_package%%.*}.txt"
    ls -alh "$webrtc_package"
}
