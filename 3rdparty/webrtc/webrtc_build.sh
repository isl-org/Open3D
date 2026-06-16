#!/usr/bin/env bash
set -euox pipefail

# Builds WebRTC static libraries for Open3D (Ubuntu/macOS). Windows: webrtc.yml
#
# Layout (default: repo parent holds depot_tools + webrtc):
#   <work>/
#   ├── Open3D/          # this repository
#   ├── depot_tools/
#   └── webrtc/
#       └── src/
#
# Usage:
#   cd /path/to/Open3D
#   export WEBRTC_COMMIT=...   # optional
#   source 3rdparty/webrtc/webrtc_build.sh
#   install_dependencies_ubuntu  # optional on Ubuntu
#   download_webrtc_sources
#   build_webrtc

# libwebrtc-bin M149 / Open3D target milestone
WEBRTC_COMMIT=${WEBRTC_COMMIT:-e8b4d4c5952a8fb7b35c2a6cba4e8c3de2ea2e1e}
# Optional pin; unset uses depot_tools HEAD.
DEPOT_TOOLS_COMMIT=${DEPOT_TOOLS_COMMIT:-}

GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI:-1}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)}
SUDO=${SUDO:-sudo}

webrtc_work_root() {
    if [[ -n "${WEBRTC_WORK_ROOT:-}" ]]; then
        echo "$WEBRTC_WORK_ROOT"
    else
        dirname "$PWD"
    fi
}

webrtc_setup_path() {
    export PATH="$(webrtc_work_root)/depot_tools:${PATH}"
    export DEPOT_TOOLS_UPDATE=0
}

install_dependencies_ubuntu() {
    options="$(echo "$@" | tr ' ' '|')"
    $SUDO apt-get update
    $SUDO apt-get install -y \
        apt-transport-https \
        build-essential \
        ca-certificates \
        clang \
        git \
        gnupg \
        libglib2.0-dev \
        libnss3-dev \
        libgtk-3-dev \
        ninja-build \
        python3 \
        python3-pip \
        python3-setuptools \
        pkg-config \
        software-properties-common \
        tree \
        curl
    if ! command -v cmake >/dev/null 2>&1 || [[ $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+') < "3.18" ]]; then
        curl https://apt.kitware.com/keys/kitware-archive-latest.asc \
            2>/dev/null | gpg --dearmor - |
            $SUDO tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
        source <(grep VERSION_CODENAME /etc/os-release)
        $SUDO apt-add-repository --yes "deb https://apt.kitware.com/ubuntu/ $VERSION_CODENAME main"
        $SUDO apt-get update
        $SUDO apt-get --yes install cmake
    fi
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        $SUDO apt-get clean
        $SUDO rm -rf /var/lib/apt/lists/*
    fi
}

download_webrtc_sources() {
    local root
    root="$(webrtc_work_root)"
    pushd "$root"
    if [[ ! -d depot_tools ]]; then
        git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git
    fi
    if [[ -n "$DEPOT_TOOLS_COMMIT" ]]; then
        git -C depot_tools checkout "$DEPOT_TOOLS_COMMIT"
    fi
    webrtc_setup_path
    command -V fetch

    if [[ ! -d webrtc/src ]]; then
        mkdir -p webrtc
        pushd webrtc
        fetch --nohooks webrtc
        popd
    fi

    git -C webrtc/src checkout "$WEBRTC_COMMIT"
    git -C webrtc/src submodule update --init --recursive
    pushd webrtc
    gclient sync -D --force --reset --no-history
    popd
    popd
}

build_webrtc() {
    local root open3d_dir
    open3d_dir="$PWD"
    root="$(webrtc_work_root)"
    webrtc_setup_path

    cp "$open3d_dir"/3rdparty/webrtc/{CMakeLists.txt,webrtc_common.cmake} "$root/webrtc/"
    bash "$open3d_dir"/3rdparty/webrtc/apply_webrtc_patches.sh \
        "$open3d_dir" "$root/webrtc/src"

    WEBRTC_COMMIT_SHORT=$(git -C "$root/webrtc/src" rev-parse --short=7 HEAD)

    mkdir -p "$root/webrtc/build"
    pushd "$root/webrtc/build"
    cmake -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$root/webrtc_release" \
        -DGLIBCXX_USE_CXX11_ABI="${GLIBCXX_USE_CXX11_ABI}" \
        ..
    ninja -j"${NPROC}"
    ninja install
    popd

    pushd "$root"
    tree -L 2 webrtc_release || ls -la webrtc_release
    if [[ $(uname -s) == 'Linux' ]]; then
        tar -czf \
            "$open3d_dir/webrtc_${WEBRTC_COMMIT_SHORT}_linux_cxx-abi-${GLIBCXX_USE_CXX11_ABI}.tar.gz" \
            -C "$root/webrtc_release" .
    elif [[ $(uname -s) == 'Darwin' ]]; then
        tar -czf \
            "$open3d_dir/webrtc_${WEBRTC_COMMIT_SHORT}_macos_arm64.tar.gz" \
            -C "$root/webrtc_release" .
    fi
    popd

    webrtc_package=$(ls "$open3d_dir"/webrtc_*.tar.gz | tail -1)
    cmake -E sha256sum "$webrtc_package" | tee "$open3d_dir/checksum_${webrtc_package##*/}" | sed 's|.*/||'
    ls -alh "$webrtc_package"
}
