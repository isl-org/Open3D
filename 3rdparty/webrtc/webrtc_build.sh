#!/usr/bin/env bash
# Build WebRTC static libraries for Open3D (Ubuntu/macOS).
# Windows uses download_webrtc_sources() from this file via Git Bash;
# the cmake/ninja build itself is driven by the webrtc.yml PowerShell steps.
#
# This file is sourced (not executed) by CI steps so that functions are
# available as shell commands. Sourcing applies `set -euo pipefail` to the
# calling shell for strict error checking across the entire CI step.
#
# Expected directory layout (<work> = parent of the Open3D checkout, or
# $WEBRTC_WORK_ROOT if set):
#   <work>/
#   ├── Open3D/          # this repository
#   ├── depot_tools/     # fetched by clone_depot_tools()
#   └── webrtc/
#       ├── .gclient     # created by `fetch --nohooks --no-history webrtc`
#       └── src/         # WebRTC source tree, pinned to WEBRTC_COMMIT
#
# Usage (Unix):
#   source 3rdparty/webrtc/webrtc_build.sh
#   install_dependencies_ubuntu   # Ubuntu only
#   download_webrtc_sources       # fetches depot_tools + runs gclient sync
#   build_webrtc                  # cmake/ninja build, installs, packages tar.gz

set -euo pipefail

# libwebrtc-bin M149 / Open3D target milestone
WEBRTC_COMMIT=${WEBRTC_COMMIT:-e8b4d4c5952a8fb7b35c2a6cba4e8c3de2ea2e1e}
# Pinned depot_tools (update intentionally when refreshing the WebRTC toolchain).
DEPOT_TOOLS_COMMIT=${DEPOT_TOOLS_COMMIT:-10eda50a3fd9c34ad8d31ec74e5f4eb5823d60f6}
DEPOT_TOOLS_URL="https://chromium.googlesource.com/chromium/tools/depot_tools"

GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI:-1}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
SUDO=${SUDO:-sudo}
# Parallel gclient git operations (speeds DEPS fetch on CI).
GCLIENT_JOBS=${GCLIENT_JOBS:-${NPROC}}

_OPEN3D_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

webrtc_work_root() {
    if [[ -n "${WEBRTC_WORK_ROOT:-}" ]]; then
        echo "$WEBRTC_WORK_ROOT"
    else
        dirname "$_OPEN3D_ROOT"
    fi
}

webrtc_setup_path() {
    export PATH="$(webrtc_work_root)/depot_tools:${PATH}"
    export DEPOT_TOOLS_UPDATE=0
}

# Fetch a pinned depot_tools tree via Gitiles tarball.
clone_depot_tools() {
    local root="$1"
    local dest="$root/depot_tools"
    local commit="$DEPOT_TOOLS_COMMIT"
    local stamp="$dest/.open3d_pinned_commit"

    if [[ -f "$stamp" && "$(cat "$stamp")" == "$commit" && -x "$dest/fetch" ]]; then
        return 0
    fi

    local tmp archive
    tmp="$(mktemp -d)"
    archive="$tmp/depot_tools.tar.gz"
    curl -fL --retry 3 --retry-delay 5 \
        -o "$archive" "${DEPOT_TOOLS_URL}/+archive/${commit}.tar.gz"
    rm -rf "$dest"
    mkdir -p "$dest"
    # Gitiles +archive tarballs unpack flat (fetch at archive root, not in a subdir).
    # On Windows (Git Bash), symlinks in the archive fail to extract because
    # symlink creation requires elevated privileges. Those symlinks are
    # Linux-only helper scripts (cbuildbot, luci-auth-fido2-plugin, etc.) and
    # are not needed for WebRTC builds. The 'fetch' check below validates the
    # critical tools were extracted.
    tar -xzf "$archive" -C "$dest" || true
    rm -rf "$tmp"

    if [[ ! -x "$dest/fetch" ]]; then
        echo "ERROR: depot_tools archive at ${commit} is missing fetch" >&2
        exit 1
    fi
    echo "$commit" > "$stamp"
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
    clone_depot_tools "$root"
    webrtc_setup_path
    # Verify fetch is on PATH (exits non-zero under set -e if not found).
    command -V fetch

    if [[ ! -d webrtc/src ]]; then
        mkdir -p webrtc
        pushd webrtc
        fetch --nohooks --no-history webrtc
        popd
    fi

    pushd webrtc
    gclient sync -D --force --reset --no-history \
        --jobs="${GCLIENT_JOBS}" \
        --revision "src@${WEBRTC_COMMIT}"
    popd
    popd
}

build_webrtc() {
    local root open3d_dir
    open3d_dir="$_OPEN3D_ROOT"
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
    ninja -j"${NPROC}" install
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
