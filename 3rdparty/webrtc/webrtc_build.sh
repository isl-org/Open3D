#!/usr/bin/env bash
set -euox pipefail

# This script builds WebRTC for Open3D for Ubuntu and macOS. For Windows, see
# .github/workflows/webrtc.yml
#
# Usage:
# $ bash # Start a new shell
# Specify custom configuration by exporting environment variables
# GLIBCXX_USE_CXX11_ABI, WEBRTC_REPO, WEBRTC_BRANCH, WEBRTC_COMMIT,
# DEPOT_TOOLS_COMMIT and NPROC, if required.
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
# 3) For old-ABI builds, apply the following patches to enable
#    GLIBCXX_USE_CXX11_ABI selection:
#    - 0001-build-enable-rtc_use_cxx11_abi-option.patch        # apply to webrtc/src
#    - 0001-src-enable-rtc_use_cxx11_abi-option.patch          # apply to webrtc/src/build
#    - 0001-third_party-enable-rtc_use_cxx11_abi-option.patch  # apply to webrtc/src/third_party
#    Note that these patches may or may not be compatible with your custom
#    WebRTC commits. You may have to patch them manually.

detect_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        getconf _NPROCESSORS_ONLN
    fi
}

# Use the WebRTC-SDK fork by default.
WEBRTC_REPO=${WEBRTC_REPO:-https://github.com/webrtc-sdk/webrtc.git}
WEBRTC_BRANCH=${WEBRTC_BRANCH:-m144_release}
# Optionally pin to a commit after checking out WEBRTC_BRANCH.
WEBRTC_COMMIT=${WEBRTC_COMMIT:-}
# Upstream HEAD on 2026-05-04
DEPOT_TOOLS_COMMIT=${DEPOT_TOOLS_COMMIT:-ff41874736c800b2f79aa8cf9596c7919066eb02}

GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI:-1}
NPROC=${NPROC:-$(detect_nproc)}
SUDO=${SUDO:-sudo}                           # Set to command if running inside docker
export PATH="$PWD/../depot_tools":${PATH}    # $(basename $PWD) == Open3D
export DEPOT_TOOLS_UPDATE=0

install_dependencies_ubuntu() {
    options="$(echo "$@" | tr ' ' '|')"
    # Dependencies
    # python3*      : resolve ImportError: No module named pkg_resources
    # libglib2.0-dev: resolve pkg_config("glib")
    $SUDO apt-get update
    $SUDO apt-get install -y \
        apt-transport-https \
        build-essential \
        ca-certificates \
        git \
        gnupg \
        libglib2.0-dev \
        python-is-python3 \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        software-properties-common \
        tree \
        curl
    if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
        curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key |
            gpg --dearmor - |
            $SUDO tee /etc/apt/trusted.gpg.d/apt.llvm.org.gpg >/dev/null
        source <(grep VERSION_CODENAME /etc/os-release)
        $SUDO apt-add-repository --yes "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-22 main"
        $SUDO apt-get update
        $SUDO apt-get install -y clang-22 lld-22 libclang-rt-22-dev
        $SUDO update-alternatives --install /usr/bin/clang clang /usr/bin/clang-22 100
        $SUDO update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-22 100
        for tool in lld ld.lld ld64.lld wasm-ld llvm-ar llvm-nm llvm-objcopy llvm-objdump llvm-ranlib llvm-readelf llvm-strip; do
            if [[ -x "/usr/bin/${tool}-22" ]]; then
                $SUDO update-alternatives --install "/usr/bin/${tool}" "${tool}" "/usr/bin/${tool}-22" 100
            fi
        done
        CLANG_RT_BUILTINS="$(find /usr/lib/llvm-22 /usr/lib/clang \
            -path '*/lib/linux/*' \
            \( -name libclang_rt.builtins.a -o -name libclang_rt.builtins-aarch64.a \) |
            head -n 1)"
        if [[ -z "${CLANG_RT_BUILTINS}" ]]; then
            CLANG_RT_BUILTINS="$(find /usr/lib/llvm-22 /usr/lib/clang \
            -not -path '*/lib/windows/*' \
            \( -name libclang_rt.builtins.a -o -name libclang_rt.builtins-aarch64.a \) |
            head -n 1)"
        fi
        if [[ -n "${CLANG_RT_BUILTINS}" ]]; then
            $SUDO mkdir -p /usr/lib/clang/22/lib/aarch64-unknown-linux-gnu
            $SUDO ln -sf "${CLANG_RT_BUILTINS}" \
                /usr/lib/clang/22/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a
        fi
    fi
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
    command -V gclient

    echo Get WebRTC
    mkdir webrtc
    cd webrtc
    if [[ -n "${WEBRTC_COMMIT}" ]]; then
        WEBRTC_GCLIENT_URL="${WEBRTC_REPO}@${WEBRTC_COMMIT}"
    else
        WEBRTC_GCLIENT_URL="${WEBRTC_REPO}@refs/heads/${WEBRTC_BRANCH}"
    fi
    cat > .gclient <<EOF
solutions = [
  {
    "name": "src",
    "url": "${WEBRTC_GCLIENT_URL}",
    "deps_file": "DEPS",
    "managed": False,
    "custom_deps": {},
    "custom_vars": {},
  },
]
target_os = []
EOF
    git clone --branch "${WEBRTC_BRANCH}" "${WEBRTC_REPO}" src

    # Optionally checkout to a specific version.
    # Ref: https://chromium.googlesource.com/chromium/src/+/master/docs/building_old_revisions.md
    if [[ -n "${WEBRTC_COMMIT}" ]]; then
        git -C src checkout "${WEBRTC_COMMIT}"
    fi
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
    cp 3rdparty/webrtc/{CMakeLists.txt,webrtc_common.cmake} ../webrtc
    if [[ "${GLIBCXX_USE_CXX11_ABI}" == "0" ]]; then
        echo Apply old-ABI patches
        git -C ../webrtc/src apply \
            "$OPEN3D_DIR"/3rdparty/webrtc/0001-src-enable-rtc_use_cxx11_abi-option.patch
        git -C ../webrtc/src/build apply \
            "$OPEN3D_DIR"/3rdparty/webrtc/0001-build-enable-rtc_use_cxx11_abi-option.patch
        git -C ../webrtc/src/third_party apply \
            "$OPEN3D_DIR"/3rdparty/webrtc/0001-third_party-enable-rtc_use_cxx11_abi-option.patch
    else
        echo Skip old-ABI patches
    fi
    WEBRTC_COMMIT_SHORT=$(git -C ../webrtc/src rev-parse --short=7 HEAD)

    echo "Build WebRTC using NPROC=${NPROC}"
    if [[ -x ../depot_tools/ensure_bootstrap ]]; then
        ../depot_tools/ensure_bootstrap
    else
        gclient --version >/dev/null
    fi
    mkdir ../webrtc/build
    pushd ../webrtc/build
    cmake -DCMAKE_INSTALL_PREFIX=../../webrtc_release \
        -DGLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} \
        ..
    make -j"${NPROC}"
    make install -j"${NPROC}"
    popd # PWD=Open3D
    pushd ..
    tree -L 2 webrtc_release || ls webrtc_release/*

    echo Package WebRTC
    if [[ $(uname -s) == 'Linux' ]]; then
        WEBRTC_ARCH=$(uname -m)
        if [[ "${WEBRTC_ARCH}" == "aarch64" || "${WEBRTC_ARCH}" == "arm64" ]]; then
            WEBRTC_PLATFORM="linux_arm64"
        else
            WEBRTC_PLATFORM="linux"
        fi
        tar -czf \
            "$OPEN3D_DIR/webrtc_${WEBRTC_COMMIT_SHORT}_${WEBRTC_PLATFORM}_cxx-abi-${GLIBCXX_USE_CXX11_ABI}.tar.gz" \
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
