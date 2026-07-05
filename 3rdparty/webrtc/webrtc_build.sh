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
    local root
    root="$(webrtc_work_root)"
    # On Windows the work root is a native path like 'C:\WebRTC'. The colon in
    # the drive letter is bash's PATH separator, which would split
    # 'C:\WebRTC/depot_tools' into the two entries 'C' and '\WebRTC/depot_tools'.
    # '\WebRTC/...' then causes sha256sum to prefix its output with '\' (GNU
    # coreutils escapes paths that contain backslashes), making the CIPD hash
    # check fail. cygpath is available in Git Bash / MSYS2; convert to POSIX.
    if command -v cygpath >/dev/null 2>&1; then
        root="$(cygpath -u "$root")"
    fi
    export PATH="${root}/depot_tools:${PATH}"
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
    # On Windows (Git Bash), symlinks in the archive fail because creating them
    # requires elevated privileges.  Use Python to extract while silently
    # skipping symlink/hardlink members so that critical batch/exe files are
    # always extracted.  Non-Windows uses plain tar.
    if command -v cygpath >/dev/null 2>&1; then
        python3 - "$dest" "$archive" <<'PYEOF_INNER'
import tarfile, sys
dest, archive = sys.argv[1], sys.argv[2]
with tarfile.open(archive, "r:gz") as tf:
    for m in tf.getmembers():
        if m.issym() or m.islnk():
            print(f"Skip symlink: {m.name}")
            continue
        try:
            tf.extract(m, dest)
        except Exception as e:
            print(f"Warning: cannot extract {m.name}: {e}", file=sys.stderr)
PYEOF_INNER
    else
        tar -xzf "$archive" -C "$dest"
    fi
    rm -rf "$tmp"

    if [[ ! -x "$dest/fetch" ]]; then
        echo "ERROR: depot_tools archive at ${commit} is missing fetch" >&2
        exit 1
    fi

    # On Windows, verify that critical batch wrappers were extracted from the
    # tarball.  If extraction failed silently, these files would be absent and
    # the bootstrap or GN build would later fail with a confusing error.
    if command -v cygpath >/dev/null 2>&1; then
        local _missing=()
        for _tool in "fetch.bat" "gn.bat"; do
            [[ -f "$dest/$_tool" ]] || _missing+=("$_tool")
        done
        if [[ ${#_missing[@]} -gt 0 ]]; then
            echo "ERROR: depot_tools extraction incomplete on Windows." >&2
            echo "       Missing: ${_missing[*]}" >&2
            echo "       Batch files in $dest:" >&2
            ls "$dest"/*.bat 2>/dev/null >&2 || ls "$dest" >&2 || true
            exit 1
        fi
    fi

    # Bootstrap depot_tools.
    # We must temporarily prepend depot_tools to PATH so that python scripts
    # and subprocesses launched during bootstrap (like gsutil.py calling luci-auth)
    # can find the depot_tools executables on both Unix and Windows.
    local old_path="$PATH"
    export PATH="${dest}:${PATH}"

    if [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"MSYS"* || "$(uname -s)" == *"CYGWIN"* ]]; then
        # On Windows, bootstrap Python and Git via the batch files.
        # This creates git.bat, python3.bat, and downloads cipd tools.
        # We must run them using cmd.exe inside the depot_tools directory.
        pushd "$dest"
        cmd.exe //c "cipd_bin_setup.bat"
        cmd.exe //c "bootstrap\\win_tools.bat"
        popd
    else
        # On Unix (Ubuntu/macOS), ensure_bootstrap downloads Python 3 via CIPD and writes python3_bin_reldir.txt.
        # DEPOT_TOOLS_DIR must be set so ensure_bootstrap resolves scripts correctly.
        DEPOT_TOOLS_DIR="$dest" "$dest/ensure_bootstrap"
    fi

    export PATH="$old_path"

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

    # depot_tools/ninja is a Python wrapper that locates the real ninja binary
    # by walking up from the *current directory* to find a tracked gclient
    # entry (see gclient_paths.FindGclientRoot). CMake's own Ninja-generator
    # sanity check ("ninja --version") runs with cwd = this build directory,
    # which is a sibling of webrtc/src (not nested inside it), so that lookup
    # fails and the wrapper falls back to requiring a "ninja" on PATH. Point
    # CMake directly at the real, DEPS-pinned ninja binary instead (mirrors
    # how the Windows job locates ninja.exe before prepending depot_tools to
    # PATH) so no system/apt ninja package is required.
    local ninja_bin="$root/webrtc/src/third_party/ninja/ninja"
    if [[ ! -x "$ninja_bin" ]]; then
        echo "ERROR: expected ninja binary not found at $ninja_bin" >&2
        exit 1
    fi

    mkdir -p "$root/webrtc/build"
    pushd "$root/webrtc/build"
    cmake -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$ninja_bin" \
        -DCMAKE_INSTALL_PREFIX="$root/webrtc_release" \
        -DGLIBCXX_USE_CXX11_ABI="${GLIBCXX_USE_CXX11_ABI}" \
        ..
    "$ninja_bin" -j"${NPROC}" install
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
