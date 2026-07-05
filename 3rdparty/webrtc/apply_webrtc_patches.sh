#!/usr/bin/env bash
# Apply Open3D WebRTC patches under WEBRTC_SRC (webrtc checkout src/).
set -euo pipefail

OPEN3D_DIR="${1:?Open3D repo path}"
WEBRTC_SRC="${2:?WebRTC src path}"

# Apply a patch, hard-failing if a required patch cannot be applied.
#
# Args: <patch> <dir> [required|optional]   (default: required)
#
# IMPORTANT: <dir> must be the *root* of a git checkout (i.e. the directory
# containing .git), not an arbitrary subdirectory. `git apply` run from a
# subdirectory with paths relative to that subdirectory silently no-ops
# ("Skipped patch ..." on stderr, exit 0) instead of applying or erroring, at
# least with the git version used by our CI/dev images. Patches touching
# files under a git-tracked *plain* subdirectory (e.g. third_party/protobuf,
# which is not itself a repository root) must therefore be applied from the
# enclosing repo root (third_party) with paths prefixed accordingly (e.g.
# protobuf/src/...) -- see 0006 and 0009.
#
# A patch is considered "already applied" when it applies in reverse; in that
# case it is skipped without error so the script is safe to re-run and tolerant
# of fixes that have landed upstream. A required patch that neither applies nor
# is already applied aborts the build: these patches add gn args / ABI defines
# or fix compile errors, so silently skipping them produces broken or
# confusing artifacts (e.g. an undeclared gn arg in args.gn, a C++11/C++17 ABI
# mismatch, or a hard compile error) rather than a clear failure here.
apply_one() {
    local patch="$1"
    local dir="$2"
    local required="${3:-required}"
    local name check_output check_rc
    name="$(basename "$patch")"

    # Capture combined output so we can detect git's silent "Skipped patch"
    # no-op (see note above), which must NOT be treated as a successful
    # check even though it exits 0.
    check_output="$(git -C "$dir" apply --check "$patch" 2>&1)" && check_rc=0 || check_rc=$?

    if [[ "$check_rc" -eq 0 && "$check_output" != *"Skipped patch"* ]]; then
        git -C "$dir" apply "$patch"
        echo "Applied $name in $dir"
    elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "Skip $name (already applied) in $dir"
    elif [[ "$required" == "optional" ]]; then
        echo "Skip $name (does not apply; optional) in $dir"
    else
        echo "ERROR: required patch $name does not apply in $dir." >&2
        [[ -n "$check_output" ]] && echo "       $check_output" >&2
        echo "       Refresh the patch for the pinned WebRTC commit." >&2
        exit 1
    fi
}

PATCH_DIR="$OPEN3D_DIR/3rdparty/webrtc"
# Required: declare gn args consumed by args.gn and fix GCC compile errors.
apply_one "$PATCH_DIR/0001-src-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0001-build-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC/build"
apply_one "$PATCH_DIR/0001-third_party-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC/third_party"
apply_one "$PATCH_DIR/0002-src-fix-nullptr_t-with-libstdcxx.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0004-call-payload_type_picker-gcc-flat_tree.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0005-build-win-dynamic-crt.patch" "$WEBRTC_SRC/build"
# 0006 and 0009 patch files under third_party/protobuf, a plain subdirectory
# of the third_party checkout (not its own git root) -- apply from
# $WEBRTC_SRC/third_party, not .../third_party/protobuf. See apply_one note.
apply_one "$PATCH_DIR/0006-third_party-protobuf-disable-constinit-on-apple.patch" "$WEBRTC_SRC/third_party"
apply_one "$PATCH_DIR/0009-third_party-protobuf-port-cc-disable-constinit-on-apple.patch" "$WEBRTC_SRC/third_party"
apply_one "$PATCH_DIR/0007-p2p-fix-gcc-cxx20-network-changes-meaning.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0008-pc-fix-gcc-payload-type-ambiguous-conversion.patch" "$WEBRTC_SRC"
