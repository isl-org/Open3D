#!/usr/bin/env bash
# Apply Open3D WebRTC patches under WEBRTC_SRC (webrtc checkout src/).
set -euo pipefail

OPEN3D_DIR="${1:?Open3D repo path}"
WEBRTC_SRC="${2:?WebRTC src path}"

# Apply a patch, hard-failing if a required patch cannot be applied.
#
# Args: <patch> <dir> [required|optional]   (default: required)
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
    local name
    name="$(basename "$patch")"
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
        git -C "$dir" apply "$patch"
        echo "Applied $name in $dir"
    elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "Skip $name (already applied) in $dir"
    elif [[ "$required" == "optional" ]]; then
        echo "Skip $name (does not apply; optional) in $dir"
    else
        echo "ERROR: required patch $name does not apply in $dir." >&2
        echo "       Refresh the patch for the pinned WebRTC commit." >&2
        exit 1
    fi
}

# Fix port.cc for Apple Clang (Xcode 15.4): the GlobalEmptyString (std::string)
# variable is declared with PROTOBUF_CONSTINIT which expands to `constinit` or
# [[clang::require_constant_initialization]] on Apple Clang >= 13.  Apple's
# libc++ std::string constructor performs a heap allocation, so the variable
# cannot be constant-initialized, producing a hard error.  Guard the declaration
# with !defined(__APPLE__) so PROTOBUF_CONSTINIT is omitted on Apple.
#
# This supplements the port_def.inc patch in 0006 and directly targets the
# specific failing declaration (port.cc:120 in the WebRTC M149 protobuf).
fix_protobuf_port_cc_apple() {
    local port_cc="${WEBRTC_SRC}/third_party/protobuf/src/google/protobuf/port.cc"
    [[ -f "$port_cc" ]] || return 0
    python3 - "$port_cc" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

old = ('PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT\n'
       '    PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GlobalEmptyString\n'
       '        fixed_address_empty_string{};')
# Check idempotency: already patched if __APPLE__ guard present near declaration.
if '__APPLE__' in content and 'fixed_address_empty_string' in content:
    print(f'Skip {path} (Apple constinit fix already applied)')
    sys.exit(0)
if old not in content:
    print(f'WARNING: {path}: expected PROTOBUF_CONSTINIT pattern not found; '
          f'Skipping Apple constinit fix', file=sys.stderr)
    sys.exit(0)
new = ('#if defined(__APPLE__)\n'
       '// Apple Clang (Xcode 15.4): GlobalEmptyString (std::string) requires heap\n'
       '// allocation in its constructor, which is not a constant expression.\n'
       '// Skip PROTOBUF_CONSTINIT to avoid "variable does not have a constant\n'
       '// initializer" hard error.\n'
       'PROTOBUF_ATTRIBUTE_NO_DESTROY\n'
       '    PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GlobalEmptyString\n'
       '        fixed_address_empty_string{};\n'
       '#else\n'
       'PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT\n'
       '    PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GlobalEmptyString\n'
       '        fixed_address_empty_string{};\n'
       '#endif  // !defined(__APPLE__)')
with open(path, 'w') as f:
    f.write(content.replace(old, new, 1))
print(f'Applied Apple constinit fix in {path}')
PYEOF
}

# Fix GCC C++20 "changes meaning" error for Network() in WebRTC.
# GCC treats a method name matching a class name as an error.
fix_gcc_cxx20_network_changes_meaning() {
    python3 - "${WEBRTC_SRC}/p2p/base/port_interface.h" "${WEBRTC_SRC}/p2p/base/port.h" <<'PYEOF'
import sys
import re
import os

for path in sys.argv[1:]:
    if not os.path.exists(path):
        continue
    with open(path, 'r') as f:
        content = f.read()
    
    # Replace `Network* Network()` with `::webrtc::Network* Network()`
    # but only if it's not already prefixed with `::webrtc::`.
    new_content = re.sub(r'(?<!::webrtc::)Network\*\s*Network\(\)', r'::webrtc::Network* Network()', content)
    
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f'Applied GCC C++20 Network() changes meaning fix in {path}')
    else:
        print(f'Skip GCC C++20 Network() changes meaning fix in {path} (already applied or not found)')
PYEOF
}

# Fix GCC ambiguous conversion from webrtc::PayloadType to int in used_ids.h.
# webrtc::PayloadType has multiple conversion operators (one inherited from
# StrongAlias) that GCC flags as ambiguous when assigning to int.
# idstruct->id can be an int or a PayloadType, so we conditionally unwrap using type traits.
fix_gcc_payload_type_ambiguous() {
    local used_ids="${WEBRTC_SRC}/pc/used_ids.h"
    [[ -f "$used_ids" ]] || return 0
    python3 - "$used_ids" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

helper = """
namespace {
template <typename T>
constexpr int AsInt(const T& t) {
    if constexpr (std::is_integral_v<T>) {
        return t;
    } else {
        return t.value();
    }
}
}  // namespace
"""

if 'AsInt(const T& t)' not in content:
    content = content.replace('#include "rtc_base/system/rtc_export.h"', '#include "rtc_base/system/rtc_export.h"\n#include <type_traits>\n' + helper)

new_content = content.replace('int original_id = idstruct->id;', 'int original_id = AsInt(idstruct->id);')
new_content = new_content.replace('int new_id = idstruct->id;', 'int new_id = AsInt(idstruct->id);')

if new_content != content:
    with open(path, 'w') as f:
        f.write(new_content)
    print(f'Applied GCC PayloadType ambiguous conversion fix in {path}')
else:
    print(f'Skip GCC PayloadType ambiguous conversion fix in {path} (already applied or not found)')
PYEOF
}

PATCH_DIR="$OPEN3D_DIR/3rdparty/webrtc"
# Required: declare gn args consumed by args.gn and fix GCC compile errors.
apply_one "$PATCH_DIR/0001-src-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0001-build-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC/build"
apply_one "$PATCH_DIR/0001-third_party-enable-rtc_use_cxx11_abi-option.patch" "$WEBRTC_SRC/third_party"
apply_one "$PATCH_DIR/0002-src-fix-nullptr_t-with-libstdcxx.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0004-call-payload_type_picker-gcc-flat_tree.patch" "$WEBRTC_SRC"
apply_one "$PATCH_DIR/0005-build-win-dynamic-crt.patch" "$WEBRTC_SRC/build"
# 0006 patches port_def.inc to prevent PROTOBUF_CONSTINIT from expanding to
# constinit/[[clang::require_constant_initialization]] on Apple. The
# fix_protobuf_port_cc_apple call below directly patches port.cc as an
# additional safety measure in case the port_def.inc path alone is insufficient.
apply_one "$PATCH_DIR/0006-third_party-protobuf-disable-constinit-on-apple.patch" "$WEBRTC_SRC/third_party/protobuf"
fix_protobuf_port_cc_apple
fix_gcc_cxx20_network_changes_meaning
fix_gcc_payload_type_ambiguous
