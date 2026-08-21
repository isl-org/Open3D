#!/usr/bin/env bash
# Usage: package_linux_wheel_runtime.sh DEST_DIR PROBE_ELF (e.g. libOpen3D.so.N)
set -euo pipefail
dest=$1; elf=$2
export LD_LIBRARY_PATH="$dest${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$elf" | grep -E 'libc\+\+|libc\+\+abi|libunwind' | awk '{print $3}' | sort -u | while read -r lib; do
  name=$(basename "$(readlink -f "$lib")")
  cp -L "$lib" "$dest/$(sed -E 's/(\.so\.[0-9]+)\..*/\1/' <<<"$name")"
done
