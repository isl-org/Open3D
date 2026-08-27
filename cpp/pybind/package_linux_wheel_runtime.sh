#!/usr/bin/env bash
# Usage: package_linux_wheel_runtime.sh DEST_DIR PROBE_ELF (e.g. libOpen3D.so.N)
# Copies Filament's LLVM libc++ runtime into the wheel under its DT_NEEDED name,
# which is what $ORIGIN lookup uses at import time (e.g. libc++.so.1.0.20).
set -euo pipefail
dest=$1; elf=$2
export LD_LIBRARY_PATH="$dest${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ldd "$elf" | awk '/libc\+\+|libunwind/ && $3 != "" {print $1, $3}' |
  while read -r soname path; do
    [ "$path" -ef "$dest/$soname" ] || cp -L "$path" "$dest/$soname"
  done
