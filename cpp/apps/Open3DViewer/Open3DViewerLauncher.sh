#!/bin/sh
# Linux launcher script for Open3DViewer, installed in the system PATH with name
# Open3D. The Open3D viewer binary is placed together with resources.
SCRIPT=$(readlink -f "$0")
INSTALL_DIRECTORY=$(readlink -f $(dirname "$SCRIPT")/..)
OPEN3D_PATH="$INSTALL_DIRECTORY/share/Open3D/Open3D"
"$OPEN3D_PATH" "$@" &