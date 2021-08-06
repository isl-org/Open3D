#!/usr/bin/env bash
set -euo pipefail

ccache -s
CCACHE_DIR=$(ccache -p | grep cache_dir | grep -oE "[^ ]+$")
echo "CCACHE_DIR: ${CCACHE_DIR}"

CCACHE_DIR_NAME=$(basename ${CCACHE_DIR})
CCACHE_DIR_PARENT=$(dirname ${CCACHE_DIR})
cd ${CCACHE_DIR_PARENT}
tar -czf ccache.tar.gz ${CCACHE_DIR_NAME}
mv ccache.tar.gz /
