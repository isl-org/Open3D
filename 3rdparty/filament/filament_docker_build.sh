#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.tar.gz"
FILAMENT_URL="https://github.com/isl-org/filament/archive/fcd2930eb75924bbb7afbe990de9782af4b5d1dc.tar.gz"
FILAMENT_TAR="filament-fcd2930eb75924bbb7afbe990de9782af4b5d1dc-linux-amd64.tar.gz"

pushd "${SCRIPT_DIR}"

docker build \
    -t open3d-filament:latest \
    --build-arg CMAKE_URL="${CMAKE_URL}" \
    --build-arg FILAMENT_URL="${FILAMENT_URL}" \
    --build-arg FILAMENT_TAR="${FILAMENT_TAR}" \
    -f Dockerfile.filament .

docker run \
    -v "${PWD}:/opt/mount" --rm \
    open3d-filament:latest \
    bash -c "cp /${FILAMENT_TAR} /opt/mount \
          && chown $(id -u):$(id -g) /opt/mount/${FILAMENT_TAR}"
popd
