#!/usr/bin/env bash
# Build a Linux libc++ runtime that delegates the Itanium C++ ABI to GNU libstdc++.
# Package lib/libc++stdabi.a as an additional Filament archive library.
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
    echo "This script supports Linux only." >&2
    exit 1
fi

LLVM_VERSION=17.0.6
LLVM_SHA256=58a8818c60e6627064f312dbf46c02d9949956558340938b71cf731ad8bc0813
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK_DIR=${WORK_DIR:-"${SCRIPT_DIR}/../build/libcxx-stdabi"}
INSTALL_DIR=${INSTALL_DIR:-"${WORK_DIR}/install"}
LLVM_TARBALL="${WORK_DIR}/llvm-project-${LLVM_VERSION}.src.tar.xz"
LLVM_SOURCE_DIR="${WORK_DIR}/llvm-project-${LLVM_VERSION}.src"
BUILD_DIR=${BUILD_DIR:-"${WORK_DIR}/build"}

command -v cmake >/dev/null
command -v ninja >/dev/null
command -v clang >/dev/null
command -v clang++ >/dev/null
command -v g++ >/dev/null
command -v curl >/dev/null
command -v sha256sum >/dev/null

mkdir -p "${WORK_DIR}"
if [[ ! -f "${LLVM_TARBALL}" ]]; then
    curl --fail --location --retry 3 \
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz" \
        --output "${LLVM_TARBALL}"
fi
echo "${LLVM_SHA256}  ${LLVM_TARBALL}" | sha256sum --check --status

if [[ ! -d "${LLVM_SOURCE_DIR}" ]]; then
    tar -xf "${LLVM_TARBALL}" -C "${WORK_DIR}"
fi

GCC_VERSION=$(g++ -dumpversion)
GCC_INCLUDE_DIR="/usr/include/c++/${GCC_VERSION}"
GCC_TARGET_INCLUDE_DIR="/usr/include/$(gcc -dumpmachine)/c++/${GCC_VERSION}"
GCC_LIBRARY_DIR=$(g++ -print-file-name=libstdc++.so)
GCC_LIBRARY_DIR=$(dirname "${GCC_LIBRARY_DIR}")

cmake -S "${LLVM_SOURCE_DIR}/runtimes" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_EXE_LINKER_FLAGS="-L${GCC_LIBRARY_DIR}" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${GCC_LIBRARY_DIR}" \
    -DLLVM_ENABLE_RUNTIMES=libcxx \
    -DLIBCXX_CXX_ABI=libstdc++ \
    -DLIBCXX_CXX_ABI_INCLUDE_PATHS="${GCC_INCLUDE_DIR};${GCC_TARGET_INCLUDE_DIR}" \
    -DLIBCXXABI_USE_LLVM_UNWINDER=OFF \
    -DLIBCXX_ENABLE_SHARED=OFF \
    -DLIBCXX_ENABLE_STATIC=ON \
    -DLIBCXX_ENABLE_ABI_LINKER_SCRIPT=OFF \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
cmake --build "${BUILD_DIR}" --target cxx --parallel

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/lib"
cp "${BUILD_DIR}/lib/libc++.a" "${INSTALL_DIR}/lib/libc++stdabi.a"

nm -A --defined-only "${INSTALL_DIR}/lib/libc++stdabi.a" | \
    grep -Eq ' (__cxa_|_Unwind_)' && {
    echo "libc++stdabi.a must not define exception or unwinder entry points" >&2
    exit 1
}

echo "Built libc++stdabi runtime: ${INSTALL_DIR}"