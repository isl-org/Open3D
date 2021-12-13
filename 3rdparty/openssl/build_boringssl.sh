#!/usr/bin/env bash
set -euo pipefail

boringssl_commit=edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
install_dir="${script_dir}/boringssl_install"
boringssl_dir="${install_dir}/boringssl"
rm -rf "${boringssl_dir}"
rm -rf "${install_dir}"

if [[ "$(uname)" == "Darwin" ]]
then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

git clone https://boringssl.googlesource.com/boringssl "${boringssl_dir}"
cd boringssl
git checkout ${boringssl_commit}

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.14 \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW \
      -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW \
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
      -DCMAKE_CUDA_VISIBILITY_PRESET=hidden \
      -DCMAKE_C_VISIBILITY_PRESET=hidden \
      -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
      ..
cmake --build . --config Release --parallel ${NPROC} --target ssl crypto
cmake -E copy_directory ../include          ${install_dir}/include
cmake -E copy           ssl/libssl.a        ${install_dir}/lib/libssl.a
cmake -E copy           crypto/libcrypto.a  ${install_dir}/lib/libcrypto.a

cd ${script_dir}
tar -C ${install_dir} -czvf boringssl.tar.gz include lib

rm -rf "${boringssl_dir}"
rm -rf "${install_dir}"
