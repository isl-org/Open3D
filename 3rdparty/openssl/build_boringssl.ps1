# Try to be consistent to build_boringssl.sh as much as possible.
#
# Install dependencies first.
# ```powershell
# winget install -e --id StrawberryPerl.StrawberryPerl
# winget install -e --id GoLang.Go
# ```

${boringssl_commit} = "edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd"
${boringssl_commit_short} = ${boringssl_commit}.SubString(0,7)

${script_dir} = "${PSScriptRoot}"
${install_dir} = "${script_dir}\boringssl_install"
${boringssl_dir} = "${script_dir}\boringssl"
${tar_name} = "boringssl_${boringssl_commit_short}_win_${env:PROCESSOR_ARCHITECTURE}.tar.gz".ToLower()

Remove-Item "${boringssl_dir}" -Force -Recurse -ErrorAction Ignore
Remove-Item "${install_dir}" -Force -Recurse -ErrorAction Ignore

${NPROC} = ${env:NUMBER_OF_PROCESSORS}

git clone https://boringssl.googlesource.com/boringssl "${boringssl_dir}"
cd "${boringssl_dir}"
git checkout ${boringssl_commit}

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_OSX_DEPLOYMENT_TARGET=10.14 `
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
      -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW `
      -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW `
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden `
      -DCMAKE_CUDA_VISIBILITY_PRESET=hidden `
      -DCMAKE_C_VISIBILITY_PRESET=hidden `
      -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON `
      ..
cmake --build . --config Release --parallel ${NPROC} --target ssl crypto
cmake -E copy_directory ..\include          ${install_dir}\include
cmake -E copy           ssl\libssl.a        ${install_dir}\lib\libssl.a
cmake -E copy           crypto\libcrypto.a  ${install_dir}\lib\libcrypto.a

cd ${script_dir}
tar -C ${install_dir} -czvf ${tar_name} include lib

Remove-Item "${boringssl_dir}" -Force -Recurse -ErrorAction Ignore
Remove-Item "${install_dir}" -Force -Recurse -ErrorAction Ignore
