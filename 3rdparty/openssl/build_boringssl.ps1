# Try to be consistent to build_boringssl.sh as much as possible.
#
# Install dependencies first.
# ```powershell
# winget install -e --id StrawberryPerl.StrawberryPerl
# winget install -e --id GoLang.Go
# winget install -e --id NASM.NASM
# ```
$ErrorActionPreference = "Stop"

${boringssl_commit} = "edfe4133d28c5e39d4fce6a2554f3e2b4cafc9bd"
${boringssl_commit_short} = ${boringssl_commit}.SubString(0,7)

${script_dir} = "${PSScriptRoot}"
${install_dir} = "${script_dir}\boringssl_install"
${boringssl_dir} = "${script_dir}\boringssl"
${tar_name} = "boringssl_${boringssl_commit_short}_win_${env:PROCESSOR_ARCHITECTURE}.tar.gz".ToLower()

function Remove-Dir($dir_path) {
    # https://stackoverflow.com/a/9012108/1255535
    Get-ChildItem -Path "${dir_path}" -Recurse | Remove-Item -Recurse -Force -ErrorAction Ignore
    Remove-Item "${dir_path}" -Force -Recurse -ErrorAction Ignore
}
Remove-Dir "${boringssl_dir}"
Remove-Dir "${install_dir}"

${NPROC} = ${env:NUMBER_OF_PROCESSORS}

git clone --depth 1 https://boringssl.googlesource.com/boringssl "${boringssl_dir}"
cd "${boringssl_dir}"
git fetch --depth 1 origin ${boringssl_commit}
git checkout FETCH_HEAD

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
cmake -E copy_directory ..\include                ${install_dir}\include
cmake -E copy           ssl\Release\ssl.lib       ${install_dir}\lib\ssl.lib
cmake -E copy           crypto\Release\crypto.lib ${install_dir}\lib\crypto.lib

cd ${script_dir}
tar -C ${install_dir} -czvf ${tar_name} include lib

Remove-Dir "${boringssl_dir}"
Remove-Dir "${install_dir}"
