# This script is mostly identical to build_boringssl.sh, except that it builds
# both the release and debug versions of the library.
#
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
${boringssl_build_dir} = "${script_dir}\boringssl\build"
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

mkdir "${boringssl_build_dir}"
cd "${boringssl_build_dir}"
cmake -DCMAKE_OSX_DEPLOYMENT_TARGET=10.14 `
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
      -DCMAKE_POLICY_DEFAULT_CMP0091:STRING=NEW `
      -DCMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW `
      -DCMAKE_CXX_VISIBILITY_PRESET=hidden `
      -DCMAKE_CUDA_VISIBILITY_PRESET=hidden `
      -DCMAKE_C_VISIBILITY_PRESET=hidden `
      -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON `
      ..
cmake --build . --config Release --parallel ${NPROC} --target ssl crypto
cmake --build . --config Debug --parallel ${NPROC} --target ssl crypto

cmake -E copy_directory ..\include                "${install_dir}\Release\include"
cmake -E copy           ssl\Release\ssl.lib       "${install_dir}\Release\lib\ssl.lib"
cmake -E copy           crypto\Release\crypto.lib "${install_dir}\Release\lib\crypto.lib"

cmake -E copy_directory ..\include                "${install_dir}\Debug\include"
cmake -E copy           ssl\Debug\ssl.lib         "${install_dir}\Debug\lib\ssl.lib"
cmake -E copy           crypto\Debug\crypto.lib   "${install_dir}\Debug\lib\crypto.lib"

cd "${script_dir}"
tar -C "${install_dir}" -czvf "${tar_name}" Release Debug

Remove-Dir "${boringssl_dir}"
Remove-Dir "${install_dir}"
