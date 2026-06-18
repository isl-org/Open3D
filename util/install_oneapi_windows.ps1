#Requires -RunAsAdministrator
# Install Intel oneAPI C++ essentials for Windows SYCL (xpu) CI builds.
# Pattern from oneapi-src/oneapi-ci (2025.3.0 URLs @ 0804a4c):
#   https://github.com/oneapi-src/oneapi-ci/tree/0804a4c9281440d8a91ac0680388b101e5f673ad
#
# Open3D uses CMake -T "Intel(R) oneAPI DPC++ Compiler", so VS 2022 integration is enabled
# (oneapi-ci samples disable it because they invoke icx via setvars).
$ErrorActionPreference = 'Stop'

# oneAPI 2025.3 offline webimages (oneapi-src/oneapi-ci @ 0804a4c9281440d8a91ac0680388b101e5f673ad)
$ONEAPI_BASEKIT_URL = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/1f18901e-877d-469d-a41a-a10f11b39336/intel-oneapi-base-toolkit-2025.3.0.372.exe"

# Default components: 
$ONEAPI_WIN_BASE_COMPONENTS = @(
    "intel.oneapi.win.cpp-dpcpp-common",
    "intel.oneapi.win.tbb.devel",
    "intel.oneapi.win.dpl",
    "intel.oneapi.win.ipp.devel",
    "intel.oneapi.win.mkl.devel"
) -join ":"

$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$WorkDir = Join-Path $TempRoot "open3d_oneapi_$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Push-Location $WorkDir
try {
    Write-Host "Downloading OneAPI webimage..."
    curl.exe --output webimage.exe --url $ONEAPI_BASEKIT_URL --retry 5 --retry-delay 5 --fail
    Write-Host "Extracting OneAPI webimage..."
    $ExtractProc = Start-Process -FilePath ".\webimage.exe" -ArgumentList @(
        "-s", "-x", "-f", "webimage_extracted", "--log", "extract.log"
    ) -Wait -PassThru -NoNewWindow
    Remove-Item webimage.exe -Force
    $Bootstrapper = Join-Path $WorkDir "webimage_extracted\bootstrapper.exe"
    if (-not (Test-Path $Bootstrapper)) {
        throw "bootstrapper.exe not found after extracting OneAPI webimage"
    }
    Write-Host "Installing OneAPI components: $ONEAPI_WIN_BASE_COMPONENTS"
    $InstallArgs = @(
        "-s", "--action", "install",
        "--components=$ONEAPI_WIN_BASE_COMPONENTS",
        "--eula=accept",
        "-p=NEED_VS2017_INTEGRATION=0",
        "-p=NEED_VS2019_INTEGRATION=0",
        "-p=NEED_VS2022_INTEGRATION=1",
        "--log-dir=$WorkDir"
    )
    $InstallProc = Start-Process -FilePath $Bootstrapper -ArgumentList $InstallArgs -Wait -PassThru -NoNewWindow
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}

$toolsetProps = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "*Intel*oneAPI*Compiler*.props" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $toolsetProps) {
    throw "Intel oneAPI DPC++ Visual Studio toolset was not installed correctly."
}
Write-Host "Found Intel DPC++ VS toolset: $($toolsetProps.FullName)"
