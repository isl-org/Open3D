#Requires -RunAsAdministrator
# Install Intel oneAPI C++ essentials for Windows SYCL (xpu) CI builds.
# Pattern from oneapi-src/oneapi-ci (2025.3.0 URLs @ 0804a4c):
#   https://github.com/oneapi-src/oneapi-ci/tree/0804a4c9281440d8a91ac0680388b101e5f673ad
#
# Open3D uses the setvars.bat script to set up the oneAPI environment for icx (C
# and C++ compiler). VS 2022 integration is enabled for cmake -T Intel DPC++ toolset support.
param(
    [switch]$Verbose
)

$ErrorActionPreference = 'Stop'
if ($env:OPEN3D_ONEAPI_DEBUG -eq '1') {
    $Verbose = $true
}

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

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PrintPathsScript = Join-Path $ScriptDir 'print_oneapi_windows_paths.ps1'

$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$WorkDir = Join-Path $TempRoot "open3d_oneapi_$([guid]::NewGuid().ToString('N'))"
$InstallLogArchive = Join-Path $TempRoot 'open3d_oneapi_install_logs'
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$InstallExitCode = 0
Push-Location $WorkDir
try {
    Write-Host "Downloading OneAPI webimage..."
    curl.exe --output webimage.exe --url $ONEAPI_BASEKIT_URL --retry 5 --retry-delay 5 --fail
    Write-Host "Extracting OneAPI webimage..."
    $ExtractProc = Start-Process -FilePath ".\webimage.exe" -ArgumentList @(
        "-s", "-x", "-f", "webimage_extracted", "--log", "extract.log"
    ) -Wait -PassThru -NoNewWindow
    if ($ExtractProc.ExitCode -ne 0) {
        throw "OneAPI webimage extract failed with exit code $($ExtractProc.ExitCode)"
    }
    Remove-Item webimage.exe -Force
    $Bootstrapper = Join-Path $WorkDir "webimage_extracted\bootstrapper.exe"
    if (-not (Test-Path $Bootstrapper)) {
        throw "bootstrapper.exe not found after extracting OneAPI webimage"
    }
    Write-Host "Installing OneAPI components: $ONEAPI_WIN_BASE_COMPONENTS"
    if ($Verbose) {
        Write-Host "Install work dir: $WorkDir"
        Write-Host "NEED_VS2022_INTEGRATION=1 (required for cmake -T Intel DPC++ toolset)"
    }
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
    $InstallExitCode = $InstallProc.ExitCode
    if ($InstallExitCode -ne 0) {
        throw "OneAPI bootstrapper install failed with exit code $InstallExitCode"
    }
} finally {
    if ($Verbose -or $InstallExitCode -ne 0) {
        if (Test-Path -LiteralPath $WorkDir) {
            Write-Host "Archiving installer logs to $InstallLogArchive"
            New-Item -ItemType Directory -Force -Path $InstallLogArchive | Out-Null
            Copy-Item -Path (Join-Path $WorkDir '*') -Destination $InstallLogArchive -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "See extract.log and installer logs under: $InstallLogArchive"
        }
    }
    Pop-Location
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}

Write-Host "OneAPI bootstrapper reported success (exit code 0)."
if (Test-Path -LiteralPath $PrintPathsScript) {
    if ($Verbose) {
        & $PrintPathsScript -Verbose
    } else {
        & $PrintPathsScript
    }
} else {
    Write-Warning "Missing $PrintPathsScript"
}
