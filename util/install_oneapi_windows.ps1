#Requires -RunAsAdministrator
# Install the Intel oneAPI Toolkit for Windows SYCL (xpu) CI builds.
# intel-oneapi-base-toolkit is EOL; Intel merged Base/HPC toolkits into a
# single oneAPI Toolkit installer starting with the 2026.0 release. Pin to
# the exact oneAPI version (2026.0.0) that PyTorch's +xpu wheels are built
# against.
# Uses setvars.bat for icx; enables VS 2022 integration for cmake -T Intel DPC++.
$ErrorActionPreference = 'Stop'

$ONEAPI_TOOLKIT_URL = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/bae85ab1-cfcd-4251-8d42-a0c27949ea33/intel-oneapi-toolkit-2026.0.0.193_offline.exe"
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
    curl.exe -L --output webimage.exe --url $ONEAPI_TOOLKIT_URL --retry 5 --retry-delay 5 --fail
    if ($LASTEXITCODE -ne 0) {
        throw "OneAPI download failed (curl exit $LASTEXITCODE): $ONEAPI_TOOLKIT_URL"
    }

    Write-Host "Extracting OneAPI webimage..."
    $ExtractProc = Start-Process -FilePath ".\webimage.exe" -ArgumentList @(
        "-s", "-x", "-f", "webimage_extracted"
    ) -Wait -PassThru -NoNewWindow
    if ($ExtractProc.ExitCode -ne 0) {
        throw "OneAPI webimage extract failed with exit code $($ExtractProc.ExitCode)"
    }
    Remove-Item webimage.exe -Force

    $Bootstrapper = Join-Path $WorkDir "webimage_extracted\bootstrapper.exe"
    if (-not (Test-Path $Bootstrapper)) {
        throw "bootstrapper.exe not found after extracting OneAPI webimage ($WorkDir)"
    }

    Write-Host "Installing OneAPI components..."
    $InstallProc = Start-Process -FilePath $Bootstrapper -ArgumentList @(
        "-s", "--action", "install",
        "--components=$ONEAPI_WIN_BASE_COMPONENTS",
        "--eula=accept",
        "-p=NEED_VS2017_INTEGRATION=0",
        "-p=NEED_VS2019_INTEGRATION=0",
        "-p=NEED_VS2022_INTEGRATION=1"
    ) -Wait -PassThru -NoNewWindow
    if ($InstallProc.ExitCode -ne 0) {
        throw "OneAPI bootstrapper install failed with exit code $($InstallProc.ExitCode)"
    }
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}
