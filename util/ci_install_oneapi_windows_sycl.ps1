# Install Intel oneAPI C++ essentials for Windows SYCL (xpu) CI builds.
# Pattern from oneapi-src/oneapi-ci (2025.3.0 URLs @ 0804a4c):
#   https://github.com/oneapi-src/oneapi-ci/tree/0804a4c9281440d8a91ac0680388b101e5f673ad
#
# Requires env ONEAPI_BASEKIT_URL, ONEAPI_HPCKIT_URL, and optional component lists.
# Open3D uses CMake -T "Intel(R) oneAPI DPC++ Compiler", so VS 2022 integration is enabled
# (oneapi-ci samples disable it because they invoke icx via setvars).
$ErrorActionPreference = 'Stop'

function Install-OneApiWebimage {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Components,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
    $WorkDir = Join-Path $TempRoot "open3d_oneapi_$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
    Push-Location $WorkDir
    try {
        Write-Host "Downloading $Label webimage..."
        curl.exe --output webimage.exe --url $Url --retry 5 --retry-delay 5 --fail
        Write-Host "Extracting $Label..."
        $ExtractProc = Start-Process -FilePath ".\webimage.exe" -ArgumentList @(
            "-s", "-x", "-f", "webimage_extracted", "--log", "extract.log"
        ) -Wait -PassThru -NoNewWindow
        if ($ExtractProc.ExitCode -ne 0) {
            throw "$Label webimage extract failed with exit code $($ExtractProc.ExitCode)"
        }
        Remove-Item webimage.exe -Force
        $Bootstrapper = Join-Path $WorkDir "webimage_extracted\bootstrapper.exe"
        if (-not (Test-Path $Bootstrapper)) {
            throw "bootstrapper.exe not found after extracting $Label"
        }
        Write-Host "Installing $Label components: $Components"
        $InstallArgs = @(
            "-s", "--action", "install",
            "--components=$Components",
            "--eula=accept",
            "-p=NEED_VS2017_INTEGRATION=0",
            "-p=NEED_VS2019_INTEGRATION=0",
            "-p=NEED_VS2022_INTEGRATION=1",
            "--log-dir=$WorkDir"
        )
        $InstallProc = Start-Process -FilePath $Bootstrapper -ArgumentList $InstallArgs -Wait -PassThru -NoNewWindow
        if ($InstallProc.ExitCode -ne 0) {
            throw "$Label bootstrapper install failed with exit code $($InstallProc.ExitCode)"
        }
    } finally {
        Pop-Location
        Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
    }
}

$BaseUrl = $env:ONEAPI_BASEKIT_URL
$HpcUrl = $env:ONEAPI_HPCKIT_URL
if (-not $BaseUrl -or -not $HpcUrl) {
    throw "ONEAPI_BASEKIT_URL and ONEAPI_HPCKIT_URL must be set."
}

# Base kit: DPC++ compiler (+ runtime deps), oneTBB, oneDPL, IPP (see .gitlab-ci.yml component names).
$BaseComponents = $env:ONEAPI_WIN_BASE_COMPONENTS
if (-not $BaseComponents) {
    $BaseComponents = @(
        "intel.oneapi.win.dpcpp-compiler",
        "intel.oneapi.win.tbb.devel",
        "intel.oneapi.win.dpl.devel",
        "intel.oneapi.win.ipp.devel"
    ) -join ","
}

# HPC kit: oneMKL only (SYCL build uses OPEN3D_USE_ONEAPI_PACKAGES / static MKL).
$HpcComponents = $env:ONEAPI_WIN_HPC_COMPONENTS
if (-not $HpcComponents) {
    $HpcComponents = "intel.oneapi.win.mkl.devel"
}

Install-OneApiWebimage -Url $BaseUrl -Components $BaseComponents -Label "oneAPI Base Toolkit"
Install-OneApiWebimage -Url $HpcUrl -Components $HpcComponents -Label "oneAPI HPC Toolkit"

$toolsetProps = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "*Intel*oneAPI*Compiler*.props" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $toolsetProps) {
    throw "Intel oneAPI DPC++ Visual Studio toolset was not installed correctly."
}
Write-Host "Found Intel DPC++ VS toolset: $($toolsetProps.FullName)"
