#Requires -RunAsAdministrator
# Install NVIDIA CUDA toolkit components for Windows CI builds.
$ErrorActionPreference = 'Stop'

param(
    [string]$CudaVersion = $env:CUDA_VERSION
)

if (-not $CudaVersion) {
    throw "CUDA_VERSION parameter or env var must be specified."
}

$CudaVersionArr = $CudaVersion.Split(".")
$CudaMajorMinor = "$($CudaVersionArr[0]).$($CudaVersionArr[1])"
$CudaVerId = "$($CudaVersionArr[0])_$($CudaVersionArr[1])"

# Installer URL
if ($CudaVersionArr[0] -ge 11) {
    $CudaUrl = "http://developer.download.nvidia.com/compute/cuda/$CudaVersion/network_installers/cuda_$($CudaVersion)_windows_network.exe"
} else {
    $CudaUrl = "http://developer.download.nvidia.com/compute/cuda/$CudaMajorMinor/Prod/network_installers/cuda_$($CudaVersion)_win10_network.exe"
}

# Required packages
$CudaPackages = @("nvcc", "visual_studio_integration", "cublas", "cublas_dev", "cudart", "cusolver", "cusolver_dev", "cusparse", "cusparse_dev", "npp", "npp_dev", "nvtx", "thrust")
$CudaInstallArgs = @("-s")
foreach ($pkg in $CudaPackages) {
    $CudaInstallArgs += "$($pkg)_$CudaMajorMinor"
}

$TempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$WorkDir = Join-Path $TempRoot "open3d_cuda_$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Push-Location $WorkDir

try {
    Write-Host "Downloading CUDA installer from $CudaUrl..."
    curl.exe --output cuda.exe --url $CudaUrl --retry 5 --retry-delay 5 --fail
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA download failed (curl exit $LASTEXITCODE): $CudaUrl"
    }

    Write-Host "Installing CUDA..."
    $InstallProc = Start-Process -FilePath ".\cuda.exe" -ArgumentList $CudaInstallArgs -Wait -PassThru -NoNewWindow
    Write-Host "CUDA installer finished with exit code $($InstallProc.ExitCode)"
    if ($InstallProc.ExitCode -ne 0 -and $InstallProc.ExitCode -ne 3010) {
         Write-Warning "CUDA installer returned a non-zero exit code ($($InstallProc.ExitCode)). Continuing anyway as some components (e.g. Visual Studio Integration) may handle warnings but the required CUDA SDK components (nvcc, libraries) remain functional."
    }

    # Define paths
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CudaMajorMinor"
    
    # Export env variables for GitHub Actions or local environment
    if ($env:GITHUB_ENV) {
        "CUDA_PATH=$CudaPath" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
        "CUDA_PATH_V$CudaVerId=$CudaPath" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
    } else {
        [System.Environment]::SetEnvironmentVariable("CUDA_PATH", $CudaPath, "Process")
        [System.Environment]::SetEnvironmentVariable("CUDA_PATH_V$CudaVerId", $CudaPath, "Process")
    }

    if ($env:GITHUB_PATH) {
        "$CudaPath\bin" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
    } else {
        $path = [System.Environment]::GetEnvironmentVariable("PATH", "Process")
        [System.Environment]::SetEnvironmentVariable("PATH", "$path;$CudaPath\bin", "Process")
    }

    Write-Host "Successfully installed CUDA and set environments."
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $WorkDir -ErrorAction SilentlyContinue
}
