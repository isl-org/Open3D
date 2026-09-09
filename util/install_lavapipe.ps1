# Install the pinned Windows Lavapipe runtime for software Vulkan CI.
$ErrorActionPreference = "Stop"

$url = "https://github.com/jakoch/rasterizers/releases/download/20260730/lavapipe-x64-26.1.5.zip"
$sha256 = "89a6af938bc6aa60989d468d1a7d5e05f443661ad210fb2e3b2f710834a87cdc"
$archive = Join-Path $env:RUNNER_TEMP "lavapipe-x64-26.1.5.zip"
$root = Join-Path $env:RUNNER_TEMP "lavapipe"

Invoke-WebRequest -Uri $url -OutFile $archive
if ((Get-FileHash -Path $archive -Algorithm SHA256).Hash.ToLower() -ne $sha256) {
    throw "Lavapipe archive checksum mismatch"
}

Expand-Archive -Path $archive -DestinationPath $root -Force
$icd = Join-Path $root "share\vulkan\icd.d\lvp_icd.x86_64.json"
if (-not (Test-Path $icd)) {
    throw "Lavapipe ICD not found: $icd"
}

(Join-Path $root "bin") | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
"VK_DRIVER_FILES=$icd" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append

# Lavapipe only ships the ICD (vulkan_lvp.dll); it does not provide the Vulkan
# loader that applications dynamically link against (BlueVK / vulkan.hpp call
# LoadLibraryA("vulkan-1.dll") on Windows). Install the loader via vcpkg, which
# is preinstalled on GitHub-hosted windows-2022 runners.
vcpkg install --classic vulkan-loader:x64-windows
$loaderBinPath = Join-Path $env:VCPKG_INSTALLATION_ROOT "installed\x64-windows\bin"
$loaderDll = Join-Path $loaderBinPath "vulkan-1.dll"
if (-not (Test-Path $loaderDll)) {
    throw "Vulkan loader not found: $loaderDll"
}
$loaderBinPath | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append