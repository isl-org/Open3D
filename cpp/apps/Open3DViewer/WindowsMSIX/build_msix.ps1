<#
.SYNOPSIS
    Stages, signs, and packs the Open3D Viewer as a side-load MSIX.

.PARAMETER SrcDir
    Open3D repository root (contains cpp/apps/Open3DViewer/WindowsMSIX).
.PARAMETER InstallDir
    Directory the viewer was installed to (contains Open3D.exe + resources/).
.PARAMETER CMakeCachePath
    Path to CMakeCache.txt, used to read OPEN3D_VERSION_FULL.
.PARAMETER OutDir
    Directory to write the staged files, cert, and final .msix/.cer into.
#>
param(
    [Parameter(Mandatory = $true)][string]$SrcDir,
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$CMakeCachePath,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = 'Stop'

# Build a four-part MSIX version from the Open3D version in CMakeCache.
# MSIX requires Major.Minor.Patch.Build (all numeric).
$RAW_VER = (Select-String -Path $CMakeCachePath `
    -Pattern "OPEN3D_VERSION_FULL:STRING=").Line.Split('=')[1].Trim()
$MSIX_VERSION = ($RAW_VER -replace '[^0-9.]', '').TrimEnd('.') + '.0'
$OPEN3D_VERSION = $RAW_VER -replace '[^0-9.]', ''

# Stage: start from the already-installed viewer folder and add manifest + assets.
$STAGING = Join-Path $OutDir "msix-staging"
New-Item -ItemType Directory -Force -Path $STAGING | Out-Null
Copy-Item -Recurse (Join-Path $InstallDir "*") $STAGING
$SRC_MSIX = Join-Path $SrcDir "cpp\apps\Open3DViewer\WindowsMSIX"
Copy-Item (Join-Path $SRC_MSIX "AppxManifest.xml") $STAGING
Copy-Item -Recurse (Join-Path $SRC_MSIX "Assets") $STAGING

# Substitute version placeholder in the staged manifest.
$manifestPath = Join-Path $STAGING "AppxManifest.xml"
(Get-Content $manifestPath) -replace '@OPEN3D_MSIX_VERSION@', $MSIX_VERSION |
    Set-Content $manifestPath

# Generate self-signed cert whose subject matches Publisher="CN=Open3D".
# Export .cer (public key only) so users can install it to trust the package.
winapp cert generate `
    --manifest $manifestPath `
    --output (Join-Path $OutDir "Open3D.pfx") `
    --export-cer `
    --if-exists overwrite

# Pack and sign the MSIX.
$MSIX_NAME = "Open3DViewer-$OPEN3D_VERSION-x64.msix"
winapp pack $STAGING `
    --output (Join-Path $OutDir $MSIX_NAME) `
    --cert (Join-Path $OutDir "Open3D.pfx")

echo "MSIX_NAME=$MSIX_NAME" | Out-File -FilePath $Env:GITHUB_ENV -Encoding utf8 -Append
