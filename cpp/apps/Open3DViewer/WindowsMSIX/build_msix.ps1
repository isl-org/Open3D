<#
.SYNOPSIS
    Stages, signs, and packs the Open3D Viewer as a side-load MSIX.

.PARAMETER InstallDir
    Directory the viewer was installed to (contains Open3D.exe + resources/).
.PARAMETER Version
    Three-part Open3D version from CMake.
.PARAMETER OutDir
    Directory to write the staged files, cert, and final .msix/.cer into.
#>
param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = 'Stop'

$OPEN3D_VERSION = $Version
$MSIX_VERSION = "$Version.0"

# Stage the already-installed viewer and add the manifest.
$STAGING = Join-Path $OutDir "msix-staging"
Remove-Item -Recurse -Force $STAGING -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $STAGING | Out-Null
Copy-Item (Join-Path $InstallDir "Open3D.exe") $STAGING
Copy-Item (Join-Path $InstallDir "tbb*.dll") $STAGING
Copy-Item -Recurse (Join-Path $InstallDir "resources") $STAGING
Copy-Item (Join-Path $PSScriptRoot "AppxManifest.xml") $STAGING

# Substitute version placeholder in the staged manifest.
$manifestPath = Join-Path $STAGING "AppxManifest.xml"
(Get-Content $manifestPath) -replace '@OPEN3D_MSIX_VERSION@', $MSIX_VERSION |
    Set-Content $manifestPath

# Generate self-signed cert whose subject matches Publisher="CN=Open3D".
# Export .cer (public key only) so users can install it to trust the package.
$MSIX_NAME = "Open3DViewer-$OPEN3D_VERSION-x64.msix"
$MSIX_PATH = Join-Path $OutDir $MSIX_NAME
$PFX_PATH = Join-Path $OutDir "Open3D.pfx"
try {
    winapp cert generate `
        --manifest $manifestPath `
        --output $PFX_PATH `
        --export-cer `
        --if-exists overwrite

    # Pack and sign the MSIX. The private key is removed immediately after use.
    winapp pack $STAGING `
        --output $MSIX_PATH `
        --cert $PFX_PATH
}
finally {
    Remove-Item $PFX_PATH -Force -ErrorAction SilentlyContinue
}

# If running in GitHub Actions, export the MSIX name and path to the environment.
if ($env:GITHUB_ENV) {
    echo "MSIX_NAME=$MSIX_NAME" | Out-File -FilePath $Env:GITHUB_ENV -Encoding utf8 -Append
    echo "MSIX_PATH=$MSIX_PATH" | Out-File -FilePath $Env:GITHUB_ENV -Encoding utf8 -Append
}
