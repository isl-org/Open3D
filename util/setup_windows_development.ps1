# Bootstrap the tools needed to build Open3D on Windows.
# Example: .\util\setup_windows_development.ps1 -CloneDirectory Open3D

[CmdletBinding()]
param(
    [string]$CloneDirectory,
    [string]$EnvironmentDirectory = '.venv',
    [switch]$InstallJupyter
)

# Add winget to path.
$env:Path += ";$env:LOCALAPPDATA\Microsoft\WindowsApps"

$gitPackage = 'Git.Git'
$vsCodePackage = 'Microsoft.VisualStudioCode'
$buildToolsPackage = 'Microsoft.VisualStudio.2022.BuildTools'
$buildToolsOverride = '--wait --passive --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100 --add Microsoft.VisualStudio.Component.VC.CMake.Project'
$cmakePackage = 'Kitware.CMake'
$pythonPackage = 'Python.Python.3.14'
$vulkanSdkPackage = 'KhronosGroup.VulkanSDK'

function Install-WingetPackage([string]$PackageId, [string]$Override = '') {
    if (-not (winget.exe list --id $PackageId --exact --accept-source-agreements |
            Select-String -SimpleMatch $PackageId)) {
        $arguments = @(
            'install', '--id', $PackageId, '--exact', '--source', 'winget', '--silent',
            '--accept-package-agreements', '--accept-source-agreements',
            '--disable-interactivity'
        )
        if ($Override) {
            $arguments += @('--override', $Override)
        }
        & winget.exe @arguments
    }
}

Install-WingetPackage $gitPackage
Install-WingetPackage $vsCodePackage
Install-WingetPackage $buildToolsPackage $buildToolsOverride
Install-WingetPackage $cmakePackage
Install-WingetPackage $pythonPackage
Install-WingetPackage $vulkanSdkPackage

$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
    [System.Environment]::GetEnvironmentVariable('Path', 'User')

if ($CloneDirectory) {
    if ([System.IO.Path]::IsPathRooted($CloneDirectory)) {
        $destination = [System.IO.Path]::GetFullPath($CloneDirectory)
    } else {
        $destination = Join-Path (Get-Location) $CloneDirectory
    }
    if (Test-Path -LiteralPath $destination) {
        if (Test-Path -LiteralPath (Join-Path $destination '.git')) {
            Write-Host "Open3D already cloned at $destination"
        } elseif (Get-ChildItem -LiteralPath $destination -Force | Select-Object -First 1) {
            throw "Clone destination is not empty: $destination"
        } else {
            git clone https://github.com/isl-org/Open3D.git $destination
        }
    } else {
        New-Item -ItemType Directory -Path $destination | Out-Null
        git clone https://github.com/isl-org/Open3D.git $destination
    }
} else {
    $destination = (Get-Location).Path
}

$environmentPath = if ([System.IO.Path]::IsPathRooted($EnvironmentDirectory)) {
    [System.IO.Path]::GetFullPath($EnvironmentDirectory)
} else {
    Join-Path $destination $EnvironmentDirectory
}
if (-not (Test-Path -LiteralPath $environmentPath)) {
    py.exe -3.14 -m venv $environmentPath
}

$pythonExecutable = Join-Path $environmentPath 'Scripts\python.exe'
& $pythonExecutable -m pip install --upgrade pip
& $pythonExecutable -m pip install -r (Join-Path $destination 'python\requirements_build.txt')
if ($InstallJupyter) {
    & $pythonExecutable -m pip install -r (Join-Path $destination 'python\requirements_jupyter_build.txt')
}