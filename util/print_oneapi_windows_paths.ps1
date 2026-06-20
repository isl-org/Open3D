# Print Intel oneAPI layout and CMake-related paths on Windows (CI debugging).
# Usage: ./util/print_oneapi_windows_paths.ps1 [-Verbose]
# Env: OPEN3D_ONEAPI_DEBUG=1 enables extra detail (same as -Verbose).
param(
    [switch]$Verbose
)

$ErrorActionPreference = 'Continue'
if ($env:OPEN3D_ONEAPI_DEBUG -eq '1') {
    $Verbose = $true
}

function Write-Section($Title) {
    Write-Host ""
    Write-Host "========== $Title =========="
}

function Test-PathReport($Label, $Path) {
    $exists = Test-Path -LiteralPath $Path
    $mark = if ($exists) { '[OK]' } else { '[MISSING]' }
    Write-Host "$mark $Label"
    Write-Host "     $Path"
    return $exists
}

function Find-FirstFile($Roots, $Pattern) {
    foreach ($root in $Roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $hit = Get-ChildItem -LiteralPath $root -Recurse -Filter $Pattern -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $hit }
    }
    return $null
}

$OneApiRoot = 'C:\Program Files (x86)\Intel\oneAPI'
$MklCmakeDir = Join-Path $OneApiRoot 'mkl\latest\lib\cmake\mkl'
$TbbRoot = Join-Path $OneApiRoot 'tbb\latest'
$CompilerLatest = Join-Path $OneApiRoot 'compiler\latest'

Write-Section 'oneAPI environment variables'
foreach ($name in @(
        'ONEAPI_ROOT', 'MKLROOT', 'TBBROOT', 'CMPLR_ROOT', 'DPL_ROOT', 'IPP_ROOT',
        'CMAKE_PREFIX_PATH', 'PATH'
    )) {
    $val = [Environment]::GetEnvironmentVariable($name, 'Process')
    if (-not $val) { $val = [Environment]::GetEnvironmentVariable($name, 'Machine') }
    if (-not $val) { $val = [Environment]::GetEnvironmentVariable($name, 'User') }
    if ($name -eq 'PATH' -and $val) {
        Write-Host "${name}: (length $($val.Length); use -Verbose for Intel entries)"
        if ($Verbose) {
            $val -split ';' | Where-Object { $_ -match 'Intel|oneAPI|icx' } | ForEach-Object { Write-Host "     $_" }
        }
    } elseif ($val) {
        Write-Host "${name}=$val"
    } else {
        Write-Host "${name}=<not set>"
    }
}

Write-Section 'oneAPI install tree (Open3D SYCL expectations)'
Test-PathReport 'oneAPI root' $OneApiRoot | Out-Null
Test-PathReport 'MKLConfig.cmake dir' $MklCmakeDir | Out-Null
Test-PathReport 'TBB package root (CMAKE_PREFIX_PATH)' $TbbRoot | Out-Null
Test-PathReport 'DPC++ compiler latest symlink' $CompilerLatest | Out-Null

$mklConfig = Join-Path $MklCmakeDir 'MKLConfig.cmake'
Test-PathReport 'MKLConfig.cmake' $mklConfig | Out-Null
$tbbConfig = Join-Path $TbbRoot 'lib\cmake\tbb\TBBConfig.cmake'
Test-PathReport 'TBBConfig.cmake' $tbbConfig | Out-Null

$icx = Join-Path $CompilerLatest 'bin\icx.exe'
Test-PathReport 'icx.exe' $icx | Out-Null

Write-Section 'OpenCL (Windows SYCL / mkl_sycl link)'
$openClLib = Find-FirstFile @($CompilerLatest) 'OpenCL.lib'
if ($openClLib) {
    Write-Host '[OK] OpenCL.lib'
    Write-Host "     $($openClLib.FullName)"
    $incCandidate = Join-Path $openClLib.Directory.Parent.FullName 'include'
    Test-PathReport 'OpenCL include (parent/include)' $incCandidate | Out-Null
} else {
    Write-Host '[MISSING] OpenCL.lib under compiler\latest'
    Write-Host "     Search root: $CompilerLatest"
}

Write-Section 'Visual Studio Intel DPC++ platform toolset (cmake -T)'
$vsRoots = @(
    Join-Path $env:ProgramFiles 'Microsoft Visual Studio'
    Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio'
)
$toolsetProps = $null
foreach ($vsRoot in $vsRoots) {
    if (-not (Test-Path -LiteralPath $vsRoot)) { continue }
    $toolsetProps = Get-ChildItem -LiteralPath $vsRoot -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq 'Toolset.props' -and
            $_.FullName -match '\\PlatformToolsets\\Intel\(R\) oneAPI DPC\+\+ Compiler\\'
        } |
        Select-Object -First 1
    if ($toolsetProps) { break }
}
if ($toolsetProps) {
    Write-Host '[OK] Intel(R) oneAPI DPC++ Compiler Toolset.props'
    Write-Host "     $($toolsetProps.FullName)"
} else {
    Write-Host '[MISSING] Platform toolset Toolset.props (CMake -T may fail at build with MSB8020)'
    foreach ($vsRoot in $vsRoots) {
        Write-Host "     Searched under: $vsRoot"
    }
}

Write-Section 'Suggested CMake configure hints (windows.yml xpu legs)'
Write-Host '-T "Intel(R) oneAPI DPC++ Compiler"'
Write-Host "-DCMAKE_PREFIX_PATH=$MklCmakeDir;$TbbRoot"
if ($openClLib) {
    $oclLib = $openClLib.FullName.Replace('\', '/')
    $oclInc = (Join-Path $openClLib.Directory.Parent.FullName 'include').Replace('\', '/')
    Write-Host "-DOpenCL_LIBRARY=$oclLib"
    Write-Host "-DOpenCL_INCLUDE_DIR=$oclInc"
} else {
    Write-Host '-DOpenCL_LIBRARY=<not found; find_package(OpenCL) may fail>'
}

if ($Verbose) {
    Write-Section 'Directory listing (verbose)'
    foreach ($sub in @('compiler', 'mkl', 'tbb', 'dpl', 'ipp')) {
        $p = Join-Path $OneApiRoot $sub
        if (Test-Path -LiteralPath $p) {
            Write-Host "--- $p ---"
            Get-ChildItem -LiteralPath $p -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "     $($_.Name)" }
        }
    }
}

Write-Section 'setvars (optional alternative to CMAKE_PREFIX_PATH)'
$setvars = Join-Path $OneApiRoot 'setvars-vcvarsall.bat'
Test-PathReport 'setvars-vcvarsall.bat' $setvars | Out-Null
Write-Host 'Local builds often run: cmd /c "setvars-vcvarsall.bat" vs2022 before cmake.'
