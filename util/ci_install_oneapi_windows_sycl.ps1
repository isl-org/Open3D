# Install oneMKL and Intel oneAPI DPC++/C++ Compiler for Windows SYCL (xpu) CI builds.
# Requires env ONEAPI_MKL_URL and ONEAPI_COMPILER_URL (DPC++/C++ offline installer, not Fortran).
# See https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html
$ErrorActionPreference = 'Stop'

$MKL_URL = $env:ONEAPI_MKL_URL
$DPCPP_URL = $env:ONEAPI_COMPILER_URL
if (-not $MKL_URL -or -not $DPCPP_URL) {
    throw "ONEAPI_MKL_URL and ONEAPI_COMPILER_URL must be set."
}

Write-Host "Downloading oneMKL..."
Invoke-WebRequest -Uri $MKL_URL -OutFile "intel-onemkl.exe"
Write-Host "Downloading Intel DPC++ Compiler..."
Invoke-WebRequest -Uri $DPCPP_URL -OutFile "intel-dpcpp-compiler.exe"
Write-Host "Installing oneMKL..."
$p1 = Start-Process -FilePath ".\intel-onemkl.exe" -ArgumentList "-s -a --silent --eula accept" -Wait -PassThru
if ($p1.ExitCode -ne 0) { throw "oneMKL installation failed with exit code $($p1.ExitCode)" }
Write-Host "Installing Intel DPC++ Compiler..."
$p2 = Start-Process -FilePath ".\intel-dpcpp-compiler.exe" -ArgumentList "-s -a --silent --eula accept" -Wait -PassThru
if ($p2.ExitCode -ne 0) { throw "Intel DPC++ Compiler installation failed with exit code $($p2.ExitCode)" }

$toolsetProps = Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter "*Intel*oneAPI*Compiler*.props" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $toolsetProps) {
    throw "Intel oneAPI DPC++ Visual Studio toolset was not installed correctly."
}
Write-Host "Found Intel DPC++ VS toolset: $($toolsetProps.FullName)"
