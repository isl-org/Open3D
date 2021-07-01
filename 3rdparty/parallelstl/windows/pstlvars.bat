@echo off
REM ===-- pstlvars.bat ------------------------------------------------------===##
REM
REM Copyright (C) 2017-2019 Intel Corporation
REM
REM SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
REM
REM This file incorporates work covered by the following copyright and permission
REM notice:
REM
REM Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
REM See https://llvm.org/LICENSE.txt for license information.
REM
REM ===----------------------------------------------------------------------===##

set SCRIPT_NAME=%~nx0
set PSTL_BIN_DIR=%~d0%~p0
set PSTLROOT=%PSTL_BIN_DIR%..

:: Set the default arguments
set PSTL_TARGET_ARCH=
set PSTL_TARGET_VS=

:ParseArgs
:: Parse the incoming arguments
if /i "%1"==""        goto SetEnv
if /i "%1"=="ia32"         (set PSTL_TARGET_ARCH=ia32)    & shift & goto ParseArgs
if /i "%1"=="intel64"      (set PSTL_TARGET_ARCH=intel64) & shift & goto ParseArgs
if /i "%1"=="vs2013"       (set PSTL_TARGET_VS=vs2013)      & shift & goto ParseArgs
if /i "%1"=="vs2015"       (set PSTL_TARGET_VS=vs2015)      & shift & goto ParseArgs
if /i "%1"=="vs2017"       (set PSTL_TARGET_VS=vs2017)      & shift & goto ParseArgs
if /i "%1"=="vs2019"       (set PSTL_TARGET_VS=vs2019)      & shift & goto ParseArgs
if /i "%1"=="all"          (set PSTL_TARGET_VS=all)     & shift & goto ParseArgs
:: for any other incoming arguments values
goto Syntax

:SetEnv
:: target architecture is a required argument
if "%PSTL_TARGET_ARCH%"=="" goto Syntax
:: PSTL_TARGET_VS default value is 'vc_mt' (all)
if "%PSTL_TARGET_VS%"=="" set PSTL_TARGET_VS=all

if exist "%PSTLROOT%\..\tbb\bin\tbbvars.bat" @call "%PSTLROOT%\..\tbb\bin\tbbvars.bat" %PSTL_TARGET_ARCH% %PSTL_TARGET_VS%

set INCLUDE=%PSTLROOT%\include;%INCLUDE%
set CPATH=%PSTLROOT%\include;%CPATH%

goto End

:Syntax
echo Syntax:
echo  %SCRIPT_NAME% ^<arch^> ^<vs^>
echo    ^<arch^> must be one of the following
echo        ia32         : Set up for IA-32  architecture
echo        intel64      : Set up for Intel(R) 64  architecture
echo    ^<vs^> should be one of the following
echo        vs2013      : Set to use with Microsoft Visual Studio 2013 runtime DLLs
echo        vs2015      : Set to use with Microsoft Visual Studio 2015 runtime DLLs
echo        vs2017      : Set to use with Microsoft Visual Studio 2017 runtime DLLs
echo        vs2019      : Set to use with Microsoft Visual Studio 2019 runtime DLLs
echo        all         : Set PSTL to use TBB statically linked with Microsoft Visual C++ runtime
echo    if ^<vs^> is not set PSTL will use TBB statically linked with Microsoft Visual C++ runtime.
exit /B 1

:End
exit /B 0