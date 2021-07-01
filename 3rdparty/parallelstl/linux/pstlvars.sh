#!/bin/sh
##===-- pstlvars.sh -------------------------------------------------------===##
#
# Copyright (C) 2017-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

# Parsing script arguments
# Arg1 represents target architecture. Its possible values are 'ia32' or 'intel64',
# default value equals to the value of $COMPILERVARS_ARCHITECTURE environment variable.

PSTL_TARGET_ARCH=

if [ -n "${COMPILERVARS_ARCHITECTURE}" ]; then
    PSTL_TARGET_ARCH=$COMPILERVARS_ARCHITECTURE
fi

if [ -n "$1" ]; then
    PSTL_TARGET_ARCH=$1
fi

if [ -n "${PSTL_TARGET_ARCH}" ]; then
    if [ "$PSTL_TARGET_ARCH" != "ia32" -a "$PSTL_TARGET_ARCH" != "intel64" ]; then
        echo "ERROR: Unknown switch '$PSTL_TARGET_ARCH'. Accepted values: ia32, intel64"
        PSTL_TARGET_ARCH=
        return 1;
    fi
else
    echo "ERROR: Architecture is not defined. Accepted values: ia32, intel64"
    return 1;
fi

# Arg2 represents PSTLROOT detection method. Its possible value is 'auto_pstlroot'. In which case
# the environment variable PSTLROOT is detected automatically by using the script directory path.
PSTLROOT=SUBSTITUTE_INSTALL_DIR_HERE
if [ -n "${BASH_SOURCE}" ]; then
    if [ "$2" = "auto_pstlroot" ]; then
       PSTLROOT=$(cd $(dirname ${BASH_SOURCE}) && pwd -P)/..
    fi
fi
export PSTLROOT

if [ -e $PSTLROOT/../tbb/bin/tbbvars.sh ]; then
   . $PSTLROOT/../tbb/bin/tbbvars.sh $PSTL_TARGET_ARCH
fi

if [ -z "${CPATH}" ]; then
    CPATH="${PSTLROOT}/include"; export CPATH
else
    CPATH="${PSTLROOT}/include:$CPATH"; export CPATH
fi
