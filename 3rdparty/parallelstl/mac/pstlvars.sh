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

export PSTLROOT="SUBSTITUTE_INSTALL_DIR_HERE"

if [ -e $PSTLROOT/../tbb/bin/tbbvars.sh ]; then
   . $PSTLROOT/../tbb/bin/tbbvars.sh
fi

if [ -z "${CPATH}" ]; then
    CPATH="${PSTLROOT}/include"; export CPATH
else
    CPATH="${PSTLROOT}/include:$CPATH"; export CPATH
fi
