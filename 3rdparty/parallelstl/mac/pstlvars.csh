#!/bin/csh
##===-- pstlvars.csh ------------------------------------------------------===##
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

setenv PSTLROOT "SUBSTITUTE_INSTALL_DIR_HERE"

if ( -e $PSTLROOT/../tbb/bin/tbbvars.csh ) then
   source $PSTLROOT/../tbb/bin/tbbvars.csh;
endif

if (! $?CPATH) then
    setenv CPATH "${PSTLROOT}/include"
else
    setenv CPATH "${PSTLROOT}/include:$CPATH"
endif
