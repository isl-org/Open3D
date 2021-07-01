// -*- C++ -*-
//===-- parallel_backend.h ------------------------------------------------===//
//
// Copyright (C) 2017-2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef __PSTL_parallel_backend_H
#define __PSTL_parallel_backend_H

#if __PSTL_PAR_BACKEND_TBB
#    include "parallel_backend_tbb.h"
#else
__PSTL_PRAGMA_MESSAGE("Parallel backend was not specified");
#endif

#endif /* __PSTL_parallel_backend_H */
