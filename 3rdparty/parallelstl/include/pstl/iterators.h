// -*- C++ -*-
//===-- iterators.h -------------------------------------------------------===//
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

#ifndef __PSTL_iterators_H
#define __PSTL_iterators_H

#include <tbb/tbb_stddef.h>

#if TBB_VERSION_MAJOR < 2019
#error Threading Building Blocks (TBB) 2019 is required for usage of special iterator types
#else

#include <tbb/iterators.h>

namespace pstl
{
using tbb::counting_iterator;
using tbb::make_transform_iterator;
using tbb::make_zip_iterator;
using tbb::transform_iterator;
using tbb::zip_iterator;
} //namespace __pstl

#endif //TBB_VERSION_MAJOR < 2019

#endif /* __PSTL_iterators_H */
