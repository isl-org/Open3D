// -*- C++ -*-
//===-- induction.h -------------------------------------------------------===//
//
// Copyright (C) 2019 Intel Corporation
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

#ifndef __PSTL_experimental_induction_H
#define __PSTL_experimental_induction_H

#include <type_traits>

#include "induction_impl.h"

namespace std
{
namespace experimental
{
inline namespace parallelism_v2
{

template <typename _Tp>
__pstl::__internal::__induction_object<_Tp, void>
induction(_Tp&& __var)
{
    return {std::forward<_Tp>(__var)};
}

template <typename _Tp, typename _Sp>
__pstl::__internal::__induction_object<_Tp, _Sp>
induction(_Tp&& __var, _Sp __stride)
{
    return {std::forward<_Tp>(__var), __stride};
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace std

#endif /* __PSTL_experimental_induction_H */
