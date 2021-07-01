// -*- C++ -*-
//===-- reduction.h -------------------------------------------------------===//
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

#ifndef __PSTL_experimental_reduction_H
#define __PSTL_experimental_reduction_H

#include <type_traits>
#include <functional>

#include "../../internal/utils.h"
#include "reduction_impl.h"

namespace std
{
namespace experimental
{
inline namespace parallelism_v2
{

// Reduction functions definitions

// Generic version of std::reduction, all the specific ones are implemented in terms of it.
template <typename _Tp, typename _BinaryOperation>
__pstl::__internal::__reduction_object<_Tp, _BinaryOperation>
reduction(_Tp& __var, const _Tp& __identity, _BinaryOperation __combiner)
{
    return __pstl::__internal::__reduction_object<_Tp, _BinaryOperation>(__var, __identity, __combiner);
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, std::plus<_Tp>>
reduction_plus(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, _Tp(), std::plus<_Tp>());
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, std::multiplies<_Tp>>
reduction_multiplies(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, _Tp(1), std::multiplies<_Tp>());
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, decltype(std::bit_and<_Tp>{})>
reduction_bit_and(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, ~_Tp(), std::bit_and<_Tp>{});
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, decltype(std::bit_or<_Tp>{})>
reduction_bit_or(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, _Tp(), std::bit_or<_Tp>{});
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, decltype(std::bit_xor<_Tp>{})>
reduction_bit_xor(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, _Tp(), std::bit_xor<_Tp>{});
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, decltype(__pstl::__internal::__pstl_min{})>
reduction_min(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, __var, __pstl::__internal::__pstl_min{});
}

template <typename _Tp>
__pstl::__internal::__reduction_object<_Tp, decltype(__pstl::__internal::__pstl_max{})>
reduction_max(_Tp& __var)
{
    return std::experimental::parallelism_v2::reduction(__var, __var, __pstl::__internal::__pstl_max{});
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace std

#endif /* __PSTL_experimental_reduction_H */
