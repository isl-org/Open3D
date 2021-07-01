// -*- C++ -*-
//===-- induction_impl.h --------------------------------------------------===//
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

#ifndef __PSTL_experimental_induction_impl_H
#define __PSTL_experimental_induction_impl_H

#include <type_traits>

namespace __pstl
{
namespace __internal
{

// Maps _Tp according to following rules:
// _Tp& -> _Tp&
// const _Tp& -> _Tp
// _Tp -> _Tp
// We're not interested in T&& here as the rvalue-ness is stripped from T before
// construction the induction object
template <typename _Tp>
using __induction_value_type =
    typename std::conditional<std::is_lvalue_reference<_Tp>::value &&
                                  !std::is_const<typename std::remove_reference<_Tp>::type>::value,
                              _Tp, typename std::remove_cv<typename std::remove_reference<_Tp>::type>::type>::type;

// Definition of induction_object structure to represend "induction" object.

template <typename _Tp, typename _Sp>
class __induction_object
{
    using __value_type = __induction_value_type<_Tp>;

    __value_type __var_;
    const _Sp __stride_;

  public:
    __induction_object(__value_type __var, _Sp __stride) : __var_(__var), __stride_(__stride) {}

    __induction_object&
    operator=(const __induction_object& __other)
    {
        __var_ = __other.__var_;
        /* stride is always const */
        return *this;
    }

    template <typename _Index>
    typename std::remove_reference<__value_type>::type
    __get_induction_or_reduction_value(_Index __p)
    {
        return __var_ + __p * __stride_;
    }

    void
    __combine(const __induction_object&)
    {
    }

    template <typename _RangeSize>
    void
    __finalize(const _RangeSize __n)
    {
        // This value is discarded if var is not a reference
        __var_ = __n * __stride_;
    }
};

template <typename _Tp>
class __induction_object<_Tp, void>
{
    using __value_type = __induction_value_type<_Tp>;

    __value_type __var_;

  public:
    __induction_object(__value_type __var) : __var_(__var) {}

    __induction_object&
    operator=(const __induction_object& __other)
    {
        __var_ = __other.__var_;
        return *this;
    }

    template <typename _Index>
    typename std::remove_reference<__value_type>::type
    __get_induction_or_reduction_value(_Index __p)
    {
        return __var_ + __p;
    }

    void
    __combine(const __induction_object&)
    {
    }

    template <typename _RangeSize>
    void
    __finalize(const _RangeSize __n)
    {
        // This value is discarded if var is not a reference
        __var_ = __n;
    }
};

} // namespace __internal
} // namespace __pstl

#endif /* __PSTL_experimental_induction_impl_H */
