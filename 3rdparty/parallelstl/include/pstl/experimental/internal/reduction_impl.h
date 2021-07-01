// -*- C++ -*-
//===-- reduction_impl.h --------------------------------------------------===//
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

#ifndef __PSTL_experimental_reduction_impl_H
#define __PSTL_experimental_reduction_impl_H

#include <type_traits>
#include <algorithm>

namespace __pstl
{
namespace __internal
{

template <typename _Tp, typename _Combiner>
class __reduction_object
{
    static_assert(std::is_copy_constructible<_Tp>::value, "_Tp shall be CopyConstructible");
    static_assert(std::is_move_assignable<_Tp>::value, "_Tp shall be MoveAssignable");

    // Reference to the original variable. It's only used at the end of execution
    // when finalize method is called and accumulated value stored in acc is written
    // to value.
    // TODO: we probably don't need to keep this reference for each object, as we need it
    // only at the end of execution. Need to check whether it can be impelemented efficently.
    // I.e. by intoducing an internal reduction object on which the operation are applied.
    _Tp& __value_;
    // Current accumulated value.
    _Tp __acc_;
    _Combiner __combiner_;

  public:
    __reduction_object(_Tp& __value, _Tp __identity, _Combiner __combiner)
        : __value_(__value), __acc_(__identity), __combiner_(__combiner)
    {
    }

    __reduction_object(const __reduction_object& __other) = default;

    __reduction_object&
    operator=(__reduction_object&& __other)
    {
        // No need to move, passed by reference
        __value_ = __other.__value_;
        __acc_ = std::move(__other.__acc_);
        __combiner_ = std::move(__other.__combiner_);
        return *this;
    }

    // Get a reference to current accumulated value.
    // The reference is required so that it could be modified from a functor.
    // The parameter is here only for consistency with induction object and ignored here.
    template <typename _Index>
    _Tp& __get_induction_or_reduction_value(_Index)
    {
        return __acc_;
    }

    // Combine 2 reduction objects together.
    void
    __combine(const __reduction_object& __other)
    {
        __acc_ = __combiner_(__acc_, __other.__acc_);
    }

    // This method is called on the last object in the ruduction sequence which accumulates
    // the final result in acc field. It simply adds acc to value using user's combimer.
    // The parameter is here only for consistency with induction object and ignored here.
    template <typename _RangeSize>
    void
    __finalize(const _RangeSize)
    {
        __value_ = __combiner_(__value_, __acc_);
    }
};

} // namespace __internal
} // namespace __pstl

#endif /* __PSTL_experimental_reduction_impl_H */
