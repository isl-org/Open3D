// -*- C++ -*-
//===-- iterator_defs.h ---------------------------------------------------===//
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

#ifndef __PSTL_iterator_defs_H
#define __PSTL_iterator_defs_H

#include <iterator>
#include <type_traits>

namespace __pstl
{
namespace __internal
{

// Define __void_type via this structure to handle redefinition issue.
// See CWG 1558 for information about it.
template <typename... _Ts>
struct __make_void_type
{
    using __type = void;
};

template <typename... _Ts>
using __void_type = typename __make_void_type<_Ts...>::__type;

// Internal wrapper around std::iterator_traits as it is required to be
// SFINAE-friendly(not produce "hard" error when _Ip is not an interator)
// only starting with C++17. Alghough many standard library implmentations
// provide it for older versions, we cannot rely on that.
template <typename _Ip, typename = void>
struct __iterator_traits
{
};

template <typename _Ip>
struct __iterator_traits<_Ip,
                         __void_type<typename _Ip::iterator_category, typename _Ip::value_type,
                                     typename _Ip::difference_type, typename _Ip::pointer, typename _Ip::reference>>
    : std::iterator_traits<_Ip>
{
};

// Handles _Tp* and const _Tp* specializations
template <typename _Tp>
struct __iterator_traits<_Tp*, void> : std::iterator_traits<_Tp*>
{
};

// Make is_random_access_iterator not to fail with a 'hard' error when it's used in SFINAE with
// a non-iterator type by providing a default value.
template <typename _IteratorType, typename = void>
struct __is_random_access_iterator_impl : std::false_type
{
};

template <typename _IteratorType>
struct __is_random_access_iterator_impl<_IteratorType,
                                        __void_type<typename __iterator_traits<_IteratorType>::iterator_category>>
    : std::is_same<typename __iterator_traits<_IteratorType>::iterator_category, std::random_access_iterator_tag>
{
};

/* iterator */
template <typename _IteratorType, typename... _OtherIteratorTypes>
struct __is_random_access_iterator
    : std::conditional<__is_random_access_iterator_impl<_IteratorType>::value,
                       __is_random_access_iterator<_OtherIteratorTypes...>, std::false_type>::type
{
};

template <typename _IteratorType>
struct __is_random_access_iterator<_IteratorType> : __is_random_access_iterator_impl<_IteratorType>
{
};

} // namespace __internal
} // namespace __pstl

#endif /* __PSTL_iterator_defs_H */
