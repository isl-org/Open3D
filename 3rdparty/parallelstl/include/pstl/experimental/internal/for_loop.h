// -*- C++ -*-
//===-- for_loop.h --------------------------------------------------------===//
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

#ifndef __PSTL_experimental_for_loop_H
#define __PSTL_experimental_for_loop_H

#include <tuple>

#include "../../internal/pstl_config.h"
#include "../../internal/execution_impl.h"
#include "../../internal/utils.h"
#include "for_loop_impl.h"

namespace std
{
namespace experimental
{
inline namespace parallelism_v2
{

// TODO: type_identity should be available in type_traits starting from C++20
// Perhaps we need to add an internal structure if PSTL is used with older versions
template <typename _Tp>
struct type_identity
{
    using type = _Tp;
};
template <typename _Tp>
using type_identity_t = typename type_identity<_Tp>::type;

// TODO: add static asserts for parameters according to the requirements
template <typename _ExecutionPolicy, typename _Ip, typename... _Rest>
void
for_loop(_ExecutionPolicy&& __exec, type_identity_t<_Ip> __start, _Ip __finish, _Rest&&... __rest)
{
    __pstl::__internal::__for_loop_repack(std::forward<_ExecutionPolicy>(__exec), __start, __finish,
                                          __pstl::__internal::__single_stride_type{},
                                          std::forward_as_tuple(std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Sp, typename... _Rest>
void
for_loop_strided(_ExecutionPolicy&& __exec, type_identity_t<_Ip> __start, _Ip __finish, _Sp __stride, _Rest&&... __rest)
{
    __pstl::__internal::__for_loop_repack(std::forward<_ExecutionPolicy>(__exec), __start, __finish, __stride,
                                          std::forward_as_tuple(std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename... _Rest>
void
for_loop_n(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Rest&&... __rest)
{
    __pstl::__internal::__for_loop_repack_n(std::forward<_ExecutionPolicy>(__exec), __start, __n,
                                            __pstl::__internal::__single_stride_type{},
                                            std::forward_as_tuple(std::forward<_Rest>(__rest)...));
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Sp, typename... _Rest>
void
for_loop_n_strided(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Sp __stride, _Rest&&... __rest)
{
    __pstl::__internal::__for_loop_repack_n(std::forward<_ExecutionPolicy>(__exec), __start, __n, __stride,
                                            std::forward_as_tuple(std::forward<_Rest>(__rest)...));
}

// Serial implementations
template <typename _Ip, typename... _Rest>
void
for_loop(type_identity_t<_Ip> __start, _Ip __finish, _Rest&&... __rest)
{
    std::experimental::parallelism_v2::for_loop(__pstl::execution::v1::seq, __start, __finish,
                                                std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Sp, typename... _Rest>
void
for_loop_strided(type_identity_t<_Ip> __start, _Ip __finish, _Sp __stride, _Rest&&... __rest)
{
    std::experimental::parallelism_v2::for_loop_strided(__pstl::execution::v1::seq, __start, __finish, __stride,
                                                        std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Size, typename... _Rest>
void
for_loop_n(_Ip __start, _Size __n, _Rest&&... __rest)
{
    std::experimental::parallelism_v2::for_loop_n(__pstl::execution::v1::seq, __start, __n,
                                                  std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Size, typename _Sp, typename... _Rest>
void
for_loop_n_strided(_Ip __start, _Size __n, _Sp __stride, _Rest&&... __rest)
{
    std::experimental::parallelism_v2::for_loop_n_strided(__pstl::execution::v1::seq, __start, __n, __stride,
                                                          std::forward<_Rest>(__rest)...);
}

} // namespace parallelism_v2
} // namespace experimental
} // namespace std

#endif /* __PSTL_experimental_for_loop_H */
