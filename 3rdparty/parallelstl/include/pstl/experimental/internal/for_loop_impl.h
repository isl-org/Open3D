// -*- C++ -*-
//===-- for_loop_impl.h ---------------------------------------------------===//
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

#ifndef __PSTL_experimental_for_loop_impl_H
#define __PSTL_experimental_for_loop_impl_H

#include <iterator>
#include <type_traits>
#include <utility>
#include <tuple>

#include "../../internal/algorithm_impl.h"
#include "../../internal/execution_impl.h"
#include "../../internal/iterator_impl.h"
#include "../../internal/iterator_defs.h"
#include "../../internal/utils.h"

#if __PSTL_USE_PAR_POLICIES
#    include "../../internal/parallel_backend.h"
#endif

namespace __pstl
{
namespace __internal
{

// Generalization of std::advance to work with an argitraty integral type
template <typename _Ip, typename _Diff>
typename std::enable_if<std::is_integral<_Ip>::value>::type
__advance(_Ip& __val, _Diff __diff)
{
    __val += __diff;
}

template <typename _Ip, typename _Diff>
typename std::enable_if<!std::is_integral<_Ip>::value>::type
__advance(_Ip& __val, _Diff __diff)
{
    std::advance(__val, __diff);
}

// This helper is required to correctly detect difference type for both integral types and iterators
template <typename _Ip, typename = void>
struct __difference;

template <typename _Ip>
struct __difference<_Ip, typename std::enable_if<std::is_integral<_Ip>::value>::type>
{
    // Define the type similar to C++20's incrementable_traits
    using __type = typename std::make_signed<decltype(std::declval<_Ip>() - std::declval<_Ip>())>::type;
};

template <typename _Ip>
struct __difference<_Ip, typename std::enable_if<!std::is_integral<_Ip>::value>::type>
{
    using __type = typename __pstl::__internal::__iterator_traits<_Ip>::difference_type;
};

// This type is used as a stride value when it's known that stride == 1 at compile time(the case of for_loop and for_loop_n).
// Based on that we can use a compile-time dispatching to choose a function overload without stride-related runtime overhead.
struct __single_stride_type
{
};

template <typename _Ip, typename _Sp>
typename __difference<_Ip>::__type
__calculate_input_sequence_length(const _Ip __first, const _Ip __last, const _Sp __stride)
{
    assert(__stride != 0);

    return (__stride > 0) ? ((__last - __first + (__stride - 1)) / __stride)
                          : ((__first - __last - (__stride + 1)) / -__stride);
}

template <typename _Ip>
typename __difference<_Ip>::__type
__calculate_input_sequence_length(const _Ip __first, const _Ip __last, __single_stride_type)
{
    return __last - __first;
}

// A tag for compiler to distinguish between copy and variadic argument constructors.
struct __reduction_pack_tag
{
};

// A wrapper class to store all the reduction and induction objects.
template <typename... _Ts>
class __reduction_pack
{
    // No matter how the Ts objects are provided(lvalue or rvalue) we need to store copies of them,
    // to avoid modification of the original ones.
    std::tuple<typename std::remove_cv<typename std::remove_reference<_Ts>::type>::type...> __objects_;

    template <typename _Fp, typename _Ip, typename _Position, std::size_t... _Is>
    void
    __apply_func_impl(_Fp&& __f, _Ip __current, _Position __p, __pstl::__internal::index_sequence<_Is...>)
    {
        std::forward<_Fp>(__f)(__current, std::get<_Is>(__objects_).__get_induction_or_reduction_value(__p)...);
    }

    template <std::size_t... _Is>
    void
    __combine_impl(const __reduction_pack& __other, __pstl::__internal::index_sequence<_Is...>)
    {
        (void)std::initializer_list<int>{
            0, ((void)std::get<_Is>(__objects_).__combine(std::get<_Is>(__other.__objects_)), 0)...};
    }

    template <typename _RangeSize, std::size_t... _Is>
    void
    __finalize_impl(const _RangeSize __n, __pstl::__internal::index_sequence<_Is...>)
    {
        (void)std::initializer_list<int>{0, ((void)std::get<_Is>(__objects_).__finalize(__n), 0)...};
    }

  public:
    template <typename... _Args>
    __reduction_pack(__reduction_pack_tag, _Args&&... __args) : __objects_(std::make_tuple(__args...))
    {
    }

    __reduction_pack(const __reduction_pack&) = default;
    __reduction_pack&
    operator=(__reduction_pack&& __other)
    {
        __objects_ = std::move(__other.__objects_);
        return *this;
    }

    void
    __combine(const __reduction_pack& __other)
    {
        __combine_impl(__other, __pstl::__internal::make_index_sequence<sizeof...(_Ts)>{});
    }

    template <typename _Fp, typename _Ip, typename _Position>
    void
    __apply_func(_Fp&& __f, _Ip __current, _Position __p)
    {
        __apply_func_impl(std::forward<_Fp>(__f), __current, __p,
                          __pstl::__internal::make_index_sequence<sizeof...(_Ts)>{});
    }

    template <typename _RangeSize>
    void
    __finalize(const _RangeSize __n)
    {
        __finalize_impl(__n, __pstl::__internal::make_index_sequence<sizeof...(_Ts)>{});
    }
};

// Sequenced version of for_loop_n
template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&&, _Ip __first, _Size __n, _Function __f, __single_stride_type,
                     /*vector=*/std::false_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    for (_Size __i = 0; __i < __n; ++__i, ++__first)
        __pack.__apply_func(__f, __first, __i);

    __pack.__finalize(__n);
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename _Sp, typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&&, _Ip __first, _Size __n, _Function __f, _Sp __stride,
                     /*vector=*/std::false_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    // Simple loop from 0 to __n is not suitable here as we need to ensure that __first is always
    // <= than the end iterator, even if it's not dereferenced. Some implementation might place
    // validation checks to enforce this invariant.
    if (__n > 0)
    {
        for (_Size __i = 0; __i < __n - 1; ++__i, __pstl::__internal::__advance(__first, __stride))
        {
            __pack.__apply_func(__f, __first, __i);
        }

        __pack.__apply_func(__f, __first, __n - 1);
    }

    __pack.__finalize(__n);
}

// Helper structure which helps us to detect whether type I can be randomly accessed(incremented/decremented by an arbitrary value)
template <typename _Ip, typename = void>
struct __is_random_access_or_integral : std::false_type
{
};

template <typename _Ip>
struct __is_random_access_or_integral<_Ip, typename std::enable_if<std::is_integral<_Ip>::value>::type> : std::true_type
{
};

template <typename _Ip>
struct __is_random_access_or_integral<
    _Ip, typename std::enable_if<__pstl::__internal::__is_random_access_iterator<_Ip>::value>::type> : std::true_type
{
};

// Sequenced version of for_loop for RAI and integral types
template <typename _ExecutionPolicy, typename _Ip, typename _Function, typename _Sp, typename... _Rest>
typename std::enable_if<__is_random_access_or_integral<_Ip>::value>::type
__pattern_for_loop(_ExecutionPolicy&& __exec, _Ip __first, _Ip __last, _Function __f, _Sp __stride,
                   /*vector=*/std::false_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __pstl::__internal::__pattern_for_loop_n(
        std::forward<_ExecutionPolicy>(__exec), __first,
        __pstl::__internal::__calculate_input_sequence_length(__first, __last, __stride), __f, __stride,
        std::false_type{}, std::false_type{}, std::forward<_Rest>(__rest)...);
}

template <typename _Ip, typename _Function, typename _Sp, typename _Pack, typename _IndexType>
typename std::enable_if<std::is_same<typename __pstl::__internal::__iterator_traits<_Ip>::iterator_category,
                                     std::bidirectional_iterator_tag>::value,
                        _IndexType>::type
__execute_loop_strided(_Ip __first, _Ip __last, _Function __f, _Sp __stride, _Pack& __pack, _IndexType) noexcept
{
    _IndexType __ordinal_position = 0;

    // __stride == 1 is handled separately as it doesn't require a check for i % stride inside a loop
    assert(__stride != 1);

    if (__stride > 0)
    {
        for (_IndexType __i = 0; __first != __last; ++__first, ++__i)
        {
            if (__i % __stride == 0)
            {
                __pack.__apply_func(__f, __first, __ordinal_position);
                ++__ordinal_position;
            }
        }
    }
    else
    {
        for (_IndexType __i = 0; __first != __last; --__first, ++__i)
        {
            if (__i % __stride == 0)
            {
                __pack.__apply_func(__f, __first, __ordinal_position);
                ++__ordinal_position;
            }
        }
    }

    return __ordinal_position;
}

template <typename _Ip, typename _Function, typename _Sp, typename _Pack, typename _IndexType>
typename std::enable_if<std::is_same<typename __pstl::__internal::__iterator_traits<_Ip>::iterator_category,
                                     std::forward_iterator_tag>::value ||
                            std::is_same<typename __pstl::__internal::__iterator_traits<_Ip>::iterator_category,
                                         std::input_iterator_tag>::value,
                        _IndexType>::type
__execute_loop_strided(_Ip __first, _Ip __last, _Function __f, _Sp __stride, _Pack& __pack, _IndexType) noexcept
{
    _IndexType __ordinal_position = 0;

    assert(__stride > 0);

    for (_IndexType __i = 0; __first != __last; ++__first, ++__i)
    {
        if (__i % __stride == 0)
        {
            __pack.__apply_func(__f, __first, __ordinal_position);
            ++__ordinal_position;
        }
    }

    return __ordinal_position;
}

// Sequenced version of for_loop for non-RAI and non-integral types
template <typename _ExecutionPolicy, typename _Ip, typename _Function, typename... _Rest>
typename std::enable_if<!__is_random_access_or_integral<_Ip>::value>::type
__pattern_for_loop(_ExecutionPolicy&&, _Ip __first, _Ip __last, _Function __f, __single_stride_type,
                   /*vector=*/std::false_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    // Make sure that our index type is able to hold all the possible values
    using __index_type = typename __difference<_Ip>::__type;
    __index_type __ordinal_position = 0;

    // Avoid check for i % stride on each iteration for the most common case.
    for (; __first != __last; ++__first, ++__ordinal_position)
        __pack.__apply_func(__f, __first, __ordinal_position);

    __pack.__finalize(__ordinal_position);
}

template <typename _ExecutionPolicy, typename _Ip, typename _Function, typename _Sp, typename... _Rest>
typename std::enable_if<!__is_random_access_or_integral<_Ip>::value>::type
__pattern_for_loop(_ExecutionPolicy&&, _Ip __first, _Ip __last, _Function __f, _Sp __stride,
                   /*vector=*/std::false_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    // Make sure that our index type is able to hold all the possible values
    using __index_type = typename __difference<_Ip>::__type;
    __index_type __ordinal_position = 0;

    if (__stride == 1)
    {
        // Avoid check for i % stride on each iteration for the most common case.
        for (; __first != __last; ++__first, ++__ordinal_position)
            __pack.__apply_func(__f, __first, __ordinal_position);
    }
    else
    {
        __ordinal_position =
            __pstl::__internal::__execute_loop_strided(__first, __last, __f, __stride, __pack,
                                                       // Only passed to deduce the type for internal counter
                                                       __index_type{});
    }

    __pack.__finalize(__ordinal_position);
}

// Vectorized version of for_loop_n
template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&& exec, _Ip __first, _Size __n, _Function __f, __single_stride_type,
                     /*vector=*/std::true_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    __pstl::__internal::__brick_walk1(
        __n, [&__pack, __first, __f](_Size __idx) { __pack.__apply_func(__f, __first + __idx, __idx); },
        std::true_type{});

    __pack.__finalize(__n);
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename _Sp, typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&& exec, _Ip __first, _Size __n, _Function __f, _Sp __stride,
                     /*vector=*/std::true_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __reduction_pack<_Rest...> __pack(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    __pstl::__internal::__brick_walk1(
        __n,
        [&__pack, __first, __f, __stride](_Size __idx) { __pack.__apply_func(__f, __first + __idx * __stride, __idx); },
        std::true_type{});

    __pack.__finalize(__n);
}

// Vectorized version of for_loop
template <typename _ExecutionPolicy, typename _Ip, typename _Function, typename _Sp, typename... _Rest>
void
__pattern_for_loop(_ExecutionPolicy&& __exec, _Ip __first, _Ip __last, _Function __f, _Sp __stride,
                   /*vector=*/std::true_type, /*parallel=*/std::false_type, _Rest&&... __rest) noexcept
{
    __pstl::__internal::__pattern_for_loop_n(
        std::forward<_ExecutionPolicy>(__exec), __first,
        __pstl::__internal::__calculate_input_sequence_length(__first, __last, __stride), __f, __stride,
        std::true_type{}, std::false_type{}, std::forward<_Rest>(__rest)...);
}

#if __PSTL_USE_PAR_POLICIES
// Parallel version of for_loop_n

// TODO: Using parallel_reduce when we don't have a reduction object in the pack might be ineffective,
// perhaps it's better to check for presence of reduction object and call parallel_for routine instead.
// TODO: need to add a static_assert for match between rest and f's arguments, currently there is a lot
// of unclear error in cast of mismatch.
template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename _IsVector,
          typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&& __exec, _Ip __first, _Size __n, _Function __f, __single_stride_type,
                     _IsVector __is_vector, /*parallel=*/std::true_type, _Rest&&... __rest)
{
    using __pack_type = __reduction_pack<_Rest...>;

    // Create an identity pack object, operations are done on copies of it.
    const __pack_type __identity(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    __pstl::__internal::__except_handler([&]() {
        return __par_backend::__parallel_reduce(std::forward<_ExecutionPolicy>(__exec), _Size(0), __n, __identity,
                                                [__is_vector, __first, __f](_Size __i, _Size __j, __pack_type __value) {
                                                    const auto __subseq_start = __first + __i;
                                                    const auto __length = __j - __i;

                                                    __pstl::__internal::__brick_walk1(
                                                        __length,
                                                        [&__value, __f, __i, __subseq_start](_Size __idx) {
                                                            __value.__apply_func(__f, __subseq_start + __idx,
                                                                                 __i + __idx);
                                                        },
                                                        __is_vector);

                                                    return __value;
                                                },
                                                [](__pack_type __lhs, const __pack_type& __rhs) {
                                                    __lhs.__combine(__rhs);
                                                    return __lhs;
                                                })
            .__finalize(__n);
    });
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Function, typename _Sp, typename _IsVector,
          typename... _Rest>
void
__pattern_for_loop_n(_ExecutionPolicy&& __exec, _Ip __first, _Size __n, _Function __f, _Sp __stride,
                     _IsVector __is_vector, /*parallel=*/std::true_type, _Rest&&... __rest)
{
    using __pack_type = __reduction_pack<_Rest...>;

    // Create an identity pack object, operations are done on copies of it.
    const __pack_type __identity(__reduction_pack_tag(), std::forward<_Rest>(__rest)...);

    __pstl::__internal::__except_handler([&]() {
        return __par_backend::__parallel_reduce(
                   std::forward<_ExecutionPolicy>(__exec), _Size(0), __n, __identity,
                   [__is_vector, __first, __f, __stride](_Size __i, _Size __j, __pack_type __value) {
                       const auto __subseq_start = __first + __i * __stride;
                       const auto __length = __j - __i;

                       __pstl::__internal::__brick_walk1(__length,
                                                         [&__value, __f, __i, __subseq_start, __stride](_Size __idx) {
                                                             __value.__apply_func(
                                                                 __f, __subseq_start + __idx * __stride, __i + __idx);
                                                         },
                                                         __is_vector);

                       return __value;
                   },
                   [](__pack_type __lhs, const __pack_type& __rhs) {
                       __lhs.__combine(__rhs);
                       return __lhs;
                   })
            .__finalize(__n);
    });
}

template <typename _ExecutionPolicy, typename _Ip, typename _Function, typename _Sp, typename _IsVector,
          typename... _Rest>
void
__pattern_for_loop(_ExecutionPolicy&& __exec, _Ip __first, _Ip __last, _Function __f, _Sp __stride,
                   _IsVector __is_vector,
                   /*parallel=*/std::true_type, _Rest&&... __rest)
{
    __pstl::__internal::__pattern_for_loop_n(
        std::forward<_ExecutionPolicy>(__exec), __first,
        __pstl::__internal::__calculate_input_sequence_length(__first, __last, __stride), __f, __stride, __is_vector,
        std::true_type{}, std::forward<_Rest>(__rest)...);
}
#endif // __PSTL_USE_PAR_POLICIES

// Helper structure to split code functions for integral and iterator types so the return
// value can be successfully deduced.
template <typename _Ip, typename = void>
struct __use_par_vec_helper;

template <typename _Ip>
struct __use_par_vec_helper<_Ip, typename std::enable_if<std::is_integral<_Ip>::value>::type>
{
    template <typename _ExecutionPolicy>
    static constexpr auto
    __use_vector(_ExecutionPolicy&& __exec) -> decltype(__exec.__allow_vector())
    {
        return __exec.__allow_vector();
    }

    template <typename _ExecutionPolicy>
    static constexpr auto
    __use_parallel(_ExecutionPolicy&& __exec) -> decltype(__exec.__allow_parallel())
    {
        return __exec.__allow_parallel();
    }
};

template <typename _Ip>
struct __use_par_vec_helper<_Ip, typename std::enable_if<!std::is_integral<_Ip>::value>::type>
{
    template <typename _ExecutionPolicy>
    static constexpr auto
    __use_vector(_ExecutionPolicy&& __exec) -> decltype(
        __pstl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _Ip>(std::forward<_ExecutionPolicy>(__exec)))
    {
        return __pstl::__internal::__is_vectorization_preferred<_ExecutionPolicy, _Ip>(
            std::forward<_ExecutionPolicy>(__exec));
    }

    template <typename _ExecutionPolicy>
    static constexpr auto
    __use_parallel(_ExecutionPolicy&& __exec)
        -> decltype(__pstl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _Ip>(
            std::forward<_ExecutionPolicy>(__exec)))
    {
        return __pstl::__internal::__is_parallelization_preferred<_ExecutionPolicy, _Ip>(
            std::forward<_ExecutionPolicy>(__exec));
    }
};

// Special versions for for_loop: handles both iterators and integral types(treated as random access iterators)
template <typename _ExecutionPolicy, typename _Ip>
auto
__use_vectorization(_ExecutionPolicy&& __exec)
    -> decltype(__use_par_vec_helper<_Ip>::__use_vector(std::forward<_ExecutionPolicy>(__exec)))
{
    return __use_par_vec_helper<_Ip>::__use_vector(std::forward<_ExecutionPolicy>(__exec));
}

template <typename _ExecutionPolicy, typename _Ip>
auto
__use_parallelization(_ExecutionPolicy&& __exec)
    -> decltype(__use_par_vec_helper<_Ip>::__use_parallel(std::forward<_ExecutionPolicy>(__exec)))
{
    return __use_par_vec_helper<_Ip>::__use_parallel(std::forward<_ExecutionPolicy>(__exec));
}

// Helper functions to extract to separate a Callable object from the pack of reductions and inductions
template <typename _ExecutionPolicy, typename _Ip, typename _Fp, typename _Sp, typename... _Rest, std::size_t... _Is>
void
__for_loop_impl(_ExecutionPolicy&& __exec, _Ip __start, _Ip __finish, _Fp&& __f, _Sp __stride,
                std::tuple<_Rest...>&& __t, __pstl::__internal::index_sequence<_Is...>)
{
    __pstl::__internal::__pattern_for_loop(std::forward<_ExecutionPolicy>(__exec), __start, __finish, __f, __stride,
                                           __pstl::__internal::__use_vectorization<_ExecutionPolicy, _Ip>(__exec),
                                           __pstl::__internal::__use_parallelization<_ExecutionPolicy, _Ip>(__exec),
                                           std::get<_Is>(std::move(__t))...);
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Fp, typename _Sp, typename... _Rest,
          std::size_t... _Is>
void
__for_loop_n_impl(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Fp&& __f, _Sp __stride,
                  std::tuple<_Rest...>&& __t, __pstl::__internal::index_sequence<_Is...>)
{
    __pstl::__internal::__pattern_for_loop_n(std::forward<_ExecutionPolicy>(__exec), __start, __n, __f, __stride,
                                             __pstl::__internal::__use_vectorization<_ExecutionPolicy, _Ip>(__exec),
                                             __pstl::__internal::__use_parallelization<_ExecutionPolicy, _Ip>(__exec),
                                             std::get<_Is>(std::move(__t))...);
}

template <typename _ExecutionPolicy, typename _Ip, typename _Sp, typename... _Rest>
void
__for_loop_repack(_ExecutionPolicy&& __exec, _Ip __start, _Ip __finish, _Sp __stride, std::tuple<_Rest...>&& __t)
{
    // Extract a callable object from the parameter pack and put it before the other elements
    __pstl::__internal::__for_loop_impl(std::forward<_ExecutionPolicy>(__exec), __start, __finish,
                                        std::get<sizeof...(_Rest) - 1>(__t), __stride, std::move(__t),
                                        __pstl::__internal::make_index_sequence<sizeof...(_Rest) - 1>());
}

template <typename _ExecutionPolicy, typename _Ip, typename _Size, typename _Sp, typename... _Rest>
void
__for_loop_repack_n(_ExecutionPolicy&& __exec, _Ip __start, _Size __n, _Sp __stride, std::tuple<_Rest...>&& __t)
{
    // Extract a callable object from the parameter pack and put it before the other elements
    __pstl::__internal::__for_loop_n_impl(std::forward<_ExecutionPolicy>(__exec), __start, __n,
                                          std::get<sizeof...(_Rest) - 1>(__t), __stride, std::move(__t),
                                          __pstl::__internal::make_index_sequence<sizeof...(_Rest) - 1>());
}

} // namespace __internal
} // namespace __pstl

#endif /* __PSTL_experimental_for_loop_impl_H */
