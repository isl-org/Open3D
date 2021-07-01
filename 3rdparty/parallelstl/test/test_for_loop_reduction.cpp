// -*- C++ -*-
//===-- test_for_loop_reduction.cpp ---------------------------------------===//
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

#include <type_traits>
#include <utility>

#include "pstl/execution"
#include "pstl/experimental/algorithm"

#include "utils.h"

using namespace TestUtils;

template <typename Policy, typename Iterator, typename Size>
void
test_body_reduction(Policy&& exec, Iterator first, Iterator last, Iterator /* expected_first */,
                    Iterator /* expected_last */, Size n)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    static_assert(std::is_arithmetic<T>::value, "Currently the testcase only works with arithmetic types");

    // Init with different arbitrary values on each iteration
    const T var1_init = n % 11;
    const T var2_init = n % 97;

    T var1 = var1_init;
    T var2 = var2_init;

    std::experimental::for_loop(std::forward<Policy>(exec), first, last,
                                std::experimental::reduction(var1, T(0), std::plus<T>{}),
                                std::experimental::reduction(var2, T(var2_init), __pstl::__internal::__pstl_min{}),
                                [](Iterator iter, T& acc1, T& acc2) {
                                    acc1 += *iter;
                                    acc2 = std::min(acc2, *iter);
                                });

    T var1_exp = var1_init;
    T var2_exp = var2_init;

    for (auto iter = first; iter != last; ++iter)
    {
        var1_exp += *iter;
        var2_exp = std::min(var2_exp, *iter);
    }

    EXPECT_TRUE(var1 == var1_exp, "wrong result of reduction 1");
    EXPECT_TRUE(var2 == var2_exp, "wrong result of reduction 2");
}

struct test_body
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n)
    {
        test_body_reduction(std::forward<Policy>(exec), first, last, expected_first, expected_last, n);
    }
};

template <typename T>
void
test()
{
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout(n, [](long int k) { return T(k % 5 != 1 ? 3 * k - 7 : 0); });
        Sequence<T> expected = inout;
        invoke_on_all_policies(test_body(), inout.begin(), inout.end(), expected.begin(), expected.end(), inout.size());
    }
}

struct test_body_predefined
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator /*expected_first*/, Iterator /*expected_last*/,
               Size n)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        static_assert(std::is_arithmetic<T>::value, "Currently the testcase only works with arithmetic types");

        // Initialize with arbitrary values
        T plus_var = 10, plus_exp = 10;
        T mult_var = 4, mult_exp = 4;
        T min_var = 15, min_exp = 15;
        T max_var = 5, max_exp = 5;

        std::experimental::for_loop(
            std::forward<Policy>(exec), first, last, std::experimental::reduction_plus(plus_var),
            std::experimental::reduction_multiplies(mult_var), std::experimental::reduction_min(min_var),
            std::experimental::reduction_max(max_var),
            [](Iterator iter, T& plus_acc, T& mult_acc, T& min_acc, T& max_acc) {
                plus_acc += *iter;
                mult_acc *= *iter;
                min_acc = std::min(min_acc, *iter);
                max_acc = std::max(max_acc, *iter);
            });

        for (auto iter = first; iter != last; ++iter)
        {
            plus_exp += *iter;
            mult_exp *= *iter;
            min_exp = std::min(min_exp, *iter);
            max_exp = std::max(max_exp, *iter);
        }

        EXPECT_TRUE(plus_var == plus_exp, "wrong result of reduction_plus");
        EXPECT_TRUE(mult_var == mult_exp, "wrong result of reduction_multiplies");
        EXPECT_TRUE(min_var == min_exp, "wrong result of reduction_min");
        EXPECT_TRUE(max_var == max_exp, "wrong result of reduction_max");
    }
};

struct test_body_predefined_bits
{
    template <typename Policy, typename Iterator, typename Size>
    typename std::enable_if<!std::is_floating_point<typename std::iterator_traits<Iterator>::value_type>::value,
                            void>::type
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator /*expected_first*/, Iterator /*expected_last*/,
               Size n)
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        static_assert(std::is_arithmetic<T>::value, "Currently the testcase only works with arithmetic types");

        // Initialize with arbitrary values
        T bit_or_var = 10, bit_or_exp = 10;
        T bit_xor_var = 4, bit_xor_exp = 4;
        T bit_and_var = 15, bit_and_exp = 15;

        std::experimental::for_loop(
            std::forward<Policy>(exec), first, last, std::experimental::reduction_bit_or(bit_or_var),
            std::experimental::reduction_bit_and(bit_and_var), std::experimental::reduction_bit_xor(bit_xor_var),
            [](Iterator iter, T& bit_or_acc, T& bit_and_acc, T& bit_xor_acc) {
                bit_or_acc |= *iter;
                bit_and_acc &= *iter;
                bit_xor_acc ^= *iter;
            });

        for (auto iter = first; iter != last; ++iter)
        {
            bit_or_exp |= *iter;
            bit_and_exp &= *iter;
            bit_xor_exp ^= *iter;
        }

        EXPECT_TRUE(bit_or_exp == bit_or_var, "wrong result of reduction_bit_or");
        EXPECT_TRUE(bit_and_exp == bit_and_var, "wrong result of reduction_bit_and");
        EXPECT_TRUE(bit_xor_exp == bit_xor_var, "wrong result of reduction_bit_xor");
    }

    template <typename Policy, typename Iterator, typename Size>
    typename std::enable_if<std::is_floating_point<typename std::iterator_traits<Iterator>::value_type>::value,
                            void>::type
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator /*expected_first*/, Iterator /*expected_last*/,
               Size n)
    {
        // no-op for floats
    }
};

template <typename T>
void
test_predefined(std::initializer_list<T> init_list)
{
    // Just arbitrary numbers
    Sequence<T> inout = init_list;
    Sequence<T> expected = inout;
    invoke_on_all_policies(test_body_predefined(), inout.begin(), inout.end(), expected.begin(), expected.end(),
                           inout.size());
    invoke_on_all_policies(test_body_predefined_bits(), inout.begin(), inout.end(), expected.begin(), expected.end(),
                           inout.size());
}

void
test_predef()
{
    // Test with arbitrary values
    test_predefined({1, 20, -14, 0, -100, 150});
    test_predefined(std::initializer_list<int>{});
    test_predefined({10, 20});
    test_predefined({1.f, 20.f, -14.f, 0.f, -100.f, 150.f});
}

int32_t
main()
{
    test<int32_t>();
    test<float64_t>();

    test_predef();

    std::cout << done() << std::endl;
    return 0;
}
