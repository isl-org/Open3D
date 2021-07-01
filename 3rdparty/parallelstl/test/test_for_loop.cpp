// -*- C++ -*-
//===-- test_for_loop.cpp -------------------------------------------------===//
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

#include "pstl/execution"
#include "pstl/algorithm"
#include "pstl/experimental/algorithm"

#include "utils.h"

using namespace TestUtils;

template <typename Type>
struct Gen
{
    Type
    operator()(std::size_t k)
    {
        return Type(k % 5 != 1 ? 3 * k - 7 : 0);
    };
};

template <typename T>
struct Flip
{
    int32_t val;
    Flip(int32_t y) : val(y) {}
    T
    operator()(T& x) const
    {
        return x = val - x;
    }
};

template <typename Policy, typename Iterator, typename Size>
void
test_body_for_loop(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last,
                   Size n)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

    auto flip = Flip<T>(1);

    std::experimental::for_loop(exec, first, last, [&flip](Iterator iter) { flip(*iter); });

    std::for_each(expected_first, expected_last, flip);
    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop");

    std::experimental::for_loop_n(exec, first, n, [&flip](Iterator iter) { flip(*iter); });
    std::for_each_n(pstl::execution::seq, expected_first, n, flip);
    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_n");
}

template <typename Policy, typename Iterator, typename Size>
void
test_body_for_loop_integral(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                            Iterator expected_last, Size n)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

    auto flip = Flip<T>(1);

    std::experimental::for_loop(exec, Size(0), n, [&flip, first](Size idx) {
        auto iter = first;
        std::advance(iter, idx);
        flip(*iter);
    });

    std::for_each(expected_first, expected_last, flip);
    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop with integral");

    std::experimental::for_loop_n(exec, Size(0), n, [&flip, first](Size idx) {
        auto iter = first;
        std::advance(iter, idx);
        flip(*iter);
    });

    std::for_each_n(pstl::execution::seq, expected_first, n, flip);
    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_n with integral");
}

template <typename Policy, typename Iterator, typename Size>
void
test_body_for_loop_strided(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                           Iterator expected_last, Size n, long loop_stride)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

#ifdef __PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN
    if (isReverse<Iterator>::value)
        return;
#endif

    auto flip = Flip<T>(1);

    char single_stride = loop_stride > 0 ? 1 : -1;

    std::experimental::for_loop_strided(exec, first, last, loop_stride, [&flip](Iterator iter) { flip(*iter); });

    typename std::make_signed<Size>::type idx = 0;
    for (auto iter = expected_first; iter != expected_last; std::advance(iter, single_stride), ++idx)
    {
        if (idx % loop_stride != 0)
            continue;

        flip(*iter);
    }

    if (loop_stride > 0)
        EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_strided");
    else
        // EXPECT_EQ_N use ++ for iterators, for negative stride our last points to the beginning
        // of the container, so just use it here.
        EXPECT_EQ_N(expected_last, last, n, "wrong effect from for_loop_strided");
}

template <typename Policy, typename Iterator, typename Size>
void
test_body_for_loop_strided_n(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                             Iterator expected_last, Size n, size_t loop_stride)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

    auto flip = Flip<T>(1);

    auto num_iters = n / loop_stride + !!(n % loop_stride);

    std::experimental::for_loop_n_strided(exec, first, num_iters, loop_stride, [&flip](Iterator iter) { flip(*iter); });

    size_t idx = 0;
    for (auto iter = expected_first; iter != expected_last; ++iter, ++idx)
    {
        if (idx % loop_stride != 0)
            continue;

        flip(*iter);
    }

    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_strided");
}

template <typename Policy, typename Iterator, typename Size>
void
test_body_for_loop_strided_integral(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                                    Iterator expected_last, Size n, size_t loop_stride)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

    auto flip = Flip<T>(1);

    std::experimental::for_loop_strided(exec, Size(0), n, loop_stride, [&flip, first](Size idx) {
        auto iter = first;
        std::advance(iter, idx);
        flip(*iter);
    });

    size_t idx = 0;
    for (auto iter = expected_first; iter != expected_last; ++iter, ++idx)
    {
        if (idx % loop_stride != 0)
            continue;

        flip(*iter);
    }

    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_strided");
}

// Test for for_loop_n_strided, it works for both positive and negative stride values.
template <typename Policy, typename Iterator, typename Size, typename S>
void
test_body_for_loop_strided_n_integral(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                                      Iterator expected_last, Size n, S loop_stride)
{
    using T = typename std::iterator_traits<Iterator>::value_type;

#ifdef __PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN
    if (isReverse<Iterator>::value)
        return;
#endif

    auto flip = Flip<T>(1);

    auto loop_stride_abs = std::abs(loop_stride);

    auto num_iters = n / loop_stride_abs + !!(n % loop_stride_abs);

    // Iterate over sequence of numbers starting from 0,
    // update the elements of first-last sequence at std::abs(idx) positions,
    // this works for both positive and negative values of idx and there is no
    // need to care about which base iterator to use: simply use first for both cases.
    std::experimental::for_loop_n_strided(exec, S(0), S(num_iters), loop_stride, [&flip, first](S idx) {
        auto iter = first;
        std::advance(iter, std::abs(idx));
        flip(*iter);
    });

    size_t idx = 0;
    for (auto iter = expected_first; iter != expected_last; ++iter, ++idx)
    {
        if (idx % loop_stride_abs != 0)
            continue;

        flip(*iter);
    }

    EXPECT_EQ_N(expected_first, first, n, "wrong effect from for_loop_strided");
}

struct test_for_loop_impl
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n)
    {
        test_body_for_loop(std::forward<Policy>(exec), first, last, expected_first, expected_last, n);
        test_body_for_loop_integral(std::forward<Policy>(exec), first, last, expected_first, expected_last, n);
    }
};

template <typename Policy, typename Iterator, typename Size, typename S>
typename std::enable_if<
    !std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::forward_iterator_tag>::value,
    void>::type
test_body_for_loop_strided_neg(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                               Iterator expected_last, Size n, S loop_stride)
{
    assert(loop_stride < 0);

    using Ssize = typename std::make_signed<Size>::type;

    // Test negative stride value with non-forward iterators on range (first - 1, first)
    auto new_first = first;
    std::advance(new_first, std::max(Ssize(0), Ssize(n) - 1));

    auto new_last = first;

    auto new_expected_first = expected_first;
    std::advance(new_expected_first, std::max(Ssize(0), Ssize(n) - 1));

    auto new_expected_last = expected_first;

    test_body_for_loop_strided(std::forward<Policy>(exec), new_first, new_last, new_expected_first, new_expected_last,
                               n > 0 ? n - 1 : 0, loop_stride);
}

template <typename Policy, typename Iterator, typename Size, typename S>
typename std::enable_if<
    std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::forward_iterator_tag>::value,
    void>::type
test_body_for_loop_strided_neg(Policy&& exec, Iterator first, Iterator last, Iterator expected_first,
                               Iterator expected_last, Size n, S loop_stride)
{
    // no-op for forward iterators. As it's not possible to iterate backwards.
}

struct test_for_loop_strided_impl
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n,
               size_t stride)
    {
        test_body_for_loop_strided(std::forward<Policy>(exec), first, last, expected_first, expected_last, n, stride);
        test_body_for_loop_strided_n(std::forward<Policy>(exec), first, last, expected_first, expected_last, n, stride);

        test_body_for_loop_strided_integral(std::forward<Policy>(exec), first, last, expected_first, expected_last, n,
                                            stride);

        test_body_for_loop_strided_n_integral(std::forward<Policy>(exec), first, last, expected_first, expected_last, n,
                                              (long)stride);

        // Additionally check negative stride with integral and iterator sequence.
        test_body_for_loop_strided_n_integral(std::forward<Policy>(exec), first, last, expected_first, expected_last, n,
                                              -(long)stride);
        test_body_for_loop_strided_neg(std::forward<Policy>(exec), first, last, expected_first, expected_last, n,
                                       -(long)stride);
    }
};

template <typename T>
void
test_for_loop()
{
    for (size_t n = 0; n <= 10000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout(n, Gen<T>());
        Sequence<T> expected = inout;

        invoke_on_all_policies(test_for_loop_impl(), inout.begin(), inout.end(), expected.begin(), expected.end(),
                               inout.size());
    }
}

template <typename T>
void
test_for_loop_strided()
{
    for (size_t n = 0; n <= 10000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout(n, Gen<T>());
        Sequence<T> expected = inout;

        std::vector<size_t> strides = {1, 2, 10, n > 1 ? n - 1 : 1, n > 0 ? n : 1, n + 1};
        for (size_t stride : strides)
        {
            invoke_on_all_policies(test_for_loop_strided_impl(), inout.begin(), inout.end(), expected.begin(),
                                   expected.end(), inout.size(), stride);
        }
    }
}

template <typename T>
void
test()
{
    test_for_loop<T>();
    test_for_loop_strided<T>();
}

int32_t
main()
{
    test<int32_t>();
    test<float64_t>();

    std::cout << done() << std::endl;
    return 0;
}
