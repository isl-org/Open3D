// -*- C++ -*-
//===-- test_for_loop_induction.cpp ---------------------------------------===//
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
test_body_induction(Policy&& exec, Iterator first, Iterator last, Iterator /* expected_first */,
                    Iterator /* expected_last */, Size n)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    static_assert(std::is_arithmetic<T>::value, "Currently the testcase only works with arithmetic types");

    // Init with different arbitrary values on each iteration
    const T ind_init = n % 97;
    const size_t stride = n % 97;

    T lval_ind = ind_init;
    const T clval_ind = ind_init;
    T rval_ind = ind_init;

    // Values for induction with stride
    T lval_sind = ind_init;
    const T clval_sind = ind_init;
    T rval_sind = ind_init;

    std::experimental::for_loop(
        std::forward<Policy>(exec), Size(0), n, std::experimental::induction(lval_ind),
        std::experimental::induction(clval_ind), std::experimental::induction(std::move(rval_ind)),
        std::experimental::induction(lval_sind, stride), std::experimental::induction(clval_sind, stride),
        std::experimental::induction(std::move(rval_sind), stride),
        [ind_init, stride](Size idx, T ind1, T ind2, T ind3, T sind1, T sind2, T sind3) {
            EXPECT_TRUE(ind1 == ind2, "wrong induction value");
            EXPECT_TRUE(ind1 == ind3, "wrong induction value");
            EXPECT_TRUE(ind1 == (ind_init + idx), "wrong induction value");

            EXPECT_TRUE(sind1 == sind2, "wrong induction value");
            EXPECT_TRUE(sind1 == sind3, "wrong induction value");
            EXPECT_TRUE(sind1 == (ind_init + idx * stride), "wrong induction value");
        });

    EXPECT_TRUE(lval_ind == n, "wrong result of induction");
    EXPECT_TRUE(rval_ind == ind_init, "wrong result of induction");
    EXPECT_TRUE(clval_ind == ind_init, "wrong result of induction");

    EXPECT_TRUE(lval_sind == n * stride, "wrong result of induction");
    EXPECT_TRUE(rval_sind == ind_init, "wrong result of induction");
    EXPECT_TRUE(clval_sind == ind_init, "wrong result of induction");
}

template <typename Policy, typename Iterator, typename Size>
void
test_body_induction_strided(Policy&& exec, Iterator first, Iterator last, Iterator /* expected_first */,
                            Iterator /* expected_last */, Size n)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    static_assert(std::is_arithmetic<T>::value, "Currently the testcase only works with arithmetic types");

    for (int loop_stride : {-1, 1, 10, -5})
    {
        // Init with different arbitrary values on each iteration
        const T ind_init = n % 97;
        T lval_ind = ind_init;

        using Ssize = typename std::make_signed<Size>::type;

        std::experimental::for_loop_n_strided(std::forward<Policy>(exec), Ssize(0), Ssize(n), loop_stride,
                                              std::experimental::induction(lval_ind),
                                              [ind_init, loop_stride](Ssize val, T ind) {
                                                  // We have either 0, stride, 2 * stride, .. or 0, -|stride|, 2 * -|stride|
                                                  // sequences, so current index can be simply calculated with this.
                                                  auto real_idx = val / loop_stride;

                                                  EXPECT_TRUE(ind == (ind_init + real_idx), "wrong induction value");
                                              });

        EXPECT_TRUE(lval_ind == n, "wrong result of induction");

        // Negative strides are not allowed with forward iterators
        if (loop_stride < 0 &&
            std::is_same<typename std::iterator_traits<Iterator>::iterator_category, std::forward_iterator_tag>::value)
            continue;

        auto new_first = first;
        auto new_last = last;

        // In case of a negative stride we reverse first and last but since we cannot get end-like iterator before first in some cases
        // just use 'first' for this purpose. Meaning that the resulting sequence shrinks to n - 1 element.
        if (loop_stride < 0 && n > 0)
        {
            std::advance(new_first, n - 1);
            new_last = first;
        }

        // Re-init the value after for_loop_n_strided
        lval_ind = ind_init;

        std::experimental::for_loop_strided(
            std::forward<Policy>(exec), new_first, new_last, loop_stride, std::experimental::induction(lval_ind),
            [ind_init, loop_stride, new_first](Iterator iter, T ind) {
                auto dist = (loop_stride > 0) ? std::distance(new_first, iter) : std::distance(iter, new_first);
                auto real_idx = dist / std::abs(loop_stride);

                EXPECT_TRUE(ind == (ind_init + real_idx), "wrong induction value");
            });

        if (loop_stride < 0 && n > 0)
        {
            EXPECT_TRUE(lval_ind == ((n - 1) / std::abs(loop_stride) + !!((n - 1) % std::abs(loop_stride))),
                        "wrong result of induction");
        }
        else
        {
            EXPECT_TRUE(lval_ind == (n / loop_stride + !!(n % loop_stride)), "wrong result of induction");
        }
    }
}

struct test_body
{
    template <typename Policy, typename Iterator, typename Size>
    void
    operator()(Policy&& exec, Iterator first, Iterator last, Iterator expected_first, Iterator expected_last, Size n)
    {
        test_body_induction(std::forward<Policy>(exec), first, last, expected_first, expected_last, n);
        test_body_induction_strided(std::forward<Policy>(exec), first, last, expected_first, expected_last, n);
    }
};

template <typename T>
void
test()
{
    for (size_t n = 0; n <= 10000; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> inout(n, [](long int k) { return T(k % 5 != 1 ? 3 * k - 7 : 0); });
        Sequence<T> expected = inout;
        invoke_on_all_policies(test_body(), inout.begin(), inout.end(), expected.begin(), expected.end(), inout.size());
    }
}

int32_t
main()
{
    test<int32_t>();
    test<float64_t>();

    std::cout << done() << std::endl;
    return 0;
}
