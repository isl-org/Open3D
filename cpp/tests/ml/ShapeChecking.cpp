// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/ml/ShapeChecking.h"

#include "tests/Tests.h"

using namespace open3d::ml::op_util;

TEST(ShapeChecking, CheckShapeReturnValues) {
    auto status_ok = CheckShape({10, 20}, 10, 20);
    EXPECT_TRUE(std::get<0>(status_ok));
    EXPECT_TRUE(std::get<1>(status_ok).empty());

    auto status_err = CheckShape({10, 20}, 1, 2);
    EXPECT_FALSE(std::get<0>(status_err));
    EXPECT_FALSE(std::get<1>(status_err).empty());
}

TEST(ShapeChecking, CheckShapeRankZero) {
    Dim h("h");
    Dim w("w");
    EXPECT_FALSE(std::get<0>(CheckShape({}, h, w)));
}

TEST(ShapeChecking, CheckShapeCommonUseCases) {
    Dim h("h");
    Dim w("w");
    EXPECT_FALSE(h.constant());
    EXPECT_FALSE(w.constant());
    EXPECT_TRUE(std::get<0>(CheckShape({30}, h)));
    EXPECT_TRUE(std::get<0>(CheckShape({30, 40}, h, w)));
    EXPECT_FALSE(std::get<0>(CheckShape({3, 40}, h, w)));
    EXPECT_TRUE(h.constant());
    EXPECT_EQ(h.value(), 30);
    EXPECT_TRUE(w.constant());
    EXPECT_EQ(w.value(), 40);

    EXPECT_TRUE(std::get<0>(CheckShape({30}, Dim())));
    EXPECT_TRUE(std::get<0>(CheckShape({30, 40}, Dim(), w)));

    EXPECT_TRUE(
            std::get<0>(CheckShape<CSOpt::IGNORE_FIRST_DIMS>({20, 30, 40}, w)));
    EXPECT_TRUE(std::get<0>(
            CheckShape<CSOpt::IGNORE_FIRST_DIMS>({20, 30, 40}, Dim(), w)));
    EXPECT_TRUE(
            std::get<0>(CheckShape<CSOpt::IGNORE_LAST_DIMS>({40, 30, 20}, w)));
    EXPECT_TRUE(std::get<0>(
            CheckShape<CSOpt::IGNORE_LAST_DIMS>({40, 30, 20}, w, Dim())));

    EXPECT_TRUE(std::get<0>(
            CheckShape<CSOpt::COMBINE_FIRST_DIMS>({2, 20, 123}, w, Dim())));
    EXPECT_TRUE(std::get<0>(
            CheckShape<CSOpt::COMBINE_LAST_DIMS>({123, 2, 20}, Dim(), w)));

    EXPECT_TRUE(std::get<0>(CheckShape({40}, w || 5)));
    EXPECT_TRUE(std::get<0>(CheckShape({5}, w || 5)));
    EXPECT_TRUE(std::get<0>(CheckShape({10, 40}, Dim(), w || 5)));
    EXPECT_TRUE(std::get<0>(CheckShape({10, 5}, Dim(), w || 5)));

    {
        Dim d("d");
        EXPECT_TRUE(std::get<0>(CheckShape({10, 123}, Dim(), d || 5)));
        EXPECT_TRUE(d.constant());
        EXPECT_EQ(d.value(), 123);
    }

    {
        Dim d("d");
        EXPECT_TRUE(std::get<0>(CheckShape({10, 123}, Dim(), d + 23)));
        EXPECT_TRUE(d.constant());
        EXPECT_EQ(d.value(), 100);
    }

    {
        Dim d("d");
        EXPECT_TRUE(
                std::get<0>(CheckShape({10, UnknownValue()}, Dim(), d + 23)));
        EXPECT_FALSE(d.constant());
    }
    {
        Dim d("d");
        EXPECT_TRUE(std::get<0>(CheckShape({UnknownValue()}, d + 23)));
        EXPECT_FALSE(d.constant());
    }

    EXPECT_FALSE(
            std::get<0>(CheckShape({40, 40, 0, 40, 40}, 40, 40, w, 40, 40)));
    EXPECT_TRUE(std::get<0>(CheckShape({1, 1}, 1, (h + (w * 2)) / 3 || 1)));
}

TEST(ShapeChecking, CheckShapeExceptions) {
    Dim d("d");
    EXPECT_THROW(CheckShape({1, 2, 3}, 1, 2, 3 * d), std::runtime_error);

    EXPECT_THROW(CheckShape({1, 2, 3}, 1, 2, d + d), std::runtime_error);

    Dim d2;
    EXPECT_THROW(CheckShape({1, 2, 3}, 1, 2, d2 + d), std::runtime_error);
}
